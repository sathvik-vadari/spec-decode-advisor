"""The arithmetic behind whether speculative decoding pays.

The quantity to reason about is `p`, the probability that the target accepts a
single draft token. It is a property of the *pair* of models and the workload,
and it is roughly invariant in the draft depth `k` -- which is what makes it
useful. The naive alternative, accepted/proposed, is not invariant: raising `k`
adds deep positions that are unlikely to be reached, so the ratio falls even
when nothing about the models changed.

Verification is sequential: the target accepts draft tokens until the first
rejection, then emits one corrected token of its own. So with acceptance
probability `p` and depth `k`, the expected number of accepted draft tokens is

    E[j] = sum_{i=1..k} p^i = p(1 - p^k) / (1 - p)

and each round emits E[j] + 1 tokens for one target forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def expected_accepted(p: float, k: int) -> float:
    """Expected accepted draft tokens per round."""
    if k <= 0:
        return 0.0
    if p >= 1.0:
        return float(k)
    return p * (1.0 - p**k) / (1.0 - p)


def expected_tokens_per_round(p: float, k: int) -> float:
    """Tokens emitted per target forward pass, including the target's own."""
    return expected_accepted(p, k) + 1.0


def acceptance_from_measurement(mean_accepted: float, k: int) -> float:
    """Recover `p` from the measured mean accepted-per-round, by bisection.

    E[j] is strictly increasing in p, so bisection is exact enough and cannot
    fail to converge.
    """
    if k <= 0 or mean_accepted <= 0:
        return 0.0
    if mean_accepted >= k:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if expected_accepted(mid, k) < mean_accepted:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass(frozen=True, slots=True)
class CostModel:
    """Cost of one speculative round, in units of one plain target pass.

        round_cost(k) = fixed_cost + draft_cost_ratio * k

    The obvious calibration -- time each model generating on its own and take
    the ratio -- is wrong twice over, and experiment 01 measured both errors.

    First, `fixed_cost` is not 1.0. A round pays for cache bookkeeping the model
    has no term for: `_rewind_cache` trims both KV caches back to the accepted
    prefix every round, whether or not anything was rejected. Fitted at 1.37 on
    the Qwen2.5 1.5B/0.5B pair against a modelled 1.0.

    Second, `draft_cost_ratio` is not the draft's cost alone. It is the cost of
    one more draft *position*, and a position costs two things: a draft pass, and
    one more token for the target to verify. Verification is only free if the
    target pass is bandwidth-bound with compute to spare. On an M4 it is not:
    experiment 03 timed the 7B pass at 48 ms for one token and +15 ms per extra
    token, so each verified position costs ~0.3 of a target pass on top of the
    draft pass. That is why the slope fitted from speedups (0.38, 0.54, 0.70 for
    0.5B, 1.5B, 3B drafts) sits a near-constant 0.25-0.29 above each draft's
    standalone pass cost (0.13, 0.25, 0.45). The premium belongs to the target,
    and it is a hardware constant: on an H100 it should be close to zero.

    So calibrate with `fit`, which prices whatever a position actually costs,
    rather than by timing the two models separately -- and read the slope as
    draft + verification, not draft.
    """

    draft_cost_ratio: float = 0.25
    fixed_cost: float = 1.0

    def round_cost(self, k: int) -> float:
        return self.fixed_cost + self.draft_cost_ratio * k

    def speedup(self, p: float, k: int) -> float:
        """Predicted speedup over plain autoregressive decoding.

        Plain decoding emits one token per target pass, so the baseline cost per
        token is 1.0. Below 1.0 here means speculation is a net loss.
        """
        return expected_tokens_per_round(p, k) / self.round_cost(k)

    def best_depth(self, p: float, max_k: int = 12) -> tuple[int, float]:
        """The draft depth maximising predicted speedup, and that speedup.

        k=0 means do not speculate.
        """
        options = [(0, 1.0)] + [(k, self.speedup(p, k)) for k in range(1, max_k + 1)]
        return max(options, key=lambda kv: kv[1])

    def breakeven_acceptance(self, k: int, tol: float = 1e-6) -> float:
        """Lowest `p` at which depth `k` beats not speculating at all."""
        lo, hi = 0.0, 1.0
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if self.speedup(mid, k) < 1.0:
                lo = mid
            else:
                hi = mid
        return hi


def fit(observations: Sequence[tuple[int, float, float]]) -> CostModel:
    """Recover a cost model from measured speedups.

    Each observation is `(k, p, measured_speedup)`. Since

        speedup = expected_tokens_per_round(p, k) / round_cost(k)

    every observation pins one round cost exactly, and round_cost is linear in
    k, so two or more observations at different depths determine both terms by
    least squares. This is the calibration that works: it prices whatever the
    engine actually does per round, including the parts not in the model.
    """
    if len(observations) < 2:
        raise ValueError("need at least two depths to separate the two terms")
    ks = np.array([k for k, _, _ in observations], dtype=float)
    costs = np.array(
        [expected_tokens_per_round(p, k) / s for k, p, s in observations],
        dtype=float,
    )
    slope, intercept = np.polyfit(ks, costs, 1)
    return CostModel(draft_cost_ratio=float(slope), fixed_cost=float(intercept))


# ---- choosing between drafts ----------------------------------------------
# A bigger draft accepts more and costs more per position. Neither effect is
# linear in parameter count and they pull in opposite directions, so whether the
# trade pays is not monotone in draft size and cannot be settled by a rule of
# thumb. It is a search over measured candidates: each candidate carries its own
# acceptance and its own fitted cost model, because both change with the draft.


@dataclass(frozen=True, slots=True)
class Candidate:
    """One draft model, as measured against the target on a workload."""

    name: str
    p: float
    cost: CostModel


@dataclass(frozen=True, slots=True)
class Recommendation:
    """`draft is None` means do not speculate at all.

    `ranking` holds every candidate's best (name, k, speedup), best first, so a
    caller can see how close the runner-up was rather than only who won.
    """

    draft: str | None
    k: int
    speedup: float
    ranking: tuple[tuple[str, int, float], ...]


def recommend(candidates: Sequence[Candidate], max_k: int = 12) -> Recommendation:
    """The (draft, depth) with the highest predicted speedup, or none.

    A candidate whose best depth is 0 never beats plain decoding, so if the
    top-ranked candidate is at k=0 the answer is not to speculate.
    """
    ranking = []
    for c in candidates:
        k, s = c.cost.best_depth(c.p, max_k)
        ranking.append((c.name, k, s))
    ranking.sort(key=lambda t: t[2], reverse=True)
    if not ranking or ranking[0][1] == 0:
        return Recommendation(draft=None, k=0, speedup=1.0, ranking=tuple(ranking))
    name, k, s = ranking[0]
    return Recommendation(draft=name, k=k, speedup=s, ranking=tuple(ranking))


# ---- calibrating from direct timings ----------------------------------------
# `fit` needs speedups at two or more depths, which means the whole sweep.
# Experiment 03 showed the slope is a draft pass plus a verified token, both of
# which are single-pass timings (see `calibration.py`). That leaves only the
# fixed cost to pin, and one speculative depth does it.


def from_components(c_draft: float, c_verify: float, fixed_cost: float = 1.0) -> CostModel:
    """A cost model whose slope is built from directly timed parts."""
    return CostModel(draft_cost_ratio=c_draft + c_verify, fixed_cost=fixed_cost)


def fit_fixed_cost(k: int, p: float, measured_speedup: float, slope: float) -> CostModel:
    """Pin the fixed cost from a single speculative depth, given the slope.

        speedup = E[tokens](p, k) / (fixed + slope * k)
        fixed   = E[tokens](p, k) / speedup - slope * k

    One observation, one unknown. Everything the model cannot see per round --
    cache rewinds, host-side work, memory residency -- lands here.
    """
    if k <= 0:
        raise ValueError("need a speculative depth (k >= 1) to observe a round")
    if measured_speedup <= 0:
        raise ValueError("speedup must be positive")
    fixed = expected_tokens_per_round(p, k) / measured_speedup - slope * k
    return CostModel(draft_cost_ratio=slope, fixed_cost=fixed)


# ---- a cost model with no fitted parameters -----------------------------------
# Experiment 06 measured round costs of 1.14, 1.23, 1.43, 1.78, 2.61 target
# passes at k = 1, 2, 3, 4, 6 on AC power. That is convex, and a straight line
# through it puts the intercept at 0.67 -- below one target pass, which is
# impossible. The curvature is the target's own pass cost over k+1 tokens: flat
# while the extra positions hide under the weight read, then a ramp, then a
# staircase of GEMM tiles (experiment 05). Subtract the *measured* pass-cost
# curve and k draft passes from those round costs and what remains is zero to
# within 0.1. There is no fixed cost. There never was one; it was the
# linearisation error of a convex curve.


@dataclass(frozen=True, slots=True)
class MeasuredCostModel:
    """Round cost from two direct timings and nothing fitted.

        round_cost(k) = k * c_draft + V(k + 1)

    `pass_curve` is V: the target's forward pass over T tokens relative to its
    pass over one token, so V(1) == 1. Measured with `calibration.time_passes`
    at T = 1 .. max_k + 1; between measured points it is interpolated, beyond
    the last one extrapolated along the final segment. `c_draft` is the draft's
    one-token pass over the target's.

    Same interface as `CostModel`, so the recommender takes either.
    """

    c_draft: float
    pass_curve: dict[int, float]
    overhead: float = 0.0
    """Per-round host work the passes do not contain -- the accept loop, the
    cache rewind, copying tokens back -- in target passes. Experiment 07 found
    0.01-0.09 of a pass on AC, a few milliseconds, largest at k=1. Pin it from
    the one speculative run the tool makes anyway (`pin_overhead`); it is the
    only number here not from a direct timing, and it is small."""

    def __post_init__(self) -> None:
        if 1 not in self.pass_curve or len(self.pass_curve) < 2:
            raise ValueError("pass_curve needs V(1) and at least one more point")
        if abs(self.pass_curve[1] - 1.0) > 1e-9:
            raise ValueError("pass_curve must be relative: V(1) == 1")

    @classmethod
    def from_latency(cls, c_draft: float, latency: Mapping[int, float]) -> "MeasuredCostModel":
        """From raw pass latencies (any unit); normalises by the one-token pass."""
        one = latency[1]
        return cls(c_draft=c_draft, pass_curve={int(t): v / one for t, v in latency.items()})

    def verify_cost(self, tokens: int) -> float:
        """V(tokens): what a pass over `tokens` positions costs, in one-token passes."""
        ts = sorted(self.pass_curve)
        if tokens in self.pass_curve:
            return self.pass_curve[tokens]
        if tokens < ts[0]:
            return self.pass_curve[ts[0]]
        if tokens > ts[-1]:
            a, b = ts[-2], ts[-1]
            slope = (self.pass_curve[b] - self.pass_curve[a]) / (b - a)
            return self.pass_curve[b] + slope * (tokens - b)
        lo = max(t for t in ts if t < tokens)
        hi = min(t for t in ts if t > tokens)
        f = (tokens - lo) / (hi - lo)
        return self.pass_curve[lo] * (1 - f) + self.pass_curve[hi] * f

    def round_cost(self, k: int) -> float:
        return self.overhead + k * self.c_draft + self.verify_cost(k + 1)

    def speedup(self, p: float, k: int) -> float:
        return expected_tokens_per_round(p, k) / self.round_cost(k)

    def best_depth(self, p: float, max_k: int = 12) -> tuple[int, float]:
        options = [(0, 1.0)] + [(k, self.speedup(p, k)) for k in range(1, max_k + 1)]
        return max(options, key=lambda kv: kv[1])

    def pin_overhead(self, k: int, p: float, measured_speedup: float) -> "MeasuredCostModel":
        """The same model with `overhead` set so it reproduces one measured depth.

        Everything else is timed; this is the one residual. If it comes out
        well below zero the timings overstate the loop (a busy host, a throttled
        GPU); well above 0.3 and something per round is not in the model."""
        if k <= 0 or measured_speedup <= 0:
            raise ValueError("need a speculative depth and a positive speedup")
        implied = expected_tokens_per_round(p, k) / measured_speedup
        h = implied - (k * self.c_draft + self.verify_cost(k + 1))
        return MeasuredCostModel(c_draft=self.c_draft, pass_curve=self.pass_curve, overhead=h)

    def breakeven_acceptance(self, k: int, tol: float = 1e-6) -> float:
        lo, hi = 0.0, 1.0
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if self.speedup(mid, k) < 1.0:
                lo = mid
            else:
                hi = mid
        return hi
