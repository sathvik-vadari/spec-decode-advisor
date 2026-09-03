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
from typing import Sequence

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

    Second, `draft_cost_ratio` measured standalone understates the in-loop cost.
    Timing the draft model by itself gave 0.468; fitting the observed speedup
    curve gave 0.618. A draft pass inside the speculative loop is more expensive
    than the same model generating alone, because of the per-token sync the loop
    forces.

    So calibrate with `fit`, which infers both terms from measured speedups,
    rather than by timing the two models separately.
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
