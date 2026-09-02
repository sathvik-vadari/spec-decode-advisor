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
    """Relative cost of one draft pass to one target pass.

    A round costs one target forward pass over k+1 positions plus k draft
    passes. `draft_cost_ratio` is the per-pass cost of the draft model relative
    to the target, and is the term that decides whether deeper drafting pays.
    """

    draft_cost_ratio: float = 0.25

    def round_cost(self, k: int) -> float:
        return 1.0 + self.draft_cost_ratio * k

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
