"""Estimating acceptance from observed rounds, without assuming the model.

`model.py` assumes a single per-token acceptance probability `p` that does not
depend on how deep into the draft you are. That assumption is what lets one
measured number predict every depth, so it is the load-bearing claim of the
whole tool -- and recovering `p` by inverting the truncated geometric sum
cannot test it, because that inversion already assumes it. It fits the model to
data generated under the model and reports a number either way.

So estimate acceptance *per draft position* instead, as a hazard:

    p_j = P(accept at position j | positions 1..j-1 were all accepted)
        = #(a >= j) / #(a >= j-1)

where `a` is the number of draft tokens accepted in a round. Verification stops
at the first rejection, so position j is only observed when everything before it
was accepted -- exactly the conditioning the hazard assumes. No model, no
inversion. If p_1 ~= p_2 ~= ... ~= p_k the geometric model is earned and the
scalar `p` is meaningful. If the hazards decline in j, deeper positions really
are harder and a single number is the wrong summary.

Pooling those hazards gives the maximum-likelihood estimate in closed form.
Under the geometric model a round of depth k contributes `a` acceptance events,
plus one rejection event if a < k (a round with a == k is right-censored: it
ran out of draft, not out of agreement). The log-likelihood is

    S log p + C log(1 - p)      S = total accepted, C = rounds with a < k

which maximises at p = S / (S + C). Note what that is: accepted over accepted-
plus-observed-rejections. The naive `accepted / proposed` differs precisely by
counting positions that were never reached, which is why it collapses as k
grows while this does not.

Uncertainty is bootstrapped over *prompts*, not rounds. Rounds within one
generation are strongly autocorrelated -- a hard passage stays hard for several
consecutive rounds -- so any interval that treats rounds as independent draws
is too narrow. Prompts are the independent unit here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class PromptRounds:
    """Accepted-token counts for every usable round of one generation."""

    prompt_id: str
    domain: str
    k: int
    accepted: tuple[int, ...]

    @property
    def n_rounds(self) -> int:
        return len(self.accepted)


@dataclass(frozen=True, slots=True)
class Hazard:
    """Acceptance at one draft position, among rounds that reached it."""

    position: int
    at_risk: int
    accepted: int

    @property
    def p(self) -> float:
        return self.accepted / self.at_risk if self.at_risk else float("nan")


def pooled_acceptance(rounds: Sequence[PromptRounds]) -> float:
    """MLE of `p` under the geometric model: accepted / (accepted + rejections).

    Right-censoring matters: a round that accepted all k draft tokens observed
    no rejection, so it contributes to the numerator only.
    """
    accepted = sum(a for r in rounds for a in r.accepted)
    rejections = sum(1 for r in rounds for a in r.accepted if a < r.k)
    if accepted + rejections == 0:
        return float("nan")
    if rejections == 0:
        return 1.0
    return accepted / (accepted + rejections)


def mean_accepted(rounds: Sequence[PromptRounds]) -> float:
    counts = [a for r in rounds for a in r.accepted]
    return float(np.mean(counts)) if counts else float("nan")


def naive_ratio(rounds: Sequence[PromptRounds]) -> float:
    """`accepted / proposed` -- the estimator this project argues against.

    Kept so the two can be reported side by side. It counts draft positions that
    verification never reached as rejections, so it falls as k rises even when
    acceptance itself has not moved.
    """
    accepted = sum(a for r in rounds for a in r.accepted)
    proposed = sum(r.k for r in rounds for _ in r.accepted)
    return accepted / proposed if proposed else float("nan")


def position_hazards(rounds: Sequence[PromptRounds], k: int) -> list[Hazard]:
    """Per-position acceptance among rounds that actually reached each position."""
    counts = np.array([a for r in rounds for a in r.accepted if r.k == k], dtype=int)
    hazards = []
    for j in range(1, k + 1):
        at_risk = int(np.sum(counts >= j - 1))
        accepted = int(np.sum(counts >= j))
        hazards.append(Hazard(position=j, at_risk=at_risk, accepted=accepted))
    return hazards


def hazard_spread(rounds: Sequence[PromptRounds], k: int) -> float:
    """p_1 - p_k: how much harder the deepest draft position is than the first.

    The quantity the invariance claim lives or dies on. Zero means the geometric
    model's single `p` is the right summary.
    """
    hz = position_hazards(rounds, k)
    if not hz or hz[0].at_risk == 0 or hz[-1].at_risk == 0:
        return float("nan")
    return hz[0].p - hz[-1].p


def bootstrap_ci(
    rounds: Sequence[PromptRounds],
    statistic: Callable[[Sequence[PromptRounds]], float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap resampling whole prompts, which are the independent unit."""
    if len(rounds) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(rounds)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[i] = statistic([rounds[j] for j in idx])
    lo = float(np.nanpercentile(stats, 100 * (1 - ci) / 2))
    hi = float(np.nanpercentile(stats, 100 * (1 + ci) / 2))
    return (lo, hi)


def filter_rounds(
    rounds: Iterable[PromptRounds],
    *,
    k: int | None = None,
    domain: str | None = None,
) -> list[PromptRounds]:
    out = list(rounds)
    if k is not None:
        out = [r for r in out if r.k == k]
    if domain is not None:
        out = [r for r in out if r.domain == domain]
    return out
