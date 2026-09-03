"""The estimator, and the claim that the naive ratio is the wrong one."""

from __future__ import annotations

import numpy as np

from spec_decode_advisor.acceptance import (
    PromptRounds,
    bootstrap_ci,
    hazard_spread,
    mean_accepted,
    naive_ratio,
    pooled_acceptance,
    position_hazards,
)
from spec_decode_advisor.model import acceptance_from_measurement


def _synthetic(p: float, k: int, n_prompts: int, rounds_each: int, seed: int = 0):
    """Rounds drawn from the geometric model the tool assumes.

    Acceptance is i.i.d. per position, so per-position hazards are flat by
    construction and any estimator that works must recover `p`.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_prompts):
        counts = []
        for _ in range(rounds_each):
            a = 0
            while a < k and rng.random() < p:
                a += 1
            counts.append(a)
        out.append(PromptRounds(f"p{i}", "synthetic", k, tuple(counts)))
    return out



def test_pooled_acceptance_recovers_p_at_every_depth():
    # the point of the estimator: the same p comes back regardless of k
    for k in (1, 2, 4, 8):
        rounds = _synthetic(p=0.8, k=k, n_prompts=40, rounds_each=60, seed=k)
        assert abs(pooled_acceptance(rounds) - 0.8) < 0.02, k


def test_naive_ratio_does_not_recover_p():
    # the estimator this project argues against: it falls as k rises even
    # though the generating p never moved
    ratios = []
    for k in (1, 2, 4, 8):
        rounds = _synthetic(p=0.8, k=k, n_prompts=40, rounds_each=60, seed=k)
        ratios.append(naive_ratio(rounds))
    assert ratios[0] > ratios[-1] + 0.15, ratios
    assert ratios == sorted(ratios, reverse=True), ratios


def test_pooled_matches_method_of_moments_under_the_model():
    # inverting the truncated geometric sum and the MLE should agree when the
    # model actually holds; divergence is a signal the model does not
    rounds = _synthetic(p=0.7, k=4, n_prompts=40, rounds_each=60, seed=3)
    mom = acceptance_from_measurement(mean_accepted(rounds), k=4)
    assert abs(pooled_acceptance(rounds) - mom) < 0.01


def test_pooled_acceptance_by_hand():
    # rounds of depth 2: accepted 2 (censored), 0, 1
    # accepted = 3, observed rejections = 2 (the 0 and the 1)
    rounds = [PromptRounds("a", "d", 2, (2, 0, 1))]
    assert pooled_acceptance(rounds) == 3 / 5


def test_pooled_acceptance_all_accepted_is_one():
    rounds = [PromptRounds("a", "d", 2, (2, 2))]
    assert pooled_acceptance(rounds) == 1.0


def test_position_hazards_by_hand():
    # depth 3, counts 3, 1, 0, 2
    #   j=1: at risk 4, reached 3  -> 3/4
    #   j=2: at risk 3, reached 2  -> 2/3
    #   j=3: at risk 2, reached 1  -> 1/2
    rounds = [PromptRounds("a", "d", 3, (3, 1, 0, 2))]
    hz = position_hazards(rounds, k=3)
    assert [(h.at_risk, h.accepted) for h in hz] == [(4, 3), (3, 2), (2, 1)]
    assert hz[0].p == 0.75


def test_position_hazards_flat_under_the_model():
    rounds = _synthetic(p=0.8, k=4, n_prompts=40, rounds_each=80, seed=7)
    hz = position_hazards(rounds, k=4)
    assert max(h.p for h in hz) - min(h.p for h in hz) < 0.05


def test_hazard_spread_detects_position_dependence():
    # hand-built rounds where deep positions are much harder than shallow ones
    # every round reaches position 1 and clears it; almost none clear position 3
    counts = tuple([1] * 50 + [2] * 30 + [3] * 2)
    rounds = [PromptRounds("a", "d", 3, counts)]
    assert hazard_spread(rounds, k=3) > 0.5


def test_hazard_spread_near_zero_under_the_model():
    rounds = _synthetic(p=0.75, k=6, n_prompts=40, rounds_each=80, seed=11)
    assert abs(hazard_spread(rounds, k=6)) < 0.08


def test_bootstrap_ci_brackets_the_truth():
    rounds = _synthetic(p=0.75, k=4, n_prompts=30, rounds_each=40, seed=5)
    lo, hi = bootstrap_ci(rounds, pooled_acceptance, n_resamples=500, seed=1)
    assert lo < 0.75 < hi
    assert hi - lo < 0.1


def test_bootstrap_ci_is_deterministic_given_a_seed():
    rounds = _synthetic(p=0.7, k=2, n_prompts=20, rounds_each=30, seed=2)
    a = bootstrap_ci(rounds, pooled_acceptance, n_resamples=200, seed=42)
    b = bootstrap_ci(rounds, pooled_acceptance, n_resamples=200, seed=42)
    assert a == b


def test_bootstrap_ci_needs_more_than_one_prompt():
    rounds = [PromptRounds("a", "d", 2, (1, 2))]
    lo, hi = bootstrap_ci(rounds, pooled_acceptance)
    assert np.isnan(lo) and np.isnan(hi)
