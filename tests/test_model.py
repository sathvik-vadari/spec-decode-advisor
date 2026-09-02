"""The acceptance arithmetic. Each test pins a property the advisor relies on."""

import pytest

from spec_decode_advisor.model import (
    CostModel,
    acceptance_from_measurement,
    expected_accepted,
    expected_tokens_per_round,
)


def test_no_drafting_accepts_nothing():
    assert expected_accepted(0.9, 0) == 0.0
    assert expected_tokens_per_round(0.9, 0) == 1.0     # just the target's token


def test_a_draft_that_is_never_accepted_still_costs_a_round():
    assert expected_accepted(0.0, 4) == 0.0
    assert expected_tokens_per_round(0.0, 4) == 1.0


def test_a_perfect_draft_accepts_every_position():
    assert expected_accepted(1.0, 5) == 5.0


def test_acceptance_is_monotone_in_p_and_in_depth():
    assert expected_accepted(0.5, 4) < expected_accepted(0.9, 4)
    assert expected_accepted(0.7, 2) < expected_accepted(0.7, 6)


def test_deeper_drafting_has_diminishing_returns():
    # each extra position is reached only if all earlier ones were accepted
    gains = [expected_accepted(0.7, k + 1) - expected_accepted(0.7, k) for k in range(1, 6)]
    assert gains == sorted(gains, reverse=True)


@pytest.mark.parametrize("p", [0.2, 0.5, 0.77, 0.95])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_recovering_p_from_a_measurement_inverts_the_model(p, k):
    assert acceptance_from_measurement(expected_accepted(p, k), k) == pytest.approx(p, abs=1e-4)


def test_p_is_invariant_in_depth_but_accepted_over_proposed_is_not():
    # the reason the tool reports p: raising k adds positions unlikely to be
    # reached, so accepted/proposed falls even though the models are unchanged
    p = 0.8
    ratios = [expected_accepted(p, k) / k for k in (1, 2, 4, 8)]
    assert ratios == sorted(ratios, reverse=True)
    recovered = [acceptance_from_measurement(expected_accepted(p, k), k) for k in (1, 2, 4, 8)]
    assert all(r == pytest.approx(p, abs=1e-4) for r in recovered)


def test_speculation_loses_when_acceptance_is_too_low():
    cm = CostModel(draft_cost_ratio=0.25)
    assert cm.speedup(0.1, 4) < 1.0
    assert cm.best_depth(0.1)[0] == 0          # advises not to speculate


def test_breakeven_rises_with_depth():
    cm = CostModel(draft_cost_ratio=0.25)
    breakevens = [cm.breakeven_acceptance(k) for k in (1, 2, 4, 8)]
    assert breakevens == sorted(breakevens)


def test_a_more_expensive_draft_raises_the_bar():
    cheap = CostModel(draft_cost_ratio=0.1).breakeven_acceptance(4)
    dear = CostModel(draft_cost_ratio=0.5).breakeven_acceptance(4)
    assert dear > cheap


def test_speedup_at_breakeven_is_one():
    cm = CostModel(draft_cost_ratio=0.3)
    for k in (1, 3, 6):
        assert cm.speedup(cm.breakeven_acceptance(k), k) == pytest.approx(1.0, abs=1e-3)


def test_best_depth_beats_every_other_depth():
    cm = CostModel(draft_cost_ratio=0.25)
    k, best = cm.best_depth(0.85, max_k=12)
    assert all(cm.speedup(0.85, other) <= best + 1e-12 for other in range(1, 13))
