"""The acceptance arithmetic. Each test pins a property the advisor relies on."""

import pytest

from spec_decode_advisor import model
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


# ---- fitted calibration -------------------------------------------------
# Timing each model on its own misprices a round: it assumes the target pass
# costs exactly 1.0 and that a draft pass costs the same inside the speculative
# loop as outside it. Experiment 01 found both false. `fit` recovers both terms
# from measured speedups instead.


def test_fixed_cost_defaults_to_one_target_pass():
    assert CostModel(0.5).round_cost(4) == 1.0 + 0.5 * 4


def test_round_cost_uses_fitted_intercept():
    m = CostModel(draft_cost_ratio=0.6, fixed_cost=1.4)
    assert m.round_cost(0) == 1.4
    assert abs(m.round_cost(3) - (1.4 + 1.8)) < 1e-12


def test_fit_recovers_a_known_cost_model():
    truth = CostModel(draft_cost_ratio=0.619, fixed_cost=1.378)
    p = 0.73
    obs = [(k, p, truth.speedup(p, k)) for k in (1, 2, 4, 8)]
    got = model.fit(obs)
    assert abs(got.draft_cost_ratio - 0.619) < 1e-6
    assert abs(got.fixed_cost - 1.378) < 1e-6


def test_fit_recovers_the_model_from_varying_p():
    # p drifts slightly with depth in real data; the fit should still hold
    truth = CostModel(draft_cost_ratio=0.5, fixed_cost=1.2)
    obs = [(k, p, truth.speedup(p, k)) for k, p in ((1, 0.72), (2, 0.73), (4, 0.74), (8, 0.75))]
    got = model.fit(obs)
    assert abs(got.draft_cost_ratio - 0.5) < 1e-6
    assert abs(got.fixed_cost - 1.2) < 1e-6


def test_fit_needs_two_depths():
    with pytest.raises(ValueError):
        model.fit([(2, 0.7, 1.1)])


def test_fitted_model_can_say_do_not_speculate():
    # the honest answer on the measured pair: overhead eats the whole win
    m = CostModel(draft_cost_ratio=0.619, fixed_cost=1.378)
    assert m.best_depth(0.805)[0] == 0
    assert m.best_depth(0.61)[0] == 0


# ---- choosing between drafts ---------------------------------------------
# A bigger draft accepts more and costs more per position, so which draft wins
# is not monotone in draft size. `recommend` searches measured candidates.

from spec_decode_advisor.model import Candidate, recommend  # noqa: E402


def test_the_cheaper_draft_wins_when_acceptance_has_saturated():
    cheap = Candidate("small", p=0.95, cost=CostModel(draft_cost_ratio=0.1))
    dear = Candidate("large", p=0.97, cost=CostModel(draft_cost_ratio=0.5))
    rec = recommend([dear, cheap])
    assert rec.draft == "small"
    assert rec.speedup > 1.0


def test_the_dearer_draft_wins_when_it_buys_enough_acceptance():
    cheap = Candidate("small", p=0.40, cost=CostModel(draft_cost_ratio=0.1))
    dear = Candidate("large", p=0.85, cost=CostModel(draft_cost_ratio=0.3))
    assert recommend([cheap, dear]).draft == "large"


def test_no_draft_is_recommended_when_none_pays():
    cands = [
        Candidate("a", p=0.15, cost=CostModel(draft_cost_ratio=0.3)),
        Candidate("b", p=0.20, cost=CostModel(draft_cost_ratio=0.6, fixed_cost=1.4)),
    ]
    rec = recommend(cands)
    assert rec.draft is None
    assert rec.k == 0
    assert rec.speedup == 1.0
    assert len(rec.ranking) == 2          # the losers are still reported


def test_ranking_is_complete_and_sorted_best_first():
    cands = [Candidate(str(i), p=0.5 + 0.1 * i, cost=CostModel(0.2)) for i in range(4)]
    rec = recommend(cands)
    speedups = [s for _, _, s in rec.ranking]
    assert speedups == sorted(speedups, reverse=True)
    assert {n for n, _, _ in rec.ranking} == {"0", "1", "2", "3"}


def test_a_single_candidate_agrees_with_best_depth():
    cm = CostModel(draft_cost_ratio=0.25, fixed_cost=1.1)
    rec = recommend([Candidate("only", p=0.8, cost=cm)])
    k, s = cm.best_depth(0.8)
    assert (rec.draft, rec.k, rec.speedup) == ("only", k, s)


def test_recommendation_survives_an_empty_candidate_list():
    rec = recommend([])
    assert rec.draft is None and rec.ranking == ()


# ---- calibrating from direct timings -------------------------------------

from spec_decode_advisor.model import fit, fit_fixed_cost, from_components  # noqa: E402


def test_components_add_up_to_the_slope():
    m = from_components(c_draft=0.10, c_verify=0.25, fixed_cost=1.1)
    assert m.draft_cost_ratio == pytest.approx(0.35)
    assert m.round_cost(2) == pytest.approx(1.1 + 0.70)


def test_one_depth_pins_the_fixed_cost_exactly_when_the_slope_is_right():
    truth = CostModel(draft_cost_ratio=0.35, fixed_cost=1.15)
    p = 0.7
    observed = truth.speedup(p, 2)
    m = fit_fixed_cost(2, p, observed, slope=0.35)
    assert m.fixed_cost == pytest.approx(1.15)
    # and it then predicts every other depth
    for k in (1, 3, 4, 8):
        assert m.speedup(p, k) == pytest.approx(truth.speedup(p, k))


def test_a_slope_that_is_too_low_inflates_the_fixed_cost():
    # the two terms trade off at the observed depth: underestimate the slope by d
    # and the fixed cost absorbs k*d, which then mispredicts other depths
    truth = CostModel(draft_cost_ratio=0.40, fixed_cost=1.10)
    p = 0.7
    m = fit_fixed_cost(2, p, truth.speedup(p, 2), slope=0.35)
    assert m.fixed_cost == pytest.approx(1.10 + 2 * 0.05)
    assert m.speedup(p, 2) == pytest.approx(truth.speedup(p, 2))   # exact where pinned
    assert m.speedup(p, 8) > truth.speedup(p, 8)                    # optimistic further out


def test_one_depth_agrees_with_the_full_fit_on_consistent_data():
    truth = CostModel(draft_cost_ratio=0.5, fixed_cost=1.3)
    p = 0.75
    obs = [(k, p, truth.speedup(p, k)) for k in (1, 2, 4)]
    full = fit(obs)
    one = fit_fixed_cost(1, p, obs[0][2], slope=full.draft_cost_ratio)
    assert one.fixed_cost == pytest.approx(full.fixed_cost)


def test_fixed_cost_needs_a_speculative_depth_and_a_positive_speedup():
    with pytest.raises(ValueError):
        fit_fixed_cost(0, 0.7, 1.0, slope=0.3)
    with pytest.raises(ValueError):
        fit_fixed_cost(2, 0.7, 0.0, slope=0.3)


# ---- the measured cost model: no fitted parameters -----------------------

from spec_decode_advisor.model import MeasuredCostModel  # noqa: E402


def test_a_flat_pass_curve_is_free_verification():
    # verifying k+1 tokens costs the same as one: round cost is one pass plus k draft passes
    m = MeasuredCostModel(c_draft=0.1, pass_curve={1: 1.0, 2: 1.0, 5: 1.0, 9: 1.0})
    for k in (1, 2, 4, 8):
        assert m.round_cost(k) == pytest.approx(1.0 + 0.1 * k)
        assert m.speedup(0.7, k) == pytest.approx(CostModel(draft_cost_ratio=0.1).speedup(0.7, k))


def test_a_linear_pass_curve_reproduces_the_linear_model():
    curve = {t: 1.0 + 0.3 * (t - 1) for t in (1, 2, 3, 5, 9)}
    m = MeasuredCostModel(c_draft=0.1, pass_curve=curve)
    lin = CostModel(draft_cost_ratio=0.4, fixed_cost=1.0)
    for k in (1, 2, 4, 8):
        assert m.round_cost(k) == pytest.approx(lin.round_cost(k))


def test_interpolates_between_and_extrapolates_beyond_measured_points():
    m = MeasuredCostModel(c_draft=0.0, pass_curve={1: 1.0, 3: 1.0, 5: 1.4})
    assert m.verify_cost(2) == pytest.approx(1.0)
    assert m.verify_cost(4) == pytest.approx(1.2)
    assert m.verify_cost(7) == pytest.approx(1.8)      # along the last segment


def test_from_latency_normalises_by_the_one_token_pass():
    m = MeasuredCostModel.from_latency(0.1, {1: 50.0, 2: 50.0, 5: 75.0})
    assert m.pass_curve == {1: 1.0, 2: 1.0, 5: 1.5}
    with pytest.raises(ValueError):
        MeasuredCostModel(c_draft=0.1, pass_curve={1: 2.0, 2: 2.0})
    with pytest.raises(ValueError):
        MeasuredCostModel(c_draft=0.1, pass_curve={2: 1.0, 3: 1.1})


def test_a_convex_curve_gives_an_interior_best_depth():
    # free to 4 tokens then steep: the sweet spot is where the free region ends
    curve = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.1, 5: 1.4, 6: 1.7, 7: 2.1, 9: 2.6}
    m = MeasuredCostModel(c_draft=0.1, pass_curve=curve)
    k, s = m.best_depth(0.7, max_k=8)
    assert k in (2, 3)
    assert s > m.speedup(0.7, 1) and s > m.speedup(0.7, 6)


def test_experiment_06_speedups_are_predicted_with_nothing_fitted():
    # committed pass-cost curve from experiment 05 (AC, idle), ms per pass over T tokens
    curve_ms = {1: 47.3, 2: 48.0, 3: 48.4, 4: 54.1, 5: 65.7, 6: 81.0, 7: 100.4}
    m = MeasuredCostModel.from_latency(c_draft=0.098, latency=curve_ms)
    measured = {1: (0.682, 1.479), 2: (0.688, 1.759), 3: (0.690, 1.741), 4: (0.688, 1.524), 6: (0.698, 1.166)}
    for k, (p, s) in measured.items():
        assert m.speedup(p, k) == pytest.approx(s, abs=0.06), k
    # and the residual "fixed cost" is nothing
    for k, (p, s) in measured.items():
        implied_round_cost = expected_tokens_per_round(p, k) / s
        assert abs(implied_round_cost - m.round_cost(k)) < 0.15, k


def test_pinning_the_overhead_reproduces_that_depth_and_moves_the_others_together():
    curve = {1: 1.0, 2: 1.0, 3: 1.02, 4: 1.15, 5: 1.4, 7: 2.1}
    base = MeasuredCostModel(c_draft=0.1, pass_curve=curve)
    truth = MeasuredCostModel(c_draft=0.1, pass_curve=curve, overhead=0.08)
    pinned = base.pin_overhead(1, 0.7, truth.speedup(0.7, 1))
    assert pinned.overhead == pytest.approx(0.08)
    for k in (2, 3, 4, 6):
        assert pinned.speedup(0.7, k) == pytest.approx(truth.speedup(0.7, k))
    with pytest.raises(ValueError):
        base.pin_overhead(0, 0.7, 1.2)
