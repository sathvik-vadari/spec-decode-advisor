"""The direct cost estimators, tested on synthetic latencies with known answers."""

import pytest

from spec_decode_advisor.calibration import PassLatency, draft_pass_ratio, verification_cost


def test_free_verification_has_zero_cost():
    # a pass over any number of tokens takes the same time: pure bandwidth-bound decode
    assert verification_cost({1: 0.050, 2: 0.050, 5: 0.050, 9: 0.050}) == pytest.approx(0.0)


def test_linear_growth_recovers_the_slope_in_pass_units():
    # 50 ms for one token, +12.5 ms per extra token -> 0.25 of a pass each
    lat = {n: 0.050 + 0.0125 * (n - 1) for n in (1, 2, 3, 5, 9)}
    assert verification_cost(lat) == pytest.approx(0.25)


def test_slope_is_least_squares_not_two_point():
    # one noisy point should not dominate the way an endpoint ratio would
    lat = {1: 0.050, 2: 0.0625, 3: 0.075, 5: 0.100, 9: 0.170}   # last point 20 ms high
    two_point = ((0.170 - 0.050) / 8) / 0.050
    ls = verification_cost(lat)
    assert 0.25 < ls < two_point


def test_needs_the_one_token_point_and_a_second():
    with pytest.raises(ValueError):
        verification_cost({2: 0.06, 3: 0.07})
    with pytest.raises(ValueError):
        verification_cost({1: 0.05})


def test_draft_pass_ratio_is_one_token_over_one_token():
    d = PassLatency({1: 0.005, 2: 0.006})
    t = PassLatency({1: 0.050, 2: 0.062})
    assert draft_pass_ratio(d, t) == pytest.approx(0.1)


def test_experiment_03_numbers_reproduce():
    # committed medians for the 7B target, in ms; the experiment reported mean 0.246
    lat = {1: 57.5, 2: 70.0, 3: 87.1, 5: 106.6, 9: 194.4}
    assert verification_cost(lat) == pytest.approx(0.29, abs=0.02)
