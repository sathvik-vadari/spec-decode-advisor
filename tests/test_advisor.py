"""The analysis half of the advisor, on synthetic runs with known answers."""

import json

import pytest

from spec_decode_advisor.advisor import analyse_draft, build_report, render, to_json
from spec_decode_advisor.calibration import PassLatency
from spec_decode_advisor.harness import RunResult
from spec_decode_advisor.model import CostModel, expected_tokens_per_round

TARGET = PassLatency({1: 0.050, 2: 0.0625, 3: 0.075, 5: 0.100, 9: 0.150})   # c_verify = 0.25
DRAFT = PassLatency({1: 0.005, 2: 0.005})                                    # c_draft = 0.10


def _run(pid, dom, k, tps, rounds=(), text="same"):
    n = 100
    return RunResult(prompt_id=pid, domain=dom, k=k, n_tokens=n, wall_s=n / tps, text=text,
                     rounds=tuple(rounds), dropped_rounds=0, finish_reason="stop")


def _synthetic(p_by_domain, fixed_cost, n_per_domain=3):
    """Runs whose k=1 speedup is exactly what the cost model says for the domain's p."""
    slope = 0.10 + 0.25
    cm = CostModel(draft_cost_ratio=slope, fixed_cost=fixed_cost)
    runs = []
    for dom, p in p_by_domain.items():
        for i in range(n_per_domain):
            pid = f"{dom}{i}"
            base_tps = 20.0
            runs.append(_run(pid, dom, 0, base_tps))
            # a round accepts its one draft token with probability p: emit rounds
            # whose empirical acceptance equals p exactly
            n_rounds = 100
            accepted = [1] * int(round(p * n_rounds)) + [0] * (n_rounds - int(round(p * n_rounds)))
            runs.append(_run(pid, dom, 1, base_tps * cm.speedup(p, 1), rounds=accepted))
    return runs


def test_recovers_acceptance_and_fixed_cost_from_k1_alone():
    runs = _synthetic({"code": 0.8, "prose": 0.6}, fixed_cost=1.15)
    rep = analyse_draft("d", runs, DRAFT, TARGET)
    assert rep.c_verify == pytest.approx(0.25)
    assert rep.c_draft == pytest.approx(0.10)
    assert rep.p == pytest.approx(0.7, abs=1e-9)
    assert rep.by_domain["code"].p == pytest.approx(0.8)
    assert rep.by_domain["prose"].p == pytest.approx(0.6)
    # under the synthetic linear pass curve, overhead = fixed_cost - 1; the domains
    # were generated at different speedups so the pooled value is close, not exact
    assert rep.overhead == pytest.approx(0.15, abs=0.03)
    assert rep.lossless_mismatches == 0
    assert rep.n_prompts == 6


def test_single_domain_overhead_is_exact():
    runs = _synthetic({"workload": 0.75}, fixed_cost=1.2)
    rep = analyse_draft("d", runs, DRAFT, TARGET)
    assert rep.overhead == pytest.approx(0.2)
    assert rep.cost.speedup(0.75, 1) == pytest.approx(rep.measured_speedup_k1)
    assert rep.pass_curve[5] == pytest.approx(2.0)          # 100 ms over 50 ms


def test_lossless_mismatches_are_counted():
    runs = _synthetic({"w": 0.7}, fixed_cost=1.1)
    runs[1] = RunResult(**{**runs[1].__dict__, "text": "different"}) if hasattr(runs[1], "__dict__") else runs[1]
    # RunResult is slots-only; rebuild the one spec run with different text
    r = runs[1]
    runs[1] = RunResult(r.prompt_id, r.domain, r.k, r.n_tokens, r.wall_s, "different", r.rounds, r.dropped_rounds, r.finish_reason)
    assert analyse_draft("d", runs, DRAFT, TARGET).lossless_mismatches == 1


def test_needs_a_baseline_for_every_speculative_run():
    runs = _synthetic({"w": 0.7}, fixed_cost=1.1)
    runs = [r for r in runs if not (r.k == 0 and r.prompt_id == "w0")]
    with pytest.raises(ValueError, match="without a baseline"):
        analyse_draft("d", runs, DRAFT, TARGET)


def test_report_picks_the_cheaper_draft_when_acceptance_is_similar():
    cheap = analyse_draft("small", _synthetic({"w": 0.72}, 1.1), PassLatency({1: 0.005, 2: 0.005}), TARGET)
    dear = analyse_draft("large", _synthetic({"w": 0.76}, 1.2), PassLatency({1: 0.020, 2: 0.020}), TARGET)
    rep = build_report("t", TARGET, [cheap, dear])
    assert rep.overall.draft == "small"
    assert rep.by_domain["w"].draft == "small"
    assert [n for n, _, _ in rep.overall.ranking] == ["small", "large"]


def test_report_declines_when_nothing_pays():
    d = analyse_draft("d", _synthetic({"w": 0.2}, 1.3), DRAFT, TARGET)
    rep = build_report("t", TARGET, [d])
    assert rep.overall.draft is None
    assert "do not speculate" in render(rep)


def test_render_and_json_carry_the_recommendation_and_the_constants():
    d = analyse_draft("small", _synthetic({"code": 0.8, "prose": 0.6}, 1.1), DRAFT, TARGET)
    rep = build_report("t", TARGET, [d], max_k=6)
    text = render(rep)
    assert "recommendation: small at k=" in text
    assert "0.25 of a pass" in text
    assert "by domain" in text and "code" in text and "prose" in text
    j = json.loads(json.dumps(to_json(rep)))          # must be serialisable
    assert j["overall"]["draft"] == "small"
    assert j["drafts"][0]["by_domain"]["code"]["p"] == pytest.approx(0.8)
    assert j["max_k"] == 6
    assert j["drafts"][0]["overhead"] == pytest.approx(0.1, abs=0.03)
    assert j["drafts"][0]["pass_curve"]["9"] == pytest.approx(3.0)


def test_target_drift_across_draft_sessions_is_reported():
    d1 = analyse_draft("a", _synthetic({"w": 0.7}, 1.1), DRAFT, TARGET)
    slow = PassLatency({n: t * 1.5 for n, t in TARGET.seconds.items()})     # same ratios, 50% slower
    d2 = analyse_draft("b", _synthetic({"w": 0.7}, 1.1), DRAFT, slow)
    rep = build_report("t", [TARGET, slow], [d1, d2])
    assert rep.target_drift == pytest.approx(0.5)
    assert "WARNING" in render(rep) and "50%" in render(rep)
    assert to_json(rep)["target_pass_ms_by_draft"] == {"a": pytest.approx(50.0), "b": pytest.approx(75.0)}


def test_no_drift_no_warning_and_one_timing_is_accepted_for_many_drafts():
    d1 = analyse_draft("a", _synthetic({"w": 0.7}, 1.1), DRAFT, TARGET)
    d2 = analyse_draft("b", _synthetic({"w": 0.7}, 1.1), DRAFT, TARGET)
    rep = build_report("t", TARGET, [d1, d2])
    assert rep.target_drift == 0.0
    assert "WARNING" not in render(rep)
    with pytest.raises(ValueError):
        build_report("t", [TARGET, TARGET, TARGET], [d1, d2])


def test_a_negative_overhead_is_flagged():
    # generate the k=1 runs as if the loop were cheaper than its timed parts
    runs = _synthetic({"w": 0.75}, fixed_cost=0.70)
    d = analyse_draft("d", runs, DRAFT, TARGET)
    assert d.overhead == pytest.approx(-0.30)
    text = render(build_report("t", TARGET, [d]))
    assert "WARNING" in text and "negative per-round overhead" in text


def test_a_large_overhead_is_flagged_and_a_sane_one_is_not():
    big = analyse_draft("d", _synthetic({"w": 0.75}, fixed_cost=1.5), DRAFT, TARGET)
    assert "large per-round overhead" in render(build_report("t", TARGET, [big]))
    sane = analyse_draft("d", _synthetic({"w": 0.75}, fixed_cost=1.1), DRAFT, TARGET)
    assert "WARNING" not in render(build_report("t", TARGET, [sane]))
