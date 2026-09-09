"""Do two direct timings and nothing fitted predict the measured speedup curve?

Experiments 01-04 modelled a round as `fixed + slope * k` and fitted both terms
from speedups. Experiment 06 broke that: on AC power the round costs at k = 1,
2, 3, 4, 6 are convex, and a straight line through them puts the intercept at
0.67 target passes, below the one pass every round must contain. Experiment 05
showed why: the target's pass over T tokens is flat for small T, then a ramp,
then a staircase. The linear model's "fixed cost" was the curvature of that
curve, absorbed into an intercept.

So the model becomes

    round_cost(k) = k * c_draft + V(k + 1)

with V the target's measured pass-cost curve and c_draft the draft's measured
pass ratio. No fitted parameters, no speculative run needed for the cost side
at all; the k=1 run is only for acceptance. This measures V and c_draft fresh
(about a minute, with the corrected warm-up), then predicts experiment 06's
five measured speedups.

Checks, fixed before running:

C1. Every predicted speedup at k in {1,2,3,4,6} is within 0.06 of the measured
    value from experiment 06. The linear fit's worst miss was 0.25.
C2. The implied residual per round, measured round cost minus the model's, is
    within 0.15 target passes at every depth. That is the claim that there is
    no fixed cost.
C4. Pin the one residual, per-round overhead, from k=1 alone and predict the
    held-out depths 2, 3, 4, 6. Within 0.06 of measured at every one. This is
    the procedure the tool will use: two timings plus the k=1 run it makes for
    acceptance anyway.

C3. The same model built from experiment 03's battery-power pass curve and
    experiment 02's battery c_draft, applied to experiment 02's battery
    speedups, is worse. Expected: the two were measured on different nights
    under different conditions, and under a capped clock the pass curve has
    no free region. Reported, not scored as pass/fail.

Conditions recorded, because they are the point.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spec_decode_advisor.calibration import time_passes  # noqa: E402
from spec_decode_advisor.model import CostModel, MeasuredCostModel, expected_tokens_per_round  # noqa: E402
from spec_decode_advisor.prompts import PROMPTS  # noqa: E402

TARGET = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DRAFT = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
TOKENS = (1, 2, 3, 4, 5, 6, 7, 9)
AC = ROOT / "results" / "06_power_state.json"
BATTERY_SWEEP = ROOT / "results" / "02_draft_comparison.json"
BATTERY_CURVE = ROOT / "results" / "03_verification_cost.json"
OUT = ROOT / "results" / "07_zero_parameter_model.json"
TOL_SPEEDUP, TOL_RESIDUAL = 0.06, 0.15


def conditions() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"lowpowermode": sh("pmset -g | awk '/lowpowermode/ {print $2}'"),
            "power": sh("pmset -g batt | head -1"),
            "battery": sh("pmset -g batt | sed -n 2p | awk '{print $3, $4}'"),
            "memory_free": sh("memory_pressure | grep -i 'system-wide'")}


def evaluate(model, by_depth: dict) -> dict:
    out = {}
    for k, d in by_depth.items():
        p, meas = d["pooled_p"], d["measured_speedup"]
        pred = model.speedup(p, int(k))
        implied = expected_tokens_per_round(p, int(k)) / meas
        out[k] = {"p": p, "measured": meas, "predicted": pred, "error": pred - meas,
                  "implied_round_cost": implied, "model_round_cost": model.round_cost(int(k)),
                  "residual": implied - model.round_cost(int(k))}
    return out


def main() -> None:
    t0 = time.perf_counter()
    cond = conditions()
    print("conditions:", json.dumps(cond), flush=True)

    target, tok = load(TARGET)
    ids = mx.array(tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": PROMPTS[6].text}], add_generation_prompt=True, tokenize=False)))
    t_lat = time_passes(target, ids, tokens=TOKENS, repeats=25)
    draft, _ = load(DRAFT)
    d_lat = time_passes(draft, ids, tokens=(1,), repeats=25)
    c_draft = d_lat.one_token / t_lat.one_token
    curve_ms = {t: s * 1e3 for t, s in t_lat.seconds.items()}
    print("target pass ms: " + "  ".join(f"T={t}: {v:.1f}" for t, v in curve_ms.items()), flush=True)
    print(f"draft pass {d_lat.one_token * 1e3:.2f} ms, c_draft {c_draft:.3f}", flush=True)

    ac = json.loads(AC.read_text())
    zero = MeasuredCostModel.from_latency(c_draft, curve_ms)
    linear = CostModel(**ac["fitted"])
    ac_zero = evaluate(zero, ac["by_depth"])
    ac_linear = evaluate(linear, ac["by_depth"])

    # C3: battery-night data, battery-night curve, different night
    bat_sweep = json.loads(BATTERY_SWEEP.read_text())["sessions"]["0.5B"]
    bat_curve = {int(t): v["median_ms"] for t, v in json.loads(BATTERY_CURVE.read_text())["latency_ms"]["7B"].items()}
    bat_c_draft = json.loads(BATTERY_CURVE.read_text())["draft_pass_ratio"]["0.5B"]
    bat_zero = evaluate(MeasuredCostModel.from_latency(bat_c_draft, bat_curve), bat_sweep["by_depth"])
    bat_linear = evaluate(CostModel(**bat_sweep["fitted"]), bat_sweep["by_depth"])

    k1 = ac["by_depth"]["1"]
    pinned = zero.pin_overhead(1, k1["pooled_p"], k1["measured_speedup"])
    ac_pinned = evaluate(pinned, {k: v for k, v in ac["by_depth"].items() if k != "1"})

    checks = {
        "C1_ac_within_tol": all(abs(v["error"]) <= TOL_SPEEDUP for v in ac_zero.values()),
        "C2_no_fixed_cost": all(abs(v["residual"]) <= TOL_RESIDUAL for v in ac_zero.values()),
        "C4_pinned_overhead_held_out": all(abs(v["error"]) <= TOL_SPEEDUP for v in ac_pinned.values()),
    }
    payload = {
        "config": {"target": TARGET, "draft": DRAFT, "tokens": list(TOKENS), "tolerances": {"speedup": TOL_SPEEDUP, "residual": TOL_RESIDUAL}},
        "conditions": cond,
        "measured_now": {"target_pass_ms": curve_ms, "draft_pass_ms": d_lat.one_token * 1e3, "c_draft": c_draft,
                         "pass_curve": zero.pass_curve},
        "ac": {"zero_parameter": ac_zero, "linear_fit": ac_linear, "linear_fit_params": ac["fitted"],
               "pinned_overhead": pinned.overhead, "pinned_held_out": ac_pinned},
        "battery": {"zero_parameter": bat_zero, "linear_fit": bat_linear, "curve_ms": bat_curve, "c_draft": bat_c_draft},
        "checks": checks,
        "elapsed_s": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print()
    print("AC (experiment 06): measured | zero-parameter model | linear fit      residual round cost (zero-param)")
    for k in ac_zero:
        z, l = ac_zero[k], ac_linear[k]
        print(f"  k={k}: {z['measured']:.3f} | {z['predicted']:.3f} ({z['error']:+.3f}) | {l['predicted']:.3f} ({l['error']:+.3f})      {z['residual']:+.3f}")
    print(f"AC, overhead pinned from k=1 ({pinned.overhead:+.3f} passes), held-out depths:")
    for k, v in ac_pinned.items():
        print(f"  k={k}: {v['measured']:.3f} | {v['predicted']:.3f} ({v['error']:+.3f})")
    print("battery (experiment 02, curve from experiment 03, different night):")
    for k in bat_zero:
        z, l = bat_zero[k], bat_linear[k]
        print(f"  k={k}: {z['measured']:.3f} | {z['predicted']:.3f} ({z['error']:+.3f}) | {l['predicted']:.3f} ({l['error']:+.3f})      {z['residual']:+.3f}")
    print()
    for c, ok in checks.items():
        print(f"{c}: {'PASS' if ok else 'FAIL'}")
    print(f"wrote {OUT}  ({payload['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
