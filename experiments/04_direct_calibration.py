"""Can two 30-second timings plus one speculative depth replace the depth sweep?

Experiment 02 calibrated the cost model by fitting `fixed + slope * k` to
measured speedups at k = 1, 2, 4 -- 360 generations, 49 minutes. Experiment 03
showed the slope decomposes into a draft pass and a verified token, both single
forward-pass timings. If those timings plus a speculative run at *one* depth
predict the speedups at the other depths, the advisor's cost calibration drops
from a sweep to two timings and one run, and the sweep is only needed for what
it is actually good for: measuring acceptance.

This is an analysis of committed results, not a new measurement. The inputs
were on disk before this was written; the pass/fail thresholds below were fixed
before the numbers were computed.

Checks:

C1. The direct slope (c_draft + c_verify from experiment 03) is within 0.10 of
    the slope experiment 02 fitted from speedups, for every draft. If not, the
    decomposition in finding 9 is incomplete.

C2. Held-out prediction. Pin the fixed cost from k=1 alone using the direct
    slope, then predict pooled speedup at k=2 and k=4. Within 0.05 absolute of
    the measured value, for every draft. This is the check that matters: k=2
    and k=4 were never used to build the predicting model.

C3. The fixed cost pinned from k=1 is within 0.10 of the three-depth fit's.

C4. The zero-run model -- direct slope, fixed cost assumed 1.0, no speculative
    generation at all -- misses C2's threshold somewhere. If it does not, the
    speculative run is unnecessary too. Expected to fail (finding 10 put the
    fixed cost at 1.11-1.27), which is why one depth is kept.

Per-domain predictions are reported as well, using each domain's own `p` with
the pooled model, since that is how the advisor will use it.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spec_decode_advisor.calibration import verification_cost  # noqa: E402
from spec_decode_advisor.model import (  # noqa: E402
    CostModel,
    fit_fixed_cost,
    from_components,
)
from spec_decode_advisor.prompts import DOMAINS  # noqa: E402

SWEEP = ROOT / "results" / "02_draft_comparison.json"
TIMINGS = ROOT / "results" / "03_verification_cost.json"
OUT = ROOT / "results" / "04_direct_calibration.json"
PIN_K = 1
HELD_OUT = ("2", "4")
TOL_SLOPE, TOL_SPEEDUP, TOL_FIXED = 0.10, 0.05, 0.10


def main() -> None:
    sweep = json.loads(SWEEP.read_text())["sessions"]
    timings = json.loads(TIMINGS.read_text())
    lat = {m: {int(n): v["median_ms"] for n, v in d.items()} for m, d in timings["latency_ms"].items()}
    c_verify = verification_cost(lat["7B"])

    per_draft = {}
    for name, s in sweep.items():
        c_draft = lat[name][1] / lat["7B"][1]
        direct_slope = c_draft + c_verify
        fitted = CostModel(**s["fitted"])
        bd = s["by_depth"]
        pin = bd[str(PIN_K)]
        one_depth = fit_fixed_cost(PIN_K, pin["pooled_p"], pin["measured_speedup"], direct_slope)
        zero_run = from_components(c_draft, c_verify, fixed_cost=1.0)

        pooled = {}
        for k in HELD_OUT:
            p_k, meas = bd[k]["pooled_p"], bd[k]["measured_speedup"]
            pooled[k] = {
                "measured": meas,
                "one_depth": one_depth.speedup(p_k, int(k)),
                "zero_run": zero_run.speedup(p_k, int(k)),
                "full_fit": fitted.speedup(p_k, int(k)),
            }
        domains = {}
        for dom in DOMAINS:
            dd = s["by_domain"][dom]["by_depth"]
            domains[dom] = {
                k: {"measured": dd[k]["measured_speedup"],
                    "one_depth": one_depth.speedup(dd[k]["pooled_p"], int(k)),
                    "full_fit": fitted.speedup(dd[k]["pooled_p"], int(k))}
                for k in HELD_OUT
            }
        dom_err = [v[k]["one_depth"] - v[k]["measured"] for v in domains.values() for k in HELD_OUT]
        per_draft[name] = {
            "c_draft_direct": c_draft,
            "c_verify_direct": c_verify,
            "slope_direct": direct_slope,
            "slope_fitted": fitted.draft_cost_ratio,
            "fixed_one_depth": one_depth.fixed_cost,
            "fixed_full_fit": fitted.fixed_cost,
            "pooled_held_out": pooled,
            "by_domain_held_out": domains,
            "domain_error_one_depth": {"mean": statistics.mean(dom_err),
                                       "max_abs": max(abs(e) for e in dom_err)},
            "checks": {
                "C1_slope_within_tol": abs(direct_slope - fitted.draft_cost_ratio) <= TOL_SLOPE,
                "C2_held_out_within_tol": all(abs(v["one_depth"] - v["measured"]) <= TOL_SPEEDUP for v in pooled.values()),
                "C3_fixed_within_tol": abs(one_depth.fixed_cost - fitted.fixed_cost) <= TOL_FIXED,
                "C4_zero_run_misses": any(abs(v["zero_run"] - v["measured"]) > TOL_SPEEDUP for v in pooled.values()),
            },
        }

    overall = {c: all(d["checks"][c] for d in per_draft.values()) for c in next(iter(per_draft.values()))["checks"]}
    payload = {
        "inputs": {"sweep": SWEEP.name, "timings": TIMINGS.name, "pin_k": PIN_K,
                   "held_out_k": [int(k) for k in HELD_OUT],
                   "tolerances": {"slope": TOL_SLOPE, "speedup": TOL_SPEEDUP, "fixed": TOL_FIXED}},
        "c_verify_least_squares": c_verify,
        "per_draft": per_draft,
        "checks_all_drafts": overall,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"c_verify (least squares over 1..9 tokens) = {c_verify:.3f}")
    print()
    print(f"{'draft':>5} {'c_draft':>8} {'slope direct':>13} {'slope fit':>10} {'fixed k=1':>10} {'fixed fit':>10}")
    for n, d in per_draft.items():
        print(f"{n:>5} {d['c_draft_direct']:>8.3f} {d['slope_direct']:>13.3f} {d['slope_fitted']:>10.3f} "
              f"{d['fixed_one_depth']:>10.3f} {d['fixed_full_fit']:>10.3f}")
    print()
    print("held-out pooled speedup: measured | one-depth (k=1 + timings) | zero-run (timings only) | full fit")
    for n, d in per_draft.items():
        for k, v in d["pooled_held_out"].items():
            print(f"{n:>5} k={k}: {v['measured']:.3f} | {v['one_depth']:.3f} ({v['one_depth'] - v['measured']:+.3f}) "
                  f"| {v['zero_run']:.3f} ({v['zero_run'] - v['measured']:+.3f}) | {v['full_fit']:.3f} ({v['full_fit'] - v['measured']:+.3f})")
    print()
    print("per-domain held-out error, one-depth model: mean / max |err|")
    for n, d in per_draft.items():
        e = d["domain_error_one_depth"]
        print(f"{n:>5} {e['mean']:+.3f} / {e['max_abs']:.3f}")
    print()
    for c, ok in overall.items():
        print(f"{c}: {'PASS' if ok else 'FAIL'}  " + "  ".join(f"{n}={'ok' if d['checks'][c] else 'X'}" for n, d in per_draft.items()))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
