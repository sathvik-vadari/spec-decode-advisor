"""Is `p` really invariant in draft depth, and does it differ by task domain?

Two questions, one sweep, because they need the same measurements.

Q1. The tool's premise is that one measured acceptance probability predicts
    every depth. Last run gave 0.785 at k=2 and 0.741 at k=4 -- recovered by
    inverting the truncated geometric sum, which assumes the invariance it was
    being used to check. This measures acceptance per draft *position* instead,
    which tests the assumption without making it.

Q2. If acceptance differs by domain, a single global draft depth is the wrong
    default and the tool has a job. Predicted ordering, stated before the run:
    copy > code > factual ~ chat > prose.

Everything is greedy. `mlx_lm` verifies by exact token match rather than the
Leviathan rejection rule, so above temperature 0 its acceptance is a lower bound
on a production engine's rather than an estimate of it. Greedy is where these
numbers transfer.

Greedy also gives a free correctness check: speculative decoding is lossless, so
every depth must emit a byte-identical continuation. If they diverge, the
instrumentation is wrong and no number here means anything.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spec_decode_advisor.acceptance import (  # noqa: E402
    bootstrap_ci,
    filter_rounds,
    hazard_spread,
    mean_accepted,
    naive_ratio,
    pooled_acceptance,
    position_hazards,
)
from spec_decode_advisor.harness import Harness, to_prompt_rounds  # noqa: E402
from spec_decode_advisor.model import CostModel, acceptance_from_measurement  # noqa: E402
from spec_decode_advisor.prompts import DOMAINS, PROMPTS  # noqa: E402

TARGET = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DRAFT = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEPTHS = (0, 1, 2, 4, 8)
MAX_TOKENS = 128
OUT = Path(__file__).resolve().parents[1] / "results" / "01_depth_invariance.json"


def main() -> None:
    t0 = time.perf_counter()
    h = Harness(TARGET, DRAFT, max_tokens=MAX_TOKENS).load()

    print("calibrating draft cost ratio...", flush=True)
    calib_prompts = [PROMPTS[i] for i in range(0, len(PROMPTS), 5)]
    c, calib_runs = h.measure_cost_ratio(calib_prompts)
    print(f"  measured c = {c:.3f}  (parameter ratio would suggest ~0.33)", flush=True)

    print(f"sweeping {len(PROMPTS)} prompts x {len(DEPTHS)} depths...", flush=True)
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        for k in DEPTHS:
            results.append(h.run(prompt, k))
        print(f"  [{i:2d}/{len(PROMPTS)}] {prompt.prompt_id}", flush=True)

    # ---- correctness: speculation is lossless, so depth must not change output
    mismatches = []
    for prompt in PROMPTS:
        texts = {r.k: r.text for r in results if r.prompt_id == prompt.prompt_id}
        base = texts[0]
        for k, t in texts.items():
            if k and t != base:
                mismatches.append({"prompt_id": prompt.prompt_id, "k": k})

    rounds = to_prompt_rounds(results)
    cost = CostModel(draft_cost_ratio=c)

    # ---- Q1: per-position hazards
    by_depth = {}
    for k in [d for d in DEPTHS if d > 0]:
        rk = filter_rounds(rounds, k=k)
        p_hat = pooled_acceptance(rk)
        lo, hi = bootstrap_ci(rk, pooled_acceptance, seed=k)
        hz = position_hazards(rk, k)
        spread = hazard_spread(rk, k)
        s_lo, s_hi = bootstrap_ci(rk, lambda r, _k=k: hazard_spread(r, _k), seed=100 + k)

        base_tps = {
            r.prompt_id: r.tokens_per_s for r in results if r.k == 0
        }
        spec = [r for r in results if r.k == k]
        measured_speedup = sum(
            r.tokens_per_s / base_tps[r.prompt_id] for r in spec
        ) / len(spec)

        by_depth[k] = {
            "pooled_p": p_hat,
            "pooled_p_ci": [lo, hi],
            "naive_ratio": naive_ratio(rk),
            "mean_accepted": mean_accepted(rk),
            "mom_p": acceptance_from_measurement(mean_accepted(rk), k),
            "n_rounds": sum(r.n_rounds for r in rk),
            "hazards": [
                {"position": z.position, "at_risk": z.at_risk, "p": z.p} for z in hz
            ],
            "hazard_spread": spread,
            "hazard_spread_ci": [s_lo, s_hi],
            "predicted_speedup": cost.speedup(p_hat, k),
            "measured_speedup": measured_speedup,
        }

    # ---- Q2: acceptance by domain
    by_domain = {}
    for dom in DOMAINS:
        entry = {}
        for k in [d for d in DEPTHS if d > 0]:
            rk = filter_rounds(rounds, k=k, domain=dom)
            lo, hi = bootstrap_ci(rk, pooled_acceptance, seed=k)
            base_tps = {r.prompt_id: r.tokens_per_s for r in results if r.k == 0}
            spec = [r for r in results if r.k == k and r.domain == dom]
            entry[k] = {
                "pooled_p": pooled_acceptance(rk),
                "pooled_p_ci": [lo, hi],
                "n_rounds": sum(r.n_rounds for r in rk),
                "best_depth": cost.best_depth(pooled_acceptance(rk))[0],
                "predicted_speedup": cost.speedup(pooled_acceptance(rk), k),
                "measured_speedup": sum(
                    r.tokens_per_s / base_tps[r.prompt_id] for r in spec
                ) / len(spec),
            }
        by_domain[dom] = entry

    payload = {
        "config": {
            "target": TARGET,
            "draft": DRAFT,
            "depths": list(DEPTHS),
            "max_tokens": MAX_TOKENS,
            "n_prompts": len(PROMPTS),
            "sampling": "greedy",
            "verification": "exact-match (mlx_lm), equals Leviathan rule at temp 0",
        },
        "cost_ratio_measured": c,
        "lossless_check": {
            "n_mismatches": len(mismatches),
            "mismatches": mismatches,
        },
        "by_depth": by_depth,
        "by_domain": by_domain,
        "runs": [
            {
                "prompt_id": r.prompt_id,
                "domain": r.domain,
                "k": r.k,
                "n_tokens": r.n_tokens,
                "wall_s": r.wall_s,
                "tokens_per_s": r.tokens_per_s,
                "rounds": list(r.rounds),
                "dropped_rounds": r.dropped_rounds,
                "finish_reason": r.finish_reason,
            }
            for r in results
        ],
        "calibration_runs": [
            {"prompt_id": r.prompt_id, "which": r.domain, "n_tokens": r.n_tokens,
             "wall_s": r.wall_s} for r in calib_runs
        ],
        "elapsed_s": time.perf_counter() - t0,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    # ---- readable summary
    print()
    print(f"lossless check: {len(mismatches)} mismatches across depths")
    print(f"measured draft cost ratio c = {c:.3f}")
    print()
    print("Q1  acceptance by depth")
    print(f"{'k':>3} {'pooled p':>9} {'95% CI':>16} {'naive':>7} {'MoM':>7} "
          f"{'spread':>8} {'spread CI':>16} {'pred':>6} {'meas':>6}")
    for k, d in by_depth.items():
        lo, hi = d["pooled_p_ci"]
        slo, shi = d["hazard_spread_ci"]
        print(f"{k:>3} {d['pooled_p']:>9.3f} [{lo:.3f}, {hi:.3f}] {d['naive_ratio']:>7.3f} "
              f"{d['mom_p']:>7.3f} {d['hazard_spread']:>8.3f} [{slo:+.3f}, {shi:+.3f}] "
              f"{d['predicted_speedup']:>6.2f} {d['measured_speedup']:>6.2f}")
    print()
    print("    per-position hazards")
    for k, d in by_depth.items():
        cells = "  ".join(
            f"p{z['position']}={z['p']:.3f}(n={z['at_risk']})" for z in d["hazards"]
        )
        print(f"  k={k}: {cells}")
    print()
    print("Q2  acceptance by domain (k=4)")
    print(f"{'domain':>9} {'pooled p':>9} {'95% CI':>16} {'best k':>7} {'meas speedup':>13}")
    for dom in DOMAINS:
        d = by_domain[dom][4]
        lo, hi = d["pooled_p_ci"]
        print(f"{dom:>9} {d['pooled_p']:>9.3f} [{lo:.3f}, {hi:.3f}] {d['best_depth']:>7} "
              f"{d['measured_speedup']:>13.2f}")
    print()
    print(f"wrote {OUT}  ({payload['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
