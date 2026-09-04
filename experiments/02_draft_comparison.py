"""Against one target, which draft model should you use -- and does it depend
on the workload?

Experiment 01 fixed the pair and varied depth. This fixes the target and varies
the draft: Qwen2.5-7B-Instruct-4bit verified against 0.5B, 1.5B and 3B drafts
from the same family, on the same 30 prompts over 5 domains, at k in {0,1,2,4}.
The 7B target also answers the scope question finding 5 left open: on a 1.5B
target speculation never paid because fixed per-round overhead dominated a cheap
target pass. A target pass ~5x dearer should shrink that overhead relatively.

Predictions, stated before the run so they can fail:

P1. Acceptance `p` rises with draft size in every domain, 0.5B < 1.5B < 3B,
    with diminishing returns per doubling. Domain ordering from experiment 01
    (copy > code > factual > chat > prose) holds for every draft.

P2. The fitted `draft_cost_ratio` rises with draft size but exceeds the
    parameter ratio (0.064, 0.20, 0.41) most for the smallest draft: a 0.5B
    pass on this machine is overhead-bound, not bandwidth-bound, so its cost
    does not fall in proportion to its size. The standalone-timed ratio
    understates the in-loop ratio for every draft, as it did in experiment 01.

P3. The fitted `fixed_cost` falls from experiment 01's 1.378 toward 1.0. The
    per-round overhead (cache rewind, sync) is roughly constant in wall time
    while the target pass is now several times longer, so it shrinks in units
    of a target pass. If `fixed_cost` stays near 1.4 the overhead is not a
    constant bookkeeping cost but scales with the target, and the mechanism
    story from experiment 01 is wrong.

P4. Speculation now pays: best measured speedup > 1.0 in copy and code for at
    least one draft. This is the test of finding 5's scope.

P5. The best draft depends on the domain. Where acceptance is high for every
    draft (copy, code) the cheapest draft wins because extra acceptance is not
    worth the extra pass cost. Where acceptance is low (prose, chat) a larger
    draft wins if anything does. So there is no single best draft, which is the
    "which draft" half of the tool's justification.

Controls. Each draft gets its own contemporaneous k=0 baseline, interleaved
prompt-by-prompt, so speedups within a draft are ratios against a baseline that
ran seconds earlier rather than against a baseline from a different thermal or
memory state. Baselines across the three sessions are then compared to each
other as a drift check, and their greedy outputs must be byte-identical (same
target, same prompt) or the instrument is broken. Drafts are swapped, not
reloaded alongside each other: peak memory is target + one draft.

Everything is greedy, for the transfer reason documented in the harness.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spec_decode_advisor.acceptance import (  # noqa: E402
    bootstrap_ci,
    filter_rounds,
    hazard_spread,
    naive_ratio,
    pooled_acceptance,
    position_hazards,
)
from spec_decode_advisor.harness import Harness, to_prompt_rounds  # noqa: E402
from spec_decode_advisor.model import Candidate, CostModel, fit, recommend  # noqa: E402
from spec_decode_advisor.prompts import DOMAINS, PROMPTS  # noqa: E402

TARGET = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DRAFTS = {
    "0.5B": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "1.5B": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "3B": "mlx-community/Qwen2.5-3B-Instruct-4bit",
}
# Qwen2.5 parameter counts: 0.49B, 1.54B, 3.09B against 7.62B.
PARAM_RATIO = {"0.5B": 0.064, "1.5B": 0.202, "3B": 0.406}
DEPTHS = (0, 1, 2, 4)
SPEC_DEPTHS = tuple(k for k in DEPTHS if k > 0)
MAX_TOKENS = 128
OUT = Path(__file__).resolve().parents[1] / "results" / "02_draft_comparison.json"


def _peak_memory_gb() -> float | None:
    try:
        import mlx.core as mx

        try:
            return mx.get_peak_memory() / 1e9
        except AttributeError:
            return mx.metal.get_peak_memory() / 1e9
    except Exception:  # noqa: BLE001
        return None


def _measured_speedup(results, k, domain=None):
    base = {r.prompt_id: r.tokens_per_s for r in results if r.k == 0}
    spec = [r for r in results if r.k == k and (domain is None or r.domain == domain)]
    return sum(r.tokens_per_s / base[r.prompt_id] for r in spec) / len(spec)


def main() -> None:
    t0 = time.perf_counter()
    names = list(DRAFTS)
    h = Harness(TARGET, DRAFTS[names[0]], max_tokens=MAX_TOKENS).load()
    calib_prompts = [PROMPTS[i] for i in range(0, len(PROMPTS), 5)]

    sessions = {}
    all_runs = []
    all_calib = []
    for si, name in enumerate(names):
        if si > 0:
            print(f"swapping draft -> {name}", flush=True)
            h.swap_draft(DRAFTS[name])
        s0 = time.perf_counter()

        print(f"[{name}] standalone cost ratio...", flush=True)
        c_naive, calib_runs = h.measure_cost_ratio(calib_prompts)
        print(f"  standalone c = {c_naive:.3f}  (parameter ratio {PARAM_RATIO[name]})", flush=True)
        for r in calib_runs:
            all_calib.append({"draft": name, "prompt_id": r.prompt_id, "which": r.domain,
                              "n_tokens": r.n_tokens, "wall_s": r.wall_s})

        print(f"[{name}] sweeping {len(PROMPTS)} prompts x {DEPTHS}...", flush=True)
        h.run(PROMPTS[0], max(DEPTHS))  # warm-up, discarded
        results = []
        for i, prompt in enumerate(PROMPTS, 1):
            for k in DEPTHS:
                results.append(h.run(prompt, k))
            print(f"  [{i:2d}/{len(PROMPTS)}] {prompt.prompt_id}", flush=True)
        all_runs.extend((name, r) for r in results)

        # lossless: within a session every depth must reproduce k=0
        mismatches = []
        for prompt in PROMPTS:
            texts = {r.k: r.text for r in results if r.prompt_id == prompt.prompt_id}
            for k, t in texts.items():
                if k and t != texts[0]:
                    mismatches.append({"prompt_id": prompt.prompt_id, "k": k})

        rounds = to_prompt_rounds(results)
        by_depth = {}
        observations = []
        for k in SPEC_DEPTHS:
            rk = filter_rounds(rounds, k=k)
            p_hat = pooled_acceptance(rk)
            lo, hi = bootstrap_ci(rk, pooled_acceptance, seed=k)
            meas = _measured_speedup(results, k)
            observations.append((k, p_hat, meas))
            by_depth[k] = {
                "pooled_p": p_hat,
                "pooled_p_ci": [lo, hi],
                "naive_ratio": naive_ratio(rk),
                "n_rounds": sum(r.n_rounds for r in rk),
                "hazards": [{"position": z.position, "at_risk": z.at_risk, "p": z.p}
                            for z in position_hazards(rk, k)],
                "hazard_spread": hazard_spread(rk, k),
                "measured_speedup": meas,
            }
        cost = fit(observations)
        for k in SPEC_DEPTHS:
            by_depth[k]["fitted_speedup"] = cost.speedup(by_depth[k]["pooled_p"], k)

        by_domain = {}
        for dom in DOMAINS:
            rd = filter_rounds(rounds, domain=dom)
            p_dom = pooled_acceptance(rd)
            lo, hi = bootstrap_ci(rd, pooled_acceptance, seed=7)
            per_k = {}
            for k in SPEC_DEPTHS:
                rk = filter_rounds(rd, k=k)
                per_k[k] = {
                    "pooled_p": pooled_acceptance(rk),
                    "n_rounds": sum(r.n_rounds for r in rk),
                    "measured_speedup": _measured_speedup(results, k, dom),
                    "fitted_speedup": cost.speedup(pooled_acceptance(rk), k),
                }
            best_k, best_s = cost.best_depth(p_dom)
            by_domain[dom] = {
                "pooled_p": p_dom,
                "pooled_p_ci": [lo, hi],
                "by_depth": per_k,
                "predicted_best_k": best_k,
                "predicted_best_speedup": best_s,
                "measured_best_k": max(SPEC_DEPTHS, key=lambda k: per_k[k]["measured_speedup"]),
                "measured_best_speedup": max(v["measured_speedup"] for v in per_k.values()),
            }

        base_tps = {r.prompt_id: r.tokens_per_s for r in results if r.k == 0}
        sessions[name] = {
            "draft": DRAFTS[name],
            "param_ratio": PARAM_RATIO[name],
            "standalone_cost_ratio": c_naive,
            "fitted": {"fixed_cost": cost.fixed_cost, "draft_cost_ratio": cost.draft_cost_ratio},
            "pooled_p_all_depths": pooled_acceptance(rounds),
            "lossless_check": {"n_mismatches": len(mismatches), "mismatches": mismatches},
            "by_depth": by_depth,
            "by_domain": by_domain,
            "baseline_tokens_per_s": base_tps,
            "baseline_texts": {r.prompt_id: r.text for r in results if r.k == 0},
            "peak_memory_gb": _peak_memory_gb(),
            "elapsed_s": time.perf_counter() - s0,
        }
        print(f"  fitted round_cost(k) = {cost.fixed_cost:.3f} + {cost.draft_cost_ratio:.3f}k"
              f"   p = {sessions[name]['pooled_p_all_depths']:.3f}"
              f"   {len(mismatches)} mismatches   ({sessions[name]['elapsed_s']:.0f}s)", flush=True)

    # ---- cross-session checks: same target, same prompts, so the baselines must
    # agree in output and should agree in speed
    base_text_mismatch = [
        pid for pid in sessions[names[0]]["baseline_texts"]
        if len({sessions[n]["baseline_texts"][pid] for n in names}) > 1
    ]
    drift = {}
    for pid in sessions[names[0]]["baseline_tokens_per_s"]:
        xs = [sessions[n]["baseline_tokens_per_s"][pid] for n in names]
        drift[pid] = statistics.pstdev(xs) / statistics.mean(xs)
    session_means = {n: statistics.mean(sessions[n]["baseline_tokens_per_s"].values()) for n in names}

    # ---- the advisor's answer per domain, predicted vs measured
    advisor = {}
    for dom in DOMAINS:
        cands = [
            Candidate(n, sessions[n]["by_domain"][dom]["pooled_p"],
                      CostModel(draft_cost_ratio=sessions[n]["fitted"]["draft_cost_ratio"],
                                fixed_cost=sessions[n]["fitted"]["fixed_cost"]))
            for n in names
        ]
        rec = recommend(cands, max_k=8)
        measured = {
            n: (sessions[n]["by_domain"][dom]["measured_best_k"],
                sessions[n]["by_domain"][dom]["measured_best_speedup"])
            for n in names
        }
        best_meas_draft = max(measured, key=lambda n: measured[n][1])
        advisor[dom] = {
            "predicted": {"draft": rec.draft, "k": rec.k, "speedup": rec.speedup,
                          "ranking": [list(t) for t in rec.ranking]},
            "measured": {
                "draft": best_meas_draft if measured[best_meas_draft][1] > 1.0 else None,
                "k": measured[best_meas_draft][0] if measured[best_meas_draft][1] > 1.0 else 0,
                "speedup": max(1.0, measured[best_meas_draft][1]),
                "per_draft": {n: {"k": k, "speedup": s} for n, (k, s) in measured.items()},
            },
        }

    payload = {
        "config": {
            "target": TARGET, "drafts": DRAFTS, "depths": list(DEPTHS),
            "max_tokens": MAX_TOKENS, "n_prompts": len(PROMPTS), "sampling": "greedy",
            "verification": "exact-match (mlx_lm), equals Leviathan rule at temp 0",
            "design": "one target loaded once; drafts swapped; per-draft interleaved k=0 baseline",
        },
        "sessions": sessions,
        "cross_session": {
            "baseline_text_mismatches": base_text_mismatch,
            "baseline_tps_cv_by_prompt": drift,
            "baseline_tps_cv_median": statistics.median(drift.values()),
            "baseline_tps_session_means": session_means,
        },
        "advisor": advisor,
        "runs": [
            {"draft": n, "prompt_id": r.prompt_id, "domain": r.domain, "k": r.k,
             "n_tokens": r.n_tokens, "wall_s": r.wall_s, "tokens_per_s": r.tokens_per_s,
             "rounds": list(r.rounds), "dropped_rounds": r.dropped_rounds,
             "finish_reason": r.finish_reason}
            for n, r in all_runs
        ],
        "calibration_runs": all_calib,
        "elapsed_s": time.perf_counter() - t0,
    }
    for n in names:  # texts were needed for the check, not for the record
        del payload["sessions"][n]["baseline_texts"]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    # ---- readable summary
    print()
    print(f"baseline outputs differ across sessions on {len(base_text_mismatch)} prompts; "
          f"baseline tok/s CV across sessions median {payload['cross_session']['baseline_tps_cv_median']:.3f}; "
          f"session means {', '.join(f'{n} {m:.1f}' for n, m in session_means.items())}")
    print()
    print("cost model per draft")
    print(f"{'draft':>6} {'param':>6} {'standalone c':>13} {'fitted c':>9} {'fixed':>6} {'p(all)':>7} {'mism':>5}")
    for n in names:
        s = sessions[n]
        print(f"{n:>6} {s['param_ratio']:>6.3f} {s['standalone_cost_ratio']:>13.3f} "
              f"{s['fitted']['draft_cost_ratio']:>9.3f} {s['fitted']['fixed_cost']:>6.3f} "
              f"{s['pooled_p_all_depths']:>7.3f} {s['lossless_check']['n_mismatches']:>5}")
    print()
    print("measured speedup by draft and depth (pooled p in brackets)")
    print(f"{'draft':>6} " + " ".join(f"{'k=' + str(k):>16}" for k in SPEC_DEPTHS))
    for n in names:
        cells = " ".join(
            f"{sessions[n]['by_depth'][k]['measured_speedup']:>6.2f} "
            f"[{sessions[n]['by_depth'][k]['pooled_p']:.3f}]  " for k in SPEC_DEPTHS)
        print(f"{n:>6} {cells}")
    print()
    print("acceptance by domain and draft")
    print(f"{'domain':>9} " + " ".join(f"{n:>8}" for n in names))
    for dom in DOMAINS:
        print(f"{dom:>9} " + " ".join(f"{sessions[n]['by_domain'][dom]['pooled_p']:>8.3f}" for n in names))
    print()
    print("advisor: which draft, what depth   (predicted from fitted model | measured best)")
    print(f"{'domain':>9} {'pred draft':>10} {'k':>2} {'x':>5}  | {'meas draft':>10} {'k':>2} {'x':>5}")
    for dom in DOMAINS:
        a = advisor[dom]
        pd, md = a["predicted"], a["measured"]
        print(f"{dom:>9} {str(pd['draft']):>10} {pd['k']:>2} {pd['speedup']:>5.2f}  | "
              f"{str(md['draft']):>10} {md['k']:>2} {md['speedup']:>5.2f}")
    print()
    print(f"wrote {OUT}  ({payload['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
