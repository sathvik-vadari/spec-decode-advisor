"""Does the GPU's power state change the answer, not just the constants?

Experiment 02 measured the 0.5B draft against the 7B target on battery power:
best depth 1-2, speedup 1.15x, and a per-position verification cost near 0.3
of a target pass. Experiment 05, on AC power with the machine idle, timed the
same target pass as flat up to 4 total tokens: verifying three draft tokens
costs nothing extra. The two are consistent with one mechanism. Lowering the
GPU clock lowers peak compute but not memory bandwidth, so the roofline's ridge
moves left and a pass over a handful of tokens becomes compute-bound where at
full clock it was still hiding under the weight read.

If that is right, the payoff of speculation on this machine depends on the
power state, and so does the optimal depth. This runs experiment 02's 0.5B
session again, on AC, idle, with depth 3 and 6 added, and compares against the
committed battery-power results. Same prompts, same models, same code path.

Predictions, written before the run:

P1. The best measured depth is 3 or deeper, not 1 or 2.
P2. The best pooled speedup is at least 1.35x (battery: 1.15x). The roofline
    argument says about 1.7x at k=3; 1.35 is the bar this can fail at.
P3. The fitted per-position slope is below 0.25 (battery: 0.383).
P4. Acceptance per depth is identical to experiment 02 to three decimals.
    Greedy is deterministic; this is a check on the instrument, not a finding.
P5. The domain ordering of speedups is unchanged: code and copy on top, prose
    at the bottom.

Not a controlled experiment: the battery arm is three nights old and ran under
memory pressure. The drift check then was 7% and the baseline decode was 17
tok/s against about 20 tonight, so the target pass itself moved ~15% while the
extra-token cost moved ~4x. That asymmetry is the signature the roofline
argument predicts, and it is what P1-P3 test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spec_decode_advisor.acceptance import bootstrap_ci, filter_rounds, pooled_acceptance  # noqa: E402
from spec_decode_advisor.calibration import time_passes, verification_cost  # noqa: E402
from spec_decode_advisor.harness import Harness, to_prompt_rounds  # noqa: E402
from spec_decode_advisor.model import fit  # noqa: E402
from spec_decode_advisor.prompts import DOMAINS, PROMPTS  # noqa: E402

TARGET = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DRAFT = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEPTHS = (0, 1, 2, 3, 4, 6)
MAX_TOKENS = 128
BATTERY = ROOT / "results" / "02_draft_comparison.json"
OUT = ROOT / "results" / "06_power_state.json"


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


def speedup(results, k, domain=None):
    base = {r.prompt_id: r.tokens_per_s for r in results if r.k == 0}
    spec = [r for r in results if r.k == k and (domain is None or r.domain == domain)]
    return sum(r.tokens_per_s / base[r.prompt_id] for r in spec) / len(spec)


def main() -> None:
    t0 = time.perf_counter()
    cond = conditions()
    print("conditions:", json.dumps(cond), flush=True)
    h = Harness(TARGET, DRAFT, max_tokens=MAX_TOKENS).load()
    ids = h.encode_ids(PROMPTS[6])

    print("timing passes", flush=True)
    t_lat = time_passes(h.target, ids)
    d_lat = time_passes(h.draft, ids, tokens=(1,))
    c_verify = verification_cost(t_lat.seconds)
    c_draft = d_lat.one_token / t_lat.one_token
    print(f"  target {t_lat.one_token * 1e3:.1f} ms, c_verify {c_verify:.3f}, c_draft {c_draft:.3f}", flush=True)

    print(f"sweeping {len(PROMPTS)} prompts x {DEPTHS}", flush=True)
    h.run(PROMPTS[0], max(DEPTHS))
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        for k in DEPTHS:
            results.append(h.run(prompt, k))
        if i % 5 == 0:
            print(f"  [{i:2d}/{len(PROMPTS)}]", flush=True)
    t_lat_end = time_passes(h.target, ids)

    rounds = to_prompt_rounds(results)
    spec_depths = [k for k in DEPTHS if k > 0]
    by_depth = {}
    obs = []
    for k in spec_depths:
        rk = filter_rounds(rounds, k=k)
        p = pooled_acceptance(rk)
        s = speedup(results, k)
        obs.append((k, p, s))
        by_depth[k] = {"pooled_p": p, "pooled_p_ci": list(bootstrap_ci(rk, pooled_acceptance, seed=k)),
                       "n_rounds": sum(r.n_rounds for r in rk), "measured_speedup": s}
    cm = fit(obs)
    for k in spec_depths:
        by_depth[k]["fitted_speedup"] = cm.speedup(by_depth[k]["pooled_p"], k)
    by_domain = {dom: {k: {"pooled_p": pooled_acceptance(filter_rounds(rounds, k=k, domain=dom)),
                           "measured_speedup": speedup(results, k, dom)} for k in spec_depths}
                 for dom in DOMAINS}
    mismatches = sum(1 for pr in PROMPTS for r in results
                     if r.prompt_id == pr.prompt_id and r.k and r.text != next(x.text for x in results if x.prompt_id == pr.prompt_id and x.k == 0))
    best_k = max(spec_depths, key=lambda k: by_depth[k]["measured_speedup"])

    battery = json.loads(BATTERY.read_text())["sessions"]["0.5B"]
    comparison = {
        str(k): {"p_ac": by_depth[k]["pooled_p"], "p_battery": battery["by_depth"][str(k)]["pooled_p"],
                 "speedup_ac": by_depth[k]["measured_speedup"], "speedup_battery": battery["by_depth"][str(k)]["measured_speedup"]}
        for k in spec_depths if str(k) in battery["by_depth"]
    }
    base_tps = sum(r.tokens_per_s for r in results if r.k == 0) / len(PROMPTS)

    payload = {
        "config": {"target": TARGET, "draft": DRAFT, "depths": list(DEPTHS), "max_tokens": MAX_TOKENS,
                   "n_prompts": len(PROMPTS), "sampling": "greedy", "battery_arm": BATTERY.name},
        "conditions": cond,
        "timings": {"target_pass_ms": t_lat.one_token * 1e3, "target_pass_ms_end": t_lat_end.one_token * 1e3,
                    "target_latency_ms": {str(n): s * 1e3 for n, s in t_lat.seconds.items()},
                    "c_verify": c_verify, "c_draft": c_draft},
        "fitted": {"fixed_cost": cm.fixed_cost, "draft_cost_ratio": cm.draft_cost_ratio},
        "fitted_battery": battery["fitted"],
        "baseline_tokens_per_s": base_tps,
        "baseline_tokens_per_s_battery": sum(battery["baseline_tokens_per_s"].values()) / 30,
        "by_depth": by_depth, "by_domain": by_domain, "comparison": comparison,
        "best_measured_k": best_k, "lossless_mismatches": mismatches,
        "checks": {"P1_best_k_ge_3": best_k >= 3,
                   "P2_best_speedup_ge_1.35": by_depth[best_k]["measured_speedup"] >= 1.35,
                   "P3_slope_lt_0.25": cm.draft_cost_ratio < 0.25,
                   "P4_acceptance_identical": all(abs(v["p_ac"] - v["p_battery"]) < 5e-4 for v in comparison.values())},
        "runs": [{"prompt_id": r.prompt_id, "domain": r.domain, "k": r.k, "n_tokens": r.n_tokens, "wall_s": r.wall_s,
                  "tokens_per_s": r.tokens_per_s, "rounds": list(r.rounds), "finish_reason": r.finish_reason} for r in results],
        "elapsed_s": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print()
    print(f"target pass {t_lat.one_token * 1e3:.1f} -> {t_lat_end.one_token * 1e3:.1f} ms over the run; baseline {base_tps:.1f} tok/s (battery {payload['baseline_tokens_per_s_battery']:.1f})")
    print(f"fitted AC: round_cost(k) = {cm.fixed_cost:.3f} + {cm.draft_cost_ratio:.3f}k    battery: {battery['fitted']['fixed_cost']:.3f} + {battery['fitted']['draft_cost_ratio']:.3f}k")
    print(f"{'k':>2} {'p AC':>6} {'p batt':>7} {'speedup AC':>11} {'batt':>6} {'fit':>6}")
    for k in spec_depths:
        d = by_depth[k]; b = battery["by_depth"].get(str(k))
        print(f"{k:>2} {d['pooled_p']:>6.3f} {(b['pooled_p'] if b else float('nan')):>7.3f} {d['measured_speedup']:>11.3f} {(b['measured_speedup'] if b else float('nan')):>6.3f} {d['fitted_speedup']:>6.3f}")
    print(f"best measured depth {best_k} at {by_depth[best_k]['measured_speedup']:.3f}x; lossless mismatches {mismatches}")
    print("by domain, best AC depth and speedup: " + "; ".join(
        f"{dom} k{max(spec_depths, key=lambda k: by_domain[dom][k]['measured_speedup'])} {max(v['measured_speedup'] for v in by_domain[dom].values()):.2f}" for dom in DOMAINS))
    for c, ok in payload["checks"].items():
        print(f"{c}: {'PASS' if ok else 'FAIL'}")
    print(f"wrote {OUT}  ({payload['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
