"""How do the cost terms scale with batch size, and where does the model say
speculation stops paying under load?

Everything so far is batch 1: one request, decode bandwidth-bound, spare
compute that makes verifying k+1 tokens cheap. A serving engine decodes many
requests per step. Once the step is compute-bound, an extra verified position
costs the same as an extra request, and the free lunch speculative decoding is
built on is gone. Published work puts the energy crossover at batch sizes in
the tens to hundreds on datacentre GPUs. This machine cannot run that
experiment -- mlx_lm's speculative path is batch 1 -- but it can measure the
mechanism: the target's pass time over B rows and n positions, and the draft's
over B rows. Acceptance does not depend on batch, so the cost side is the whole
question.

Measured: for the 7B target, pass latency at batch B in {1,2,4,8,16} and
n in {1,2,3,5} verified positions per row, warm cache, prompt replicated B
times. For each draft, pass latency at each B with n=1. From these, per B:

    throughput(B)   = B / t_target(B, 1)              plain batched decode
    c_verify(B)     = slope_n t_target(B, n) / t_target(B, 1)
    c_draft(B)      = t_draft(B, 1) / t_target(B, 1)

Predicted, not measured: speedup at batch B for depth k, using experiment 02's
acceptance for the pair and the batch-1 fixed cost, which is assumed constant
in B. The crossover batch is where the best depth no longer beats 1.0.

Predictions, written before the run:

P1. The target's one-token pass is flat in B while bandwidth-bound, then grows
    about linearly. On this machine the flat region is short: B_free <= 4 for
    the 7B, judging from the 1.5B (flat to 4, 2.7x at 16).

P2. c_verify(B) rises from ~0.3 at B=1 toward 1.0. Fully compute-bound, an
    extra position costs exactly one extra row's worth, so c_verify -> 1. It
    exceeds 0.8 by B=16.

P3. c_draft(B) falls with B. The draft is smaller, stays overhead- or
    bandwidth-bound longer, so its pass grows more slowly than the target's
    until both saturate, where the ratio approaches the parameter ratio
    (0.064 for 0.5B, 0.20 for 1.5B).

P4. Net, c_verify wins: the slope c_draft + c_verify rises with B, and the
    predicted k=1 speedup for the 0.5B draft (p = 0.69) drops below 1.0 at
    some B in [4, 16] on this machine. The crossover would sit at a larger B
    on hardware with more compute per byte of bandwidth, which is the whole
    reason datacentre GPUs see it in the tens to hundreds.

Conditions are recorded in the output, because finding 12 showed they matter.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spec_decode_advisor.calibration import time_passes, verification_cost  # noqa: E402
from spec_decode_advisor.model import CostModel, expected_tokens_per_round  # noqa: E402
from spec_decode_advisor.prompts import PROMPTS  # noqa: E402

TARGET = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DRAFTS = {"0.5B": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "1.5B": "mlx-community/Qwen2.5-1.5B-Instruct-4bit"}
BATCHES = (1, 2, 4, 8, 16)
POSITIONS = (1, 2, 3, 5)
REPEATS, WARMUP = 15, 3
FINE_T = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 28, 32, 33, 40, 48, 56, 64, 65, 96)
# from experiment 02 / results 04: pooled acceptance and the batch-1 pinned fixed cost per draft
ACCEPTANCE = {"0.5B": 0.688, "1.5B": 0.739}
FIXED_B1 = {"0.5B": 1.103, "1.5B": 1.156}
OUT = ROOT / "results" / "05_batch_scaling.json"


def conditions() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {
        "lowpowermode": sh("pmset -g | awk '/lowpowermode/ {print $2}'"),
        "power": sh("pmset -g batt | head -1"),
        "battery": sh("pmset -g batt | sed -n 2p | awk '{print $3, $4}'"),
        "memory_free": sh("memory_pressure | grep -i 'system-wide'"),
    }


def free(model):
    del model
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()


def main() -> None:
    t0 = time.perf_counter()
    cond = conditions()
    print("conditions:", json.dumps(cond), flush=True)

    target, tok = load(TARGET)
    ids = mx.array(tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": PROMPTS[6].text}], add_generation_prompt=True, tokenize=False)))

    print("target passes", flush=True)
    target_lat = {}
    for B in BATCHES:
        lat = time_passes(target, ids, tokens=POSITIONS, repeats=REPEATS, warmup=WARMUP, batch=B)
        target_lat[B] = {n: s * 1e3 for n, s in lat.seconds.items()}
        print(f"  B={B:>2}: " + "  ".join(f"n={n}: {ms:7.1f} ms" for n, ms in target_lat[B].items()), flush=True)

    # A fine sweep of total tokens at batch 1, because the grid above is not
    # smooth: the cost looks like a step function of B * n. This is the direct
    # look at the kernel's tiling, and it is what c_verify is a linearisation of.
    print("fine sweep, batch 1", flush=True)
    fine = time_passes(target, ids, tokens=FINE_T, repeats=REPEATS, warmup=WARMUP)
    fine_ms = {T: s * 1e3 for T, s in fine.seconds.items()}
    for T in FINE_T:
        print(f"  T={T:>3}: {fine_ms[T]:7.1f} ms  x1={fine_ms[T] / fine_ms[1]:.2f}", flush=True)

    draft_lat = {}
    for name, mid in DRAFTS.items():
        draft, _ = load(mid)
        draft_lat[name] = {}
        for B in BATCHES:
            lat = time_passes(draft, ids, tokens=(1,), repeats=REPEATS, warmup=WARMUP, batch=B)
            draft_lat[name][B] = lat.seconds[1] * 1e3
        print(f"draft {name}: " + "  ".join(f"B={B}: {ms:6.1f} ms" for B, ms in draft_lat[name].items()), flush=True)
        free(draft)

    per_batch = {}
    for B in BATCHES:
        t1 = target_lat[B][1]
        entry = {
            "target_pass_ms": t1,
            "throughput_tok_s": B / (t1 / 1e3),
            "pass_vs_b1": t1 / target_lat[1][1],
            "c_verify": verification_cost({n: ms for n, ms in target_lat[B].items()}),
            "drafts": {},
        }
        for name in DRAFTS:
            c_draft = draft_lat[name][B] / t1
            cm = CostModel(draft_cost_ratio=c_draft + entry["c_verify"], fixed_cost=FIXED_B1[name])
            p = ACCEPTANCE[name]
            best_k, best_s = cm.best_depth(p, max_k=8)
            entry["drafts"][name] = {
                "draft_pass_ms": draft_lat[name][B],
                "c_draft": c_draft,
                "slope": c_draft + entry["c_verify"],
                "predicted_speedup_k1": cm.speedup(p, 1),
                "predicted_speedup_k2": cm.speedup(p, 2),
                "predicted_best_k": best_k,
                "predicted_best_speedup": best_s,
            }
        per_batch[B] = entry

    crossover = {}
    for name in DRAFTS:
        xs = [B for B in BATCHES if per_batch[B]["drafts"][name]["predicted_best_speedup"] <= 1.0]
        crossover[name] = xs[0] if xs else None

    payload = {
        "config": {"target": TARGET, "drafts": DRAFTS, "batches": list(BATCHES), "positions": list(POSITIONS),
                   "repeats": REPEATS, "warmup": WARMUP, "acceptance_assumed": ACCEPTANCE,
                   "fixed_cost_assumed_b1": FIXED_B1, "prompt_id": PROMPTS[6].prompt_id},
        "conditions": cond,
        "target_latency_ms": {str(B): {str(n): ms for n, ms in d.items()} for B, d in target_lat.items()},
        "target_pass_ms_vs_total_tokens_b1": {str(T): ms for T, ms in fine_ms.items()},
        "draft_latency_ms": {name: {str(B): ms for B, ms in d.items()} for name, d in draft_lat.items()},
        "per_batch": {str(B): e for B, e in per_batch.items()},
        "predicted_crossover_batch": crossover,
        "elapsed_s": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print()
    print(f"{'B':>3} {'pass ms':>8} {'x B=1':>6} {'tok/s':>7} {'c_verify':>9} | " +
          " | ".join(f"{n:>4} c_draft slope  k=1  best" for n in DRAFTS))
    for B in BATCHES:
        e = per_batch[B]
        cells = " | ".join(f"{e['drafts'][n]['c_draft']:>12.3f} {e['drafts'][n]['slope']:.3f} {e['drafts'][n]['predicted_speedup_k1']:.2f}  "
                           f"k{e['drafts'][n]['predicted_best_k']} {e['drafts'][n]['predicted_best_speedup']:.2f}" for n in DRAFTS)
        print(f"{B:>3} {e['target_pass_ms']:>8.1f} {e['pass_vs_b1']:>6.2f} {e['throughput_tok_s']:>7.1f} {e['c_verify']:>9.3f} | {cells}")
    print()
    print("predicted crossover batch (best depth no longer beats 1.0):", crossover)
    print(f"wrote {OUT}  ({payload['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
