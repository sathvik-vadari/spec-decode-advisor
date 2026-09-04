"""Is verifying k+1 tokens in one target pass as cheap as generating one?

Speculative decoding's premise is that it is: decode at batch 1 is
memory-bandwidth-bound, so a forward pass over k+1 positions reads the same
weights as a pass over one and should cost about the same. Experiment 02 fitted
a per-position cost 0.25-0.29 target-passes *above* each draft's standalone
pass cost, and that premium was the same for a 0.5B, 1.5B and 3B draft. A
draft-independent per-position cost is not a draft cost. The candidate is the
target: if a pass over k+1 tokens costs (1 + c_verify * k) passes, c_verify
lands in the fitted slope and gets misattributed to the draft.

Prediction, stated before the run: the 7B pass grows roughly linearly in token
count with slope about 0.27 of a one-token pass, matching the fitted premium.
If the slope is near zero the premium is something else (sync, Python) and the
attribution in experiment 02's writeup is wrong.

Method: prefill one prompt into a KV cache, then time a forward pass over n
fresh tokens with that cache, trimming the cache back after each, n in
{1,2,3,5,9}, 25 repeats after 3 discarded warm-ups, median and quartiles. Same
for each draft alone, which also gives the draft's pass cost directly rather
than through stream_generate's per-token overhead.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as C

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spec_decode_advisor.prompts import PROMPTS  # noqa: E402

MODELS = {
    "7B": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "0.5B": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "1.5B": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "3B": "mlx-community/Qwen2.5-3B-Instruct-4bit",
}
TOKENS = (1, 2, 3, 5, 9)
REPEATS = 25
WARMUP = 3
OUT = Path(__file__).resolve().parents[1] / "results" / "03_verification_cost.json"


def bench(model_id: str) -> dict:
    model, tok = load(model_id)
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPTS[6].text}], add_generation_prompt=True, tokenize=False
    )
    prompt = mx.array(tok.encode(text))
    kv = C.make_prompt_cache(model)
    mx.eval(model(prompt[None], cache=kv))
    out = {}
    for n in TOKENS:
        y = mx.array([100 + i for i in range(n)])  # any valid ids; timing does not depend on which
        ts = []
        for i in range(WARMUP + REPEATS):
            t = time.perf_counter()
            mx.eval(model(y[None], cache=kv))
            dt = time.perf_counter() - t
            C.trim_prompt_cache(kv, n)
            if i >= WARMUP:
                ts.append(dt)
        q = statistics.quantiles(ts, n=4)
        out[n] = {"median_ms": statistics.median(ts) * 1e3, "q1_ms": q[0] * 1e3, "q3_ms": q[2] * 1e3}
    del model, kv
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()
    return out


def main() -> None:
    t0 = time.perf_counter()
    results = {}
    for name, mid in MODELS.items():
        print(f"{name}...", flush=True)
        results[name] = bench(mid)

    t1 = results["7B"][1]["median_ms"]
    # slope in target-pass units, from each multi-token point
    slopes = {n: (results["7B"][n]["median_ms"] - t1) / (n - 1) / t1 for n in TOKENS if n > 1}
    payload = {
        "config": {"models": MODELS, "tokens": list(TOKENS), "repeats": REPEATS, "warmup": WARMUP,
                   "prompt_id": PROMPTS[6].prompt_id},
        "latency_ms": results,
        "target_pass_ms": t1,
        "verify_cost_per_position": slopes,
        "verify_cost_per_position_mean": statistics.mean(slopes.values()),
        "draft_pass_ratio": {n: results[n][1]["median_ms"] / t1 for n in MODELS if n != "7B"},
        "elapsed_s": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print()
    print(f"{'model':>5} " + " ".join(f"{'n=' + str(n):>14}" for n in TOKENS))
    for name in MODELS:
        r = results[name]
        print(f"{name:>5} " + " ".join(f"{r[n]['median_ms']:7.1f} x{r[n]['median_ms'] / r[1]['median_ms']:4.2f} " for n in TOKENS))
    print()
    print("7B verification cost per extra position, in one-token passes: "
          + ", ".join(f"n={n}: {s:.3f}" for n, s in slopes.items())
          + f"   mean {payload['verify_cost_per_position_mean']:.3f}")
    print("draft one-token pass / target one-token pass: "
          + ", ".join(f"{n}: {r:.3f}" for n, r in payload["draft_pass_ratio"].items()))
    print(f"wrote {OUT}  ({payload['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
