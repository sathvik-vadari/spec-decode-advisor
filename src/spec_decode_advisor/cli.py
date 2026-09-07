"""`spec-decode-advisor`: should you enable speculative decoding on your workload,
with which draft, at what depth.

    spec-decode-advisor --target mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --draft mlx-community/Qwen2.5-0.5B-Instruct-4bit \\
        --draft mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
        --prompts my_workload.jsonl --json report.json

Cost: two forward-pass timings per model plus two generations per prompt per
draft (one plain, one at k=1). Nothing else. See `advisor.py` for why that is
enough and experiment 04 for the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .advisor import analyse_draft, build_report, render, to_json
from .calibration import time_passes
from .harness import Harness
from .prompts import PROMPTS, load_prompts


def _short(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="spec-decode-advisor", description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", required=True, help="target model id (HF hub or local path)")
    ap.add_argument("--draft", action="append", required=True, help="candidate draft; repeat for several")
    ap.add_argument("--prompts", help=".jsonl (text, id?, domain?) or one prompt per line; default: built-in 30-prompt set")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-k", type=int, default=8, help="deepest draft depth to consider")
    ap.add_argument("--repeats", type=int, default=25, help="timing repeats per pass length")
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr, flush=True))
    prompts = load_prompts(args.prompts) if args.prompts else PROMPTS
    t0 = time.perf_counter()

    say(f"loading {_short(args.target)} + {_short(args.draft[0])}")
    h = Harness(args.target, args.draft[0], max_tokens=args.max_tokens).load()
    ids = h.encode_ids(prompts[0])
    say("timing target passes")
    target_latency = time_passes(h.target, ids, repeats=args.repeats)

    reports = []
    for i, draft_id in enumerate(args.draft):
        if i > 0:
            say(f"swapping draft -> {_short(draft_id)}")
            h.swap_draft(draft_id)
        say(f"[{_short(draft_id)}] timing draft passes")
        draft_latency = time_passes(h.draft, ids, repeats=args.repeats)
        say(f"[{_short(draft_id)}] running {len(prompts)} prompts at k=0 and k=1")
        h.run(prompts[0], 1)  # warm-up, discarded
        runs = []
        for j, prompt in enumerate(prompts, 1):
            runs.append(h.run(prompt, 0))
            runs.append(h.run(prompt, 1))
            if j % 5 == 0 or j == len(prompts):
                say(f"  {j}/{len(prompts)}")
        reports.append(analyse_draft(_short(draft_id), runs, draft_latency, target_latency))

    rep = build_report(_short(args.target), target_latency, reports, max_k=args.max_k)
    print(render(rep))
    print(f"\n{time.perf_counter() - t0:.0f}s")
    if args.json:
        payload = to_json(rep)
        payload["config"] = {"target": args.target, "drafts": args.draft, "prompts": args.prompts or "builtin",
                             "n_prompts": len(prompts), "max_tokens": args.max_tokens, "repeats": args.repeats}
        payload["elapsed_s"] = time.perf_counter() - t0
        args.json.write_text(json.dumps(payload, indent=2))
        say(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
