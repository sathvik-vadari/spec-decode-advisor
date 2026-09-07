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


def _run_pair(h, prompt, say, retries: int = 1):
    """The k=0 and k=1 generations for one prompt, or None if the GPU fails twice.

    A Metal command-buffer timeout kills one generation and leaves the process
    fine; the first 7B run of this tool died that way 25 prompts into its third
    draft and took the two finished drafts with it. Retry once, then drop the
    prompt and say so. Both generations are dropped together so every k=1 run
    keeps its own baseline.
    """
    for attempt in range(retries + 1):
        try:
            return h.run(prompt, 0), h.run(prompt, 1)
        except RuntimeError as e:
            first = str(e).splitlines()[0][:90] if str(e) else type(e).__name__
            say(f"  {prompt.prompt_id}: {first}" + (" -- retrying" if attempt < retries else " -- skipped"))
    return None


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

    config = {"target": args.target, "drafts": args.draft, "prompts": args.prompts or "builtin",
              "n_prompts": len(prompts), "max_tokens": args.max_tokens, "repeats": args.repeats}
    skipped: dict[str, list[str]] = {}

    def checkpoint(reports, partial: bool) -> None:
        if not args.json:
            return
        rep = build_report(_short(args.target), target_latency, reports, max_k=args.max_k)
        payload = to_json(rep)
        payload.update(config=config, skipped=skipped, partial=partial,
                       elapsed_s=time.perf_counter() - t0)
        args.json.write_text(json.dumps(payload, indent=2))

    reports = []
    for i, draft_id in enumerate(args.draft):
        name = _short(draft_id)
        if i > 0:
            say(f"swapping draft -> {name}")
            h.swap_draft(draft_id)
        say(f"[{name}] timing draft passes")
        draft_latency = time_passes(h.draft, ids, repeats=args.repeats)
        say(f"[{name}] running {len(prompts)} prompts at k=0 and k=1")
        h.run(prompts[0], 1)  # warm-up, discarded
        runs = []
        for j, prompt in enumerate(prompts, 1):
            pair = _run_pair(h, prompt, say)
            if pair is None:
                skipped.setdefault(name, []).append(prompt.prompt_id)
            else:
                runs.extend(pair)
            if j % 5 == 0 or j == len(prompts):
                say(f"  {j}/{len(prompts)}")
        reports.append(analyse_draft(name, runs, draft_latency, target_latency))
        checkpoint(reports, partial=i + 1 < len(args.draft))  # a crash later loses one draft, not all

    rep = build_report(_short(args.target), target_latency, reports, max_k=args.max_k)
    print(render(rep))
    if skipped:
        print("  skipped after a GPU failure: " + "; ".join(f"{n}: {', '.join(ps)}" for n, ps in skipped.items()))
    print(f"\n{time.perf_counter() - t0:.0f}s")
    if args.json:
        say(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
