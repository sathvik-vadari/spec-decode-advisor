# spec-decode-advisor

Tells you whether speculative decoding will pay off on *your* workload, and at what draft depth.

## The problem

Speculative decoding is advertised as a 2–3× speedup. Whether you get it depends on how often your
target model accepts your draft model's guesses, which depends on the model pair, the task, and the
batch size. Below roughly 50% acceptance the verification overhead makes it a **net loss** — you do
more total work and go slower.

So the honest answer to "should I turn this on?" is "measure it," and there is no tool that does the
measuring. Teams enable it, see a number, and can't tell whether a different draft model or a
different depth would be better — or whether they should have left it off.

## The approach

Run the workload, count accepted tokens, recover one number, then predict everything else.

The number is **`p`, the probability the target accepts a single draft token.** It's a property of
the model pair and the workload, and it is roughly invariant in the draft depth `k` — which is what
makes it useful for prediction.

The obvious alternative, `accepted / proposed`, is **not** invariant and will mislead you. Measured
on a real run tonight:

| depth | accepted / proposed | mean accepted per round | recovered `p` |
|---|---|---|---|
| k=2 | 0.70 | 1.40 | **0.785** |
| k=4 | 0.50 | 2.00 | **0.741** |

The naive ratio collapses from 0.70 to 0.50 while nothing about the models changed. Raising `k` just
adds deep positions that are unlikely to be reached, since verification stops at the first
rejection. `p` barely moves.

Once you have `p`, the rest is arithmetic. Verification is sequential, so expected accepted tokens
per round is a truncated geometric sum, and a round costs one target pass plus `k` draft passes:

```
E[accepted] = p(1 - p^k) / (1 - p)
speedup(p, k) = (E[accepted] + 1) / (1 + c·k)        c = draft cost / target cost
```

Which gives the two outputs that actually matter:

**Optimal depth.** At p=0.77 and c=0.25, predicted speedup peaks at **k=3, 1.61×** — going deeper
makes it worse, and the measured run agreed: k=4 was slower than k=2 despite emitting more tokens
per target pass.

**Breakeven acceptance** — below this, speculation loses outright:

| depth | `p` must exceed |
|---|---|
| k=1 | 0.250 |
| k=2 | 0.366 |
| k=4 | 0.519 |
| k=8 | 0.677 |

## The application

A CLI: point it at your prompts and candidate draft models, and it reports whether to enable
speculative decoding, which draft model to use, and at what depth.

## Status

Early. The acceptance model is built and tested; measurement harness and CLI are next.

- [x] Acceptance model: recover `p`, predict speedup, optimal depth, breakeven
- [ ] Measurement harness over a prompt set with `mlx_lm`
- [ ] Per-task-domain acceptance — code vs prose vs chat differ, and that is the point
- [ ] Draft model comparison
- [ ] Energy: rejected drafts are burned compute, so speculation trades joules for latency
- [ ] CLI

## Honest limits

`p` is not perfectly invariant — 0.785 at k=2 versus 0.741 at k=4 in tonight's run. That is one
short generation over ~25 rounds, so it may be sampling noise, or deeper positions may genuinely be
harder to predict. Measuring which is a task, not an assumption.

The cost model treats a draft pass as a fixed fraction of a target pass. That ignores batch-size
effects, and batch size is known to matter: published benchmarks find speculation saves energy at
small batch and *costs* ~26% more at batch 128.

## Prior art

Well-trodden, and the point here is a working tool rather than novelty.
[Leviathan et al.](https://arxiv.org/abs/2211.17192) introduced the method and the acceptance
arithmetic. [An Interpretable Latency Model for Speculative Decoding in LLM
Serving](https://arxiv.org/pdf/2605.15051) models the latency side.
[Benchmarking the Energy Savings with Speculative Decoding
Strategies](https://arxiv.org/pdf/2602.09113) measured the energy crossover and found it is driven
by batch size.

## Running

```bash
uv sync
uv run pytest
```
