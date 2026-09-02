# Logbook

## 2026-09-02 — the acceptance model

Q: what single number predicts whether speculative decoding pays, and can it be measured?

`p`, the per-token acceptance probability. Built the arithmetic (`model.py`), 27 tests pass.

Measured a real pair first — Qwen2.5-1.5B target, 0.5B draft, on the M4:

    no speculation   72 tok  2.20s  32.8 tok/s
    k=2              72 tok  0.92s  78.1 tok/s   accepted 42/60,  2.40 tok/target pass
    k=4              72 tok  0.96s  75.1 tok/s   accepted 48/96,  3.00 tok/target pass

**k=4 emits more tokens per target pass and is still slower than k=2.** That is the whole tension in
one line: deeper drafting buys more tokens per verification but pays for more draft passes and hits
lower acceptance at the deep positions.

The thing I nearly got wrong: `accepted/proposed` looks like the acceptance rate and is not. It read
0.70 at k=2 and 0.50 at k=4 — a huge swing with the models unchanged. Verification stops at the
first rejection, so raising k just adds positions that are rarely reached. Recovering the per-token
`p` by inverting the truncated geometric sum gives 0.785 and 0.741, roughly stable. `p` is the
invariant, and it is what transfers across depths.

`mlx_lm` gives this away free: `GenerationResponse.from_draft` per token. Rounds = count of
non-draft tokens, accepted = count of draft tokens. No patching needed.

Model then predicts optimal k=3 at 1.61x for p=0.77, c=0.25 — testable, and the k=4-worse-than-k=2
measurement is consistent with a peak below 4.

Not yet honest about: whether p really is depth-invariant (one short run, ~25 rounds, could be
noise) and batch-size effects, which published work says drive the energy crossover.

Next: the measurement harness over a real prompt set, and per-domain acceptance — code vs prose
should differ, and if they do, a single global draft depth is the wrong default.
