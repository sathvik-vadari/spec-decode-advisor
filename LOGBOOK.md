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

## 2026-09-03 — the invariance test, and the cost model was the problem

Q: is `p` really invariant in draft depth, and does acceptance differ by task domain?

Built the measurement harness, the prompt set, and a per-position estimator. 54 tests pass.
Numbers in `results/01_depth_invariance.json`.

**Caught the instrument before trusting it.** I was recovering `p` by inverting
`E[j] = p(1-p^k)/(1-p)`, which is derived *from* the assumption that `p` is constant across
positions. So it was being used to test the thing it presupposes -- feed it data where acceptance
collapses at depth 3 and it still returns a tidy scalar. Fixed by estimating acceptance per
position as a hazard, `p_j = #(a >= j) / #(a >= j-1)`, which is a conditional frequency count and
assumes nothing. Verification stops at the first rejection, so position `j` is only observed when
everything before it was accepted -- exactly the conditioning a hazard wants. Lesson worth keeping:
if the estimator is a rearrangement of the model, it cannot falsify the model.

**Answer to Q1 is yes and no, and the "no" is the interesting half.** Pooled `p` is stable --
0.722, 0.726, 0.728, 0.735 at k=1,2,4,8, CIs overlapping almost entirely -- so last night's
0.785-vs-0.741 was noise, and the prediction layer holds. But the hazards *rise*: 0.668 at position
1 to 0.880 at position 6, and the bootstrap CI on `p1 - pk` excludes zero at k=2, 4 and 8. So
acceptance is not i.i.d. and the geometric model is misspecified.

The mechanism is survivor selection, which is frailty in the survival-analysis sense: the only
rounds reaching position 6 are rounds inside easy stretches of text, so conditioning on survival
selects a progressively easier subpopulation and manufactures a rising hazard out of a mixture of
constant ones. Tested it by stratifying within domain, which should shrink the rise -- pooled
-0.158, code -0.013, prose -0.087, but copy -0.174 and factual -0.196. So domain explains about a
third and the rest is heterogeneity inside a single response: markdown scaffolding and list bullets
accept near 1.0, content words much lower. Acceptance is a property of the local context, not of
the model pair. `p` survives as a summary; the mechanism story in the README did not.

**Q2 came out exactly as pre-registered.** copy 0.865, code 0.805, factual 0.750, chat 0.668, prose
0.610. Wrote the predicted ordering into the prompt-set docstring before running so it could fail,
and it didn't. 0.255 spread and optimal depth 1 to 3 on one model pair -- that is the argument for
the tool existing.

**The real finding is that my cost calibration was wrong twice.** Measured speedup came out below
1.0 at every depth -- 0.87, 0.86, 0.76, 0.56 -- against a predicted 1.17 at k=1. Solving for the
missing per-round cost gave 0.510, 0.688, 1.000, 1.574, growing with k, so not one omitted
constant. Two errors: `fixed_cost` is 1.378 not 1.0, because every round trims both KV caches back
to the accepted prefix through `_rewind_cache` whether anything was rejected or not; and a draft
pass costs 0.619 inside the loop against 0.468 timed standalone, because the loop forces a
per-token sync. Fitted, the model reproduces the measured curve to 0.01 at all four depths -- the
linear form was fine, the calibration wasn't.

That is a design change, not a constant change. Timing the two models separately is *structurally*
unable to see per-round overhead, so the advisor now fits both terms from observed speedups at two
or more depths. On this pair it flips four of five domains from a predicted win to "do not
speculate", and the best case, copy, only reaches 1.01x. Honest answer for a 1.5B/0.5B pair on an
M4: don't.

**Two things about mlx_lm that matter for transfer.** It verifies by exact token match -- samples
the target token independently and accepts only on equality -- so acceptance is `sum p(x)q(x)`,
where production engines use the Leviathan rule and get `sum min(p(x), q(x))`, which is never
smaller. Above temperature 0 anything measured here is a lower bound on vLLM, not an estimate, and
the gap widens as distributions flatten. At temperature 0 both reduce to argmax equality. That is
why the whole sweep is greedy.

And the free correctness check failed in an informative way. Greedy speculation must be
byte-identical at every depth; 18 of 120 comparisons diverged, from 5 prompts, all in prose/chat/
factual and none in code/copy, with every speculative depth agreeing with the others and only the
unspeculated baseline differing, at the same character position across depths. That is batch
invariance: verification runs the target over k+1 positions, plain decode over 1, different matmul
shapes take different reduction orders, and the argmax flips wherever the top two tokens are nearly
tied. So speculation preserves the output distribution but not bitwise reproducibility against
plain decode -- which will show up as spurious golden-output test failures on any real rollout,
clustered in the high-entropy outputs.

Not yet honest about: batch size, still entirely unmeasured and still the largest hole, since batch
1 is the regime where the spare FLOPs verification depends on actually exist.

Next: draft-model comparison, so the tool can answer "which draft model" and not only "what depth".
