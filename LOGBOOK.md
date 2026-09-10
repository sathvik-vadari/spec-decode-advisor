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

## 2026-09-04 — which draft, and where the in-loop premium actually lives

Q: against one target, which draft should you use, and does it depend on the domain? And does
finding 5's "never pays" survive a bigger target?

Built draft swapping, a cross-draft recommender, experiment 02 (7B target x {0.5B, 1.5B, 3B} x 30
prompts x k in {0,1,2,4}, 49 min under memory pressure) and experiment 03 (the target pass timed
against tokens verified). 62 tests. Five predictions written into the experiment 02 docstring and
committed before the run.

Scorecard. P1, acceptance rises with draft size in every domain: yes, +0.03 to +0.05 per doubling,
but code overtook copy on the 7B target for every draft. P2, per-position cost exceeds the parameter
ratio most for the smallest draft: yes, 6x for 0.5B against 1.7x for 3B, and standalone timing
understates it for every draft. P3, fixed cost falls toward 1.0: yes for the 0.5B draft, 1.378 to
1.112 -- but it rises with draft size (1.16, 1.27), which I did not predict. P4, speculation pays on
7B: yes, 1.15x pooled, 1.26x on code, every domain above 1.0 with the 0.5B draft. So finding 5 was
about target size, not about the machine. P5, the best draft depends on the domain: **no.** 0.5B
wins everywhere, and at k <= 2 the 3B could not have matched it even at perfect acceptance. The
"which draft" question has a null answer on this hardware: smallest.

The finding I was not looking for. The fitted per-position premium over each draft's standalone
pass cost was 0.25, 0.29, 0.25 for three drafts of very different size. A draft-independent cost is
not a draft cost. Timed the 7B pass at 1/2/3/5/9 tokens with a warm cache: +0.21 to +0.30 of a
one-token pass per extra token, mean 0.25. Verification is not free on an M4 -- a 5-token pass costs
about 2x a 1-token pass -- and that is the whole premium. Experiment 01's "per-token sync" story was
wrong; corrected in the README and the CostModel docstring. `c_verify` is the constant that decides
whether speculation pays, it is hardware-specific, and it is a 30-second measurement. The advisor
should measure it directly rather than infer it from the fit.

Instrument checks: baseline outputs byte-identical across the three sessions on all 30 prompts;
baseline tok/s CV 7% across sessions; 16/17/14 of 90 lossless mismatches, same batch-invariance
signature as before. Memory pressure was real (about 5 GB free, 3B session peaked at 6.3 GB); a
first run of experiment 03 gave 48 ms and 0.28-0.30, the committed rerun 57 ms and 0.25.

Dead end: tried to explain the draft-dependent fixed cost by the draft's prompt prefill. Fitting per
domain (copy prompts are 5x longer) shows nothing, and it is quantitatively too small anyway. Left
open, with memory residency as the untested candidate.

Not yet honest about: batch size, still. And the 70B-on-H100 regime where c_verify goes to zero is
a roofline argument, not a measurement.

Next: measure c_verify inside the advisor instead of inferring it, then the CLI.

## 2026-09-06 — two timings and one depth replace the sweep

Q: can the cost side of the advisor be calibrated from two 30-second forward-pass timings plus a
single speculative depth, instead of a depth sweep?

Analysis only, against experiments 02 and 03 already on disk, with the pass/fail thresholds fixed
before computing. Built `calibration.py` (least-squares verification cost, draft pass ratio, the
timing wrapper), `from_components` and `fit_fixed_cost`. 73 tests.

Yes, and more cleanly than expected. Least-squares `c_verify` over 1-9 tokens is 0.295; the
mean-of-slopes 0.25 I quoted on the 4th was biased low by the noisy short points, and the README
now carries both. Direct slope = `c_draft` + `c_verify` lands within 0.02 of the fitted slope for all
three drafts. Pin the fixed cost from k=1 and predict k=2 and k=4 held out: max error 0.016 pooled;
per domain mean +0.03, worst 0.15, which is the six-prompt noise floor. Timings only with fixed=1.0
misses by 0.03-0.12, so the one speculative run is necessary -- and it is the same run that
measures `p`. All four checks pass.

What it means for the tool: baseline, one run at k=1, two timings. The sweep was the instrument
that earned this; it is no longer the procedure.

Not yet honest about: batch size, still.

Next: the CLI, which now has a complete and cheap procedure to wrap.

## 2026-09-07 — the CLI, and a replication that half-failed

Q: does the k=1-only procedure, run as the tool, reproduce experiment 02's answer on a different
day?

Built `advisor.py` (analysis, pure, tested on synthetic runs), `cli.py`, prompt-file loading. First
7B run died on a Metal command-buffer GPU timeout 25 prompts into the third draft and, because the
report was written only at the end, lost the two finished drafts with it. Added a one-retry-then-skip
per prompt and a checkpoint after every draft; the rerun hit the same timeout on copy_translate and
survived it. 91 tests.

The replication. Acceptance identical to three decimals everywhere -- greedy is deterministic, so
that is not evidence. k=1 speedups replicated for 0.5B and 1.5B (1.119/1.126, 1.040/1.034), and the
recommendation held: 0.5B, ~1.12x. But the target pass was 89 ms against 57, baseline 10 tok/s
against 17, c_verify 0.38 against 0.30, every c_draft up a third. Cause found mid-run: Low Power
Mode on, battery 24%, Chrome at most of a core. A capped GPU clock slows the compute-bound parts
(small draft passes, extra verified tokens) more than the bandwidth-bound target pass, so the
*ratios* moved, not just the absolutes. Nothing measured in seconds is invariant to machine state,
and the whole cost side is measured in seconds.

The unphysical number: the 3B's k=1 speedup went 0.90 -> 1.08 and its pinned fixed cost came out
0.75, below one target pass. The loop paid less than the sum of its timed parts. Best guess is host
contention: per-token host overhead in plain decode went from ~2 ms to ~9 ms, and a speculative
round amortises it over ~1.8 tokens. A synchronous pass timing can't see that; pipelined generation
hides it. Untested.

Shipped because of it: target timed per draft session with a drift warning; a warning when the
pinned fixed cost is below 0.95; pass time printed first. Tonight's constants would mispredict
exp 02's k=2/k=4 by -0.04 to -0.14, so: plugged in, Low Power Mode off, idle machine, and compare
the pass time with the last run.

Not yet honest about: batch size, still. And the host-contention mechanism is a guess.

Next: energy, or batch size. Batch size is the bigger hole and needs a batched engine; mlx_lm's
speculative path is batch 1. Decide next session.

## 2026-09-09 — batch, the staircase, and the night the answer changed

Q1: how do the cost terms scale with batch? Q2 (unplanned): why did tonight's batch-1 verification
cost read 0.07 when it read 0.29 on the 4th?

Experiment 05 (AC, idle): the 7B pass is flat to batch 4, 2x at 8, 3.3x at 16. c_verify rises
0.09 -> 0.53 (predicted -> 1.0: half right). c_draft is flat, not falling (predicted falling:
wrong). Predicted crossover batch 8 for 0.5B, 2 for 1.5B -- model outputs, since mlx_lm can't
speculate at batch. The fine sweep in total tokens is the real result: flat 1-4, ramp 5-12, then a
staircase with 32-token treads of ~145 ms each. A GEMM tile. ~3.2 TFLOP/s on the treads against a
~95 GB/s weight read on the floor.

Q2 became experiment 06: rerun the 0.5B session on AC, idle, depths 0-6, predictions committed
first. 1.76x at k=2 against 1.15x on the 4th. Code 2.40x. Prose 1.40x. Acceptance identical to three
decimals. P2 (>= 1.35x) passed; P1 (best k >= 3) failed by a hair, k=2 1.759 vs k=3 1.741; P3
(slope < 0.25) failed because the linear model is the wrong shape here. Fitted intercept 0.67 --
below one pass, impossible.

That impossibility was the lead. Round costs 1.14, 1.23, 1.43, 1.78, 2.61 at k=1,2,3,4,6 are
convex; subtract k draft passes and the *measured* pass curve over k+1 tokens and the residual is
~0. Experiment 07: round_cost(k) = k*c_draft + V(k+1), nothing fitted, predicts all five speedups
within 0.05. The 2-parameter fit misses by 0.25 in-sample. The "fixed cost" of experiments 01-04
was curvature in an intercept. Finding 10's draft-dependence of it is *not* explained by this and
stays open.

Two instrument lessons. (1) A freshly loaded model's first timings read ~1.8x high and three
warm-ups didn't clear it; one untimed pass over the schedule does (4.41 vs 4.36 ms). This inflated
c_draft in every CLI run so far. (2) The one-token pass, which normalises everything, read 46.1,
50.7, 43.7 ms in three runs minutes apart while every other count agreed within 1 ms. Now measured
twice, first and last, samples pooled. A per-round residual of 0.02-0.14 passes remains; pinning it
from the k=1 run holds held-out error to ~0.06 regardless.

On cause: the 4th's curve had no free region; tonight's does, on AC *and* on battery at 93% with
LPM off. So it isn't the power source. The 4th followed a 49-min sweep under memory pressure on
~50% charge. Thermal, memory pressure, charge -- not separated. LPM is proven (finding 12). Written
as unresolved.

The advisor now runs on the measured model: pass curve over 1..max_k+1, draft pass, k=0 and k=1
runs, pin the overhead, search. 98 tests.

Not yet honest about: batch-size *speedup* (predicted, not measured); the cause of the 4th's state.

Next: a project-level decision. The tool is complete for its stated question. What remains is either
the batched engine question, which this machine cannot answer, or a new project.

## 2026-09-10 — figures, and an experiment stopped for the right reason

Made the four figures the README had been missing, from committed results via `analysis/figures.py`:
the pass-cost staircase, the two-nights speedup curves with the zero-parameter model, acceptance by
domain and draft, and the batch-scaling pair. Three-slot categorical palette, validated.

Then experiment 08: a 2x2 on what put the machine in Sep 4's state -- thermal history (15-min matmul
burn, its own throughput logged) x memory pressure (host memory held near 15% free) -- with four
predictions committed before the run. Three attempts, none completed, each taught something:

- Attempt 1: 10 GB of `np.ones` allocated in 2.5 s and free memory went *up*. Identical pages
  compress to nothing under the macOS memory compressor. Pressure has to be incompressible.
- Attempt 2, with random data: 3.5 GB took free memory from 33% to 12%. The pressured pass curve
  was the one real datum of the night: one-token pass 43.6 -> 54.3 ms, **but the free region
  stayed** -- V(2) 1.00, V(3) 1.06. Pressure slows the bandwidth-bound weight read; it does not make
  the extra tokens expensive. Sep 4 had both, so pressure alone is not Sep 4. Then the first
  speculative generation at 12% free died with a GPU timeout. Third such timeout in the project,
  every one with memory tight: that correlation is now a recorded fact.
- Attempt 3, with retries and 15%: the reference cell's pass read 57.7 ms against 43.6 fifteen
  minutes earlier, and a GPU timeout hit the warm-up before any pressure was applied. `ps` showed
  why -- an npm build, a Next.js dev server and a VM, each at a quarter to a third of a core.
  Sathvik was working on the machine. Stopped it.

Stopping was the correct reading of findings 12 and 14: a machine-state experiment run under
someone else's workload measures the workload. The code is committed and the design stands. It
needs: plugged in, Low Power Mode off, no dev servers or VMs (`ps -Ao pcpu,comm -r | head`), and 30
idle minutes before the cold cells. Battery went 65% -> 48% in 20 minutes of these runs; that is
also a reason to wait for AC.

Instrument note: V(2) and V(3) read below 1.0 in two cells tonight (0.89, 0.89 and 1.00, 0.97),
i.e. the two- and three-token passes timed faster than the one-token pass. With the first count
measured twice and pooled that should not happen unless the whole run is being perturbed at the
~5 ms level. It was. Under contention the pass curve's first point is the least trustworthy.

Not yet honest about: what put the machine in Sep 4's state. Pressure is now ruled out as the sole
cause; thermal is untested.

Next: rerun 08 on an idle, plugged-in machine. Then the project-level decision.
