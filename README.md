# spec-decode-advisor

Tells you whether speculative decoding will pay off on *your* workload, and at what draft depth.

## The question

Speculative decoding is advertised as a 2–3× speedup. Whether you get it depends on how often
your target model accepts your draft model's guesses, which depends on the model pair, the task,
and the batch size. Below roughly 50% acceptance the verification overhead makes it a net loss.

So the honest answer to "should I turn this on?" is "measure it" — and there is no tool that does
the measuring. Teams enable it, see a number, and can't tell whether a different draft model or a
different depth would be better, or whether they should have left it off.

## The answers so far

Findings 1–6 are from experiment 01: Qwen2.5-1.5B-Instruct-4bit with a 0.5B draft, 30 prompts
across 5 task domains, depths {0,1,2,4,8}. Findings 7–10 are from experiments 02 and 03: a 7B
target against 0.5B, 1.5B and 3B drafts at depths {0,1,2,4}, and a direct timing of the target's
verification pass. All greedy, M4 / 16 GB. Full numbers in `results/`.

**1. Acceptance varies enough by task domain that a single global draft depth is wrong.**
The ordering was predicted in the prompt set before the run and came out exactly as stated.

| domain | `p` | 95% CI | optimal `k` |
|---|---|---|---|
| copy (output restates the prompt) | **0.865** | [0.822, 0.923] | 3 |
| code | 0.805 | [0.761, 0.847] | 2 |
| factual | 0.750 | [0.694, 0.796] | 2 |
| chat | 0.668 | [0.634, 0.708] | 1 |
| prose | **0.610** | [0.583, 0.637] | 1 |

A 0.255 spread in `p` on one model pair, and optimal depth ranging 1→3. This is the tool's reason
to exist, and it is now measured rather than asserted.

**2. `p` is stable in draft depth, so one measurement does predict the others.**

| depth | pooled `p` | 95% CI | `accepted / proposed` |
|---|---|---|---|
| k=1 | 0.722 | [0.683, 0.765] | 0.722 |
| k=2 | 0.726 | [0.689, 0.764] | 0.622 |
| k=4 | 0.728 | [0.692, 0.766] | 0.472 |
| k=8 | 0.735 | [0.698, 0.773] | 0.302 |

The naive ratio collapses from 0.722 to 0.302 while `p` moves 0.013 and every CI overlaps.
Verification stops at the first rejection, so raising `k` just adds deep positions that are rarely
reached; `accepted / proposed` counts those as rejections and `p` does not.

**3. But acceptance is not i.i.d. across draft positions — it is context-dependent, and the
geometric model is a useful misspecification rather than a true one.**

Estimating acceptance *per position* as a hazard, `p_j = #(a ≥ j) / #(a ≥ j−1)`, at k=8:

```
p1=0.668  p2=0.699  p3=0.730  p4=0.760  p5=0.813  p6=0.880  p7=0.842  p8=0.826
```

Acceptance *rises* with depth, and the bootstrap CI on `p_1 − p_k` excludes zero at k=2, 4 and 8.
This is survivor selection, not deep positions being genuinely easier: the only rounds that reach
position 6 are rounds sitting in easy stretches of text, so conditioning on survival selects a
progressively easier subpopulation. Stratifying by domain removes about a third of the rise
(pooled −0.158; within-domain −0.013 for code, −0.087 for prose), and the remainder is
heterogeneity *within* a single response — markdown scaffolding and list boilerplate accept near
1.0 while content words accept far lower.

The practical consequence is narrow: `p` survives as a summary statistic, because it is stable and
predictive, which is all the prediction layer needs. What does not survive is the claim that
acceptance is a per-token property of the model pair. It is a property of the *local context*.

**4. Calibrating the cost model by timing the two models separately is structurally wrong.**

Measured speedups were below 1.0 at every depth — 0.87, 0.86, 0.76, 0.56 at k=1,2,4,8 — while the
cost model predicted 1.17 at k=1. Solving for the missing per-round cost gave 0.510, 0.688, 1.000,
1.574, which grows with `k`, so it is two wrong terms rather than one omitted constant:

```
timed separately:   round_cost(k) = 1.000 + 0.468k
fitted from data:   round_cost(k) = 1.378 + 0.619k
```

`fixed_cost` is not one target pass, because every round trims both KV caches back to the accepted
prefix via `_rewind_cache` whether or not anything was rejected. And a draft *position* costs more
(0.619) than a standalone draft pass (0.468). Experiment 01 blamed a per-token sync in the loop;
experiment 03 found the actual cause, which is that the target's verification pass is not free per
position (finding 9). Fitted, the model reproduces the measured curve to within 0.01 at all four
depths — the linear form was right, only the calibration was wrong.

So the advisor fits `round_cost` from observed speedups at two or more depths. Timing the models
separately cannot see per-round overhead at all.

**5. On this pair, on this hardware, speculation never pays.** Best case is `copy` at 1.01×; the
other four domains come out as "do not speculate". That is the correct output for a 1.5B target
with a 0.5B draft on an M4, where fixed per-pass overhead dominates, and it is a real result rather
than a broken measurement. Finding 7 shows the same draft pays once the target is 7B, so this is a
statement about target size, not about the machine.

**6. Speculative decoding is lossless in distribution but not bitwise-reproducible against
non-speculative decoding.** Greedy decoding must emit identical text at every depth. It didn't: 18
of 120 comparisons diverged, from 5 prompts, all in the high-entropy domains (prose, chat, factual)
and none in code or copy. Every speculative depth agreed with every other and only the unspeculated
baseline differed, with divergence at the same character position across depths.

That signature is batch invariance. Verification runs the target over `k+1` positions in one pass;
plain decode runs one position. Different matmul shapes take different kernel and reduction paths,
logits differ in the low bits, and the argmax flips wherever the top two tokens are near-tied.
High-entropy text has near-ties constantly; code does not. Anyone running golden-output regression
tests across a speculative-decoding rollout should expect spurious failures clustered in exactly
the high-entropy outputs.

**7. Speculation pays on a 7B target, with the smallest draft, in every domain.** Same 30 prompts,
Qwen2.5-7B-Instruct-4bit target, 0.5B draft. Measured speedup at the best depth per domain: code
1.26×, chat 1.21×, copy 1.19×, factual 1.15×, prose 1.05×; pooled 1.13× at k=1, 1.15× at k=2,
1.03× at k=4. Finding 5's "never" was about a 1.5B target whose pass was too cheap to amortize the
per-round overhead, not about this hardware. Controls: each draft ran against its own interleaved
k=0 baseline; the three sessions' baselines were byte-identical on all 30 prompts and their speed
agreed to a 7% median CV; 16, 17 and 14 of 90 lossless comparisons diverged, the same
batch-invariance signature as finding 6.

**8. A bigger draft loses everywhere, and on these cost ratios it cannot win.** This is the
prediction that failed. Written before the run: the best draft would depend on the domain, with a
larger draft winning where acceptance is low. Measured:

| draft | param ratio | pooled `p` | fitted per-position cost | fitted fixed cost | best measured speedup |
|---|---|---|---|---|---|
| 0.5B | 0.064 | 0.687 | 0.383 | 1.112 | **1.15×** |
| 1.5B | 0.202 | 0.739 | 0.535 | 1.156 | 1.03× |
| 3B | 0.406 | 0.773 | 0.702 | 1.268 | 0.90× |

Acceptance rises 0.03–0.05 per doubling of draft size, in every domain. Per-position cost rises
0.15–0.17 per step. The trade never closes: at k=1 or k=2 the 3B draft could not match the 0.5B's
measured speedup even at perfect acceptance (ceilings 1.01× and 1.12× against 1.13× and 1.15×),
and at k=4 it would need `p` = 0.91 against a measured 0.77. The 1.5B would need 0.85–0.90 against
0.74. So on this machine "which draft" has a boring answer, smallest, and the advisor reports it
that way. Whether that holds where a draft pass is cheap relative to the target (a 70B target on an
H100) is what the tool must measure rather than assume.

Domain ordering held with one swap: code overtook copy on the 7B target, for all three drafts
(0.830 vs 0.800 with the 0.5B draft). One reading is that the 7B paraphrases where the 1.5B
restated, so "copy" acceptance depends on how literally the *target* copies. Untested.

**9. Verification is not free on this hardware, and that is where the "in-loop premium" lives.**
The fitted per-position cost sat a near-constant 0.25–0.29 target-passes above each draft's
standalone pass cost (0.13, 0.25, 0.45), for three drafts of very different size. A
draft-independent cost is not a draft cost. Timing the 7B forward pass directly with a warm cache:
57 ms for one token, and +0.21 to +0.30 of that per additional token verified in the same pass,
mean 0.25 (`results/03_verification_cost.json`). So

```
round_cost(k) = fixed_cost + (c_draft + c_verify) · k        c_verify ≈ 0.25 here
```

and `fit` recovers the sum. Speculative decoding's premise is that a k+1-token pass reads the same
weights as a 1-token pass and costs about the same. On an M4 with 4-bit weights a 5-token
verification costs 1.85–2.2× a 1-token pass. On an H100, where decode is deeply bandwidth-bound,
`c_verify` should be near zero, which is why the 2–3× headline numbers come from that class of
hardware. It is a 30-second measurement on any machine, and the advisor should take it directly
rather than infer it.

**10. The fixed per-round cost fell toward 1.0 as predicted for the smallest draft, and rose with
draft size, which was not predicted.** 1.378 on the 1.5B target became 1.112 on the 7B target with
the same 0.5B draft: a constant wall-clock bookkeeping cost shrinks in units of a longer target
pass. But 1.156 and 1.268 for the 1.5B and 3B drafts say the fixed cost has a draft-dependent part.
The obvious candidate, the draft's prompt prefill (once per generation, proportional to draft size),
was tested by fitting per domain — copy prompts are 5× longer — and does not show; it is also
quantitatively too small. Per-domain fits scatter ±0.2 on six prompts each, so only the 3B's excess
is clearly real. Open. One untested candidate is memory residency: the 3B session peaked at 6.3 GB
on a machine with about 5 GB free, so the draft's weights may be evicted while the target runs and
re-faulted each round. That would not exist with everything resident in HBM.

## The approach

Run the workload, count accepted tokens per round, and estimate two things.

**Acceptance**, per draft position, as a hazard — because verification stops at the first rejection,
position `j` is only observed when everything before it was accepted, which is exactly the
conditioning a hazard assumes. Pooling gives the MLE in closed form:

```
p = accepted / (accepted + observed rejections)
```

A round that accepted all `k` draft tokens is right-censored — it ran out of draft, not out of
agreement — so it contributes to the numerator only. That censoring is the whole difference from
`accepted / proposed`.

**Cost**, by fitting `round_cost(k) = fixed_cost + draft_cost_ratio · k` to measured speedups, since

```
speedup(p, k) = (E[accepted] + 1) / round_cost(k)      E[accepted] = p(1 − p^k)/(1 − p)
```

pins one round cost per observation.

Uncertainty is bootstrapped over *prompts*, not rounds: rounds within one generation are strongly
autocorrelated, so treating them as independent draws gives intervals that don't survive a second
run.

## The application

A CLI: point it at your prompts and candidate draft models, and it reports whether to enable
speculative decoding, which draft model to use, and at what depth.

## Status

- [x] Acceptance model: expected tokens per round, optimal depth, breakeven
- [x] Per-position hazard estimator with prompt-level bootstrap
- [x] Measurement harness over a prompt set with `mlx_lm`
- [x] Per-task-domain acceptance
- [x] Fitted cost calibration
- [x] Draft model comparison
- [x] Verification cost per position, measured directly
- [ ] Energy: rejected drafts are burned compute, so speculation trades joules for latency
- [ ] Batch-size effects, which published work says drive the energy crossover
- [ ] CLI

## Threats to validity

**Batch size is unmeasured, and it is the single largest omission.** Everything here is batch 1,
where decode is memory-bandwidth-bound and the spare FLOPs that make verification cheap actually
exist. At production batch sizes the engine approaches compute-saturation, those spare FLOPs are
gone, and speculation spends real compute on tokens it discards. Published benchmarks find
speculation saves energy at small batch and costs ~26% more at batch 128. No conclusion here about
whether to enable speculation transfers to a loaded server.

**The cost calibration is hardware- and scale-specific, and does not transfer.** Every fitted
constant here is an M4 number. The one that matters most is `c_verify` ≈ 0.25: on this GPU each
extra token the target verifies costs a quarter of a full pass, and that alone caps speedup well
below what an H100 would show for the same pair, where the pass is bandwidth-bound and `c_verify`
should be near zero. The small drafts are also overhead-bound here (a 0.5B pass costs 6× its
parameter ratio), which is why finding 8's "smallest draft wins" may invert where a 3B pass is
genuinely 0.04 of a 70B pass. The *method* — fit the terms from measured speedups, and measure
`c_verify` directly — is what transfers. The constants do not.

**Experiments 02 and 03 ran under memory pressure.** The machine had roughly 5 GB free with other
applications open; the 3B session peaked at 6.3 GB. Interleaved per-prompt baselines control for
drift (session means 16.9, 15.3 and 17.5 tok/s; per-prompt CV 7%), and a first run of experiment
03 gave 48 ms and 0.28–0.30 against the committed 57 ms and 0.25. Finding 10's draft-dependent
fixed cost may be a residency artifact of exactly this.

**Acceptance is measured greedily on purpose, and temperature > 0 would understate it.** `mlx_lm`
verifies by exact token match: it samples the target token independently and accepts only when it
equals the draft's, giving acceptance `Σ p(x)q(x)`. Production engines use the Leviathan rejection
rule, `min(1, p/q)`, giving `Σ min(p(x), q(x))`, which is never smaller. Above temperature 0 an
`mlx_lm` measurement is a lower bound on what vLLM would achieve with the same pair, not an estimate
of it — and the gap widens as distributions flatten, so it is worst exactly where you would most
want the number. At temperature 0 both rules reduce to argmax equality and agree exactly.

**Rounds are censored three ways, all biasing `p` downward, and all dropped rather than corrected.**
`speculative_generate_step` sets `num_draft = min(max_tokens − ntoks, k)`, so rounds near the end of
a generation draft fewer than `k` tokens; hitting `max_tokens` mid-verification ends a round with no
target token; and `stream_generate` breaks on EOS before a round finishes. Dropping them costs
sample size but avoids the bias.

**One model family, two target sizes.** Every acceptance number is Qwen2.5 drafting for Qwen2.5.
The 7B target is the largest that fits comfortably in 16 GB, so the regime where a draft is cheap
relative to its target is extrapolated from the roofline, not measured. The domain *ordering* was
stable across four pairs except that code and copy swapped at 7B; the levels are pair-specific.

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
uv run python experiments/01_depth_invariance.py   # ~6 min
uv run python experiments/02_draft_comparison.py   # ~50 min; downloads 7B and 3B (~6 GB)
uv run python experiments/03_verification_cost.py  # ~1 min
```
