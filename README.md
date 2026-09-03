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

Measured on Qwen2.5-1.5B-Instruct-4bit with a 0.5B draft, 30 prompts across 5 task domains,
depths {0,1,2,4,8}, greedy, M4 / 16 GB. Full numbers in `results/01_depth_invariance.json`.

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
prefix via `_rewind_cache` whether or not anything was rejected. And a draft pass costs more inside
the speculative loop (0.619) than the same model generating alone (0.468), because the loop forces
a per-token sync. Fitted, the model reproduces the measured curve to within 0.01 at all four
depths — the linear form was right, only the calibration was wrong.

So the advisor fits `round_cost` from observed speedups at two or more depths. Timing the models
separately cannot see per-round overhead at all.

**5. On this pair, on this hardware, speculation never pays.** Best case is `copy` at 1.01×; the
other four domains come out as "do not speculate". That is the correct output for a 1.5B target
with a 0.5B draft on an M4, where fixed per-pass overhead dominates, and it is a real result rather
than a broken measurement — see *Threats to validity* for what it does and does not generalise to.

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
- [ ] Draft model comparison
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

**The cost calibration is hardware- and scale-specific, and does not transfer.** `fixed_cost` of
1.378 and `draft_cost_ratio` of 0.619 are M4 numbers for a 1.5B/0.5B pair, where fixed per-pass
overhead dominates because the weights are small enough to stream quickly. On an H100 with a 70B
target, a forward pass is bandwidth-bound and the ratio tracks parameter count much more closely.
The *method* — fit both terms from measured speedups — is what transfers. The constants do not, and
finding 5 ("speculation never pays") is a statement about this pair on this machine only.

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

**One model pair.** Every acceptance number is Qwen2.5 1.5B/0.5B. The domain *ordering* should be
robust — it follows from how predictable each domain's text is — but the levels are pair-specific.

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
uv run python experiments/01_depth_invariance.py
```
