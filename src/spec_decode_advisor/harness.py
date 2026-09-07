"""Running a workload and recovering per-round acceptance from it.

`mlx_lm` hands this over without any patching: `GenerationResponse.from_draft`
says whether each emitted token came from the draft model. A round is a run of
accepted draft tokens terminated by exactly one token from the target, so the
token stream segments into rounds by looking for the `from_draft == False`
boundaries, and the number of `True`s before each boundary is that round's
accepted count.

Three ways that count is silently wrong, all of which bias `p` downward, and all
of which are handled here rather than assumed away:

1.  `speculative_generate_step` sets `num_draft = min(max_tokens - ntoks, k)`,
    so rounds near the end of a generation draft *fewer* than k tokens. Such a
    round can accept every token it drafted and still look like an early
    rejection. Rounds whose effective depth was reduced are dropped.

2.  Hitting `max_tokens` inside the accept loop ends the round with no target
    token, so the observed count is a lower bound on what the round would have
    accepted. Such a round has no terminator and is dropped.

3.  `stream_generate` breaks on EOS *before* the round finishes. If EOS arrived
    as an accepted draft token the round is truncated the same way; if it
    arrived as the target's own token the round closed normally and is real.

All three reduce to one rule: a round counts only if it is terminated by a
`from_draft == False` token and its effective depth was the full k.

Everything is measured greedily, and that is a transferability decision rather
than a convenience. `mlx_lm` verifies by exact token match -- it samples the
target token independently and accepts only when it equals the draft's -- so its
acceptance probability is sum_x p(x)q(x). Production engines use the Leviathan
rejection rule, which accepts with probability sum_x min(p(x), q(x)), and that
is never smaller. At temperature > 0 an `mlx_lm` measurement is therefore a
lower bound on what vLLM would achieve with the same pair, not an estimate of
it. At temperature 0 both rules collapse to argmax equality and agree exactly,
so greedy is the only setting where these numbers transfer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from .acceptance import PromptRounds
from .prompts import Prompt


def segment_rounds(from_draft: Sequence[bool], k: int, max_tokens: int) -> list[int]:
    """Accepted-token counts for each usable round, given the per-token draft flags.

    Pure so the censoring rules can be tested without loading a model.
    """
    rounds: list[int] = []
    start = 0
    current = 0
    for i, is_draft in enumerate(from_draft):
        if is_draft:
            current += 1
            continue
        # A target token closes the round.
        if min(max_tokens - start, k) == k:
            rounds.append(current)
        start = i + 1
        current = 0
    # Anything left has no terminating target token: truncated, so discard it.
    return rounds


@dataclass(frozen=True, slots=True)
class RunResult:
    prompt_id: str
    domain: str
    k: int
    n_tokens: int
    wall_s: float
    text: str
    rounds: tuple[int, ...]
    dropped_rounds: int
    finish_reason: str

    @property
    def tokens_per_s(self) -> float:
        return self.n_tokens / self.wall_s if self.wall_s else float("nan")


@dataclass
class Harness:
    """Holds the model pair so a sweep loads weights once."""

    target_id: str
    draft_id: str
    max_tokens: int = 128
    _target: object = field(default=None, repr=False)
    _draft: object = field(default=None, repr=False)
    _tokenizer: object = field(default=None, repr=False)

    def load(self) -> "Harness":
        from mlx_lm import load

        self._target, self._tokenizer = load(self.target_id)
        self._draft, draft_tok = load(self.draft_id)
        self._check_vocab(draft_tok)
        return self

    def _check_vocab(self, draft_tok) -> None:
        # Speculative decoding compares raw token ids, so a mismatched vocabulary
        # would not error -- it would just never accept anything.
        if draft_tok.vocab_size != self._tokenizer.vocab_size:
            raise ValueError(
                f"tokenizer mismatch: target vocab {self._tokenizer.vocab_size}, "
                f"draft vocab {draft_tok.vocab_size}"
            )

    def swap_draft(self, draft_id: str) -> "Harness":
        """Replace the draft model and keep the target where it is.

        Comparing candidate drafts against one target should not reload the
        target per candidate: the load is slow, and a fresh load changes the
        memory layout the timing runs against. The old draft is released before
        the new one is loaded so peak memory is target + one draft, not target +
        every candidate -- on a 16 GB machine with a 7B target that is the
        difference between fitting and swapping.
        """
        import gc

        import mlx.core as mx
        from mlx_lm import load

        self._draft = None
        gc.collect()
        try:
            mx.clear_cache()
        except AttributeError:  # older mlx
            mx.metal.clear_cache()
        draft, draft_tok = load(draft_id)
        self._check_vocab(draft_tok)
        self._draft = draft
        self.draft_id = draft_id
        return self

    @property
    def target(self):
        return self._target

    @property
    def draft(self):
        return self._draft

    @property
    def tokenizer(self):
        return self._tokenizer

    def _encode(self, prompt: Prompt) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.text}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def encode_ids(self, prompt: Prompt):
        """The chat-templated prompt as a token-id array, for timing passes."""
        import mlx.core as mx

        return mx.array(self._tokenizer.encode(self._encode(prompt)))

    def run(self, prompt: Prompt, k: int) -> RunResult:
        """One greedy generation. `k = 0` means no speculation at all."""
        from mlx_lm import stream_generate

        text_prompt = self._encode(prompt)
        kwargs = {}
        if k > 0:
            kwargs["draft_model"] = self._draft
            kwargs["num_draft_tokens"] = k

        flags: list[bool] = []
        chunks: list[str] = []
        finish_reason = "length"
        tic = time.perf_counter()
        for resp in stream_generate(
            self._target,
            self._tokenizer,
            text_prompt,
            max_tokens=self.max_tokens,
            **kwargs,
        ):
            flags.append(bool(resp.from_draft))
            chunks.append(resp.text)
            if resp.finish_reason:
                finish_reason = resp.finish_reason
        wall = time.perf_counter() - tic

        if k > 0:
            rounds = segment_rounds(flags, k, self.max_tokens)
            total_rounds = sum(1 for f in flags if not f)
            dropped = total_rounds - len(rounds)
        else:
            rounds, dropped = [], 0

        return RunResult(
            prompt_id=prompt.prompt_id,
            domain=prompt.domain,
            k=k,
            n_tokens=len(flags),
            wall_s=wall,
            text="".join(chunks),
            rounds=tuple(rounds),
            dropped_rounds=dropped,
            finish_reason=finish_reason,
        )

    def run_bare(self, prompt: Prompt, which: str) -> RunResult:
        """Non-speculative generation with one model alone, for cost calibration.

        The cost model's `c` -- a draft pass as a fraction of a target pass -- is
        currently a guessed 0.25. Timing each model by itself on the same prompts
        measures it, which matters because `c` is what decides whether deeper
        drafting pays.
        """
        from mlx_lm import stream_generate

        model = self._target if which == "target" else self._draft
        text_prompt = self._encode(prompt)
        n = 0
        finish_reason = "length"
        tic = time.perf_counter()
        for resp in stream_generate(
            model, self._tokenizer, text_prompt, max_tokens=self.max_tokens
        ):
            n += 1
            if resp.finish_reason:
                finish_reason = resp.finish_reason
        wall = time.perf_counter() - tic
        return RunResult(
            prompt_id=prompt.prompt_id,
            domain=f"_{which}",
            k=0,
            n_tokens=n,
            wall_s=wall,
            text="",
            rounds=(),
            dropped_rounds=0,
            finish_reason=finish_reason,
        )

    def measure_cost_ratio(
        self, prompts: Sequence[Prompt]
    ) -> tuple[float, list[RunResult]]:
        """Seconds per draft pass over seconds per target pass, measured."""
        runs = []
        for which in ("target", "draft"):
            self.run_bare(prompts[0], which)  # warm-up, discarded
            for prompt in prompts:
                runs.append(self.run_bare(prompt, which))
        per_token = {}
        for which in ("target", "draft"):
            rs = [r for r in runs if r.domain == f"_{which}"]
            per_token[which] = sum(r.wall_s for r in rs) / sum(r.n_tokens for r in rs)
        return per_token["draft"] / per_token["target"], runs

    def sweep(
        self, prompts: Sequence[Prompt], depths: Sequence[int], warmup: bool = True
    ) -> list[RunResult]:
        """Every prompt at every depth. Depth order is inner so drift in machine
        state (thermal, memory pressure) hits all depths of a prompt alike rather
        than accumulating against whichever depth ran last."""
        if warmup and prompts:
            self.run(prompts[0], max(depths))  # discard: first run pays lazy init
        results = []
        for prompt in prompts:
            for k in depths:
                results.append(self.run(prompt, k))
        return results


def to_prompt_rounds(results: Sequence[RunResult]) -> list[PromptRounds]:
    return [
        PromptRounds(
            prompt_id=r.prompt_id, domain=r.domain, k=r.k, accepted=r.rounds
        )
        for r in results
        if r.k > 0 and r.rounds
    ]
