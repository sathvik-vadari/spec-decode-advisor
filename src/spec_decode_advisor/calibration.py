"""Measuring the cost terms directly, in seconds, instead of inferring them.

Experiment 02 recovered `round_cost(k) = fixed_cost + slope * k` by fitting
measured speedups at three depths, which needs the whole sweep. Experiment 03
then showed what the slope is made of: a draft pass plus one more token for the
target to verify,

    slope = c_draft + c_verify

and both of those are timings of a single forward pass with a warm KV cache.
Thirty seconds each, no speculative run at all.

The fixed cost is different in kind. It is whatever the loop does per round
beyond the passes -- cache rewinds, host-side bookkeeping, and on this machine
possibly memory residency -- and there is no single pass to time for it. So the
direct route still needs one speculative depth to pin it. Whether "two timings
plus one depth" predicts the remaining depths as well as the full fit does is
experiment 04's question.

The estimators here are pure so they can be tested without a model; the timing
wrapper is the only thing that touches MLX.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class PassLatency:
    """Median seconds for one forward pass over `n` new tokens, per `n`."""

    seconds: dict[int, float]

    @property
    def one_token(self) -> float:
        return self.seconds[1]


def verification_cost(latency: Mapping[int, float]) -> float:
    """Extra cost per verified token, in units of a one-token pass.

    Least-squares slope of pass latency against token count, divided by the
    one-token latency. Zero means verifying k+1 tokens costs the same as
    generating one -- the premise speculative decoding is sold on. Experiment 03
    measured about 0.25 on an M4 for a 4-bit 7B.
    """
    if 1 not in latency or len(latency) < 2:
        raise ValueError("need the one-token latency and at least one other point")
    ns = np.array(sorted(latency), dtype=float)
    ts = np.array([latency[int(n)] for n in ns], dtype=float)
    slope, _ = np.polyfit(ns, ts, 1)
    return float(slope / latency[1])


def draft_pass_ratio(draft: PassLatency, target: PassLatency) -> float:
    """A draft's one-token pass as a fraction of the target's."""
    return draft.one_token / target.one_token


def time_passes(
    model,
    prompt_tokens,
    tokens: Sequence[int] = (1, 2, 3, 5, 9),
    repeats: int = 25,
    warmup: int = 3,
) -> PassLatency:
    """Prefill `prompt_tokens` once, then time a pass over `n` fresh tokens with
    that cache for each `n`, trimming the cache back after every pass so the
    context length is identical across repeats. Median of `repeats` after
    discarding `warmup`. Token identity does not affect timing, so the ids are
    arbitrary but valid."""
    import mlx.core as mx
    from mlx_lm.models import cache as C

    kv = C.make_prompt_cache(model)
    mx.eval(model(prompt_tokens[None], cache=kv))
    out: dict[int, float] = {}
    for n in tokens:
        y = mx.array([100 + i for i in range(n)])
        ts = []
        for i in range(warmup + repeats):
            t = time.perf_counter()
            mx.eval(model(y[None], cache=kv))
            dt = time.perf_counter() - t
            C.trim_prompt_cache(kv, n)
            if i >= warmup:
                ts.append(dt)
        out[n] = statistics.median(ts)
    return PassLatency(out)
