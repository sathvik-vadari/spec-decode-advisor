"""The prompt set, tagged by task domain.

Acceptance is a property of the model pair *and the workload*. The whole claim
of this tool is that the workload half matters enough that a single global
draft depth is the wrong default -- so the prompt set has to span domains that
should differ, and differ for reasons that are stateable in advance:

    copy       output largely restates text already in the prompt. The draft
               model does not need to be smart, only to copy. Should be the
               easiest domain by a wide margin.
    code       highly structured, low branching factor, much of it boilerplate
               the draft model has seen. Should be easy.
    factual    short, constrained, but the specific tokens are recall rather
               than structure -- a small model knows fewer of them.
    chat       conversational; medium entropy, medium structure.
    prose      open-ended and creative. Many tokens are defensible at any
               position, so the draft and target agree least. Should be hardest.

If measured acceptance does not order roughly copy > code > chat/factual >
prose, the prediction was wrong and that is the finding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prompt:
    prompt_id: str
    domain: str
    text: str


_PASSAGE = (
    "The 1815 eruption of Mount Tambora in the Dutch East Indies was the largest "
    "volcanic eruption in recorded history. It ejected roughly 150 cubic kilometres "
    "of material and injected sulfate aerosols into the stratosphere, which lowered "
    "global average temperatures by about 0.5 degrees Celsius. The following year, "
    "1816, became known as the Year Without a Summer. Crops failed across the "
    "northern hemisphere, food prices rose sharply, and famine followed in parts of "
    "Europe and North America."
)

_CONTRACT = (
    "SECTION 4. TERMINATION. Either party may terminate this Agreement upon thirty "
    "(30) days written notice to the other party. Upon termination, the Client shall "
    "pay all fees accrued through the effective date of termination. Sections 6, 7, "
    "and 9 shall survive termination of this Agreement."
)

_LOG = (
    "2026-08-14T02:11:09Z ERROR worker-7 connection reset by peer (attempt 3/5)\n"
    "2026-08-14T02:11:11Z WARN  worker-7 backing off for 4000ms\n"
    "2026-08-14T02:11:15Z ERROR worker-7 connection reset by peer (attempt 4/5)\n"
    "2026-08-14T02:11:23Z FATAL worker-7 giving up after 5 attempts, shedding queue"
)


PROMPTS: tuple[Prompt, ...] = (
    # ---- copy: the answer is mostly already in the prompt ----
    Prompt("copy_tambora", "copy", f"Summarize the following passage.\n\n{_PASSAGE}"),
    Prompt("copy_contract", "copy", f"Restate the following clause in plain English.\n\n{_CONTRACT}"),
    Prompt("copy_log", "copy", f"Explain what these log lines show.\n\n{_LOG}"),
    Prompt("copy_extract", "copy", f"List every date and number mentioned in this passage.\n\n{_PASSAGE}"),
    Prompt("copy_quote", "copy", f"Quote the sentence in this passage that states the temperature effect, then explain it.\n\n{_PASSAGE}"),
    Prompt("copy_translate", "copy", f"Rewrite this clause as a bulleted list, keeping the exact obligations.\n\n{_CONTRACT}"),

    # ---- code: structured, low branching factor ----
    Prompt("code_binsearch", "code", "Write a Python function that performs binary search on a sorted list and returns the index or -1."),
    Prompt("code_dataclass", "code", "Write a Python dataclass named ServerConfig with fields for host, port, timeout, and max_retries, with sensible defaults."),
    Prompt("code_retry", "code", "Write a Python decorator that retries a function with exponential backoff."),
    Prompt("code_sql", "code", "Write a SQL query that finds the top 10 customers by total order value in the last 90 days."),
    Prompt("code_dockerfile", "code", "Write a Dockerfile for a Python 3.12 FastAPI application using uv."),
    Prompt("code_test", "code", "Write pytest unit tests for a function `def normalize(xs: list[float]) -> list[float]` that scales a list to sum to 1."),

    # ---- factual: constrained but recall-heavy ----
    Prompt("fact_planets", "factual", "List the planets of the solar system in order from the sun, with one short fact about each."),
    Prompt("fact_http", "factual", "What do HTTP status codes 301, 401, 403, 429, and 503 mean?"),
    Prompt("fact_tcp", "factual", "Explain the TCP three-way handshake."),
    Prompt("fact_capitals", "factual", "Name the capital cities of Australia, Canada, Brazil, Turkey, and South Africa."),
    Prompt("fact_bigo", "factual", "What is the average and worst case time complexity of quicksort, mergesort, and heapsort?"),
    Prompt("fact_units", "factual", "Explain the difference between a watt, a watt-hour, and a joule."),

    # ---- chat: conversational, medium entropy ----
    Prompt("chat_debug", "chat", "My Kubernetes pod keeps getting OOMKilled. What should I check first?"),
    Prompt("chat_advice", "chat", "I have two job offers, one at a large stable company and one at an early startup. How should I think about the decision?"),
    Prompt("chat_explain", "chat", "Explain what a load balancer does to someone who has never worked in infrastructure."),
    Prompt("chat_opinion", "chat", "Is it worth learning Rust in 2026 if I already know Go and Python?"),
    Prompt("chat_plan", "chat", "I want to run a 10k in three months and currently run twice a week. How should I train?"),
    Prompt("chat_followup", "chat", "My CI pipeline got slower over the last month and nobody changed it. Where do I start looking?"),

    # ---- prose: open-ended, high entropy ----
    Prompt("prose_story", "prose", "Write the opening paragraph of a short story about a lighthouse keeper who stops receiving mail."),
    Prompt("prose_letter", "prose", "Write a warm letter to a friend who has just moved to a city where they know nobody."),
    Prompt("prose_desc", "prose", "Describe a night market in a coastal town, focusing on sound and smell rather than sight."),
    Prompt("prose_poem", "prose", "Write a short free-verse poem about the last day of summer."),
    Prompt("prose_speech", "prose", "Write the closing lines of a retirement speech for a bridge engineer."),
    Prompt("prose_scene", "prose", "Write a scene where two strangers share a table in a crowded cafe and neither speaks."),
)


DOMAINS: tuple[str, ...] = ("copy", "code", "factual", "chat", "prose")


def by_domain(domain: str) -> tuple[Prompt, ...]:
    return tuple(p for p in PROMPTS if p.domain == domain)


def load_prompts(path: str) -> tuple[Prompt, ...]:
    """A workload from disk.

    `.jsonl`: one object per line with `text`, and optionally `id` and `domain`.
    Anything else: one prompt per line, blank lines skipped, all in one domain
    called "workload". Domains only matter if you want the per-domain breakdown;
    the pooled recommendation does not need them.
    """
    import json
    from pathlib import Path

    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    prompts: list[Prompt] = []
    if path.endswith(".jsonl"):
        for i, ln in enumerate(lines, 1):
            obj = json.loads(ln)
            if "text" not in obj:
                raise ValueError(f"{path}:{i}: missing 'text'")
            prompts.append(
                Prompt(str(obj.get("id", f"p{i:03d}")), str(obj.get("domain", "workload")), obj["text"])
            )
    else:
        prompts = [Prompt(f"p{i:03d}", "workload", ln) for i, ln in enumerate(lines, 1)]
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return tuple(prompts)
