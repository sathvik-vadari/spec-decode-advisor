"""The CLI's failure handling, without a GPU."""

from spec_decode_advisor.cli import _run_pair
from spec_decode_advisor.prompts import PROMPTS


class _FlakyHarness:
    """Raises on the first `fail_first` calls, then returns the (prompt, k) it was given."""

    def __init__(self, fail_first):
        self.fail_first = fail_first
        self.calls = 0

    def run(self, prompt, k):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("[METAL] Command buffer execution failed: Caused GPU Timeout Error")
        return (prompt.prompt_id, k)


def test_a_single_gpu_failure_is_retried_and_both_runs_are_kept():
    msgs = []
    pair = _run_pair(_FlakyHarness(fail_first=1), PROMPTS[0], msgs.append)
    assert pair == ((PROMPTS[0].prompt_id, 0), (PROMPTS[0].prompt_id, 1))
    assert len(msgs) == 1 and "retrying" in msgs[0]


def test_two_failures_skip_the_prompt_and_say_so():
    msgs = []
    assert _run_pair(_FlakyHarness(fail_first=2), PROMPTS[0], msgs.append) is None
    assert "skipped" in msgs[-1]


def test_a_healthy_harness_runs_exactly_twice():
    h = _FlakyHarness(fail_first=0)
    _run_pair(h, PROMPTS[0], lambda *_: None)
    assert h.calls == 2
