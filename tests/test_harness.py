"""Round segmentation, and the three ways an observed round count is censored."""

from __future__ import annotations

from spec_decode_advisor.harness import segment_rounds


def test_single_clean_round():
    # three draft tokens accepted, then the target's own token closes the round
    assert segment_rounds([True, True, True, False], 4, 100) == [3]


def test_multiple_rounds():
    flags = [True, False, True, True, False, False]
    assert segment_rounds(flags, 4, 100) == [1, 2, 0]


def test_zero_accepted_round():
    # the very first draft token was rejected
    assert segment_rounds([False], 4, 100) == [0]


def test_all_accepted_round_is_kept():
    # a == k is right-censored for estimation, but it is still a real round
    assert segment_rounds([True, True, True, True, False], 4, 100) == [4]


def test_truncated_tail_is_dropped():
    # max_tokens hit inside the accept loop: no target token, so the count is
    # only a lower bound on what the round would have accepted
    assert segment_rounds([True, True, False, True, True], 4, 100) == [2]


def test_reduced_effective_depth_is_dropped():
    # near the end mlx_lm drafts min(max_tokens - ntoks, k) tokens, so a round
    # that accepted everything it drafted can look like an early rejection
    assert segment_rounds([True, False, True, False], 4, 4) == [1]


def test_k1_rounds():
    assert segment_rounds([True, False, False, True, False], 1, 100) == [1, 0, 1]


def test_empty_stream():
    assert segment_rounds([], 4, 100) == []


def test_only_draft_tokens_yields_nothing():
    # never terminated, so nothing is observable
    assert segment_rounds([True, True, True], 4, 100) == []


# ---- swapping drafts -----------------------------------------------------

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from spec_decode_advisor.harness import Harness  # noqa: E402


def test_vocab_check_rejects_a_draft_from_another_family():
    # Speculation compares raw token ids, so a vocab mismatch would not error at
    # runtime -- it would silently never accept anything.
    h = Harness("t", "d")
    h._tokenizer = SimpleNamespace(vocab_size=151643)
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        h._check_vocab(SimpleNamespace(vocab_size=32000))


def test_vocab_check_accepts_a_sibling():
    h = Harness("t", "d")
    h._tokenizer = SimpleNamespace(vocab_size=151643)
    h._check_vocab(SimpleNamespace(vocab_size=151643))  # no raise
