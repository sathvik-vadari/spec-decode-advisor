"""Loading a user's workload from a file."""

import pytest

from spec_decode_advisor.prompts import load_prompts


def test_plain_text_one_prompt_per_line(tmp_path):
    f = tmp_path / "w.txt"
    f.write_text("first prompt\n\n  \nsecond prompt\n")
    ps = load_prompts(str(f))
    assert [p.text for p in ps] == ["first prompt", "second prompt"]
    assert {p.domain for p in ps} == {"workload"}
    assert [p.prompt_id for p in ps] == ["p001", "p002"]


def test_jsonl_with_optional_fields(tmp_path):
    f = tmp_path / "w.jsonl"
    f.write_text('{"text": "a", "id": "x", "domain": "code"}\n{"text": "b"}\n')
    ps = load_prompts(str(f))
    assert (ps[0].prompt_id, ps[0].domain, ps[0].text) == ("x", "code", "a")
    assert (ps[1].prompt_id, ps[1].domain, ps[1].text) == ("p002", "workload", "b")


def test_jsonl_without_text_is_an_error(tmp_path):
    f = tmp_path / "w.jsonl"
    f.write_text('{"prompt": "a"}\n')
    with pytest.raises(ValueError, match="missing 'text'"):
        load_prompts(str(f))


def test_empty_file_is_an_error(tmp_path):
    f = tmp_path / "w.txt"
    f.write_text("\n\n")
    with pytest.raises(ValueError, match="no prompts"):
        load_prompts(str(f))
