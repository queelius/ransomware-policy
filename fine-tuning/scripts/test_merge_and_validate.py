"""Tests for merge_and_validate.py."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest

from merge_and_validate import (
    compute_coverage_stats,
    exact_dedup,
    format_coverage_report,
    fuzzy_dedup,
    jaccard_similarity,
    load_raw_examples,
    split_train_eval,
    validate_example,
)

VALID_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are an analyst."},
        {"role": "user", "content": "What is T1486?"},
        {
            "role": "assistant",
            "content": (
                "T1486 is Data Encrypted for Impact, a technique used by"
                " ransomware to encrypt files on victim systems."
            ),
        },
    ],
    "metadata": {
        "source": "mitre_attack_stix",
        "source_url": "https://github.com/mitre-attack/attack-stix-data",
        "source_section": "T1486",
        "example_type": "qa",
        "topics": ["ransomware_ttps"],
        "generation_prompt_id": "technique_qa_v1",
        "generated_at": "2026-03-17T14:30:00Z",
        "generator_model": "claude-opus-4-6",
        "quality_score": None,
    },
}


# ---------------------------------------------------------------------------
# validate_example
# ---------------------------------------------------------------------------


class TestValidateExample:
    def test_valid_example_passes(self):
        errors = validate_example(VALID_EXAMPLE)
        assert errors == []

    def test_missing_messages(self):
        ex = {"metadata": VALID_EXAMPLE["metadata"]}
        errors = validate_example(ex)
        assert any("messages must be a list" in e for e in errors)

    def test_wrong_message_count(self):
        ex = copy.deepcopy(VALID_EXAMPLE)
        ex["messages"] = ex["messages"][:2]  # only system + user
        errors = validate_example(ex)
        assert any("exactly 3 elements" in e for e in errors)

    def test_empty_assistant_content(self):
        ex = copy.deepcopy(VALID_EXAMPLE)
        ex["messages"][2]["content"] = ""
        errors = validate_example(ex)
        assert any("non-empty string" in e for e in errors)

    def test_short_assistant_content(self):
        ex = copy.deepcopy(VALID_EXAMPLE)
        ex["messages"][2]["content"] = "Too short."
        errors = validate_example(ex)
        assert any("50 characters" in e for e in errors)

    def test_missing_metadata(self):
        ex = {"messages": VALID_EXAMPLE["messages"]}
        errors = validate_example(ex)
        assert any("metadata must be a dict" in e for e in errors)

    def test_missing_metadata_field(self):
        ex = copy.deepcopy(VALID_EXAMPLE)
        del ex["metadata"]["source"]
        errors = validate_example(ex)
        assert any("source" in e for e in errors)

    def test_invalid_example_type(self):
        ex = copy.deepcopy(VALID_EXAMPLE)
        ex["metadata"]["example_type"] = "invalid_type"
        errors = validate_example(ex)
        assert any("example_type" in e for e in errors)

    def test_wrong_role_order(self):
        ex = copy.deepcopy(VALID_EXAMPLE)
        ex["messages"][0]["role"] = "user"
        ex["messages"][1]["role"] = "system"
        errors = validate_example(ex)
        assert any("system" in e for e in errors)

    def test_parse_error_example(self):
        ex = {"_parse_error": "Expecting value: line 1 column 1"}
        errors = validate_example(ex)
        assert any("JSON parse error" in e for e in errors)


# ---------------------------------------------------------------------------
# exact_dedup
# ---------------------------------------------------------------------------


class TestExactDedup:
    def test_removes_duplicates(self):
        examples = [copy.deepcopy(VALID_EXAMPLE) for _ in range(3)]
        deduped, n_removed = exact_dedup(examples)
        assert len(deduped) == 1
        assert n_removed == 2

    def test_keeps_distinct(self):
        ex1 = copy.deepcopy(VALID_EXAMPLE)
        ex2 = copy.deepcopy(VALID_EXAMPLE)
        ex2["messages"][1]["content"] = "What is T1490?"
        deduped, n_removed = exact_dedup([ex1, ex2])
        assert len(deduped) == 2
        assert n_removed == 0

    def test_empty_input(self):
        deduped, n_removed = exact_dedup([])
        assert deduped == []
        assert n_removed == 0


# ---------------------------------------------------------------------------
# jaccard_similarity
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_texts(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert jaccard_similarity("hello world", "foo bar baz") == 0.0

    def test_partial_overlap(self):
        sim = jaccard_similarity("the cat sat on the mat", "the dog sat on the rug")
        # shared: the, sat, on → 3 words; union: the, cat, sat, on, mat, dog, rug → 7
        assert 0.3 < sim < 0.5

    def test_empty_texts(self):
        assert jaccard_similarity("", "") == 1.0

    def test_one_empty(self):
        assert jaccard_similarity("hello", "") == 0.0

    def test_case_insensitive(self):
        assert jaccard_similarity("Hello World", "hello world") == 1.0


# ---------------------------------------------------------------------------
# fuzzy_dedup
# ---------------------------------------------------------------------------


class TestFuzzyDedup:
    def test_flags_near_duplicates(self):
        ex1 = copy.deepcopy(VALID_EXAMPLE)
        ex2 = copy.deepcopy(VALID_EXAMPLE)
        # Slightly different user content but very high overlap
        ex2["messages"][1]["content"] = "What is T1486?"  # identical
        kept, flagged = fuzzy_dedup([ex1, ex2], threshold=0.85)
        assert len(kept) == 1
        assert len(flagged) == 1

    def test_keeps_distinct_examples(self):
        ex1 = copy.deepcopy(VALID_EXAMPLE)
        ex2 = copy.deepcopy(VALID_EXAMPLE)
        ex2["messages"][1][
            "content"
        ] = "Describe the lifecycle of a ransomware attack from initial compromise to encryption."
        kept, flagged = fuzzy_dedup([ex1, ex2], threshold=0.85)
        assert len(kept) == 2
        assert len(flagged) == 0


# ---------------------------------------------------------------------------
# compute_coverage_stats
# ---------------------------------------------------------------------------


class TestComputeCoverageStats:
    def test_basic_stats(self):
        examples = [copy.deepcopy(VALID_EXAMPLE) for _ in range(5)]
        stats = compute_coverage_stats(examples)
        assert stats["total"] == 5
        assert stats["by_source"]["mitre_attack_stix"] == 5
        assert stats["by_type"]["qa"] == 5
        assert stats["by_topic"]["ransomware_ttps"] == 5
        assert stats["avg_user_length"] > 0
        assert stats["avg_assistant_length"] > 0

    def test_multiple_sources(self):
        ex1 = copy.deepcopy(VALID_EXAMPLE)
        ex2 = copy.deepcopy(VALID_EXAMPLE)
        ex2["metadata"]["source"] = "dfir_report"
        stats = compute_coverage_stats([ex1, ex2])
        assert stats["by_source"]["mitre_attack_stix"] == 1
        assert stats["by_source"]["dfir_report"] == 1

    def test_empty_input(self):
        stats = compute_coverage_stats([])
        assert stats["total"] == 0
        assert stats["avg_user_length"] == 0


# ---------------------------------------------------------------------------
# format_coverage_report
# ---------------------------------------------------------------------------


class TestFormatCoverageReport:
    def test_contains_key_sections(self):
        stats = compute_coverage_stats([VALID_EXAMPLE])
        report = format_coverage_report(stats)
        assert "# SFT Domain Knowledge" in report
        assert "Total examples" in report
        assert "mitre_attack_stix" in report
        assert "qa" in report
        assert "ransomware_ttps" in report


# ---------------------------------------------------------------------------
# split_train_eval
# ---------------------------------------------------------------------------


class TestSplitTrainEval:
    def test_proportions(self):
        examples = []
        for i in range(20):
            ex = copy.deepcopy(VALID_EXAMPLE)
            ex["messages"][1]["content"] = f"Question number {i} about ransomware?"
            examples.append(ex)
        train, eval_set = split_train_eval(examples, eval_fraction=0.1, seed=42)
        assert len(train) + len(eval_set) == 20
        # With 20 examples from 1 source, eval should be round(20*0.1)=2
        assert len(eval_set) == 2

    def test_stratified_by_source(self):
        examples = []
        for i in range(10):
            ex = copy.deepcopy(VALID_EXAMPLE)
            ex["messages"][1]["content"] = f"MITRE question {i}"
            ex["metadata"]["source"] = "mitre_attack_stix"
            examples.append(ex)
        for i in range(10):
            ex = copy.deepcopy(VALID_EXAMPLE)
            ex["messages"][1]["content"] = f"DFIR question {i}"
            ex["metadata"]["source"] = "dfir_report"
            examples.append(ex)
        train, eval_set = split_train_eval(examples, eval_fraction=0.1, seed=42)
        assert len(train) + len(eval_set) == 20
        # Each source contributes at least 1 to eval
        eval_sources = {ex["metadata"]["source"] for ex in eval_set}
        assert "mitre_attack_stix" in eval_sources
        assert "dfir_report" in eval_sources

    def test_single_example_per_source_goes_to_train(self):
        ex = copy.deepcopy(VALID_EXAMPLE)
        train, eval_set = split_train_eval([ex], eval_fraction=0.1, seed=42)
        assert len(train) == 1
        assert len(eval_set) == 0

    def test_deterministic(self):
        examples = []
        for i in range(20):
            ex = copy.deepcopy(VALID_EXAMPLE)
            ex["messages"][1]["content"] = f"Question {i}"
            examples.append(ex)
        train1, eval1 = split_train_eval(examples, seed=42)
        train2, eval2 = split_train_eval(examples, seed=42)
        assert len(train1) == len(train2)
        assert len(eval1) == len(eval2)


# ---------------------------------------------------------------------------
# load_raw_examples (integration with temp files)
# ---------------------------------------------------------------------------


class TestLoadRawExamples:
    def test_loads_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            fpath = raw_dir / "test_source.jsonl"
            with open(fpath, "w") as f:
                f.write(json.dumps(VALID_EXAMPLE) + "\n")
                f.write(json.dumps(VALID_EXAMPLE) + "\n")
            examples = load_raw_examples(raw_dir)
            assert len(examples) == 2
            assert examples[0]["_source_file"] == "test_source.jsonl"

    def test_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            fpath = raw_dir / "test.jsonl"
            with open(fpath, "w") as f:
                f.write(json.dumps(VALID_EXAMPLE) + "\n")
                f.write("\n")
                f.write(json.dumps(VALID_EXAMPLE) + "\n")
            examples = load_raw_examples(raw_dir)
            assert len(examples) == 2

    def test_captures_parse_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            fpath = raw_dir / "bad.jsonl"
            with open(fpath, "w") as f:
                f.write("not valid json\n")
            examples = load_raw_examples(raw_dir)
            assert len(examples) == 1
            assert "_parse_error" in examples[0]

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            examples = load_raw_examples(Path(tmpdir))
            assert examples == []
