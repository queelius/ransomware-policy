#!/usr/bin/env python3
"""Tests for generate_sft_data.py — SFT data generation orchestrator."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from generate_sft_data import (
    SFTExample,
    build_metadata,
    format_prompt,
    load_prompt_templates,
    load_source_content,
    invoke_claude,
    parse_generated_examples,
    _extract_json_array,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_prompts_dir(tmp_path):
    """Create a temporary prompts directory with sample YAML files."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    template_a = {
        "source": "mitre_attack_stix",
        "description": "Test template A",
        "templates": [
            {
                "id": "technique_qa_v1",
                "type": "qa",
                "system": "You are a security analyst.",
                "prompt": "Analyze this technique:\n{content}\n\nOutput JSON.",
                "target_count": 3,
            },
            {
                "id": "technique_instruction_v1",
                "type": "instruction",
                "system": "You are a security analyst.",
                "prompt": "Create investigation scenario:\n{content}\n\nOutput JSON.",
                "target_count": 1,
            },
        ],
    }
    template_b = {
        "source": "atomic_red_team",
        "description": "Test template B",
        "templates": [
            {
                "id": "atomic_qa_v1",
                "type": "qa",
                "system": "You are a red-team expert.",
                "prompt": "Based on this atomic test:\n{content}\n\nOutput JSON.",
                "target_count": 2,
            },
        ],
    }

    (prompts_dir / "mitre_attack.yaml").write_text(yaml.dump(template_a))
    (prompts_dir / "atomic_red_team.yaml").write_text(yaml.dump(template_b))

    # Write a file without templates key (should be skipped)
    (prompts_dir / "empty.yaml").write_text(yaml.dump({"source": "nope"}))

    return prompts_dir


@pytest.fixture
def tmp_raw_content(tmp_path):
    """Create a temporary raw_content directory with sample files."""
    raw_dir = tmp_path / "raw_content"
    src_dir = raw_dir / "mitre_attack_stix"
    src_dir.mkdir(parents=True)

    (src_dir / "T1486.txt").write_text("Technique T1486: Data Encrypted for Impact")
    (src_dir / "T1490.json").write_text('{"name": "Inhibit System Recovery"}')

    return raw_dir


# ---------------------------------------------------------------------------
# load_prompt_templates
# ---------------------------------------------------------------------------


class TestLoadPromptTemplates:
    def test_loads_all_valid_yamls(self, tmp_prompts_dir):
        results = load_prompt_templates(tmp_prompts_dir)
        assert len(results) == 2  # empty.yaml is skipped

    def test_returns_template_structures(self, tmp_prompts_dir):
        results = load_prompt_templates(tmp_prompts_dir)
        sources = {r["source"] for r in results}
        assert "mitre_attack_stix" in sources
        assert "atomic_red_team" in sources

    def test_templates_have_required_keys(self, tmp_prompts_dir):
        results = load_prompt_templates(tmp_prompts_dir)
        for result in results:
            assert "templates" in result
            for tpl in result["templates"]:
                assert "id" in tpl
                assert "type" in tpl
                assert "prompt" in tpl

    def test_missing_dir_returns_empty(self, tmp_path):
        results = load_prompt_templates(tmp_path / "nonexistent")
        assert results == []

    def test_empty_dir_returns_empty(self, tmp_path):
        empty_dir = tmp_path / "empty_prompts"
        empty_dir.mkdir()
        results = load_prompt_templates(empty_dir)
        assert results == []


# ---------------------------------------------------------------------------
# load_source_content
# ---------------------------------------------------------------------------


class TestLoadSourceContent:
    def test_loads_all_files(self, tmp_raw_content):
        chunks = load_source_content("mitre_attack_stix", tmp_raw_content)
        assert len(chunks) == 2

    def test_chunk_structure(self, tmp_raw_content):
        chunks = load_source_content("mitre_attack_stix", tmp_raw_content)
        for chunk in chunks:
            assert "section" in chunk
            assert "content" in chunk
            assert isinstance(chunk["section"], str)
            assert isinstance(chunk["content"], str)

    def test_section_is_stem(self, tmp_raw_content):
        chunks = load_source_content("mitre_attack_stix", tmp_raw_content)
        sections = {c["section"] for c in chunks}
        assert "T1486" in sections
        assert "T1490" in sections

    def test_content_read_correctly(self, tmp_raw_content):
        chunks = load_source_content("mitre_attack_stix", tmp_raw_content)
        by_section = {c["section"]: c["content"] for c in chunks}
        assert "Data Encrypted for Impact" in by_section["T1486"]

    def test_missing_source_returns_empty(self, tmp_raw_content):
        chunks = load_source_content("nonexistent_source", tmp_raw_content)
        assert chunks == []


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------


class TestFormatPrompt:
    def test_substitutes_content(self):
        template = {"prompt": "Analyze this:\n{content}\n\nDone."}
        result = format_prompt(template, "T1486 definition text")
        assert "T1486 definition text" in result
        assert "{content}" not in result

    def test_preserves_surrounding_text(self):
        template = {"prompt": "Before {content} after"}
        result = format_prompt(template, "MIDDLE")
        assert result == "Before MIDDLE after"

    def test_empty_content(self):
        template = {"prompt": "Intro: {content}"}
        result = format_prompt(template, "")
        assert result == "Intro: "

    def test_no_placeholder(self):
        template = {"prompt": "Static prompt with no placeholder"}
        result = format_prompt(template, "anything")
        assert result == "Static prompt with no placeholder"

    def test_multiple_content_placeholders(self):
        template = {"prompt": "First: {content}, Second: {content}"}
        result = format_prompt(template, "X")
        assert result == "First: X, Second: X"

    def test_missing_prompt_key(self):
        template = {"id": "test", "type": "qa"}
        result = format_prompt(template, "content")
        assert result == ""


# ---------------------------------------------------------------------------
# parse_generated_examples
# ---------------------------------------------------------------------------


class TestParseGeneratedExamples:
    # --- QA format ---
    def test_qa_valid_json(self):
        response = json.dumps([
            {"question": "What is T1486?", "answer": "Data Encrypted for Impact."},
            {"question": "How to detect?", "answer": "Check entropy of files."},
        ])
        examples = parse_generated_examples(response, "qa")
        assert len(examples) == 2
        assert examples[0].messages[0]["role"] == "user"
        assert examples[0].messages[0]["content"] == "What is T1486?"
        assert examples[0].messages[1]["role"] == "assistant"

    def test_qa_with_markdown_fence(self):
        response = '```json\n[{"question": "Q?", "answer": "A."}]\n```'
        examples = parse_generated_examples(response, "qa")
        assert len(examples) == 1

    def test_qa_with_preamble(self):
        response = 'Here are the QA pairs:\n[{"question": "Q?", "answer": "A."}]'
        examples = parse_generated_examples(response, "qa")
        assert len(examples) == 1

    # --- Instruction format ---
    def test_instruction_valid(self):
        response = json.dumps([
            {"instruction": "Investigate this host.", "response": "Check event log 4688."}
        ])
        examples = parse_generated_examples(response, "instruction")
        assert len(examples) == 1
        assert examples[0].messages[0]["role"] == "user"
        assert examples[0].messages[1]["role"] == "assistant"

    # --- Passage format ---
    def test_passage_valid(self):
        response = json.dumps([{"passage": "Ransomware encrypts files using AES-256."}])
        examples = parse_generated_examples(response, "passage")
        assert len(examples) == 1
        assert len(examples[0].messages) == 1
        assert examples[0].messages[0]["role"] == "assistant"

    # --- Error handling ---
    def test_empty_response(self):
        assert parse_generated_examples("", "qa") == []

    def test_malformed_json(self):
        assert parse_generated_examples("not json at all", "qa") == []

    def test_json_object_not_array(self):
        assert parse_generated_examples('{"question": "Q?"}', "qa") == []

    def test_missing_fields_skipped(self):
        response = json.dumps([
            {"question": "Q?", "answer": "A."},
            {"question": "Q2?"},  # missing answer
            {"wrong_key": "val"},
        ])
        examples = parse_generated_examples(response, "qa")
        assert len(examples) == 1

    def test_empty_strings_skipped(self):
        response = json.dumps([{"question": "", "answer": "A."}])
        examples = parse_generated_examples(response, "qa")
        assert len(examples) == 0

    def test_non_dict_items_skipped(self):
        response = json.dumps([
            {"question": "Q?", "answer": "A."},
            "string_item",
            42,
        ])
        examples = parse_generated_examples(response, "qa")
        assert len(examples) == 1

    def test_unknown_type_returns_empty(self):
        response = json.dumps([{"question": "Q?", "answer": "A."}])
        examples = parse_generated_examples(response, "unknown_type")
        assert len(examples) == 0


# ---------------------------------------------------------------------------
# _extract_json_array
# ---------------------------------------------------------------------------


class TestExtractJsonArray:
    def test_plain_array(self):
        result = _extract_json_array('[{"a": 1}]')
        assert result == '[{"a": 1}]'

    def test_markdown_fence(self):
        result = _extract_json_array('```json\n[{"a": 1}]\n```')
        assert result is not None
        parsed = json.loads(result)
        assert parsed == [{"a": 1}]

    def test_preamble_text(self):
        result = _extract_json_array('Here is the output:\n[{"a": 1}]')
        assert result is not None
        parsed = json.loads(result)
        assert parsed == [{"a": 1}]

    def test_no_array(self):
        result = _extract_json_array("No JSON here")
        assert result is None

    def test_nested_brackets(self):
        text = '[{"list": [1, 2, 3]}]'
        result = _extract_json_array(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed[0]["list"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    def test_returns_all_fields(self):
        meta = build_metadata(
            source_key="mitre_attack_stix",
            source_url="https://github.com/mitre-attack/attack-stix-data",
            source_section="T1486",
            example_type="qa",
            topics=["ransomware_ttps"],
            prompt_id="technique_qa_v1",
            run_id="20260317T143000Z",
        )
        assert meta["source"] == "mitre_attack_stix"
        assert meta["source_url"] == "https://github.com/mitre-attack/attack-stix-data"
        assert meta["source_section"] == "T1486"
        assert meta["example_type"] == "qa"
        assert meta["topics"] == ["ransomware_ttps"]
        assert meta["generation_prompt_id"] == "technique_qa_v1"
        assert meta["generator_model"] == "claude-opus-4-6"
        assert meta["quality_score"] is None
        assert meta["run_id"] == "20260317T143000Z"

    def test_generated_at_is_iso(self):
        meta = build_metadata(
            source_key="test",
            source_url="",
            source_section="",
            example_type="qa",
            topics=[],
            prompt_id="test_v1",
            run_id="run1",
        )
        # Should be parseable ISO datetime
        from datetime import datetime
        dt = datetime.fromisoformat(meta["generated_at"])
        assert dt.year >= 2026

    def test_multiple_topics(self):
        meta = build_metadata(
            source_key="test",
            source_url="",
            source_section="",
            example_type="qa",
            topics=["ransomware_ttps", "windows_internals", "forensic_analysis"],
            prompt_id="test_v1",
            run_id="run1",
        )
        assert len(meta["topics"]) == 3


# ---------------------------------------------------------------------------
# invoke_claude (mocked)
# ---------------------------------------------------------------------------


class TestInvokeClaude:
    @patch("generate_sft_data.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='[{"question": "Q?", "answer": "A."}]',
        )
        result = invoke_claude("prompt text", "system text")
        assert result == '[{"question": "Q?", "answer": "A."}]'
        mock_run.assert_called_once()

    @patch("generate_sft_data.subprocess.run")
    def test_retry_on_failure(self, mock_run):
        # First call fails, second succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr="error"),
            MagicMock(returncode=0, stdout="success"),
        ]
        result = invoke_claude("prompt", "system")
        assert result == "success"
        assert mock_run.call_count == 2

    @patch("generate_sft_data.subprocess.run")
    def test_both_attempts_fail(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        result = invoke_claude("prompt", "system")
        assert result == ""
        assert mock_run.call_count == 2

    @patch("generate_sft_data.subprocess.run")
    def test_timeout_retries(self, mock_run):
        mock_run.side_effect = [
            subprocess.TimeoutExpired(cmd="claude", timeout=120),
            MagicMock(returncode=0, stdout="ok"),
        ]
        result = invoke_claude("prompt", "system")
        assert result == "ok"

    @patch("generate_sft_data.subprocess.run")
    def test_command_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = invoke_claude("prompt", "system")
        assert result == ""

    @patch("generate_sft_data.subprocess.run")
    def test_passes_correct_args(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok")
        invoke_claude("my prompt", "my system")
        args, kwargs = mock_run.call_args
        assert args[0] == ["claude", "-p", "my prompt", "--no-input"]
        assert kwargs["timeout"] == 120
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True


# ---------------------------------------------------------------------------
# Integration-style test with mocked invoke_claude
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @patch("generate_sft_data.invoke_claude")
    def test_generate_for_source(self, mock_claude, tmp_path, tmp_raw_content):
        """Full pipeline: templates + content -> JSONL output."""
        from generate_sft_data import generate_for_source

        mock_claude.return_value = json.dumps([
            {"question": "What is T1486?", "answer": "Data Encrypted for Impact."},
        ])

        source = {
            "key": "mitre_attack_stix",
            "url": "https://github.com/mitre-attack/attack-stix-data",
            "topics": ["ransomware_ttps"],
        }
        templates = [
            {
                "id": "technique_qa_v1",
                "type": "qa",
                "system": "You are a security analyst.",
                "prompt": "Analyze:\n{content}\n\nJSON output.",
                "target_count": 3,
            },
        ]

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        log_path = tmp_path / "generation_log.jsonl"

        count = generate_for_source(
            source=source,
            templates=templates,
            raw_content_dir=tmp_raw_content,
            output_dir=output_dir,
            log_path=log_path,
            run_id="20260317T143000Z",
            dry_run=False,
        )

        # 2 content chunks * 1 template * 1 example each = 2 examples
        assert count == 2

        # Verify output file
        output_file = output_dir / "mitre_attack_stix.jsonl"
        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2

        # Verify structure of first line
        record = json.loads(lines[0])
        assert "messages" in record
        assert "metadata" in record
        assert record["messages"][0]["role"] == "system"
        assert record["messages"][1]["role"] == "user"
        assert record["messages"][2]["role"] == "assistant"
        assert record["metadata"]["source"] == "mitre_attack_stix"
        assert record["metadata"]["generation_prompt_id"] == "technique_qa_v1"

        # Verify log file
        assert log_path.exists()
        log_lines = log_path.read_text().strip().split("\n")
        assert len(log_lines) == 2  # one per chunk

    @patch("generate_sft_data.invoke_claude")
    def test_dry_run_no_output(self, mock_claude, tmp_path, tmp_raw_content):
        """Dry run should not invoke claude or write output."""
        from generate_sft_data import generate_for_source

        source = {
            "key": "mitre_attack_stix",
            "url": "https://example.com",
            "topics": [],
        }
        templates = [
            {
                "id": "test_v1",
                "type": "qa",
                "system": "sys",
                "prompt": "Prompt: {content}",
                "target_count": 1,
            },
        ]

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        log_path = tmp_path / "generation_log.jsonl"

        count = generate_for_source(
            source=source,
            templates=templates,
            raw_content_dir=tmp_raw_content,
            output_dir=output_dir,
            log_path=log_path,
            run_id="test_run",
            dry_run=True,
        )

        assert count == 0
        mock_claude.assert_not_called()
