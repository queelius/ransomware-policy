"""Tests for agent tools and tool-call parser."""

from datetime import datetime

import numpy as np
import pytest

from simulator.models import ContentType
from simulator.registry import FileRegistry, ProcessTable
from tools.inspection import (
    TOOL_COSTS,
    VALID_VERDICTS,
    check_process,
    execute_tool,
    inspect_file,
    scan_directory,
)
from tools.parser import (
    ParsedToolCall,
    extract_thinking,
    format_tool_result,
    is_decide_call,
    parse_all_tool_calls,
    parse_tool_call,
)


@pytest.fixture
def registry():
    r = FileRegistry()
    now = datetime(2025, 1, 1)
    r.add_file("C:/Users/A/Documents/report.docx", 1024, 4.2, ".docx",
               ContentType.DOC, now)
    r.add_file("C:/Users/A/Documents/budget.xlsx", 2048, 3.8, ".xlsx",
               ContentType.DOC, now)
    return r


@pytest.fixture
def ptable():
    pt = ProcessTable()
    pt.seed_processes(datetime(2025, 1, 1))
    return pt


# ── Tool functions ───────────────────────────────────────────────────


class TestInspectFile:
    def test_existing_file(self, registry):
        result = inspect_file(registry, "C:/Users/A/Documents/report.docx")
        assert result["entropy"] == 4.2
        assert result["size"] == 1024
        assert result["extension"] == ".docx"
        assert result["content_type"] == "doc"
        assert "modified_ts" in result

    def test_missing_file(self, registry):
        result = inspect_file(registry, "C:/nope.txt")
        assert "error" in result


class TestCheckProcess:
    def test_existing_process(self, ptable):
        result = check_process(ptable, 4)  # System process
        assert result["name"] == "System"
        assert "child_pids" in result

    def test_missing_process(self, ptable):
        result = check_process(ptable, 99999)
        assert "error" in result


class TestScanDirectory:
    def test_populated_directory(self, registry):
        result = scan_directory(registry, "C:/Users/A/Documents")
        assert len(result["files"]) == 2
        paths = {f["path"] for f in result["files"]}
        assert "C:/Users/A/Documents/report.docx" in paths

    def test_empty_directory(self, registry):
        result = scan_directory(registry, "C:/Nonexistent")
        assert result["files"] == []


class TestExecuteTool:
    def test_dispatch_inspect_file(self, registry, ptable):
        result, cost = execute_tool("inspect_file",
                                    {"path": "C:/Users/A/Documents/report.docx"},
                                    registry, ptable)
        assert result["entropy"] == 4.2
        assert cost == TOOL_COSTS["inspect_file"]

    def test_dispatch_check_process(self, registry, ptable):
        result, cost = execute_tool("check_process", {"pid": 4},
                                    registry, ptable)
        assert result["name"] == "System"
        assert cost == TOOL_COSTS["check_process"]

    def test_dispatch_decide_valid(self, registry, ptable):
        result, cost = execute_tool(
            "DECIDE", {"verdict": "alert", "explanation": "test"},
            registry, ptable)
        assert result["verdict"] == "alert"
        assert cost == 0.0

    def test_dispatch_decide_invalid(self, registry, ptable):
        result, cost = execute_tool(
            "DECIDE", {"verdict": "invalid_verdict"},
            registry, ptable)
        assert "error" in result

    def test_unknown_tool(self, registry, ptable):
        result, cost = execute_tool("fake_tool", {}, registry, ptable)
        assert "error" in result


class TestToolCosts:
    def test_all_tools_have_costs(self):
        expected_tools = {"inspect_file", "check_process", "scan_directory",
                          "recall_memory", "DECIDE"}
        assert set(TOOL_COSTS.keys()) == expected_tools

    def test_decide_is_free(self):
        assert TOOL_COSTS["DECIDE"] == 0.0

    def test_investigation_tools_have_negative_cost(self):
        for tool, cost in TOOL_COSTS.items():
            if tool != "DECIDE":
                assert cost < 0


# ── Parser ───────────────────────────────────────────────────────────


class TestParseToolCallJSON:
    def test_qwen3_json_format(self):
        text = '<tool_call>{"name": "inspect_file", "arguments": {"path": "C:/test.docx"}}</tool_call>'
        result = parse_tool_call(text)
        assert result is not None
        assert result.tool_name == "inspect_file"
        assert result.args["path"] == "C:/test.docx"

    def test_json_with_surrounding_text(self):
        text = '''<think>Let me check this file</think>
<tool_call>{"name": "check_process", "arguments": {"pid": 1234}}</tool_call>'''
        result = parse_tool_call(text)
        assert result is not None
        assert result.tool_name == "check_process"
        assert result.args["pid"] == 1234


class TestParseToolCallFunction:
    def test_decide_with_strings(self):
        text = '<tool_call>DECIDE("quarantine", "Active ransomware detected")</tool_call>'
        result = parse_tool_call(text)
        assert result is not None
        assert result.tool_name == "DECIDE"
        assert result.args["verdict"] == "quarantine"
        assert result.args["explanation"] == "Active ransomware detected"

    def test_inspect_file(self):
        text = '<tool_call>inspect_file("C:/Users/A/Documents/report.docx")</tool_call>'
        result = parse_tool_call(text)
        assert result is not None
        assert result.tool_name == "inspect_file"
        assert result.args["path"] == "C:/Users/A/Documents/report.docx"

    def test_check_process_numeric(self):
        text = '<tool_call>check_process(1234)</tool_call>'
        result = parse_tool_call(text)
        assert result is not None
        assert result.tool_name == "check_process"
        assert result.args["pid"] == 1234

    def test_recall_memory(self):
        text = '<tool_call>recall_memory("entropy spike file encryption")</tool_call>'
        result = parse_tool_call(text)
        assert result is not None
        assert result.tool_name == "recall_memory"
        assert result.args["query"] == "entropy spike file encryption"


class TestParseEdgeCases:
    def test_no_tool_call(self):
        assert parse_tool_call("Just regular text") is None

    def test_empty_tool_call(self):
        assert parse_tool_call("<tool_call></tool_call>") is None

    def test_malformed_json(self):
        text = '<tool_call>{"name": broken json</tool_call>'
        # Should fall through to function parsing, which also fails
        result = parse_tool_call(text)
        # The function parser might partially match — that's okay
        # as long as it doesn't crash

    def test_multiple_tool_calls(self):
        text = '''<tool_call>inspect_file("C:/a.txt")</tool_call>
some thinking...
<tool_call>inspect_file("C:/b.txt")</tool_call>'''
        # parse_tool_call returns first one
        result = parse_tool_call(text)
        assert result.args["path"] == "C:/a.txt"

        # parse_all_tool_calls returns all
        all_calls = parse_all_tool_calls(text)
        assert len(all_calls) == 2
        assert all_calls[1].args["path"] == "C:/b.txt"


class TestFormatToolResult:
    def test_formats_as_xml(self):
        result = format_tool_result({"entropy": 7.92, "size": 2048})
        assert result.startswith("<tool_result>")
        assert result.endswith("</tool_result>")
        assert '"entropy": 7.92' in result


class TestIsDecideCall:
    def test_decide_uppercase(self):
        assert is_decide_call("DECIDE") is True

    def test_decide_lowercase(self):
        assert is_decide_call("decide") is True

    def test_not_decide(self):
        assert is_decide_call("inspect_file") is False


class TestExtractThinking:
    def test_extracts_thinking(self):
        text = "<think>This looks suspicious</think>"
        assert extract_thinking(text) == "This looks suspicious"

    def test_no_thinking(self):
        assert extract_thinking("no think tags here") is None
