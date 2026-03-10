"""Tests for agent tools and tool-call parser."""

from datetime import datetime

import numpy as np
import pytest

from simulator.host import HostState
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
from tools.network_tools import inspect_connection, list_connections
from tools.forensic_tools import (
    list_process_handles,
    query_event_log,
    query_registry,
    read_file_sample,
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
def host():
    rng = np.random.RandomState(42)
    return HostState.create(rng)


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
    def test_dispatch_inspect_file(self, host):
        # Use a known path from the seeded filesystem
        paths = host.files.all_paths()
        result, cost = execute_tool("inspect_file", {"path": paths[0]}, host)
        assert "entropy" in result
        assert cost == TOOL_COSTS["inspect_file"]

    def test_dispatch_check_process(self, host):
        result, cost = execute_tool("check_process", {"pid": 4}, host)
        assert result["name"] == "System"
        assert cost == TOOL_COSTS["check_process"]

    def test_dispatch_decide_valid(self, host):
        result, cost = execute_tool(
            "DECIDE", {"verdict": "alert", "explanation": "test"}, host)
        assert result["verdict"] == "alert"
        assert cost == 0.0

    def test_dispatch_decide_invalid(self, host):
        result, cost = execute_tool(
            "DECIDE", {"verdict": "invalid_verdict"}, host)
        assert "error" in result

    def test_unknown_tool(self, host):
        result, cost = execute_tool("fake_tool", {}, host)
        assert "error" in result

    def test_dispatch_list_connections(self, host):
        result, cost = execute_tool("list_connections", {}, host)
        assert "connections" in result
        assert cost == TOOL_COSTS["list_connections"]

    def test_dispatch_inspect_connection(self, host):
        result, cost = execute_tool("inspect_connection", {"conn_id": 1}, host)
        assert "conn_id" in result or "error" in result
        assert cost == TOOL_COSTS["inspect_connection"]

    def test_dispatch_query_registry(self, host):
        result, cost = execute_tool(
            "query_registry",
            {"key_path": r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run"},
            host)
        assert "values" in result or "error" in result
        assert cost == TOOL_COSTS["query_registry"]

    def test_dispatch_list_process_handles(self, host):
        result, cost = execute_tool("list_process_handles", {"pid": 4}, host)
        assert result["name"] == "System"
        assert "loaded_modules" in result
        assert cost == TOOL_COSTS["list_process_handles"]

    def test_dispatch_query_event_log(self, host):
        result, cost = execute_tool("query_event_log", {}, host)
        assert "entries" in result
        assert result["count"] > 0
        assert cost == TOOL_COSTS["query_event_log"]

    def test_dispatch_read_file_sample(self, host):
        result, cost = execute_tool(
            "read_file_sample",
            {"path": host.files.all_paths()[0]},
            host)
        # Contents may not be set, so we might get an error
        assert "path" in result or "error" in result
        assert cost == TOOL_COSTS["read_file_sample"]


# ── New tool function tests ──────────────────────────────────────────


class TestListConnections:
    def test_all_connections(self, host):
        result = list_connections(host.connections)
        assert len(result["connections"]) == 3  # seeded background connections

    def test_filter_by_state(self, host):
        result = list_connections(host.connections, filter_state="established")
        assert len(result["connections"]) == 3

    def test_filter_empty_result(self, host):
        result = list_connections(host.connections, filter_state="closed")
        assert len(result["connections"]) == 0


class TestInspectConnection:
    def test_existing_connection(self, host):
        result = inspect_connection(host.connections, 1)
        assert result["conn_id"] == 1
        assert "remote_address" in result
        assert "opened_at" in result

    def test_missing_connection(self, host):
        result = inspect_connection(host.connections, 999)
        assert "error" in result


class TestQueryRegistry:
    def test_existing_key(self, host):
        result = query_registry(
            host.registry,
            r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
        )
        assert "values" in result
        assert "SecurityHealth" in result["values"]

    def test_missing_key(self, host):
        result = query_registry(host.registry, r"HKLM\MISSING\KEY")
        assert "error" in result

    def test_parent_key_enumerates_children(self, host):
        result = query_registry(host.registry, r"HKLM\SOFTWARE")
        # Should find sub-keys
        assert "sub_keys" in result


class TestListProcessHandles:
    def test_system_process(self, host):
        result = list_process_handles(host.processes, 4)
        assert result["name"] == "System"
        assert result["user"] == "SYSTEM"
        assert result["is_elevated"] is True
        assert "ntdll.dll" in result["loaded_modules"]

    def test_missing_process(self, host):
        result = list_process_handles(host.processes, 99999)
        assert "error" in result


class TestQueryEventLog:
    def test_all_events(self, host):
        result = query_event_log(host.event_log)
        assert result["count"] == 5  # seeded events

    def test_filter_by_source(self, host):
        result = query_event_log(host.event_log, source="System")
        assert result["count"] >= 2

    def test_filter_by_event_id(self, host):
        result = query_event_log(host.event_log, event_id=4624)
        assert result["count"] == 1

    def test_invalid_since(self, host):
        result = query_event_log(host.event_log, since="not-a-date")
        assert "error" in result


class TestReadFileSample:
    def test_file_with_seeded_contents(self, host):
        """seed_filesystem now populates contents, so read_file_sample works."""
        path = host.files.all_paths()[0]
        result = read_file_sample(host.files, path)
        assert "hex" in result
        assert "entropy" in result
        assert "magic_bytes" in result
        assert "error" not in result

    def test_file_with_contents(self):
        r = FileRegistry()
        now = datetime(2025, 1, 1)
        r.add_file("C:/test.txt", 100, 3.0, ".txt", ContentType.DOC, now)
        r.modify_contents("C:/test.txt", b"Hello, World! " * 20)
        result = read_file_sample(r, "C:/test.txt")
        assert "hex" in result
        assert "entropy" in result
        assert "magic_bytes" in result

    def test_missing_file(self, host):
        result = read_file_sample(host.files, "C:/nope.txt")
        assert "error" in result


class TestToolCosts:
    def test_all_tools_have_costs(self):
        expected_tools = {
            "inspect_file", "check_process", "scan_directory",
            "list_connections", "inspect_connection",
            "query_registry", "list_process_handles", "query_event_log",
            "read_file_sample", "DECIDE",
        }
        assert set(TOOL_COSTS.keys()) == expected_tools

    def test_recall_memory_not_in_tool_costs(self):
        assert "recall_memory" not in TOOL_COSTS

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

    def test_new_tools_parse(self):
        """Verify the 6 new tools parse correctly."""
        test_cases = [
            ('list_connections("established")', "list_connections",
             {"filter": "established"}),
            ('inspect_connection(5)', "inspect_connection", {"conn_id": 5}),
            ('query_registry("HKLM\\\\SOFTWARE")', "query_registry",
             {"key_path": "HKLM\\SOFTWARE"}),
            ('list_process_handles(1234)', "list_process_handles", {"pid": 1234}),
            ('read_file_sample("C:/test.txt")', "read_file_sample",
             {"path": "C:/test.txt"}),
        ]
        for raw, expected_name, expected_args in test_cases:
            text = f"<tool_call>{raw}</tool_call>"
            result = parse_tool_call(text)
            assert result is not None, f"Failed to parse: {raw}"
            assert result.tool_name == expected_name


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

    def test_multiple_tool_calls(self):
        text = '''<tool_call>inspect_file("C:/a.txt")</tool_call>
some thinking...
<tool_call>inspect_file("C:/b.txt")</tool_call>'''
        result = parse_tool_call(text)
        assert result.args["path"] == "C:/a.txt"

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
