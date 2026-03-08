"""Agent investigation tools: inspect_file, check_process, scan_directory,
plus dispatch for all tools via execute_tool.

Each tool takes simulator state and returns a JSON-serializable dict.
"""

from __future__ import annotations

from simulator.host import HostState
from simulator.registry import FileRegistry, ProcessTable

# Tool name → cost (negative values, subtracted from reward)
TOOL_COSTS: dict[str, float] = {
    "inspect_file": -0.02,
    "check_process": -0.02,
    "scan_directory": -0.05,
    "list_connections": -0.03,
    "inspect_connection": -0.03,
    "query_registry": -0.03,
    "list_process_handles": -0.03,
    "query_event_log": -0.04,
    "read_file_sample": -0.04,
    "DECIDE": 0.0,
}

VALID_VERDICTS = {"ignore", "monitor", "alert", "quarantine", "block"}


def inspect_file(registry: FileRegistry, path: str) -> dict:
    """Inspect a file's metadata. Cost: -0.02.

    Returns entropy, size, extension, modified_ts, content_type.
    """
    record = registry.get_file(path)
    if record is None:
        return {"error": f"File not found: {path}"}

    return {
        "entropy": round(record.entropy, 2),
        "size": record.size,
        "extension": record.extension,
        "modified_ts": record.modified_at.isoformat(),
        "content_type": record.content_type.value,
    }


def check_process(ptable: ProcessTable, pid: int) -> dict:
    """Check a process's details. Cost: -0.02.

    Returns name, command_line, parent process name, child PIDs.
    """
    record = ptable.get_process(pid)
    if record is None:
        return {"error": f"Process not found: pid={pid}"}

    parent = ptable.get_process(record.parent_pid)
    parent_name = parent.name if parent else "unknown"

    return {
        "name": record.name,
        "command_line": record.command_line,
        "parent": parent_name,
        "child_pids": record.child_pids,
    }


def scan_directory(registry: FileRegistry, path: str) -> dict:
    """List files in a directory with metadata summaries. Cost: -0.05.

    More expensive because it reveals more information.
    """
    files = registry.list_directory(path)
    if not files:
        return {"files": [], "note": f"No files found in {path}"}

    return {
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "entropy": round(f.entropy, 2),
                "extension": f.extension,
            }
            for f in files
        ]
    }


def execute_tool(
    tool_name: str,
    args: dict,
    host: HostState,
) -> tuple[dict, float]:
    """Execute a tool by name and return (result_dict, cost).

    Dispatches to the appropriate tool function.
    """
    if tool_name == "inspect_file":
        path = args.get("path", "")
        return inspect_file(host.files, path), TOOL_COSTS["inspect_file"]

    elif tool_name == "check_process":
        pid = args.get("pid")
        if pid is None:
            return {"error": "Missing 'pid' argument"}, TOOL_COSTS["check_process"]
        return check_process(host.processes, int(pid)), TOOL_COSTS["check_process"]

    elif tool_name == "scan_directory":
        path = args.get("path", "")
        return scan_directory(host.files, path), TOOL_COSTS["scan_directory"]

    elif tool_name == "list_connections":
        from tools.network_tools import list_connections
        filter_state = args.get("filter", args.get("filter_state"))
        return list_connections(host.connections, filter_state), \
               TOOL_COSTS["list_connections"]

    elif tool_name == "inspect_connection":
        from tools.network_tools import inspect_connection
        conn_id = args.get("conn_id")
        if conn_id is None:
            return {"error": "Missing 'conn_id' argument"}, \
                   TOOL_COSTS["inspect_connection"]
        return inspect_connection(host.connections, int(conn_id)), \
               TOOL_COSTS["inspect_connection"]

    elif tool_name == "query_registry":
        from tools.forensic_tools import query_registry
        key_path = args.get("key_path", "")
        return query_registry(host.registry, key_path), \
               TOOL_COSTS["query_registry"]

    elif tool_name == "list_process_handles":
        from tools.forensic_tools import list_process_handles
        pid = args.get("pid")
        if pid is None:
            return {"error": "Missing 'pid' argument"}, \
                   TOOL_COSTS["list_process_handles"]
        return list_process_handles(host.processes, int(pid)), \
               TOOL_COSTS["list_process_handles"]

    elif tool_name == "query_event_log":
        from tools.forensic_tools import query_event_log
        return query_event_log(
            host.event_log,
            source=args.get("source"),
            event_id=args.get("event_id"),
            since=args.get("since"),
        ), TOOL_COSTS["query_event_log"]

    elif tool_name == "read_file_sample":
        from tools.forensic_tools import read_file_sample
        path = args.get("path", "")
        offset = int(args.get("offset", 0))
        length = int(args.get("length", 256))
        return read_file_sample(host.files, path, offset, length), \
               TOOL_COSTS["read_file_sample"]

    elif tool_name == "DECIDE":
        verdict = args.get("verdict", "")
        explanation = args.get("explanation", "")
        if verdict not in VALID_VERDICTS:
            return {"error": f"Invalid verdict '{verdict}'. "
                    f"Must be one of: {', '.join(sorted(VALID_VERDICTS))}"}, 0.0
        return {"verdict": verdict, "explanation": explanation}, TOOL_COSTS["DECIDE"]

    else:
        return {"error": f"Unknown tool: {tool_name}"}, 0.0
