"""Agent investigation tools: inspect_file, check_process, scan_directory.

Each tool takes simulator state and returns a JSON-serializable dict.
"""

from __future__ import annotations

from simulator.registry import FileRegistry, ProcessTable

# Tool name → cost (negative values, subtracted from reward)
TOOL_COSTS: dict[str, float] = {
    "inspect_file": -0.02,
    "check_process": -0.02,
    "scan_directory": -0.05,
    "recall_memory": -0.03,
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
    registry: FileRegistry,
    ptable: ProcessTable,
    memory_store: object | None = None,
) -> tuple[dict, float]:
    """Execute a tool by name and return (result_dict, cost).

    Dispatches to the appropriate tool function.
    The memory_store parameter is used for recall_memory (implemented in tools/memory.py).
    """
    if tool_name == "inspect_file":
        path = args.get("path", "")
        return inspect_file(registry, path), TOOL_COSTS["inspect_file"]

    elif tool_name == "check_process":
        pid = args.get("pid")
        if pid is None:
            return {"error": "Missing 'pid' argument"}, TOOL_COSTS["check_process"]
        return check_process(ptable, int(pid)), TOOL_COSTS["check_process"]

    elif tool_name == "scan_directory":
        path = args.get("path", "")
        return scan_directory(registry, path), TOOL_COSTS["scan_directory"]

    elif tool_name == "recall_memory":
        if memory_store is None:
            return {"matches": [], "note": "Memory store not available"}, \
                   TOOL_COSTS["recall_memory"]
        # Delegate to memory module's recall function
        query = args.get("query", "")
        from tools.memory import recall_memory
        return recall_memory(memory_store, query), TOOL_COSTS["recall_memory"]

    elif tool_name == "DECIDE":
        verdict = args.get("verdict", "")
        explanation = args.get("explanation", "")
        if verdict not in VALID_VERDICTS:
            return {"error": f"Invalid verdict '{verdict}'. "
                    f"Must be one of: {', '.join(sorted(VALID_VERDICTS))}"}, 0.0
        return {"verdict": verdict, "explanation": explanation}, TOOL_COSTS["DECIDE"]

    else:
        return {"error": f"Unknown tool: {tool_name}"}, 0.0
