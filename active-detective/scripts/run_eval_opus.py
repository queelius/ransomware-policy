"""Run evaluation with Claude Opus via Anthropic API.

Uses the same DetectiveSession and tools as the MCP server,
but calls Claude's API directly for batch automation.

Usage:
    PYTHONPATH=. python scripts/run_eval_opus.py [n_scenarios] [seed]
    PYTHONPATH=. python scripts/run_eval_opus.py 20 42
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anthropic

sys.path.insert(0, ".")

from mcp_session import DetectiveSession
from scripts.generate_eval_manifest import generate_manifest

# Claude API tool definitions matching the MCP server tools
TOOLS = [
    {
        "name": "get_telemetry",
        "description": "Retrieve the host telemetry windows for analysis. Call this first to see the telemetry data before investigating. Returns current window plus prior history windows.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "inspect_file",
        "description": "Inspect a file's metadata: entropy, size, extension, content_type, modified timestamp. Cost: 1 investigation step.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path to inspect"}},
            "required": ["path"],
        },
    },
    {
        "name": "check_process",
        "description": "Check a running process: name, command line, parent, children. Cost: 1 investigation step.",
        "input_schema": {
            "type": "object",
            "properties": {"pid": {"type": "integer", "description": "Process ID"}},
            "required": ["pid"],
        },
    },
    {
        "name": "scan_directory",
        "description": "List all files in a directory with metadata summaries. Most expensive tool. Cost: 1 investigation step.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_connections",
        "description": "List active network connections with optional state filter. Cost: 1 investigation step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filter_state": {
                    "type": "string",
                    "description": "Filter by state: established, listening, closed, syn_sent",
                },
            },
            "required": [],
        },
    },
    {
        "name": "inspect_connection",
        "description": "Inspect a specific network connection in detail. Cost: 1 investigation step.",
        "input_schema": {
            "type": "object",
            "properties": {"conn_id": {"type": "integer", "description": "Connection ID"}},
            "required": ["conn_id"],
        },
    },
    {
        "name": "query_registry",
        "description": "Query a Windows registry key and its values. Useful for detecting persistence, Defender disabled, VSS deleted. Cost: 1 investigation step.",
        "input_schema": {
            "type": "object",
            "properties": {"key_path": {"type": "string", "description": "Registry key path"}},
            "required": ["key_path"],
        },
    },
    {
        "name": "list_process_handles",
        "description": "List a process's open file handles, loaded modules, and security context. Cost: 1 investigation step.",
        "input_schema": {
            "type": "object",
            "properties": {"pid": {"type": "integer", "description": "Process ID"}},
            "required": ["pid"],
        },
    },
    {
        "name": "query_event_log",
        "description": "Query Windows Event Log entries. Filter by source, event_id, and/or timestamp. Cost: 1 investigation step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Event source filter"},
                "event_id": {"type": "integer", "description": "Event ID filter"},
                "since": {"type": "string", "description": "Timestamp filter"},
            },
            "required": [],
        },
    },
    {
        "name": "read_file_sample",
        "description": "Read raw bytes from a file for forensic analysis. Returns hex dump, computed entropy, and magic bytes. Cost: 1 investigation step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "offset": {"type": "integer", "description": "Byte offset (default 0)"},
                "length": {"type": "integer", "description": "Bytes to read (default 256)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "DECIDE",
        "description": "Submit your final verdict. This ends the investigation. verdict: ignore, monitor, alert, quarantine, or block.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "description": "Your verdict"},
                "explanation": {"type": "string", "description": "Brief reasoning"},
            },
            "required": ["verdict", "explanation"],
        },
    },
]


def load_eval_prompt() -> str:
    """Load the full Opus evaluation prompt."""
    prompt_path = Path("prompts/eval_prompt.md")
    if prompt_path.exists():
        return prompt_path.read_text()
    return (
        "You are a ransomware forensic investigator. Call get_telemetry() first, "
        "then investigate with tools, then call DECIDE(verdict, explanation). "
        "Verdicts: ignore, monitor, alert, quarantine, block. Max 5 investigation steps."
    )


def execute_tool_call(session: DetectiveSession, tool_name: str, tool_input: dict) -> str:
    """Execute a tool call against the session and return the result as a string."""
    if tool_name == "get_telemetry":
        return session.get_telemetry()
    elif tool_name == "DECIDE":
        result = session.decide(
            tool_input.get("verdict", "ignore"),
            tool_input.get("explanation", ""),
        )
        return json.dumps(result, indent=2)
    else:
        result = session.execute_tool(tool_name, tool_input)
        return json.dumps(result, indent=2)


def run_scenario(
    client: anthropic.Anthropic,
    session: DetectiveSession,
    system_prompt: str,
    model: str = "claude-opus-4-6",
    max_turns: int = 15,
) -> dict:
    """Run a single scenario through Claude and return the result."""
    messages = [{"role": "user", "content": "Analyze this host for ransomware. Begin by calling get_telemetry()."}]

    for turn in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Check if model wants to use tools
        if response.stop_reason == "tool_use":
            # Process all tool calls in the response
            assistant_content = response.content
            tool_results = []

            for block in assistant_content:
                if block.type == "tool_use":
                    tool_result = execute_tool_call(session, block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result,
                    })

                    # Check if DECIDE was called
                    if block.name == "DECIDE":
                        return {
                            "verdict": block.input.get("verdict", "ignore"),
                            "explanation": block.input.get("explanation", ""),
                            "turns": turn + 1,
                        }

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Model stopped without tool use — shouldn't happen but handle it
            break

    # If we got here, model never called DECIDE — force ignore
    return {"verdict": "ignore", "explanation": "Budget exhausted without verdict", "turns": max_turns}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    model = sys.argv[3] if len(sys.argv) > 3 else "claude-opus-4-6"
    results_file = f"results/opus_eval_{n}_{seed}.jsonl"

    Path("results").mkdir(exist_ok=True)
    client = anthropic.Anthropic()
    system_prompt = load_eval_prompt()
    manifest = generate_manifest(n=n, seed=seed)

    print(f"Running {n} scenarios through {model}...")
    print(f"Results: {results_file}\n")

    for i, entry in enumerate(manifest):
        session = DetectiveSession(
            seed=entry["seed"],
            scenario_type=entry["scenario_type"],
            observability=entry["observability"],
            attack_progress=entry["attack_progress"],
            results_file=results_file,
        )

        try:
            result = run_scenario(client, session, system_prompt, model=model)

            # If DECIDE wasn't called through the session (e.g., budget issue), call it now
            if not session.is_done:
                session.decide(result["verdict"], result["explanation"])

            gt = session.episode.ground_truth
            correct = session.verdict is not None and (
                (session.verdict in ("quarantine", "block", "alert") and gt.is_ransomware)
                or (session.verdict in ("ignore", "monitor") and not gt.is_ransomware)
            )

            print(
                f"  [{i+1}/{n}] seed={entry['seed']} "
                f"type={entry['scenario_type']:20s} "
                f"verdict={result['verdict']:12s} "
                f"correct={correct} "
                f"steps={session.steps} "
                f"turns={result['turns']}"
            )
        except Exception as e:
            print(f"  [{i+1}/{n}] seed={entry['seed']} type={entry['scenario_type']:20s} ERROR: {e}")
            # Record a failed scenario
            if not session.is_done:
                session.decide("ignore", f"Error: {e}")

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    print(f"\nResults written to {results_file}")

    from scripts.compute_metrics import load_results, compute_all_metrics, format_report
    results = load_results(results_file)
    metrics = compute_all_metrics(results)
    print(f"\n{format_report(metrics)}")


if __name__ == "__main__":
    main()
