"""System prompt templates and training format builders.

The system prompt defines the agent-environment contract.
Tool descriptions are parameterized for ablation studies.
"""

from __future__ import annotations

# ── Tool descriptions ────────────────────────────────────────────────

TOOL_DESCRIPTIONS: dict[str, str] = {
    "inspect_file": (
        "inspect_file(path) — Inspect a file's metadata. "
        "Returns entropy, size, extension, modified timestamp, content type."
    ),
    "check_process": (
        "check_process(pid) — Check a running process. "
        "Returns name, command line, parent process, child PIDs."
    ),
    "scan_directory": (
        "scan_directory(path) — List files in a directory with metadata. "
        "More expensive but reveals more information."
    ),
    "list_connections": (
        "list_connections(filter?) — List active network connections. "
        "Returns conn_id, pid, remote address, protocol, state, bytes transferred."
    ),
    "inspect_connection": (
        "inspect_connection(conn_id) — Inspect a specific network connection. "
        "Returns full details including local port, timestamps, byte counts."
    ),
    "query_registry": (
        "query_registry(key_path) — Query a Windows registry key. "
        "Returns all values under the key, useful for detecting persistence and defense evasion."
    ),
    "list_process_handles": (
        "list_process_handles(pid) — List a process's handles and security context. "
        "Returns open file handles, network sockets, loaded modules, user, integrity level."
    ),
    "query_event_log": (
        "query_event_log(source?, event_id?, since?) — Query Windows Event Log. "
        "Returns matching entries. Useful for detecting Defender disable, log clearing, service changes."
    ),
    "read_file_sample": (
        "read_file_sample(path, offset?, length?) — Read raw bytes from a file. "
        "Returns hex-encoded sample, entropy, and magic bytes. Useful for content analysis."
    ),
}

DECIDE_DESCRIPTION = (
    "DECIDE(verdict, explanation) — Submit your final verdict. "
    "This ends the investigation. Verdicts: ignore, monitor, alert, quarantine, block."
)

# ── System prompt template ───────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are a ransomware detection agent analyzing host telemetry.

Your job: examine the telemetry window below, investigate suspicious activity \
using your tools, and render a verdict.

## Available tools

{tool_list}
- {decide_desc}

## Output format

1. Think step-by-step inside <think>...</think> tags.
2. Call a tool inside <tool_call>...</tool_call> tags.
3. You will receive the result in <tool_result>...</tool_result> tags.
4. Repeat steps 1-3 as needed (up to {k_max} steps).
5. When ready, call DECIDE with your verdict and a brief explanation.

## Verdicts

- **ignore** — Normal activity, no threat.
- **monitor** — Mildly suspicious, keep watching.
- **alert** — Likely malicious, notify analyst.
- **quarantine** — Active threat, isolate affected files/processes.
- **block** — Critical threat, terminate and block immediately.

## Budget

You have up to {k_max} investigation steps. Use them wisely — \
unnecessary tool calls reduce your score. If you are confident, \
call DECIDE immediately."""


def build_system_prompt(
    k_max: int = 5,
    available_tools: list[str] | None = None,
) -> str:
    """Build the system prompt with configurable tool set.

    Parameters
    ----------
    k_max:
        Maximum investigation steps (tool calls) allowed.
    available_tools:
        Subset of tool names to include. None means all tools.
        Use for ablation studies (e.g., exclude "scan_directory").
    """
    if available_tools is None:
        available_tools = list(TOOL_DESCRIPTIONS.keys())

    tool_lines = []
    for name in available_tools:
        desc = TOOL_DESCRIPTIONS.get(name)
        if desc:
            tool_lines.append(f"- {desc}")

    tool_list = "\n".join(tool_lines)

    return _SYSTEM_PROMPT_TEMPLATE.format(
        tool_list=tool_list,
        decide_desc=DECIDE_DESCRIPTION,
        k_max=k_max,
    )


def format_episode_input(system_prompt: str, telemetry_window: str) -> str:
    """Combine system prompt and telemetry into a single model input.

    For completion-style models (non-chat).
    """
    return f"{system_prompt}\n\n## Telemetry window\n\n{telemetry_window}\n"


def build_chat_messages(
    system_prompt: str,
    telemetry_window: str,
    history: list[dict] | None = None,
    history_windows: list[str] | None = None,
) -> list[dict]:
    """Build chat-format messages for chat models (e.g., Qwen3).

    Parameters
    ----------
    system_prompt:
        The system prompt text.
    telemetry_window:
        Current telemetry observation.
    history:
        Optional prior tool-call/result turns to prepend.
    history_windows:
        Optional prior telemetry windows (oldest first) to include
        as context before the current window. These are labeled
        ``Window t-N (prior)`` so the model can reason about temporal
        progression of events.

    Returns
    -------
    List of {"role": str, "content": str} message dicts.
    """
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    # Build user content with history context
    if history_windows:
        parts = []
        n = len(history_windows)
        for i, hw in enumerate(history_windows):
            label = f"Window t-{n - i} (prior)"
            parts.append(f"--- {label} ---\n{hw}")
        parts.append(f"--- Current window ---\n{telemetry_window}")
        user_content = "\n\n".join(parts)
    else:
        user_content = telemetry_window

    messages.append({
        "role": "user",
        "content": f"## Telemetry window\n\n{user_content}",
    })

    return messages
