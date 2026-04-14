"""Tool-call parser: parse <tool_call> from model output, format <tool_result>.

Supports two formats:
1. Qwen3 native JSON: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
2. Fallback function-call: <tool_call>tool_name("arg1", key=val)</tool_call>
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ParsedToolCall:
    """A successfully parsed tool call."""

    tool_name: str
    args: dict
    raw_text: str


# Regex patterns for tool call extraction
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)

# Pattern for function-call syntax: tool_name("arg1", key="val")
_FUNC_CALL_PATTERN = re.compile(
    r"^(\w+)\s*\((.*)\)$", re.DOTALL
)


def parse_tool_call(text: str) -> ParsedToolCall | None:
    """Parse a tool call from the model's generated text.

    Returns the first valid tool call found, or None.
    """
    match = _TOOL_CALL_PATTERN.search(text)
    if not match:
        return None

    raw = match.group(1).strip()

    # Try JSON format first (Qwen3 native)
    parsed = _try_json_format(raw)
    if parsed:
        return ParsedToolCall(
            tool_name=parsed[0], args=parsed[1], raw_text=raw,
        )

    # Try function-call format
    parsed = _try_function_format(raw)
    if parsed:
        return ParsedToolCall(
            tool_name=parsed[0], args=parsed[1], raw_text=raw,
        )

    return None


def parse_all_tool_calls(text: str) -> list[ParsedToolCall]:
    """Parse all tool calls from text (for multi-step rollout analysis)."""
    results = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        raw = match.group(1).strip()
        parsed = _try_json_format(raw) or _try_function_format(raw)
        if parsed:
            results.append(ParsedToolCall(
                tool_name=parsed[0], args=parsed[1], raw_text=raw,
            ))
    return results


def format_tool_result(result: dict) -> str:
    """Format a tool result dict as <tool_result> XML."""
    return f"<tool_result>{json.dumps(result)}</tool_result>"


def is_decide_call(tool_name: str) -> bool:
    """Check if this is the terminal DECIDE action."""
    return tool_name.upper() == "DECIDE"


def extract_thinking(text: str) -> str | None:
    """Extract <think>...</think> content from model output."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else None


# ── Private parsing helpers ──────────────────────────────────────────


def _try_json_format(raw: str) -> tuple[str, dict] | None:
    """Try parsing Qwen3 JSON format: {"name": "...", "arguments": {...}}"""
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "name" in data:
            name = data["name"]
            args = data.get("arguments", data.get("args", {}))
            if not isinstance(args, dict):
                args = {}
            return name, args
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _try_function_format(raw: str) -> tuple[str, dict] | None:
    """Try parsing function-call format: tool_name("arg1", key="val")

    Handles common patterns:
    - DECIDE("quarantine", "explanation text")
    - inspect_file("C:/path/to/file.docx")
    - check_process(1234)
    """
    match = _FUNC_CALL_PATTERN.match(raw)
    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2).strip()

    if not args_str:
        return tool_name, {}

    args = _parse_function_args(tool_name, args_str)
    return tool_name, args


def _parse_function_args(tool_name: str, args_str: str) -> dict:
    """Parse function arguments into a dict based on tool name.

    Maps positional arguments to known parameter names.
    """
    # Try to parse as JSON array first
    try:
        values = json.loads(f"[{args_str}]")
        return _map_positional_args(tool_name, values)
    except json.JSONDecodeError:
        pass

    # Manual parsing for common patterns
    # Handle key=value pairs
    args: dict = {}
    parts = _split_args(args_str)

    positional: list = []
    for part in parts:
        part = part.strip()
        if "=" in part and not part.startswith('"'):
            key, val = part.split("=", 1)
            args[key.strip()] = _coerce_value(val.strip())
        else:
            positional.append(_coerce_value(part))

    if positional and not args:
        return _map_positional_args(tool_name, positional)

    # Merge positional into args
    mapped = _map_positional_args(tool_name, positional)
    mapped.update(args)
    return mapped


def _map_positional_args(tool_name: str, values: list) -> dict:
    """Map positional arguments to parameter names by tool."""
    if not values:
        return {}

    from tools.inspection import TOOL_PARAMS
    params = TOOL_PARAMS.get(tool_name, ())
    result = {}
    for i, val in enumerate(values):
        if i < len(params):
            result[params[i]] = val
        else:
            result[f"arg_{i}"] = val
    return result


def _split_args(args_str: str) -> list[str]:
    """Split argument string respecting quoted strings."""
    parts: list[str] = []
    current: list[str] = []
    in_quotes = False
    quote_char = ""
    depth = 0

    for char in args_str:
        if char in ('"', "'") and depth == 0:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
            current.append(char)
        elif char in ("{", "["):
            depth += 1
            current.append(char)
        elif char in ("}", "]"):
            depth -= 1
            current.append(char)
        elif char == "," and not in_quotes and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)

    if current:
        parts.append("".join(current))

    return parts


def _coerce_value(val: str) -> str | int | float:
    """Try to coerce a string value to int or float."""
    # Strip quotes
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        return val[1:-1]

    try:
        return int(val)
    except ValueError:
        pass

    try:
        return float(val)
    except ValueError:
        pass

    return val
