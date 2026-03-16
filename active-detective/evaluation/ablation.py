"""Tool ablation runner: evaluate with restricted tool sets.

Per design doc section 5.6, runs the trained model with one tool removed
at a time to measure each tool's marginal contribution. Also tracks
which tools the agent *attempts* to call when they're unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.metrics import (
    DetectionMetrics,
    EfficiencyMetrics,
    EvalResult,
    compute_detection_metrics,
    compute_efficiency_metrics,
    compute_tool_usage,
)

# All investigation tools (excluding DECIDE which is always available)
ALL_TOOLS = [
    "inspect_file", "check_process", "scan_directory",
    "list_connections", "inspect_connection", "query_registry",
    "list_process_handles", "query_event_log", "read_file_sample",
]

# V1 tools (original 3, before HostState expansion)
V1_TOOLS = ["inspect_file", "check_process", "scan_directory"]

# Ablation variants from design doc section 5.6
ABLATION_VARIANTS: dict[str, list[str]] = {
    "full": ALL_TOOLS.copy(),
    # Tool removal variants
    "no_scan_directory": [t for t in ALL_TOOLS if t != "scan_directory"],
    "no_check_process": [t for t in ALL_TOOLS if t != "check_process"],
    "inspect_only": ["inspect_file"],
    # New tool removal variants
    "no_list_connections": [t for t in ALL_TOOLS if t != "list_connections"],
    "no_query_registry": [t for t in ALL_TOOLS if t != "query_registry"],
    "no_event_log": [t for t in ALL_TOOLS if t != "query_event_log"],
    "no_process_handles": [t for t in ALL_TOOLS if t != "list_process_handles"],
    "no_file_sample": [t for t in ALL_TOOLS if t != "read_file_sample"],
    # Category variants
    "file_tools_only": ["inspect_file", "scan_directory", "read_file_sample"],
    "network_tools_only": ["list_connections", "inspect_connection"],
    "v1_tools": V1_TOOLS.copy(),
    "no_tools": [],
}


@dataclass
class AblationResult:
    """Result of evaluating one ablation variant."""

    variant_name: str
    available_tools: list[str]
    detection: DetectionMetrics
    efficiency: EfficiencyMetrics
    tool_usage: dict[str, int]
    attempted_removed_tools: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "variant": self.variant_name,
            "available_tools": self.available_tools,
            "detection": self.detection.to_dict(),
            "efficiency": self.efficiency.to_dict(),
            "tool_usage": self.tool_usage,
            "attempted_removed_tools": self.attempted_removed_tools,
        }


def run_ablation_sweep(
    eval_fn: callable,
    scenarios: list[dict],
    variants: dict[str, list[str]] | None = None,
) -> list[AblationResult]:
    """Run ablation sweep across tool set variants.

    Parameters
    ----------
    eval_fn:
        Callable that takes (scenarios, available_tools) → list[EvalResult].
        This abstracts away the model loading/inference — the caller
        provides the evaluation function.
    scenarios:
        List of scenario dicts to evaluate on.
    variants:
        Mapping of variant name → list of available tool names.
        Defaults to ABLATION_VARIANTS.

    Returns
    -------
    List of AblationResult, one per variant.
    """
    if variants is None:
        variants = ABLATION_VARIANTS

    results = []
    for variant_name, tools in variants.items():
        eval_results = eval_fn(scenarios, tools)

        # Detect attempts to use removed tools
        removed = set(ALL_TOOLS) - set(tools)
        attempted_removed: dict[str, int] = {}
        for r in eval_results:
            for tool in r.tools_used:
                if tool in removed:
                    attempted_removed[tool] = attempted_removed.get(tool, 0) + 1

        results.append(AblationResult(
            variant_name=variant_name,
            available_tools=tools,
            detection=compute_detection_metrics(eval_results),
            efficiency=compute_efficiency_metrics(eval_results),
            tool_usage=compute_tool_usage(eval_results),
            attempted_removed_tools=attempted_removed,
        ))

    return results


def save_ablation_results(
    results: list[AblationResult],
    path: str | Path,
) -> None:
    """Save ablation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def format_ablation_table(results: list[AblationResult]) -> str:
    """Format ablation results as a markdown table for display."""
    lines = [
        "| Variant | Tools | F1 | Recall | Precision | Mean Steps | "
        "Early Stop % | Attempted Removed |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        tool_str = ", ".join(r.available_tools) if r.available_tools else "(none)"
        removed_str = (
            ", ".join(f"{k}={v}" for k, v in r.attempted_removed_tools.items())
            if r.attempted_removed_tools else "-"
        )
        lines.append(
            f"| {r.variant_name} "
            f"| {tool_str} "
            f"| {r.detection.f1:.3f} "
            f"| {r.detection.recall:.3f} "
            f"| {r.detection.precision:.3f} "
            f"| {r.efficiency.mean_steps:.1f} "
            f"| {r.efficiency.early_stop_rate:.1%} "
            f"| {removed_str} |"
        )

    return "\n".join(lines)
