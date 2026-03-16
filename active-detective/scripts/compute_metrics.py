"""Compute evaluation metrics from results JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics import (
    EvalResult,
    compute_detection_metrics,
    compute_efficiency_metrics,
    compute_phase_accuracy,
    compute_tool_usage,
)


def load_results(path: str | Path) -> list[EvalResult]:
    """Load evaluation results from a JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            results.append(
                EvalResult(
                    verdict=r["verdict"],
                    is_ransomware_gt=r["is_ransomware_gt"],
                    scenario_type=r["scenario_type"],
                    attack_phase=r.get("attack_phase"),
                    steps_taken=r["steps_taken"],
                    cumulative_cost=r["cumulative_cost"],
                    k_max=r["k_max"],
                    tools_used=r.get("tools_used", []),
                )
            )
    return results


def compute_all_metrics(results: list[EvalResult]) -> dict:
    """Compute all metrics from evaluation results."""
    detection = compute_detection_metrics(results)
    efficiency = compute_efficiency_metrics(results)
    phase_acc = compute_phase_accuracy(results)
    tool_use = compute_tool_usage(results)

    # Compute metrics by scenario type
    by_type = {}
    for stype in sorted(set(r.scenario_type for r in results)):
        subset = [r for r in results if r.scenario_type == stype]
        by_type[stype] = compute_detection_metrics(subset).to_dict()

    return {
        "detection": detection.to_dict(),
        "efficiency": efficiency.to_dict(),
        "by_scenario_type": by_type,
        "phase_accuracy": phase_acc,
        "tool_usage": tool_use,
        "n_episodes": len(results),
    }


def format_report(metrics: dict) -> str:
    """Format metrics as a human-readable report."""
    lines = []
    d = metrics["detection"]
    e = metrics["efficiency"]

    lines.append(f"=== Detection Metrics ({metrics['n_episodes']} episodes) ===")
    lines.append(f"  Precision: {d['precision']:.3f}")
    lines.append(f"  Recall:    {d['recall']:.3f}")
    lines.append(f"  F1:        {d['f1']:.3f}")
    lines.append(f"  Accuracy:  {d['accuracy']:.3f}")
    lines.append(f"  TP={d['tp']} FP={d['fp']} TN={d['tn']} FN={d['fn']}")

    lines.append(f"\n=== Efficiency Metrics ===")
    lines.append(f"  Mean steps: {e['mean_steps']:.2f}")
    lines.append(f"  Mean cost:  {e['mean_cost']:.3f}")
    lines.append(f"  Early stop: {e['early_stop_rate']:.1%}")

    lines.append(f"\n=== By Scenario Type ===")
    for stype, m in sorted(metrics["by_scenario_type"].items()):
        lines.append(f"  {stype}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")

    lines.append(f"\n=== Tool Usage ===")
    for tool, count in sorted(metrics["tool_usage"].items(), key=lambda x: -x[1]):
        lines.append(f"  {tool}: {count}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("results", nargs="+", help="JSONL result files")
    args = parser.parse_args()

    for path in args.results:
        print(f"\n{'='*60}")
        print(f"Results: {path}")
        print(f"{'='*60}")
        results = load_results(path)
        metrics = compute_all_metrics(results)
        print(format_report(metrics))
