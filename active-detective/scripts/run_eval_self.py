"""Run evaluation with Claude Code as the agent (in-session).

Generates episodes, prints telemetry, and waits for tool calls.
For batch automation, outputs scenario configs that can be fed to claude -p.
For quick self-eval, uses a simple heuristic to establish baseline.
"""
import json
import sys
sys.path.insert(0, ".")

from mcp_session import DetectiveSession
from scripts.generate_eval_manifest import generate_manifest


def run_heuristic_eval(manifest: list[dict], results_file: str = "results/heuristic_baseline.jsonl"):
    """Quick heuristic baseline: scan telemetry for extension changes."""
    import pathlib
    pathlib.Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    for entry in manifest:
        session = DetectiveSession(
            seed=entry["seed"],
            scenario_type=entry["scenario_type"],
            observability=entry["observability"],
            attack_progress=entry["attack_progress"],
            results_file=results_file,
        )
        telemetry = session.get_telemetry()
        
        # Simple heuristic: count extension changes in telemetry
        ext_changes = telemetry.count("ext_change=") - telemetry.count("ext_change=none")
        locked_count = telemetry.count(".locked") + telemetry.count(".encrypted") + telemetry.count(".enc") + telemetry.count(".cry")
        defender_disabled = "DisableAntiSpyware" in telemetry or "DisableRealtimeMonitoring" in telemetry
        
        if locked_count >= 3 or (ext_changes >= 3 and defender_disabled):
            verdict = "quarantine"
        elif locked_count >= 1 or ext_changes >= 1:
            verdict = "alert"
        elif defender_disabled:
            verdict = "monitor"
        else:
            verdict = "ignore"
        
        result = session.decide(verdict, f"Heuristic: {ext_changes} ext changes, {locked_count} locked files, defender_disabled={defender_disabled}")
        print(f"  seed={entry['seed']} type={entry['scenario_type']:20s} verdict={verdict:12s} correct={result['correct']} reward={result['reward']:.2f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    results_file = sys.argv[3] if len(sys.argv) > 3 else "results/heuristic_baseline.jsonl"
    
    manifest = generate_manifest(n=n, seed=seed)
    print(f"Running heuristic baseline on {n} scenarios...")
    run_heuristic_eval(manifest, results_file)
    print(f"\nResults written to {results_file}")
    
    # Compute metrics
    from scripts.compute_metrics import load_results, compute_all_metrics, format_report
    results = load_results(results_file)
    metrics = compute_all_metrics(results)
    print(f"\n{format_report(metrics)}")
