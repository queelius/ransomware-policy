"""Generate evaluation scenario manifests for batch evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.scenarios import DEFAULT_SCENARIO_MIX, DEFAULT_OBSERVABILITY_LEVELS
from simulator.models import ScenarioType


def generate_manifest(
    n: int = 200,
    seed: int = 42,
    scenario_mix: dict[ScenarioType, float] | None = None,
    observability_levels: list[float] | None = None,
) -> list[dict]:
    """Generate a list of scenario configurations for evaluation."""
    mix = scenario_mix or DEFAULT_SCENARIO_MIX
    obs_levels = observability_levels or DEFAULT_OBSERVABILITY_LEVELS
    rng = np.random.RandomState(seed)

    types: list[ScenarioType] = []
    weights = list(mix.values())
    type_list = list(mix.keys())
    for _ in range(n):
        idx = rng.choice(len(type_list), p=weights)
        types.append(type_list[idx])

    manifest: list[dict] = []
    for i, stype in enumerate(types):
        obs = obs_levels[rng.randint(len(obs_levels))]
        progress = 0.0 if stype == ScenarioType.BENIGN else round(rng.uniform(0.2, 0.9), 2)
        manifest.append({
            "seed": seed * 10000 + i,
            "scenario_type": stype.value,
            "observability": obs,
            "attack_progress": progress,
        })

    return manifest


def save_manifest(manifest: list[dict], path: str | Path) -> None:
    """Save manifest to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate eval manifest")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_manifest.jsonl")
    args = parser.parse_args()

    manifest = generate_manifest(n=args.n, seed=args.seed)
    save_manifest(manifest, args.output)
    print(f"Generated {len(manifest)} scenarios to {args.output}")
