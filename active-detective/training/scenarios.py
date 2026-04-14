"""Batch scenario generator for training episodes.

Generates balanced episode batches with configurable scenario mix,
observability levels, and optional memory-store seeding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from simulator.models import ScenarioType
from simulator.telemetry import Episode, generate_episode

# Default scenario mix from design doc Phase 1
DEFAULT_SCENARIO_MIX: dict[ScenarioType, float] = {
    ScenarioType.BENIGN: 0.40,
    ScenarioType.BLITZ: 0.20,
    ScenarioType.SLEEPER: 0.15,
    ScenarioType.EXFIL_FIRST: 0.15,
    ScenarioType.SEMANTIC_SHUFFLE: 0.10,
}

DEFAULT_OBSERVABILITY_LEVELS = [0.3, 0.5, 0.7, 0.9]


@dataclass
class ScenarioBatch:
    """A batch of generated training episodes with metadata."""

    episodes: list[Episode]
    seed: int

    @property
    def scenario_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ep in self.episodes:
            key = ep.scenario_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def observability_stats(self) -> dict[str, float]:
        obs = [ep.observability for ep in self.episodes]
        return {
            "min": min(obs),
            "max": max(obs),
            "mean": sum(obs) / len(obs),
        }


def generate_training_scenarios(
    n_episodes: int = 1000,
    observability_levels: list[float] | None = None,
    scenario_mix: dict[ScenarioType, float] | None = None,
    seed: int = 42,
) -> ScenarioBatch:
    """Generate a batch of training episodes.

    Parameters
    ----------
    n_episodes:
        Total number of episodes to generate.
    observability_levels:
        Observability levels to sample from uniformly.
    scenario_mix:
        Mapping of ScenarioType → probability. Must sum to ~1.0.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    ScenarioBatch with episodes and metadata.
    """
    if observability_levels is None:
        observability_levels = DEFAULT_OBSERVABILITY_LEVELS
    if scenario_mix is None:
        scenario_mix = DEFAULT_SCENARIO_MIX

    rng = np.random.RandomState(seed)

    # Build scenario schedule from mix proportions
    scenario_types = list(scenario_mix.keys())
    probabilities = np.array([scenario_mix[s] for s in scenario_types])
    # Normalize in case proportions don't exactly sum to 1
    probabilities = probabilities / probabilities.sum()

    episodes: list[Episode] = []
    for _ in range(n_episodes):
        # Sample scenario type
        idx = rng.choice(len(scenario_types), p=probabilities)
        scenario_type = scenario_types[idx]

        # Sample observability level uniformly
        obs = observability_levels[rng.randint(0, len(observability_levels))]

        # Sample attack progress (uniform 0.2–0.9 for attack scenarios)
        if scenario_type == ScenarioType.BENIGN:
            progress = 0.0
        else:
            progress = rng.uniform(0.2, 0.9)

        ep = generate_episode(
            scenario_type=scenario_type,
            observability=obs,
            rng=rng,
            attack_progress=progress,
        )
        episodes.append(ep)

    return ScenarioBatch(episodes=episodes, seed=seed)


def save_scenarios(batch: ScenarioBatch, path: str | Path) -> None:
    """Save a scenario batch to JSONL for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for ep in batch.episodes:
            record = {
                "input_text": ep.input_text,
                "scenario_type": ep.scenario_type.value,
                "is_ransomware": ep.ground_truth.is_ransomware,
                "attack_phase": ep.ground_truth.attack_phase,
                "observability": ep.observability,
                "raw_event_count": ep.raw_event_count,
                "visible_event_count": ep.visible_event_count,
            }
            f.write(json.dumps(record) + "\n")


def load_scenarios(path: str | Path) -> list[dict]:
    """Load scenarios from JSONL file.

    Returns list of dicts (not Episode objects; Episodes contain
    non-serializable state). Use for analysis and training data loading.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_scenario_plan(
    n_episodes: int = 1000,
    observability_levels: list[float] | None = None,
    scenario_mix: dict[ScenarioType, float] | None = None,
    seed: int = 42,
    n_history: int = 2,
) -> list[dict]:
    """Build a reproducible plan of scenario parameters without generating episodes.

    Returns a list of dicts, each with the exact fields that
    DetectionEnv.reset() consumes via scenario_data. This replaces the
    wasteful pattern where prepare_dataset generated full episodes just
    to read their parameters.

    The returned plan is the canonical source for both training and
    for save_scenario_plan: what is saved is exactly what the env rolls
    out at training time.
    """
    if observability_levels is None:
        observability_levels = DEFAULT_OBSERVABILITY_LEVELS
    if scenario_mix is None:
        scenario_mix = DEFAULT_SCENARIO_MIX

    rng = np.random.RandomState(seed)
    scenario_types = list(scenario_mix.keys())
    probabilities = np.array([scenario_mix[s] for s in scenario_types])
    probabilities = probabilities / probabilities.sum()

    plan: list[dict] = []
    for _ in range(n_episodes):
        idx = rng.choice(len(scenario_types), p=probabilities)
        scenario_type = scenario_types[idx]
        obs = observability_levels[rng.randint(0, len(observability_levels))]

        if scenario_type == ScenarioType.BENIGN:
            progress = 0.0
        else:
            progress = float(rng.uniform(0.2, 0.9))

        ep_seed = int(rng.randint(0, 2**31))

        plan.append({
            "scenario_type": scenario_type.value,
            "observability": float(obs),
            "attack_progress": progress,
            "seed": ep_seed,
            "n_history": n_history,
        })

    return plan


def save_scenario_plan(plan: list[dict], path: str | Path) -> None:
    """Save a scenario plan to JSONL. Each line is directly usable as
    scenario_data in DetectionEnv.reset()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in plan:
            f.write(json.dumps(entry) + "\n")


def load_scenario_plan(path: str | Path) -> list[dict]:
    """Load a scenario plan from JSONL."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
