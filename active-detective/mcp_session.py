"""DetectiveSession: testable core for MCP server state management."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from simulator.models import ScenarioType, GroundTruth, Verdict
from simulator.telemetry import Episode, generate_episode
from tools.inspection import execute_tool, TOOL_COSTS, VALID_VERDICTS
from environment.reward import compute_reward, RewardBreakdown


class DetectiveSession:
    """Manages a single investigation episode's lifecycle."""

    def __init__(
        self,
        seed: int,
        scenario_type: str,
        observability: float = 0.7,
        attack_progress: float = 0.5,
        k_max: int = 5,
        n_history: int = 2,
        results_file: str | None = None,
    ) -> None:
        self.seed = seed
        self.k_max = k_max
        self.results_file = results_file

        rng = np.random.RandomState(seed)
        stype = ScenarioType(scenario_type)
        progress = 0.0 if stype == ScenarioType.BENIGN else attack_progress

        self.episode: Episode = generate_episode(
            scenario_type=stype,
            observability=observability,
            rng=rng,
            attack_progress=progress,
            n_history=n_history,
        )
        self.host = self.episode.host

        self.steps: int = 0
        self.cumulative_cost: float = 0.0
        self.tools_used: list[str] = []
        self.verdict: str | None = None
        self.explanation: str = ""

    @property
    def is_done(self) -> bool:
        return self.verdict is not None

    @property
    def budget_remaining(self) -> int:
        return self.k_max - self.steps

    def get_telemetry(self) -> str:
        """Return formatted telemetry text with history windows."""
        parts: list[str] = []
        for i, hw in enumerate(self.episode.history_windows):
            n = len(self.episode.history_windows) - i
            parts.append(f"--- Window t-{n} (prior) ---")
            parts.append(hw)
        parts.append("--- Current Window ---")
        parts.append(self.episode.input_text)
        return "\n\n".join(parts)

    def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute an investigation tool. Returns result dict."""
        if self.is_done:
            return {"error": "Investigation already concluded. No further tools available."}
        if self.steps >= self.k_max:
            return {
                "error": f"Budget exhausted ({self.k_max} steps used). "
                "You must call DECIDE to render your verdict."
            }

        result, cost = execute_tool(tool_name, args, self.host)
        self.steps += 1
        self.cumulative_cost += cost
        self.tools_used.append(tool_name)
        return result

    def decide(self, verdict: str, explanation: str) -> dict:
        """Render final verdict. Returns result with reward breakdown."""
        if self.is_done:
            return {"error": "Verdict already rendered."}
        if verdict not in VALID_VERDICTS:
            return {
                "error": f"Invalid verdict '{verdict}'. "
                f"Must be one of: {', '.join(sorted(VALID_VERDICTS))}"
            }

        self.verdict = verdict
        self.explanation = explanation

        gt = self.episode.ground_truth
        predicted_ransomware = Verdict(verdict).is_ransomware_prediction
        correct = predicted_ransomware == gt.is_ransomware

        reward = compute_reward(
            verdict=verdict,
            ground_truth=gt,
            cumulative_cost=self.cumulative_cost,
            steps_taken=self.steps,
            max_steps=self.k_max,
            has_thinking=False,
            has_tool_call=self.steps > 0,
        )

        result_record = {
            "seed": self.seed,
            "scenario_type": self.episode.scenario_type.value,
            "is_ransomware_gt": gt.is_ransomware,
            "attack_phase": gt.attack_phase,
            "observability": self.episode.observability,
            "verdict": verdict,
            "explanation": explanation,
            "steps_taken": self.steps,
            "cumulative_cost": self.cumulative_cost,
            "tools_used": self.tools_used,
            "reward": reward.total,
            "correct": correct,
            "k_max": self.k_max,
        }

        if self.results_file:
            with open(self.results_file, "a") as f:
                f.write(json.dumps(result_record) + "\n")

        return {
            "verdict": verdict,
            "correct": correct,
            "reward": reward.total,
            "verdict_reward": reward.verdict_reward,
            "efficiency_bonus": reward.efficiency_bonus,
            "action_cost": reward.action_cost,
        }
