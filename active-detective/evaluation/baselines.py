"""Baseline implementations for controlled comparison.

Each baseline implements the same interface: given a telemetry window
and environment, produce a verdict. This enables apples-to-apples
comparison against the RL-trained agent.

Baselines:
- RandomAgent: random tool + random verdict (sanity check)
- ExhaustiveAgent: uses all k steps round-robin, then decides
- HeuristicAgent: entropy + rename rate heuristics (no LLM)
- PassiveLLM: same model but forced to DECIDE immediately (no tools)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from environment.env import RansomwareDetectionEnv
from evaluation.metrics import EvalResult
from simulator.models import ScenarioType, Verdict
from tools.inspection import VALID_VERDICTS
from tools.parser import ParsedToolCall


# ── Base interface ───────────────────────────────────────────────────


@dataclass
class BaselineConfig:
    """Configuration for baseline evaluation."""

    k_max: int = 5
    seed: int = 42


class BaselineAgent:
    """Base class for baseline agents."""

    name: str = "baseline"

    def act(
        self,
        env: RansomwareDetectionEnv,
        telemetry_text: str,
        rng: np.random.RandomState,
    ) -> EvalResult:
        """Run the agent on a single episode and return the result."""
        raise NotImplementedError


# ── Random agent ─────────────────────────────────────────────────────


class RandomAgent(BaselineAgent):
    """Random tool selection + random verdict. Sanity check baseline.

    Uses each tool step to call a random tool with random arguments,
    then issues a random verdict.
    """

    name = "random"

    def __init__(self, k_max: int = 5) -> None:
        self.k_max = k_max

    def act(self, env, telemetry_text, rng):
        tools_used = []

        # Random number of investigation steps (0 to k_max - 1)
        n_steps = rng.randint(0, self.k_max)

        for _ in range(n_steps):
            if env.is_done:
                break

            tool_choice = rng.choice([
                "check_process", "inspect_file", "scan_directory",
                "list_connections", "inspect_connection",
                "query_registry", "list_process_handles", "query_event_log",
                "read_file_sample",
            ])
            tools_used.append(tool_choice)

            if tool_choice == "check_process":
                call = ParsedToolCall(tool_choice, {"pid": 4}, "")
            elif tool_choice == "inspect_file":
                call = ParsedToolCall(tool_choice, {"path": "C:/test.txt"}, "")
            elif tool_choice == "scan_directory":
                call = ParsedToolCall(tool_choice, {"path": "C:/Users"}, "")
            elif tool_choice == "list_connections":
                call = ParsedToolCall(tool_choice, {}, "")
            elif tool_choice == "inspect_connection":
                call = ParsedToolCall(tool_choice, {"conn_id": 1}, "")
            elif tool_choice == "query_registry":
                call = ParsedToolCall(tool_choice, {"key_path": r"HKLM\SOFTWARE"}, "")
            elif tool_choice == "list_process_handles":
                call = ParsedToolCall(tool_choice, {"pid": 4}, "")
            elif tool_choice == "query_event_log":
                call = ParsedToolCall(tool_choice, {}, "")
            else:  # read_file_sample
                call = ParsedToolCall(tool_choice, {"path": "C:/test.txt"}, "")

            env.step(call)

        # Random verdict
        if not env.is_done:
            verdict = rng.choice(list(VALID_VERDICTS))
            tools_used.append("DECIDE")
            env.step(ParsedToolCall("DECIDE", {"verdict": verdict}, ""))

        rollout = env.finish()
        return EvalResult(
            verdict=rollout.verdict,
            is_ransomware_gt=rollout.ground_truth.is_ransomware,
            scenario_type=rollout.ground_truth.scenario_type.value,
            attack_phase=rollout.ground_truth.attack_phase,
            steps_taken=len(rollout.steps),
            cumulative_cost=rollout.cumulative_cost,
            k_max=self.k_max,
            tools_used=tools_used,
        )


# ── Exhaustive agent ─────────────────────────────────────────────────


class ExhaustiveAgent(BaselineAgent):
    """Uses all k-1 steps with round-robin tools, then decides.

    Tests whether the RL agent learns meaningful early stopping.
    """

    name = "exhaustive"

    TOOL_SEQUENCE = [
        ("scan_directory", {"path": "C:/Users/A/Documents"}),
        ("check_process", {"pid": 4}),
        ("list_connections", {}),
        ("query_registry", {"key_path": r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run"}),
        ("query_event_log", {}),
        ("list_process_handles", {"pid": 4}),
        ("inspect_file", {"path": "C:/Users/A/Documents"}),
        ("read_file_sample", {"path": "C:/Users/A/Documents"}),
        ("inspect_connection", {"conn_id": 1}),
    ]

    def __init__(self, k_max: int = 5) -> None:
        self.k_max = k_max

    def act(self, env, telemetry_text, rng):
        tools_used = []

        # Use k_max - 1 investigation steps
        for i in range(self.k_max - 1):
            if env.is_done:
                break

            tool_name, args = self.TOOL_SEQUENCE[i % len(self.TOOL_SEQUENCE)]
            tools_used.append(tool_name)
            env.step(ParsedToolCall(tool_name, args, ""))

        # Always decide based on simple heuristic
        if not env.is_done:
            verdict = self._heuristic_verdict(telemetry_text)
            tools_used.append("DECIDE")
            env.step(ParsedToolCall(
                "DECIDE", {"verdict": verdict, "explanation": "Exhaustive scan"}, ""))

        rollout = env.finish()
        return EvalResult(
            verdict=rollout.verdict,
            is_ransomware_gt=rollout.ground_truth.is_ransomware,
            scenario_type=rollout.ground_truth.scenario_type.value,
            attack_phase=rollout.ground_truth.attack_phase,
            steps_taken=len(rollout.steps),
            cumulative_cost=rollout.cumulative_cost,
            k_max=self.k_max,
            tools_used=tools_used,
        )

    @staticmethod
    def _heuristic_verdict(telemetry: str) -> str:
        """Simple keyword heuristic for verdict."""
        suspicious_keywords = [
            "ext_change=", ".locked", ".encrypted",
            ".enc", "encoded", "ransom",
        ]
        score = sum(1 for kw in suspicious_keywords if kw in telemetry)
        if score >= 3:
            return "quarantine"
        elif score >= 1:
            return "alert"
        return "ignore"


# ── Heuristic agent ──────────────────────────────────────────────────


class HeuristicAgent(BaselineAgent):
    """Feature-based heuristic detector. No LLM, no RL.

    Counts extension changes and size deltas from telemetry text,
    applies thresholds. This is the "passive classifier" baseline
    from the design doc. Note: entropy_delta is no longer available
    in passive telemetry (the agent must use tools to discover it).
    """

    name = "heuristic"

    def act(self, env, telemetry_text, rng):
        # Extract features directly from telemetry text (no tool calls)
        rename_count = _count_extension_changes(telemetry_text)

        if rename_count >= 2:
            verdict = "quarantine"
        elif rename_count >= 1:
            verdict = "alert"
        else:
            verdict = "ignore"

        # Decide immediately (no tools used)
        if not env.is_done:
            env.step(ParsedToolCall(
                "DECIDE",
                {"verdict": verdict,
                 "explanation": f"rename_count={rename_count}"},
                "",
            ))

        rollout = env.finish()
        return EvalResult(
            verdict=rollout.verdict,
            is_ransomware_gt=rollout.ground_truth.is_ransomware,
            scenario_type=rollout.ground_truth.scenario_type.value,
            attack_phase=rollout.ground_truth.attack_phase,
            steps_taken=1,
            cumulative_cost=0.0,
            k_max=0,
            tools_used=["DECIDE"],
        )


# ── Helpers ──────────────────────────────────────────────────────────


def _extract_entropy_deltas(text: str) -> list[float]:
    """Extract entropy_delta values from telemetry text."""
    pattern = r"entropy_delta=([+-]?\d+\.?\d*)"
    return [float(m) for m in re.findall(pattern, text)]


def _count_extension_changes(text: str) -> int:
    """Count extension change events in telemetry text."""
    return text.count("ext_change=")


# ── Evaluation runner ────────────────────────────────────────────────


def evaluate_baseline(
    agent: BaselineAgent,
    scenarios: list[dict],
    k_max: int = 5,
    seed: int = 42,
) -> list[EvalResult]:
    """Evaluate a baseline agent on a set of scenarios.

    Parameters
    ----------
    agent:
        Baseline agent to evaluate.
    scenarios:
        List of scenario dicts (from generate_training_scenarios or load).
        Each must have: scenario_type, observability.
    k_max:
        Tool call budget.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of EvalResult, one per scenario.
    """
    rng = np.random.RandomState(seed)
    results = []

    for scenario in scenarios:
        scenario_type = ScenarioType(scenario.get("scenario_type", "benign"))
        observability = scenario.get("observability", 0.5)

        env = RansomwareDetectionEnv(max_steps=k_max)
        ep_rng = np.random.RandomState(rng.randint(0, 2**31))
        telemetry = env.reset(scenario_type, observability, ep_rng)

        result = agent.act(env, telemetry, rng)
        results.append(result)

    return results
