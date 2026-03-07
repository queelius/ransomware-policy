"""End-to-end smoke tests: full pipeline from episode generation to reward.

Uses a mock model (no GPU needed) that generates realistic tool-call text.
Validates the machinery works before investing in GRPO training.
"""

from __future__ import annotations

import numpy as np
import pytest

from environment.env import RansomwareDetectionEnv, RolloutResult
from environment.reward import (
    CORRECT_REWARD,
    FALSE_NEGATIVE_REWARD,
    FORMAT_THINKING_REWARD,
    FORMAT_TOOL_CALL_REWARD,
    WRONG_REWARD,
)
from simulator.models import ScenarioType
from tools.parser import ParsedToolCall, parse_tool_call
from training.prompts import build_chat_messages, build_system_prompt
from training.scenarios import generate_training_scenarios


# ── Mock model that produces tool calls ──────────────────────────────


def _mock_investigate_then_decide(
    telemetry: str,
    verdict: str = "ignore",
    explanation: str = "Looks normal",
) -> list[str]:
    """Simulate a model's multi-turn output.

    Returns a list of generated texts (one per turn).
    """
    turns = [
        # Turn 1: think + scan directory
        '<think>Let me check the documents folder for suspicious files.</think>\n'
        '<tool_call>scan_directory("C:/Users/A/Documents")</tool_call>',
        # Turn 2: think + decide
        f'<think>Based on the scan results, this looks {verdict}.</think>\n'
        f'<tool_call>DECIDE("{verdict}", "{explanation}")</tool_call>',
    ]
    return turns


def _mock_direct_decide(
    verdict: str = "ignore",
    explanation: str = "Nothing suspicious",
) -> list[str]:
    """Model that immediately decides without investigation."""
    return [
        f'<think>Quick assessment — nothing stands out.</think>\n'
        f'<tool_call>DECIDE("{verdict}", "{explanation}")</tool_call>',
    ]


def _mock_malformed_then_decide() -> list[str]:
    """Model that produces garbage first, then a valid DECIDE."""
    return [
        "I think this might be suspicious but I'm not sure what to do",
        '<tool_call>DECIDE("monitor", "Uncertain")</tool_call>',
    ]


def _mock_json_format_model(verdict: str = "alert") -> list[str]:
    """Model that uses Qwen3 JSON tool-call format."""
    return [
        '<think>Analyzing the telemetry.</think>\n'
        '<tool_call>{"name": "check_process", "arguments": {"pid": 4}}</tool_call>',
        '<think>Process looks fine. Let me decide.</think>\n'
        f'<tool_call>{{"name": "DECIDE", "arguments": {{"verdict": "{verdict}", "explanation": "Analysis complete"}}}}</tool_call>',
    ]


# ── Helper to run a full rollout ─────────────────────────────────────


def run_mock_rollout(
    env: RansomwareDetectionEnv,
    model_turns: list[str],
) -> RolloutResult:
    """Execute a full rollout using pre-generated model outputs."""
    for turn_text in model_turns:
        if env.is_done:
            break

        result = env.step_from_text(turn_text)
        if result is None:
            # No valid tool call found — has_tool_call stays False for this turn
            pass

    return env.finish()


# ── Tests ────────────────────────────────────────────────────────────


class TestEndToEndPipeline:
    """Full pipeline: generate → prompt → (mock) model → parse → execute → reward."""

    def test_correct_benign_detection(self):
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        telemetry = env.reset(ScenarioType.BENIGN, 0.8, rng)

        turns = _mock_investigate_then_decide(telemetry, "ignore", "Normal activity")
        rollout = run_mock_rollout(env, turns)

        assert rollout.verdict == "ignore"
        assert rollout.reward.verdict_reward == CORRECT_REWARD
        assert rollout.reward.total > 0
        assert not rollout.exceeded_budget
        assert rollout.has_thinking
        assert rollout.has_tool_call

    def test_correct_ransomware_detection(self):
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        telemetry = env.reset(ScenarioType.BLITZ, 0.9, rng, attack_progress=0.7)

        turns = _mock_investigate_then_decide(
            telemetry, "quarantine", "Active ransomware")
        rollout = run_mock_rollout(env, turns)

        assert rollout.verdict == "quarantine"
        assert rollout.reward.verdict_reward == CORRECT_REWARD

    def test_false_negative(self):
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        env.reset(ScenarioType.SLEEPER, 0.5, rng, attack_progress=0.4)

        turns = _mock_direct_decide("ignore", "Everything normal")
        rollout = run_mock_rollout(env, turns)

        assert rollout.reward.verdict_reward == FALSE_NEGATIVE_REWARD
        assert rollout.reward.total < 0

    def test_false_positive(self):
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        env.reset(ScenarioType.BENIGN, 0.8, rng)

        turns = _mock_direct_decide("quarantine", "Suspicious activity")
        rollout = run_mock_rollout(env, turns)

        assert rollout.reward.verdict_reward == WRONG_REWARD

    def test_malformed_output_handled(self):
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        env.reset(ScenarioType.BENIGN, 0.8, rng)

        turns = _mock_malformed_then_decide()
        rollout = run_mock_rollout(env, turns)

        assert not rollout.has_thinking  # no <think> tags
        assert rollout.has_tool_call  # DECIDE is still a tool call
        # Only tool_call component — no thinking
        assert abs(rollout.reward.format_reward - FORMAT_TOOL_CALL_REWARD) < 1e-9
        # Should still reach a verdict
        assert rollout.verdict == "monitor"

    def test_json_format_tool_calls(self):
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        env.reset(ScenarioType.BLITZ, 0.9, rng, attack_progress=0.8)

        turns = _mock_json_format_model("alert")
        rollout = run_mock_rollout(env, turns)

        assert rollout.verdict == "alert"
        assert rollout.has_thinking
        assert rollout.has_tool_call
        assert len(rollout.steps) == 2

    def test_budget_exhaustion(self):
        env = RansomwareDetectionEnv(max_steps=2)
        rng = np.random.RandomState(42)
        env.reset(ScenarioType.BLITZ, 0.7, rng, attack_progress=0.5)

        # Two investigation steps, no DECIDE
        turns = [
            '<tool_call>check_process(4)</tool_call>',
            '<tool_call>check_process(4)</tool_call>',
        ]
        rollout = run_mock_rollout(env, turns)

        assert rollout.exceeded_budget
        assert rollout.verdict == "ignore"  # default

    def test_efficiency_bonus_for_quick_decide(self):
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        env.reset(ScenarioType.BENIGN, 0.8, rng)

        # Decide immediately (1 step, 4 unused)
        quick = _mock_direct_decide("ignore", "Clearly safe")
        rollout_quick = run_mock_rollout(env, quick)

        env2 = RansomwareDetectionEnv(max_steps=5)
        rng2 = np.random.RandomState(42)
        env2.reset(ScenarioType.BENIGN, 0.8, rng2)

        # Investigate then decide (2 steps, 3 unused)
        slow = _mock_investigate_then_decide("", "ignore", "Checked first")
        rollout_slow = run_mock_rollout(env2, slow)

        assert rollout_quick.reward.efficiency_bonus > rollout_slow.reward.efficiency_bonus


class TestBatchRollout:
    """Test running multiple episodes like a training batch."""

    def test_batch_of_mixed_scenarios(self):
        """Run 10 episodes and verify reward distribution."""
        batch = generate_training_scenarios(n_episodes=10, seed=42)
        rewards = []

        for ep in batch.episodes:
            env = RansomwareDetectionEnv(max_steps=5)
            rng = np.random.RandomState(hash(ep.input_text) % (2**31))
            env.reset(ep.scenario_type, ep.observability, rng)

            # Simple heuristic: always ignore
            turns = _mock_direct_decide("ignore", "No action needed")
            rollout = run_mock_rollout(env, turns)
            rewards.append(rollout.reward.total)

        # With ~40% benign episodes, "always ignore" should get some right
        correct_count = sum(1 for r in rewards if r > 0)
        assert correct_count > 0  # at least some benign episodes scored positive
        # But most attack episodes should score negative
        negative_count = sum(1 for r in rewards if r < 0)
        assert negative_count > 0


class TestPromptIntegration:
    """Test that system prompt + env work together."""

    def test_system_prompt_fits_in_context(self):
        prompt = build_system_prompt(k_max=5)
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        telemetry = env.reset(ScenarioType.BLITZ, 0.9, rng, attack_progress=0.5)

        messages = build_chat_messages(prompt, telemetry)
        total_chars = sum(len(m["content"]) for m in messages)
        # Rough estimate: 4 chars per token, 4096 token budget
        # System prompt + telemetry should be well under budget
        assert total_chars < 4096 * 4

    def test_tool_ablation_prompt(self):
        """System prompt with ablated tools still works with env."""
        prompt = build_system_prompt(
            k_max=3,
            available_tools=["inspect_file", "check_process"],
        )
        assert "recall_memory" not in prompt
        assert "scan_directory" not in prompt

        # Env still works (tools are in the env regardless of prompt)
        env = RansomwareDetectionEnv(max_steps=3)
        rng = np.random.RandomState(42)
        env.reset(ScenarioType.BENIGN, 0.8, rng)
        turns = _mock_direct_decide("ignore", "Safe")
        rollout = run_mock_rollout(env, turns)
        assert rollout.verdict == "ignore"


class TestRolloutTraceInspection:
    """Verify rollout traces contain enough information for debugging."""

    def test_step_results_have_tool_info(self):
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        env.reset(ScenarioType.BENIGN, 0.9, rng)

        turns = _mock_investigate_then_decide("", "ignore", "OK")
        rollout = run_mock_rollout(env, turns)

        assert len(rollout.steps) == 2
        assert rollout.steps[0].tool_name == "scan_directory"
        assert "files" in rollout.steps[0].result or "note" in rollout.steps[0].result
        assert rollout.steps[1].tool_name == "DECIDE"
        assert rollout.steps[1].is_terminal

    def test_result_text_is_formatted_xml(self):
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        env.reset(ScenarioType.BENIGN, 0.9, rng)

        turns = [
            '<tool_call>check_process(4)</tool_call>',
            '<tool_call>DECIDE("ignore", "safe")</tool_call>',
        ]
        rollout = run_mock_rollout(env, turns)

        for step in rollout.steps:
            assert step.result_text.startswith("<tool_result>")
            assert step.result_text.endswith("</tool_result>")

    def test_recall_memory_in_rollout(self):
        env = RansomwareDetectionEnv(max_steps=5)
        rng = np.random.RandomState(42)
        history = ["Entropy spike on multiple docx files detected"]
        env.reset(ScenarioType.BLITZ, 0.9, rng, history_windows=history)

        turns = [
            '<tool_call>recall_memory("entropy spike")</tool_call>',
            '<tool_call>DECIDE("alert", "Historical context confirms threat")</tool_call>',
        ]
        rollout = run_mock_rollout(env, turns)

        assert rollout.steps[0].tool_name == "recall_memory"
        assert "matches" in rollout.steps[0].result
