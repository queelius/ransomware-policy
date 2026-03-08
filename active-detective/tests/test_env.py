"""Tests for RansomwareDetectionEnv."""

import numpy as np
import pytest

from environment.env import RansomwareDetectionEnv, RolloutResult, StepResult
from simulator.models import ScenarioType, Verdict
from tools.parser import ParsedToolCall


@pytest.fixture
def env():
    return RansomwareDetectionEnv(max_steps=5)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestEnvReset:
    def test_reset_returns_text(self, env, rng):
        text = env.reset(ScenarioType.BENIGN, 0.8, rng)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_reset_clears_state(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.8, rng)
        assert env.steps_taken == 0
        assert env.budget_remaining == 5
        assert not env.is_done

    def test_reset_with_history(self, env, rng):
        history = ["File entropy spike on report.docx",
                    "Normal office activity"]
        text = env.reset(ScenarioType.BLITZ, 0.7, rng,
                         history_windows=history)
        assert isinstance(text, str)

    def test_reset_reproducible(self):
        env1 = RansomwareDetectionEnv()
        env2 = RansomwareDetectionEnv()
        t1 = env1.reset(ScenarioType.BENIGN, 0.5, np.random.RandomState(42))
        t2 = env2.reset(ScenarioType.BENIGN, 0.5, np.random.RandomState(42))
        assert t1 == t2


class TestEnvStep:
    def test_step_inspect_file(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)

        call = ParsedToolCall(
            tool_name="scan_directory",
            args={"path": "C:/Users/A/Documents"},
            raw_text="scan_directory(\"C:/Users/A/Documents\")",
        )
        result = env.step(call)
        assert isinstance(result, StepResult)
        assert result.tool_name == "scan_directory"
        assert not result.is_terminal
        assert result.cost < 0
        assert "<tool_result>" in result.result_text

    def test_step_decide_is_terminal(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)

        call = ParsedToolCall(
            tool_name="DECIDE",
            args={"verdict": "ignore", "explanation": "Looks benign"},
            raw_text='DECIDE("ignore", "Looks benign")',
        )
        result = env.step(call)
        assert result.is_terminal
        assert env.is_done

    def test_step_accumulates_cost(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)

        for _ in range(3):
            call = ParsedToolCall(
                tool_name="check_process",
                args={"pid": 4},
                raw_text="check_process(4)",
            )
            env.step(call)

        assert env.steps_taken == 3
        # 3 * -0.02 = -0.06
        assert abs(env._cumulative_cost - (-0.06)) < 1e-9

    def test_step_after_done_raises(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        env.step(ParsedToolCall("DECIDE", {"verdict": "ignore"}, ""))
        with pytest.raises(RuntimeError, match="done"):
            env.step(ParsedToolCall("check_process", {"pid": 4}, ""))

    def test_step_without_reset_raises(self):
        env = RansomwareDetectionEnv()
        with pytest.raises(RuntimeError, match="No episode"):
            env.step(ParsedToolCall("check_process", {"pid": 4}, ""))


class TestBudgetExhaustion:
    def test_budget_exhaustion_marks_done(self):
        env = RansomwareDetectionEnv(max_steps=2)
        env.reset(ScenarioType.BENIGN, 0.9, np.random.RandomState(42))

        env.step(ParsedToolCall("check_process", {"pid": 4}, ""))
        assert not env.is_done

        env.step(ParsedToolCall("check_process", {"pid": 4}, ""))
        assert env.is_done

    def test_budget_remaining_tracks(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        assert env.budget_remaining == 5

        env.step(ParsedToolCall("check_process", {"pid": 4}, ""))
        assert env.budget_remaining == 4


class TestStepFromText:
    def test_valid_tool_call(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        text = '<tool_call>check_process(4)</tool_call>'
        result = env.step_from_text(text)
        assert result is not None
        assert result.tool_name == "check_process"

    def test_no_tool_call_returns_none(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        result = env.step_from_text("Just some thinking, no tool call")
        assert result is None
        assert not env._has_tool_call

    def test_json_format_tool_call(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        text = '<tool_call>{"name": "check_process", "arguments": {"pid": 4}}</tool_call>'
        result = env.step_from_text(text)
        assert result is not None
        assert result.tool_name == "check_process"


class TestRecallMemoryRemoved:
    def test_recall_memory_returns_unknown_tool_error(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        call = ParsedToolCall(
            tool_name="recall_memory",
            args={"query": "entropy spike"},
            raw_text='recall_memory("entropy spike")',
        )
        result = env.step(call)
        assert "error" in result.result
        assert "Unknown tool" in result.result["error"]


class TestFinish:
    def test_finish_after_decide(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        env.step(ParsedToolCall("DECIDE", {"verdict": "ignore", "explanation": "safe"}, ""))

        rollout = env.finish()
        assert isinstance(rollout, RolloutResult)
        assert rollout.verdict == "ignore"
        assert rollout.explanation == "safe"
        assert rollout.reward is not None
        assert rollout.reward.verdict_reward > 0  # correct for benign
        assert not rollout.exceeded_budget

    def test_finish_after_budget_exhaustion(self):
        env = RansomwareDetectionEnv(max_steps=2)
        env.reset(ScenarioType.BLITZ, 0.7, np.random.RandomState(42),
                  attack_progress=0.5)

        env.step(ParsedToolCall("check_process", {"pid": 4}, ""))
        env.step(ParsedToolCall("check_process", {"pid": 4}, ""))

        rollout = env.finish()
        assert rollout.exceeded_budget
        assert rollout.verdict == "ignore"  # default
        # Should be very negative: false negative (-2) on ransomware
        assert rollout.reward.verdict_reward < 0

    def test_finish_without_episode_raises(self):
        env = RansomwareDetectionEnv()
        with pytest.raises(RuntimeError, match="No episode"):
            env.finish()

    def test_correct_ransomware_detection(self, rng):
        env = RansomwareDetectionEnv(max_steps=5)
        env.reset(ScenarioType.BLITZ, 0.9, rng, attack_progress=0.8)

        # Investigate then decide
        env.step(ParsedToolCall("scan_directory",
                                {"path": "C:/Users/A/Documents"}, ""))
        env.step(ParsedToolCall("DECIDE",
                                {"verdict": "quarantine",
                                 "explanation": "Ransomware detected"}, ""))

        rollout = env.finish()
        assert rollout.reward.verdict_reward > 0
        assert rollout.reward.total > 0

    def test_no_thinking_reduces_format_reward(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        # Step without thinking tags — only has_tool_call will be True
        env.step(ParsedToolCall("DECIDE", {"verdict": "ignore"}, ""))

        rollout = env.finish()
        assert not rollout.has_thinking
        assert rollout.has_tool_call
        # Only tool_call component (0.05), no thinking component
        assert abs(rollout.reward.format_reward - 0.05) < 1e-9

    def test_thinking_tags_increase_format_reward(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        env.mark_thinking()
        env.step(ParsedToolCall("DECIDE", {"verdict": "ignore"}, ""))

        rollout = env.finish()
        assert rollout.has_thinking
        assert rollout.has_tool_call
        # Both components: 0.05 + 0.05
        assert abs(rollout.reward.format_reward - 0.10) < 1e-9


class TestRolloutResult:
    def test_steps_tracked(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        env.step(ParsedToolCall("check_process", {"pid": 4}, ""))
        env.step(ParsedToolCall("check_process", {"pid": 4}, ""))
        env.step(ParsedToolCall("DECIDE", {"verdict": "ignore"}, ""))

        rollout = env.finish()
        assert len(rollout.steps) == 3
        assert rollout.steps[0].tool_name == "check_process"
        assert rollout.steps[2].is_terminal

    def test_cumulative_cost(self, env, rng):
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        env.step(ParsedToolCall("inspect_file",
                                {"path": "C:/nonexistent.txt"}, ""))
        env.step(ParsedToolCall("DECIDE", {"verdict": "ignore"}, ""))

        rollout = env.finish()
        assert rollout.cumulative_cost < 0  # at least one tool cost
