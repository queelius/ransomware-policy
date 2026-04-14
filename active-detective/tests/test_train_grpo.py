"""Tests for the GRPO training script components.

Tests everything that doesn't require GPU: DetectionEnv, reward functions,
dataset preparation, CLI config parsing.
"""

import json

import numpy as np
import pytest

from simulator.models import GroundTruth, ScenarioType, Verdict
from training.train_grpo import (
    DetectionEnv,
    TrainingConfig,
    _compute_env_reward,
    detection_reward,
    format_reward,
    prepare_dataset,
)


# ── DetectionEnv tests ───────────────────────────────────────────────


class TestDetectionEnv:
    def _make_scenario_data(self, scenario_type="benign", observability=0.8,
                            seed=42, attack_progress=0.0, n_history=2):
        return json.dumps({
            "scenario_type": scenario_type,
            "observability": observability,
            "attack_progress": attack_progress,
            "seed": seed,
            "n_history": n_history,
        })

    def test_reset_returns_telemetry_text(self):
        env = DetectionEnv()
        result = env.reset(
            scenario_data=self._make_scenario_data())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_reset_sets_ground_truth(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data("benign"))
        assert env._ground_truth is not None
        assert env._ground_truth.is_ransomware is False

    def test_reset_ransomware_ground_truth(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data(
            "blitz", attack_progress=0.5))
        assert env._ground_truth.is_ransomware is True

    def test_inspect_file_returns_json(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.inspect_file("C:/nonexistent.txt")
        data = json.loads(result)
        assert "error" in data  # file doesn't exist

    def test_check_process_system(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.check_process(4)  # System process
        data = json.loads(result)
        assert data["name"] == "System"

    def test_scan_directory(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.scan_directory("C:/Users/A/Documents")
        data = json.loads(result)
        assert "files" in data

    def test_decide_valid(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.decide("ignore", "Normal activity")
        data = json.loads(result)
        assert data["verdict"] == "ignore"
        assert env._verdict == "ignore"

    def test_decide_invalid(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.decide("invalid_verdict", "Bad")
        data = json.loads(result)
        assert "error" in data

    def test_list_connections(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.list_connections()
        data = json.loads(result)
        assert "connections" in data

    def test_inspect_connection(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.inspect_connection(1)
        data = json.loads(result)
        assert "conn_id" in data or "error" in data

    def test_query_registry(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.query_registry(
            r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run")
        data = json.loads(result)
        assert "values" in data or "error" in data

    def test_list_process_handles(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.list_process_handles(4)
        data = json.loads(result)
        assert data["name"] == "System"

    def test_query_event_log(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.query_event_log()
        data = json.loads(result)
        assert "entries" in data

    def test_read_file_sample(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        result = env.read_file_sample("C:/nonexistent.txt")
        data = json.loads(result)
        assert "error" in data

    def test_step_counting(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        assert env._steps == 0

        env.check_process(4)
        assert env._steps == 1

        env.check_process(4)
        assert env._steps == 2

    def test_cost_accumulation(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())

        env.inspect_file("C:/test.txt")  # -0.02
        env.check_process(4)              # -0.02
        env.scan_directory("C:/Users")    # -0.05

        expected = -0.02 + -0.02 + -0.05
        assert abs(env._cumulative_cost - expected) < 1e-9

    def test_host_from_episode(self):
        """DetectionEnv should use episode's host, not a fresh one."""
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        # The host should be the episode's host
        assert env._host is not None
        # Inspect a file that exists — should get real data, not error
        paths = env._host.files.all_paths()
        assert len(paths) > 0
        result = json.loads(env.inspect_file(paths[0]))
        assert "error" not in result
        assert "entropy" in result

    def test_reset_includes_history_windows(self):
        """reset() should return history windows + current window."""
        env = DetectionEnv()
        result = env.reset(scenario_data=self._make_scenario_data(
            "blitz", attack_progress=0.5))
        assert "Window t-2 (prior)" in result
        assert "Window t-1 (prior)" in result
        assert "Current window" in result

    def test_reset_no_history_when_zero(self):
        env = DetectionEnv()
        result = env.reset(scenario_data=self._make_scenario_data(n_history=0))
        assert "Window t-" not in result
        assert "Current window" not in result  # no label when no history

    def test_reset_clears_state(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._make_scenario_data())
        env.check_process(4)
        env.decide("ignore", "safe")

        # Reset again
        env.reset(scenario_data=self._make_scenario_data())
        assert env._steps == 0
        assert env._cumulative_cost == 0.0
        assert env._verdict is None


class TestDetectionEnvMultiStep:
    """Test realistic multi-step investigation sequences."""

    def test_investigate_then_decide_benign(self):
        env = DetectionEnv()
        env.reset(scenario_data=json.dumps({
            "scenario_type": "benign",
            "observability": 0.9,
            "attack_progress": 0.0,
            "seed": 42,
            "n_history": 2,
        }))

        # Step 1: scan directory
        result1 = env.scan_directory("C:/Users/A/Documents")
        assert "files" in json.loads(result1)

        # Step 2: check a process
        result2 = env.check_process(4)
        assert json.loads(result2)["name"] == "System"

        # Step 3: decide
        result3 = env.decide("ignore", "Normal activity detected")
        assert json.loads(result3)["verdict"] == "ignore"

        assert env._steps == 3
        assert env._verdict == "ignore"

    def test_investigate_then_decide_ransomware(self):
        env = DetectionEnv()
        env.reset(scenario_data=json.dumps({
            "scenario_type": "blitz",
            "observability": 0.9,
            "attack_progress": 0.8,
            "seed": 42,
            "n_history": 2,
        }))

        # Investigate
        env.scan_directory("C:/Users/A/Documents")

        # Decide
        env.decide("quarantine", "Active ransomware detected")

        assert env._verdict == "quarantine"
        assert env._ground_truth.is_ransomware is True


# ── Reward function tests ────────────────────────────────────────────


class TestComputeEnvReward:
    def _make_env(self, scenario_type="benign", is_ransomware=False,
                  verdict=None, steps=1, cost=0.0):
        env = DetectionEnv()
        env._ground_truth = GroundTruth(
            scenario_type=ScenarioType(scenario_type),
            is_ransomware=is_ransomware,
        )
        env._verdict = verdict
        env._steps = steps
        env._cumulative_cost = cost
        return env

    def test_correct_benign_ignore(self):
        env = self._make_env("benign", False, "ignore", steps=1)
        r = _compute_env_reward(env)
        assert r > 0  # correct + efficiency bonus

    def test_correct_ransomware_alert(self):
        env = self._make_env("blitz", True, "alert", steps=2, cost=-0.04)
        r = _compute_env_reward(env)
        assert r > 0

    def test_false_negative_harsh_penalty(self):
        env = self._make_env("blitz", True, "ignore", steps=1)
        r = _compute_env_reward(env)
        assert r < -1.0  # -2.0 + small bonuses

    def test_false_positive_penalty(self):
        env = self._make_env("benign", False, "quarantine", steps=1)
        r = _compute_env_reward(env)
        assert r < 0

    def test_no_verdict_penalty(self):
        env = self._make_env("benign", False, None, steps=5)
        r = _compute_env_reward(env)
        assert r < 0

    def test_no_verdict_ransomware_worst(self):
        env = self._make_env("blitz", True, None, steps=5)
        r = _compute_env_reward(env)
        assert r <= -2.0

    def test_efficiency_bonus_removed(self):
        """The reward redesign deliberately removed efficiency bonus to
        avoid rewarding 'do nothing fast'. Steps taken shouldn't affect
        the reward for a correct verdict."""
        env_fast = self._make_env("benign", False, "ignore", steps=1)
        env_slow = self._make_env("benign", False, "ignore", steps=5)
        assert _compute_env_reward(env_fast) == _compute_env_reward(env_slow)

    def test_action_cost_not_subtracted(self):
        """The reward redesign removed action cost to stop penalizing
        investigation. Tool costs are bookkept but don't reduce reward."""
        env_cheap = self._make_env("benign", False, "ignore", steps=1, cost=0.0)
        env_expensive = self._make_env("benign", False, "ignore", steps=1, cost=-0.1)
        assert _compute_env_reward(env_cheap) == _compute_env_reward(env_expensive)

    def test_reward_independent_of_k_max(self):
        """After the reward redesign, k_max does not affect the verdict
        reward (no efficiency bonus). It only controls the tool call budget
        at rollout time."""
        env10 = DetectionEnv(k_max=10)
        env10._ground_truth = GroundTruth(
            scenario_type=ScenarioType.BENIGN, is_ransomware=False)
        env10._verdict = "ignore"
        env10._steps = 1
        env10._cumulative_cost = 0.0

        env5 = DetectionEnv(k_max=5)
        env5._ground_truth = GroundTruth(
            scenario_type=ScenarioType.BENIGN, is_ransomware=False)
        env5._verdict = "ignore"
        env5._steps = 1
        env5._cumulative_cost = 0.0

        assert _compute_env_reward(env10) == _compute_env_reward(env5)


class TestDetectionReward:
    def test_batch_reward(self):
        envs = []
        for i in range(3):
            env = DetectionEnv()
            env._ground_truth = GroundTruth(
                scenario_type=ScenarioType.BENIGN,
                is_ransomware=False,
            )
            env._verdict = "ignore"
            env._steps = 1
            env._cumulative_cost = 0.0
            envs.append(env)

        rewards = detection_reward(envs)
        assert len(rewards) == 3
        assert all(r > 0 for r in rewards)  # all correct


class TestFormatReward:
    def test_both_thinking_and_tool_call(self):
        completions = ["<think>Analysis</think> <tool_call>DECIDE(\"ignore\")</tool_call>"]
        rewards = format_reward(completions)
        assert rewards[0] == 0.1  # both thinking + tool_call

    def test_thinking_only(self):
        completions = ["<think>Analysis</think> no tools used"]
        rewards = format_reward(completions)
        assert rewards[0] == 0.05

    def test_decide_in_prose_not_rewarded(self):
        """Natural language 'decide' shouldn't count as tool call."""
        completions = ["I need to decide what to do here."]
        rewards = format_reward(completions)
        assert rewards[0] == 0.0  # no format reward

    def test_no_format(self):
        completions = ["Random gibberish output"]
        rewards = format_reward(completions)
        assert rewards[0] == 0.0


# ── Dataset preparation tests ────────────────────────────────────────


class TestPrepareDataset:
    def test_correct_length(self):
        config = TrainingConfig(n_episodes=10, seed=42)
        dataset = prepare_dataset(config)
        assert len(dataset) == 10

    def test_has_required_fields(self):
        config = TrainingConfig(n_episodes=5, seed=42)
        dataset = prepare_dataset(config)
        for item in dataset:
            assert "prompt" in item
            assert "scenario_data" in item

    def test_prompt_is_conversational(self):
        config = TrainingConfig(n_episodes=3, seed=42)
        dataset = prepare_dataset(config)
        for item in dataset:
            prompt = item["prompt"]
            assert isinstance(prompt, list)
            assert prompt[0]["role"] == "system"
            assert prompt[-1]["role"] == "user"

    def test_scenario_data_is_valid_json(self):
        config = TrainingConfig(n_episodes=5, seed=42)
        dataset = prepare_dataset(config)
        for item in dataset:
            data = json.loads(item["scenario_data"])
            assert "scenario_type" in data
            assert "observability" in data
            assert "seed" in data

    def test_reproducible(self):
        config = TrainingConfig(n_episodes=5, seed=42)
        d1 = prepare_dataset(config)
        d2 = prepare_dataset(config)
        for a, b in zip(d1, d2):
            assert a["scenario_data"] == b["scenario_data"]

    def test_scenario_data_includes_n_history(self):
        config = TrainingConfig(n_episodes=3, seed=42)
        dataset = prepare_dataset(config)
        for item in dataset:
            data = json.loads(item["scenario_data"])
            assert "n_history" in data
            assert data["n_history"] == 2  # default

    def test_tool_ablation_in_prompt(self):
        config = TrainingConfig(
            n_episodes=3, seed=42,
            available_tools=["inspect_file", "check_process"],
        )
        dataset = prepare_dataset(config)
        system_msg = dataset[0]["prompt"][0]["content"]
        assert "inspect_file" in system_msg
        assert "recall_memory" not in system_msg


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.model_name == "Qwen/Qwen3.5-9B"
        assert config.k_max == 5
        assert config.group_size == 4
        assert config.lora_r == 16
        assert config.lora_alpha == 32

    def test_observability_levels_default(self):
        config = TrainingConfig()
        assert 0.3 in config.observability_levels
        assert 0.9 in config.observability_levels


# ── Phase 1 tests: budget enforcement + verdict validation ───────────


class TestBudgetEnforcement:
    """k_max budget must be enforced by DetectionEnv tool methods.

    Before Phase 1, DetectionEnv had no budget cap. An agent could call
    50+ tools, unbounded negative cost, destroying the active-investigation
    premise. These tests pin the new behavior.
    """

    def _scenario_data(self, **kwargs):
        data = {
            "scenario_type": "benign",
            "observability": 0.9,
            "attack_progress": 0.0,
            "seed": 42,
            "n_history": 0,
        }
        data.update(kwargs)
        return json.dumps(data)

    def test_budget_exhaustion_returns_error(self):
        """After k_max non-decide calls, further tool calls return error."""
        env = DetectionEnv(k_max=2)
        env.reset(scenario_data=self._scenario_data())

        env.check_process(4)  # step 1
        env.check_process(4)  # step 2 (at budget)
        result = json.loads(env.check_process(4))  # over budget
        assert "error" in result
        assert "budget" in result["error"].lower()

    def test_budget_exhaustion_does_not_add_cost(self):
        """Over-budget tool calls must not accumulate more cost."""
        env = DetectionEnv(k_max=2)
        env.reset(scenario_data=self._scenario_data())

        env.check_process(4)
        env.check_process(4)
        cost_at_budget = env._cumulative_cost

        env.check_process(4)  # over budget
        env.inspect_file("C:/x.txt")  # over budget

        assert env._cumulative_cost == cost_at_budget

    def test_decide_allowed_at_budget(self):
        """decide() should work even when budget is exhausted."""
        env = DetectionEnv(k_max=2)
        env.reset(scenario_data=self._scenario_data())

        env.check_process(4)
        env.check_process(4)
        result = json.loads(env.decide("ignore", "safe"))
        assert "verdict" in result
        assert env._verdict == "ignore"


class TestDecideValidationOrder:
    """decide() must validate before setting _verdict state.

    Before Phase 1, an invalid verdict like 'invalid_verdict' would be
    assigned to env._verdict anyway, then an error was returned. This
    corrupted the reward computation which trusted _verdict.
    """

    def test_invalid_verdict_does_not_set_state(self):
        env = DetectionEnv()
        env.reset(scenario_data=json.dumps({
            "scenario_type": "benign",
            "observability": 0.9,
            "attack_progress": 0.0,
            "seed": 42,
            "n_history": 0,
        }))
        env.decide("nonsense_verdict", "whatever")
        assert env._verdict is None

    def test_invalid_verdict_does_not_overwrite_prior_valid(self):
        """An invalid decide after a valid one should not corrupt state."""
        env = DetectionEnv()
        env.reset(scenario_data=json.dumps({
            "scenario_type": "benign",
            "observability": 0.9,
            "attack_progress": 0.0,
            "seed": 42,
            "n_history": 0,
        }))
        env.decide("ignore", "first")
        env.decide("bogus", "second")
        assert env._verdict == "ignore"


class TestToolNameEnum:
    """A ToolName enum should be the single source of truth for tool names."""

    def test_enum_matches_tool_costs(self):
        from tools.inspection import TOOL_COSTS, ToolName
        assert set(tn.value for tn in ToolName) == set(TOOL_COSTS.keys())

    def test_valid_verdicts_derived_from_verdict_enum(self):
        from tools.inspection import VALID_VERDICTS
        assert VALID_VERDICTS == {v.value for v in Verdict}


# ── Phase 4 tests: env isolation ─────────────────────────────────────


class TestEnvIsolation:
    """Each reset() produces a fresh host so GRPO siblings don't share state."""

    def _scenario_data(self, seed=42):
        return json.dumps({
            "scenario_type": "blitz",
            "observability": 0.9,
            "attack_progress": 0.7,
            "seed": seed,
            "n_history": 0,
        })

    def test_reset_produces_fresh_host_each_call(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._scenario_data(seed=1))
        host_1 = env._host
        env.reset(scenario_data=self._scenario_data(seed=2))
        host_2 = env._host
        assert host_1 is not host_2

    def test_reset_clears_prior_mutations(self):
        env = DetectionEnv()
        env.reset(scenario_data=self._scenario_data(seed=1))
        env.check_process(4)
        env.decide("alert", "suspect")
        assert env._steps > 0
        assert env._verdict == "alert"

        env.reset(scenario_data=self._scenario_data(seed=1))
        assert env._steps == 0
        assert env._verdict is None
        assert env._cumulative_cost == 0.0

    def test_two_envs_do_not_share_host(self):
        env_a = DetectionEnv()
        env_b = DetectionEnv()
        env_a.reset(scenario_data=self._scenario_data(seed=1))
        env_b.reset(scenario_data=self._scenario_data(seed=1))
        assert env_a._host is not env_b._host


# ── Phase 7 tests: harden format_reward against hacking ──────────────


class TestFormatRewardHardened:
    """format_reward requires well-formed <think> + parseable <tool_call>.
    Before hardening, emitting empty or malformed tags earned full reward.
    """

    def test_unclosed_think_not_rewarded(self):
        rewards = format_reward(["<think>no closing tag here..."])
        assert rewards[0] == 0.0

    def test_empty_thinking_not_rewarded(self):
        rewards = format_reward(["<think></think> <tool_call>x</tool_call>"])
        assert rewards[0] < 0.1

    def test_malformed_tool_call_not_rewarded(self):
        rewards = format_reward([
            "<think>analysis done</think> <tool_call>gibberish without parens</tool_call>"
        ])
        assert rewards[0] == 0.05

    def test_well_formed_tool_call_rewarded(self):
        rewards = format_reward([
            '<think>quick analysis</think> <tool_call>{"name": "inspect_file", "arguments": {"path": "C:/x"}}</tool_call>'
        ])
        assert rewards[0] == 0.1

    def test_function_call_format_rewarded(self):
        rewards = format_reward([
            '<think>check process</think> <tool_call>check_process(1234)</tool_call>'
        ])
        assert rewards[0] == 0.1
