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

    def test_efficiency_bonus(self):
        env_fast = self._make_env("benign", False, "ignore", steps=1)
        env_slow = self._make_env("benign", False, "ignore", steps=5)
        assert _compute_env_reward(env_fast) > _compute_env_reward(env_slow)

    def test_action_cost_reduces_reward(self):
        env_cheap = self._make_env("benign", False, "ignore", steps=1, cost=0.0)
        env_expensive = self._make_env("benign", False, "ignore", steps=1, cost=-0.1)
        assert _compute_env_reward(env_cheap) > _compute_env_reward(env_expensive)


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
        completions = ["<think>Analysis</think> decision made tool_call"]
        rewards = format_reward(completions)
        assert rewards[0] == 0.1  # both thinking + tool_call

    def test_thinking_only(self):
        completions = ["<think>Analysis</think> no tools used"]
        rewards = format_reward(completions)
        assert rewards[0] == 0.05

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
        assert config.model_name == "Qwen/Qwen3-8B"
        assert config.k_max == 5
        assert config.group_size == 4
        assert config.lora_r == 16
        assert config.lora_alpha == 32

    def test_observability_levels_default(self):
        config = TrainingConfig()
        assert 0.3 in config.observability_levels
        assert 0.9 in config.observability_levels
