"""Tests for the RLVR reward function."""

import pytest

from environment.reward import (
    CORRECT_REWARD,
    EFFICIENCY_BONUS_PER_STEP,
    FALSE_NEGATIVE_REWARD,
    FORMAT_REWARD,
    WRONG_REWARD,
    RewardBreakdown,
    compute_reward,
)
from simulator.models import GroundTruth, ScenarioType


@pytest.fixture
def ransomware_gt():
    return GroundTruth(
        scenario_type=ScenarioType.BLITZ,
        is_ransomware=True,
        attack_phase="encryption",
    )


@pytest.fixture
def benign_gt():
    return GroundTruth(
        scenario_type=ScenarioType.BENIGN,
        is_ransomware=False,
    )


class TestVerdictReward:
    def test_correct_alert_on_ransomware(self, ransomware_gt):
        r = compute_reward("alert", ransomware_gt, 0.0, 1, 5, True)
        assert r.verdict_reward == CORRECT_REWARD

    def test_correct_quarantine_on_ransomware(self, ransomware_gt):
        r = compute_reward("quarantine", ransomware_gt, 0.0, 1, 5, True)
        assert r.verdict_reward == CORRECT_REWARD

    def test_correct_block_on_ransomware(self, ransomware_gt):
        r = compute_reward("block", ransomware_gt, 0.0, 1, 5, True)
        assert r.verdict_reward == CORRECT_REWARD

    def test_correct_ignore_on_benign(self, benign_gt):
        r = compute_reward("ignore", benign_gt, 0.0, 1, 5, True)
        assert r.verdict_reward == CORRECT_REWARD

    def test_correct_monitor_on_benign(self, benign_gt):
        r = compute_reward("monitor", benign_gt, 0.0, 1, 5, True)
        assert r.verdict_reward == CORRECT_REWARD

    def test_false_negative_penalty(self, ransomware_gt):
        """Ignoring ransomware should incur the harshest penalty."""
        r = compute_reward("ignore", ransomware_gt, 0.0, 1, 5, True)
        assert r.verdict_reward == FALSE_NEGATIVE_REWARD

    def test_false_negative_monitor(self, ransomware_gt):
        """Monitoring (but not acting on) ransomware is also a false negative."""
        r = compute_reward("monitor", ransomware_gt, 0.0, 1, 5, True)
        assert r.verdict_reward == FALSE_NEGATIVE_REWARD

    def test_false_positive_penalty(self, benign_gt):
        """Alerting on benign activity is a false positive."""
        r = compute_reward("alert", benign_gt, 0.0, 1, 5, True)
        assert r.verdict_reward == WRONG_REWARD

    def test_false_negative_worse_than_false_positive(self):
        assert FALSE_NEGATIVE_REWARD < WRONG_REWARD

    def test_invalid_verdict(self, benign_gt):
        r = compute_reward("invalid_verdict", benign_gt, 0.0, 1, 5, True)
        assert r.verdict_reward == WRONG_REWARD


class TestActionCost:
    def test_cost_passed_through(self, benign_gt):
        r = compute_reward("ignore", benign_gt, -0.06, 3, 5, True)
        assert r.action_cost == -0.06

    def test_zero_cost(self, benign_gt):
        r = compute_reward("ignore", benign_gt, 0.0, 1, 5, True)
        assert r.action_cost == 0.0


class TestEfficiencyBonus:
    def test_full_budget_unused(self, benign_gt):
        r = compute_reward("ignore", benign_gt, 0.0, 1, 5, True)
        # 5 - 1 = 4 unused steps
        assert r.efficiency_bonus == 4 * EFFICIENCY_BONUS_PER_STEP

    def test_all_budget_used(self, benign_gt):
        r = compute_reward("ignore", benign_gt, 0.0, 5, 5, True)
        assert r.efficiency_bonus == 0.0

    def test_partial_budget(self, benign_gt):
        r = compute_reward("ignore", benign_gt, 0.0, 3, 5, True)
        assert r.efficiency_bonus == 2 * EFFICIENCY_BONUS_PER_STEP


class TestFormatReward:
    def test_well_formatted(self, benign_gt):
        r = compute_reward("ignore", benign_gt, 0.0, 1, 5, True)
        assert r.format_reward == FORMAT_REWARD

    def test_poorly_formatted(self, benign_gt):
        r = compute_reward("ignore", benign_gt, 0.0, 1, 5, False)
        assert r.format_reward == 0.0


class TestRewardBreakdown:
    def test_total_computation(self):
        rb = RewardBreakdown(
            verdict_reward=1.0,
            action_cost=-0.04,
            efficiency_bonus=0.03,
            format_reward=0.1,
        )
        expected = 1.0 + (-0.04) + 0.03 + 0.1
        assert abs(rb.total - expected) < 1e-9

    def test_typical_correct_scenario(self, ransomware_gt):
        """A correct detection with 2 tool calls should net positive."""
        r = compute_reward("quarantine", ransomware_gt, -0.04, 2, 5, True)
        assert r.total > 0

    def test_typical_false_negative(self, ransomware_gt):
        """Missing ransomware should always be negative."""
        r = compute_reward("ignore", ransomware_gt, 0.0, 1, 5, True)
        assert r.total < 0
