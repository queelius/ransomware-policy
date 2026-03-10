"""Reward function for RLVR: verdict checking, action costs, format reward.

Deterministic, rule-based reward — no reward model needed.
The simulator provides ground truth, making rewards verifiable.
"""

from __future__ import annotations

from dataclasses import dataclass

from simulator.models import GroundTruth, Verdict


@dataclass
class RewardBreakdown:
    """Itemized reward components for logging and analysis."""

    verdict_reward: float
    action_cost: float
    efficiency_bonus: float
    format_reward: float

    @property
    def total(self) -> float:
        return (
            self.verdict_reward
            + self.action_cost
            + self.efficiency_bonus
            + self.format_reward
        )


# ── Constants ────────────────────────────────────────────────────────

CORRECT_REWARD = 1.0
WRONG_REWARD = -1.0
FALSE_NEGATIVE_REWARD = -2.0  # missed ransomware is worse than false alarm
EFFICIENCY_BONUS_PER_STEP = 0.05
FORMAT_THINKING_REWARD = 0.05
FORMAT_TOOL_CALL_REWARD = 0.05


def compute_reward(
    verdict: str,
    ground_truth: GroundTruth,
    cumulative_cost: float,
    steps_taken: int,
    max_steps: int,
    has_thinking: bool,
    has_tool_call: bool,
) -> RewardBreakdown:
    """Compute the RLVR reward for a completed episode.

    Parameters
    ----------
    verdict:
        The agent's final verdict string (should be a valid Verdict value).
    ground_truth:
        Ground truth from the simulator.
    cumulative_cost:
        Sum of per-tool costs accumulated during the rollout (negative).
    steps_taken:
        Number of tool calls the agent made (including DECIDE).
    max_steps:
        Budget cap (k_max). Used to compute efficiency bonus.
    has_thinking:
        Whether the model used <think>...</think> tags.
    has_tool_call:
        Whether the model used proper <tool_call>...</tool_call> tags.

    Returns
    -------
    RewardBreakdown with itemized components and total.
    """
    # ── Verdict reward ───────────────────────────────────────────────
    try:
        v = Verdict(verdict)
    except ValueError:
        # Invalid verdict string — treat as wrong
        verdict_reward = WRONG_REWARD
    else:
        predicted_ransomware = v.is_ransomware_prediction
        actual_ransomware = ground_truth.is_ransomware

        if predicted_ransomware == actual_ransomware:
            verdict_reward = CORRECT_REWARD
        elif actual_ransomware and not predicted_ransomware:
            # False negative: missed ransomware
            verdict_reward = FALSE_NEGATIVE_REWARD
        else:
            # False positive: false alarm
            verdict_reward = WRONG_REWARD

    # ── Efficiency bonus ─────────────────────────────────────────────
    # Reward for stopping early (unused budget steps)
    unused_steps = max(0, max_steps - steps_taken)
    efficiency_bonus = unused_steps * EFFICIENCY_BONUS_PER_STEP

    # ── Format reward (split: thinking + tool_call) ──────────────────
    format_reward = 0.0
    if has_thinking:
        format_reward += FORMAT_THINKING_REWARD
    if has_tool_call:
        format_reward += FORMAT_TOOL_CALL_REWARD

    return RewardBreakdown(
        verdict_reward=verdict_reward,
        action_cost=cumulative_cost,
        efficiency_bonus=efficiency_bonus,
        format_reward=format_reward,
    )
