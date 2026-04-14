"""RansomwareDetectionEnv: RLVR environment with tool-execution rollouts.

Not a Gymnasium env — operates on model text output.
Parses tool calls, executes against simulator state, injects results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from environment.reward import RewardBreakdown, compute_reward
from simulator.host import HostState
from simulator.models import GroundTruth, ScenarioType
from simulator.telemetry import Episode, generate_episode
from tools.inspection import TOOL_COSTS, VALID_VERDICTS, execute_tool
from tools.parser import (
    ParsedToolCall,
    extract_thinking,
    format_tool_result,
    is_decide_call,
    parse_tool_call,
)


@dataclass
class StepResult:
    """Result of a single tool execution step."""

    tool_name: str
    args: dict
    result: dict
    cost: float
    is_terminal: bool
    result_text: str  # formatted <tool_result>...</tool_result>


@dataclass
class RolloutResult:
    """Complete rollout result for reward computation and logging."""

    verdict: str | None
    explanation: str
    ground_truth: GroundTruth
    steps: list[StepResult]
    cumulative_cost: float
    has_thinking: bool
    has_tool_call: bool
    reward: RewardBreakdown | None = None
    exceeded_budget: bool = False


class RansomwareDetectionEnv:
    """Tool-execution environment for ransomware detection rollouts.

    Usage::

        env = RansomwareDetectionEnv(max_steps=5)
        prompt = env.reset(scenario_type, observability, rng)

        # In the GRPO loop, the model generates text from `prompt`.
        # For each tool call in the output:
        step = env.step(parsed_tool_call)
        # Inject step.result_text back into the model's context.

        # After DECIDE or budget exhaustion:
        rollout = env.finish()
    """

    def __init__(
        self,
        max_steps: int = 5,
    ) -> None:
        self.max_steps = max_steps

        # State set by reset()
        self._episode: Episode | None = None
        self._host: HostState | None = None
        self._steps: list[StepResult] = []
        self._cumulative_cost: float = 0.0
        self._has_thinking: bool = False
        self._has_tool_call: bool = False
        self._verdict: str | None = None
        self._explanation: str = ""
        self._done: bool = False

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def steps_taken(self) -> int:
        return len(self._steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_steps - self.steps_taken)

    @property
    def host(self) -> HostState | None:
        """The HostState bound to this episode (read-only reference).

        Exposed so evaluation scripts can inspect simulator state
        without reaching into private attributes.
        """
        return self._host

    def reset(
        self,
        scenario_type: ScenarioType,
        observability: float,
        rng: np.random.RandomState,
        attack_progress: float = 0.5,
        n_history: int = 2,
    ) -> str:
        """Reset environment for a new episode.

        Parameters
        ----------
        scenario_type:
            Which scenario to generate.
        observability:
            Fraction of events visible (0.0 to 1.0).
        rng:
            Random state for reproducibility.
        attack_progress:
            How far along the attack is (0.0 to 1.0).
        n_history:
            Number of prior telemetry windows to generate for temporal
            context.  Stored on ``self._episode.history_windows``.

        Returns
        -------
        The initial telemetry text (to be embedded in the system prompt).
        """
        self._episode = generate_episode(
            scenario_type, observability, rng,
            attack_progress=attack_progress,
            n_history=n_history,
        )

        # Use the episode's host for tool execution — ensures tools
        # query the same state that generated the telemetry.
        self._host = self._episode.host

        # Reset rollout state
        self._steps = []
        self._cumulative_cost = 0.0
        self._has_thinking = False
        self._has_tool_call = False
        self._verdict = None
        self._explanation = ""
        self._done = False

        return self._episode.input_text

    def step(self, tool_call: ParsedToolCall) -> StepResult:
        """Execute a single tool call and return the result.

        Raises RuntimeError if the episode is already done.
        """
        if self._done:
            raise RuntimeError("Episode is done — call reset() first")
        if self._episode is None:
            raise RuntimeError("No episode loaded — call reset() first")

        tool_name = tool_call.tool_name
        args = tool_call.args

        # Track format compliance
        self._has_tool_call = True

        # Execute the tool
        result, cost = execute_tool(tool_name, args, self._host)
        self._cumulative_cost += cost

        # Check for terminal DECIDE action
        is_terminal = is_decide_call(tool_name)
        if is_terminal:
            self._verdict = args.get("verdict", "")
            self._explanation = args.get("explanation", "")
            self._done = True

        result_text = format_tool_result(result)

        step_result = StepResult(
            tool_name=tool_name,
            args=args,
            result=result,
            cost=cost,
            is_terminal=is_terminal,
            result_text=result_text,
        )
        self._steps.append(step_result)

        # Check budget
        if not self._done and self.steps_taken >= self.max_steps:
            self._done = True

        return step_result

    def step_from_text(self, model_output: str) -> StepResult | None:
        """Parse a tool call from model output and execute it.

        Returns None if no valid tool call was found.
        Also tracks whether thinking tags are present.
        """
        # Track thinking
        if extract_thinking(model_output) is not None:
            self._has_thinking = True

        parsed = parse_tool_call(model_output)
        if parsed is None:
            return None
        return self.step(parsed)

    def mark_thinking(self) -> None:
        """Flag that the model produced thinking output."""
        self._has_thinking = True

    def finish(self) -> RolloutResult:
        """Finalize the episode and compute reward.

        Can be called after DECIDE or after budget exhaustion.
        If the agent ran out of budget without calling DECIDE,
        the verdict defaults to "ignore" (no action taken).
        """
        if self._episode is None:
            raise RuntimeError("No episode loaded — call reset() first")

        # Default verdict if budget exhausted without DECIDE
        exceeded = False
        if self._verdict is None:
            self._verdict = "ignore"
            self._explanation = "Budget exhausted without decision"
            exceeded = True

        reward = compute_reward(
            verdict=self._verdict,
            ground_truth=self._episode.ground_truth,
            cumulative_cost=self._cumulative_cost,
            steps_taken=self.steps_taken,
            max_steps=self.max_steps,
            has_thinking=self._has_thinking,
            has_tool_call=self._has_tool_call,
        )

        return RolloutResult(
            verdict=self._verdict,
            explanation=self._explanation,
            ground_truth=self._episode.ground_truth,
            steps=list(self._steps),
            cumulative_cost=self._cumulative_cost,
            has_thinking=self._has_thinking,
            has_tool_call=self._has_tool_call,
            reward=reward,
            exceeded_budget=exceeded,
        )
