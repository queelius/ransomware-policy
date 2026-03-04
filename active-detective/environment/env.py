"""RansomwareDetectionEnv: RLVR environment with tool-execution rollouts.

Not a Gymnasium env — operates on model text output.
Parses tool calls, executes against simulator state, injects results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from environment.reward import RewardBreakdown, compute_reward
from simulator.models import GroundTruth, ScenarioType
from simulator.telemetry import Episode, generate_episode
from tools.inspection import TOOL_COSTS, VALID_VERDICTS, execute_tool
from tools.memory import MemoryStore
from tools.parser import (
    ParsedToolCall,
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
    well_formatted: bool
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
        memory_top_k: int = 3,
    ) -> None:
        self.max_steps = max_steps
        self.memory_top_k = memory_top_k

        # State set by reset()
        self._episode: Episode | None = None
        self._registry = None
        self._ptable = None
        self._memory: MemoryStore | None = None
        self._steps: list[StepResult] = []
        self._cumulative_cost: float = 0.0
        self._well_formatted: bool = True
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

    def reset(
        self,
        scenario_type: ScenarioType,
        observability: float,
        rng: np.random.RandomState,
        attack_progress: float = 0.5,
        history_windows: list[str] | None = None,
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
        history_windows:
            Optional list of past telemetry window texts to seed memory.

        Returns
        -------
        The initial telemetry text (to be embedded in the system prompt).
        """
        self._episode = generate_episode(
            scenario_type, observability, rng,
            attack_progress=attack_progress,
        )

        # Re-create host state for tool execution.
        # We re-seed with the same rng state approach but use fresh state
        # so the tools see the same filesystem the episode was generated from.
        from simulator.registry import FileRegistry, ProcessTable
        from datetime import datetime

        now = datetime(2025, 6, 15, 10, 0, 0)
        # Use a separate RNG so we don't disturb the caller's sequence
        host_rng = np.random.RandomState(rng.randint(0, 2**31))
        self._registry = FileRegistry()
        self._registry.seed_filesystem(host_rng, now)
        self._ptable = ProcessTable()
        self._ptable.seed_processes(now)

        # Initialize memory store with optional history
        self._memory = MemoryStore(top_k=self.memory_top_k)
        if history_windows:
            for i, window_text in enumerate(history_windows):
                self._memory.add_window(window_text, {"window_id": f"hist-{i}"})

        # Reset rollout state
        self._steps = []
        self._cumulative_cost = 0.0
        self._well_formatted = True
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

        # Execute the tool
        result, cost = execute_tool(
            tool_name, args,
            self._registry, self._ptable, self._memory,
        )
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

        Returns None if no valid tool call was found (marks as poorly formatted).
        """
        parsed = parse_tool_call(model_output)
        if parsed is None:
            self._well_formatted = False
            return None
        return self.step(parsed)

    def mark_format_error(self) -> None:
        """Flag that the model produced poorly formatted output."""
        self._well_formatted = False

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
            well_formatted=self._well_formatted,
        )

        return RolloutResult(
            verdict=self._verdict,
            explanation=self._explanation,
            ground_truth=self._episode.ground_truth,
            steps=list(self._steps),
            cumulative_cost=self._cumulative_cost,
            well_formatted=self._well_formatted,
            reward=reward,
            exceeded_budget=exceeded,
        )
