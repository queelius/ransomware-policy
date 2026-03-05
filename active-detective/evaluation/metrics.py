"""Detection accuracy, efficiency, and Pareto frontier metrics.

Computes precision, recall, F1 (binary: ransomware vs benign),
attack phase accuracy, efficiency metrics, and Pareto frontiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simulator.models import Verdict


@dataclass
class DetectionMetrics:
    """Binary detection metrics (ransomware vs benign)."""

    tp: int = 0  # true positive: ransomware correctly detected
    fp: int = 0  # false positive: benign flagged as ransomware
    tn: int = 0  # true negative: benign correctly ignored
    fn: int = 0  # false negative: ransomware missed

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class EfficiencyMetrics:
    """Tool usage and decision efficiency metrics."""

    total_episodes: int = 0
    total_steps: int = 0
    total_cost: float = 0.0
    early_stops: int = 0  # episodes where agent used < k_max steps

    @property
    def mean_steps(self) -> float:
        return self.total_steps / self.total_episodes if self.total_episodes > 0 else 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.total_episodes if self.total_episodes > 0 else 0.0

    @property
    def early_stop_rate(self) -> float:
        return self.early_stops / self.total_episodes if self.total_episodes > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total_episodes": self.total_episodes,
            "mean_steps": round(self.mean_steps, 3),
            "mean_cost": round(self.mean_cost, 4),
            "early_stop_rate": round(self.early_stop_rate, 3),
        }


@dataclass
class EvalResult:
    """Result of evaluating a single episode."""

    verdict: str
    is_ransomware_gt: bool
    scenario_type: str
    attack_phase: str | None
    steps_taken: int
    cumulative_cost: float
    k_max: int
    tools_used: list[str] = field(default_factory=list)


def compute_detection_metrics(results: list[EvalResult]) -> DetectionMetrics:
    """Compute binary detection metrics from evaluation results.

    Maps verdicts to binary prediction:
    - alert/quarantine/block → predicted ransomware
    - ignore/monitor → predicted benign
    """
    metrics = DetectionMetrics()

    for r in results:
        try:
            v = Verdict(r.verdict)
            predicted_ransomware = v.is_ransomware_prediction
        except ValueError:
            # Invalid verdict — treat as benign prediction
            predicted_ransomware = False

        if predicted_ransomware and r.is_ransomware_gt:
            metrics.tp += 1
        elif predicted_ransomware and not r.is_ransomware_gt:
            metrics.fp += 1
        elif not predicted_ransomware and not r.is_ransomware_gt:
            metrics.tn += 1
        else:
            metrics.fn += 1

    return metrics


def compute_efficiency_metrics(
    results: list[EvalResult],
) -> EfficiencyMetrics:
    """Compute tool usage and decision efficiency metrics."""
    eff = EfficiencyMetrics(total_episodes=len(results))

    for r in results:
        eff.total_steps += r.steps_taken
        eff.total_cost += r.cumulative_cost
        if r.steps_taken < r.k_max:
            eff.early_stops += 1

    return eff


def compute_phase_accuracy(results: list[EvalResult]) -> dict[str, float]:
    """Compute per-phase detection accuracy.

    Groups results by attack_phase and computes accuracy within each group.
    """
    phase_groups: dict[str, list[bool]] = {}

    for r in results:
        phase = r.attack_phase or "benign"
        if phase not in phase_groups:
            phase_groups[phase] = []

        try:
            v = Verdict(r.verdict)
            predicted = v.is_ransomware_prediction
        except ValueError:
            predicted = False

        correct = predicted == r.is_ransomware_gt
        phase_groups[phase].append(correct)

    return {
        phase: round(sum(correct_list) / len(correct_list), 4)
        for phase, correct_list in phase_groups.items()
        if correct_list
    }


def compute_tool_usage(results: list[EvalResult]) -> dict[str, int]:
    """Count how many times each tool was used across all episodes."""
    usage: dict[str, int] = {}
    for r in results:
        for tool in r.tools_used:
            usage[tool] = usage.get(tool, 0) + 1
    return usage


@dataclass
class ParetoPoint:
    """A point on the accuracy-cost Pareto frontier."""

    config_name: str
    accuracy: float
    mean_cost: float
    f1: float


def compute_pareto_frontier(
    named_results: dict[str, list[EvalResult]],
) -> list[ParetoPoint]:
    """Compute accuracy vs cost Pareto frontier across configurations.

    Each entry in named_results maps a config name (e.g., "k=5, obs=0.7")
    to a list of EvalResults for that configuration.

    Returns Pareto-optimal points sorted by cost (ascending).
    """
    points = []
    for name, results in named_results.items():
        dm = compute_detection_metrics(results)
        em = compute_efficiency_metrics(results)
        points.append(ParetoPoint(
            config_name=name,
            accuracy=dm.accuracy,
            mean_cost=em.mean_cost,
            f1=dm.f1,
        ))

    # Sort by cost ascending (less negative = cheaper)
    points.sort(key=lambda p: p.mean_cost, reverse=True)

    # Filter to Pareto-optimal: keep points where no other point
    # has both higher accuracy AND lower cost
    pareto = []
    best_accuracy = -1.0
    for p in points:
        if p.accuracy > best_accuracy:
            pareto.append(p)
            best_accuracy = p.accuracy

    return pareto
