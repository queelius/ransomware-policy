"""Telemetry window builder with observability control.

Collects raw events from generators, applies stochastic dropping
to simulate partial observability, and formats into text for the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from simulator.host import HostState
from simulator.models import (
    EventLogEvent,
    FileEvent,
    GroundTruth,
    NetEvent,
    ProcessEvent,
    RegistryEvent,
    ScenarioType,
    TelemetryEvent,
)
from simulator import generators as gen


@dataclass
class Episode:
    """A single training/evaluation episode."""

    input_text: str
    ground_truth: GroundTruth
    scenario_type: ScenarioType
    observability: float
    raw_event_count: int
    visible_event_count: int
    history_windows: list[str] = field(default_factory=list)
    host: HostState | None = None


def apply_observability_filter(
    events: list[TelemetryEvent],
    observability: float,
    rng: np.random.RandomState,
) -> list[TelemetryEvent]:
    """Drop events stochastically to simulate partial observability.

    Each event is independently retained with probability ``observability``.
    Events with a custom ``drop_probability`` are dropped with that
    probability instead (higher drop_probability = harder to observe).
    """
    surviving: list[TelemetryEvent] = []
    for event in events:
        # Per-event drop probability overrides global if set
        if event.drop_probability > 0:
            keep_prob = 1.0 - event.drop_probability
        else:
            keep_prob = observability

        if rng.random() < keep_prob:
            surviving.append(event)

    return surviving


def format_telemetry_window(events: list[TelemetryEvent], window_start: datetime) -> str:
    """Format surviving events into structured text for the agent.

    Events are sorted by timestamp and formatted as one line per event.
    """
    if not events:
        return "[no events observed in this window]"

    # Sort by timestamp
    sorted_events = sorted(events, key=lambda e: e.ts)
    lines: list[str] = []

    for event in sorted_events:
        offset = (event.ts - window_start).total_seconds()
        offset_str = f"t+{int(offset)}s"

        if isinstance(event, ProcessEvent):
            lines.append(
                f"[{offset_str}] PROC pid={event.pid} name={event.name} "
                f"parent={event.parent_pid} cmd=\"{event.command_line}\""
            )
        elif isinstance(event, FileEvent):
            parts = [
                f"[{offset_str}] FILE path={event.path}",
                f"size_delta={event.size_delta:+d}",
            ]
            if event.extension_change:
                parts.append(f"ext_change={event.extension_change}")
            lines.append(" ".join(parts))
        elif isinstance(event, NetEvent):
            lines.append(
                f"[{offset_str}] NET dest={event.destination} "
                f"bytes={event.bytes_transferred} proto={event.protocol} "
                f"dir={event.direction}"
            )
        elif isinstance(event, RegistryEvent):
            lines.append(
                f"[{offset_str}] REG action={event.action} "
                f"key={event.key_path} value={event.value_name}"
            )
        elif isinstance(event, EventLogEvent):
            lines.append(
                f"[{offset_str}] EVTLOG source={event.source} "
                f"id={event.event_id} msg=\"{event.message}\""
            )

    return "\n".join(lines)


_ATTACK_GENERATORS = {
    ScenarioType.BLITZ: gen.blitz_encryptor,
    ScenarioType.SLEEPER: gen.slow_sleeper,
    ScenarioType.EXFIL_FIRST: gen.exfil_first,
    ScenarioType.SEMANTIC_SHUFFLE: gen.semantic_shuffle,
}

_BENIGN_GENERATORS = [gen.office_edits, gen.browser_downloads,
                      gen.system_maintenance]

_EXTRA_BENIGN_GENERATORS = [gen.backup_operations, gen.av_scan]


def _run_generators_on_host(
    host: HostState,
    scenario_type: ScenarioType,
    rng: np.random.RandomState,
    attack_progress: float,
) -> tuple[list[TelemetryEvent], str | None]:
    """Run benign + scenario generators against a shared host.

    Returns (all_events, attack_phase). Mutates `host` in place.
    """
    all_events: list[TelemetryEvent] = []
    attack_phase: str | None = None

    # Always generate some benign background activity
    n_benign = rng.randint(1, 3)
    chosen_benign = rng.choice(len(_BENIGN_GENERATORS), size=n_benign, replace=False)
    for idx in chosen_benign:
        all_events.extend(_BENIGN_GENERATORS[idx](host, rng))

    if scenario_type == ScenarioType.BENIGN:
        if rng.random() < 0.4:
            gen_fn = _EXTRA_BENIGN_GENERATORS[rng.randint(0, len(_EXTRA_BENIGN_GENERATORS))]
            all_events.extend(gen_fn(host, rng))
    elif scenario_type in _ATTACK_GENERATORS:
        events, attack_phase = _ATTACK_GENERATORS[scenario_type](
            host, rng, progress=attack_progress)
        all_events.extend(events)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return all_events, attack_phase


def generate_episode(
    scenario_type: ScenarioType,
    observability: float,
    rng: np.random.RandomState,
    now: datetime | None = None,
    attack_progress: float = 0.5,
    n_history: int = 2,
) -> Episode:
    """Generate a single episode with telemetry window and history.

    Uses one shared HostState across all windows so history reflects
    real state evolution: files encrypted in t-2 remain encrypted in
    the current window, process lineage is preserved, etc.

    Each window uses an independent derived RNG so changing n_history
    at a fixed parent seed does not perturb the current-window draws.
    """
    now = now or datetime(2025, 6, 15, 10, 0, 0)

    # Derive independent per-window RNG streams from the parent rng.
    # This isolates each window's draws from the others, so n_history
    # variations don't shift the current window's content.
    #
    # Draw the host+current seeds FIRST so they are invariant to
    # n_history. History seeds come after.
    host_seed = int(rng.randint(0, 2**31))
    current_seed = int(rng.randint(0, 2**31))
    history_seeds = [int(rng.randint(0, 2**31)) for _ in range(n_history)]

    # ── Single shared host seeded at the earliest window time ───────
    earliest_time = now - timedelta(seconds=120 * n_history)
    host_rng = np.random.RandomState(host_seed)
    host = HostState.create(host_rng, earliest_time)

    # ── Replay history windows on the shared host ───────────────────
    history_windows: list[str] = []
    for i in range(n_history):
        window_time = now - timedelta(seconds=120 * (n_history - i))
        host.clock.reset(window_time)

        if scenario_type == ScenarioType.BENIGN:
            hist_progress = 0.0
        else:
            hist_progress = attack_progress * (i + 1) / (n_history + 1)

        window_rng = np.random.RandomState(history_seeds[i])
        hist_events, _ = _run_generators_on_host(
            host, scenario_type, window_rng, hist_progress)
        visible = apply_observability_filter(hist_events, observability, window_rng)
        history_windows.append(format_telemetry_window(visible, window_time))

    # ── Generate the current window on the same host ────────────────
    host.clock.reset(now)
    current_rng = np.random.RandomState(current_seed)
    current_events, attack_phase = _run_generators_on_host(
        host, scenario_type, current_rng, attack_progress)

    raw_count = len(current_events)
    visible_events = apply_observability_filter(current_events, observability, current_rng)
    window_text = format_telemetry_window(visible_events, now)

    is_ransomware = scenario_type != ScenarioType.BENIGN

    ground_truth = GroundTruth(
        scenario_type=scenario_type,
        is_ransomware=is_ransomware,
        attack_phase=attack_phase,
    )

    return Episode(
        input_text=window_text,
        ground_truth=ground_truth,
        scenario_type=scenario_type,
        observability=observability,
        raw_event_count=raw_count,
        visible_event_count=len(visible_events),
        history_windows=history_windows,
        host=host,
    )
