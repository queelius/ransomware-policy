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


def _generate_window(
    scenario_type: ScenarioType,
    observability: float,
    rng: np.random.RandomState,
    window_time: datetime,
    attack_progress: float,
) -> tuple[str, str | None, HostState]:
    """Generate a single telemetry window and return (formatted_text, attack_phase, host).

    Creates a fresh HostState, runs benign background + scenario generators,
    applies observability filter, and formats the result.
    """
    host = HostState.create(rng, window_time)
    all_events: list[TelemetryEvent] = []
    attack_phase: str | None = None

    # Always generate some benign background activity
    n_benign = rng.randint(1, 3)
    chosen_benign = rng.choice(len(_BENIGN_GENERATORS), size=n_benign, replace=False)
    for idx in chosen_benign:
        all_events.extend(_BENIGN_GENERATORS[idx](host, rng))

    # Add scenario-specific events
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

    # Apply observability filter
    visible_events = apply_observability_filter(all_events, observability, rng)

    # Format into text
    window_text = format_telemetry_window(visible_events, window_time)
    return window_text, attack_phase, host


def generate_episode(
    scenario_type: ScenarioType,
    observability: float,
    rng: np.random.RandomState,
    now: datetime | None = None,
    attack_progress: float = 0.5,
    n_history: int = 2,
) -> Episode:
    """Generate a single episode with telemetry window and history.

    Seeds a full host state, runs generators for one window, applies
    observability filter, and returns formatted episode.

    When ``n_history > 0``, generates prior telemetry windows at
    earlier timestamps and lower attack progress to give the agent
    temporal context.
    """
    now = now or datetime(2025, 6, 15, 10, 0, 0)

    # ── Generate history windows (earliest first) ───────────────────
    history_windows: list[str] = []
    for i in range(n_history):
        # Each history window is 120 seconds earlier than the next
        window_time = now - timedelta(seconds=120 * (n_history - i))

        if scenario_type == ScenarioType.BENIGN:
            hist_progress = 0.0
        else:
            # Linearly-spaced earlier progress values
            hist_progress = attack_progress * (i + 1) / (n_history + 1)

        hist_text, _, _ = _generate_window(
            scenario_type, observability, rng, window_time, hist_progress)
        history_windows.append(hist_text)

    # ── Generate the current window (existing logic) ────────────────
    host = HostState.create(rng, now)

    all_events: list[TelemetryEvent] = []
    attack_phase: str | None = None

    # Always generate some benign background activity
    n_benign = rng.randint(1, 3)
    chosen_benign = rng.choice(len(_BENIGN_GENERATORS), size=n_benign, replace=False)
    for idx in chosen_benign:
        all_events.extend(_BENIGN_GENERATORS[idx](host, rng))

    # Add attack-specific events
    if scenario_type == ScenarioType.BENIGN:
        # Add more benign activity for realism
        if rng.random() < 0.4:
            gen_fn = _EXTRA_BENIGN_GENERATORS[rng.randint(0, len(_EXTRA_BENIGN_GENERATORS))]
            all_events.extend(gen_fn(host, rng))
        is_ransomware = False
    elif scenario_type in _ATTACK_GENERATORS:
        events, attack_phase = _ATTACK_GENERATORS[scenario_type](
            host, rng, progress=attack_progress)
        all_events.extend(events)
        is_ransomware = True
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    raw_count = len(all_events)

    # Apply observability filter
    visible_events = apply_observability_filter(all_events, observability, rng)

    # Format into text
    window_text = format_telemetry_window(visible_events, now)

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
