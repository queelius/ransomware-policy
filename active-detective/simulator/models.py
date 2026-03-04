"""Data models for simulated host environment.

FileRecord and ProcessRecord represent mutable host state.
Telemetry event dataclasses represent observable events the agent receives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


# ── Host state (ground truth, partially hidden from agent) ───────────


class ContentType(str, Enum):
    DOC = "doc"
    CONFIG = "config"
    IMAGE = "image"
    EXECUTABLE = "executable"
    ARCHIVE = "archive"
    DATABASE = "database"


@dataclass
class FileRecord:
    """A file in the simulated host filesystem.

    ``is_encrypted`` is ground truth — hidden from the agent.
    The agent can only observe metadata via the ``inspect_file`` tool.
    """

    path: str
    size: int
    entropy: float
    extension: str
    modified_at: datetime
    content_type: ContentType
    is_encrypted: bool = False

    # Original values before any mutation (for delta computation)
    _original_entropy: float = field(init=False, repr=False)
    _original_size: int = field(init=False, repr=False)
    _original_extension: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._original_entropy = self.entropy
        self._original_size = self.size
        self._original_extension = self.extension

    @property
    def entropy_delta(self) -> float:
        return self.entropy - self._original_entropy

    @property
    def size_delta(self) -> int:
        return self.size - self._original_size

    @property
    def extension_changed(self) -> bool:
        return self.extension != self._original_extension


@dataclass
class ProcessRecord:
    """A running process on the simulated host."""

    pid: int
    name: str
    parent_pid: int
    command_line: str
    start_time: datetime
    child_pids: list[int] = field(default_factory=list)


# ── Telemetry events (observable by the agent) ───────────────────────


@dataclass
class ProcessEvent:
    """Observable process telemetry event."""

    ts: datetime
    pid: int
    name: str
    parent_pid: int
    command_line: str

    event_type: Literal["process"] = field(default="process", init=False)
    drop_probability: float = 0.0


@dataclass
class FileEvent:
    """Observable file metadata change event."""

    ts: datetime
    path: str
    size_delta: int
    entropy_delta: float
    extension_change: str | None  # e.g. ".docx -> .docx.locked", or None

    event_type: Literal["file"] = field(default="file", init=False)
    drop_probability: float = 0.0


@dataclass
class NetEvent:
    """Observable network telemetry event."""

    ts: datetime
    destination: str
    bytes_transferred: int
    protocol: str
    direction: Literal["inbound", "outbound"]

    event_type: Literal["network"] = field(default="network", init=False)
    drop_probability: float = 0.0


# Union type for all events
TelemetryEvent = ProcessEvent | FileEvent | NetEvent


# ── Scenario metadata ───────────────────────────────────────────────


class ScenarioType(str, Enum):
    BENIGN = "benign"
    BLITZ = "blitz"
    SLEEPER = "sleeper"
    EXFIL_FIRST = "exfil_first"
    SEMANTIC_SHUFFLE = "semantic_shuffle"


class Verdict(str, Enum):
    IGNORE = "ignore"
    MONITOR = "monitor"
    ALERT = "alert"
    QUARANTINE = "quarantine"
    BLOCK = "block"

    @property
    def is_ransomware_prediction(self) -> bool:
        """Whether this verdict predicts ransomware (vs benign)."""
        return self in (Verdict.ALERT, Verdict.QUARANTINE, Verdict.BLOCK)


@dataclass
class GroundTruth:
    """Ground truth for a scenario episode."""

    scenario_type: ScenarioType
    is_ransomware: bool
    attack_phase: str | None = None  # e.g. "exfiltration", "encryption"

    @property
    def label(self) -> str:
        if not self.is_ransomware:
            return "benign"
        if self.attack_phase:
            return f"{self.scenario_type.value}:{self.attack_phase}"
        return self.scenario_type.value
