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
    contents: bytes | None = None

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
    open_file_handles: list[str] = field(default_factory=list)
    open_network_sockets: list[int] = field(default_factory=list)
    loaded_modules: list[str] = field(default_factory=list)
    user: str = "SYSTEM"
    integrity_level: str = "High"
    is_elevated: bool = False


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


@dataclass
class RegistryEvent:
    """Observable registry modification event."""

    ts: datetime
    key_path: str
    value_name: str
    action: Literal["set", "delete", "create"]

    event_type: Literal["registry"] = field(default="registry", init=False)
    drop_probability: float = 0.0


@dataclass
class EventLogEvent:
    """Observable Windows Event Log entry event."""

    ts: datetime
    source: str
    event_id: int
    message: str

    event_type: Literal["eventlog"] = field(default="eventlog", init=False)
    drop_probability: float = 0.0


# ── Connection / Registry sub-record models ──────────────────────────


@dataclass
class ConnectionRecord:
    """A network connection on the simulated host."""

    conn_id: int
    pid: int
    local_port: int
    remote_address: str
    remote_port: int
    protocol: str
    state: Literal["established", "listening", "closed", "syn_sent"]
    bytes_sent: int = 0
    bytes_received: int = 0
    opened_at: datetime = field(default_factory=lambda: datetime(2025, 1, 1))


@dataclass
class RegistryValue:
    """A single value in a registry key."""

    name: str
    data: str
    value_type: str = "REG_SZ"


@dataclass
class RegistryKey:
    """A registry key with its values."""

    path: str
    values: dict[str, RegistryValue] = field(default_factory=dict)
    modified_at: datetime = field(default_factory=lambda: datetime(2025, 1, 1))


@dataclass
class EventLogEntry:
    """A stored event log entry."""

    timestamp: datetime
    source: str
    event_id: int
    level: str  # "Information", "Warning", "Error", "Critical"
    message: str


# Union type for all events
TelemetryEvent = ProcessEvent | FileEvent | NetEvent | RegistryEvent | EventLogEvent


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
