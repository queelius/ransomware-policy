"""Telemetry dataset generator (v0.1)

Generates rolling time‑window JSONL records for both benign (goodware) and ransomware scenarios.
Designed as the first scaffold – extend behaviours by editing the GoodwareGenerator
and RansomwareGenerator subclasses.
"""
from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Dict, Any

WINDOW_LEN = timedelta(seconds=120)
STRIDE = timedelta(seconds=30)

# ──────────────────────────────────────────────────────────────────────────────
# Event primitives
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessEvent:
    ts: datetime
    pid: int
    name: str
    ppid: int
    cmd: str
    integrity: str = "medium"

@dataclass
class FileEvent:
    ts: datetime
    path: str
    action: str  # read | write | rename | encrypt
    pid: int

@dataclass
class NetEvent:
    ts: datetime
    pid: int
    dst: str
    bytes: int
    proto: str = "tcp"

# ──────────────────────────────────────────────────────────────────────────────
# Telemetry container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TelemetryWindow:
    episode_id: str
    window_start: datetime
    label: str  # "benign" or "ransomware"
    stage: str | None  # encryption, exfiltration, etc.
    process_tree: List[ProcessEvent]
    file_events: List[FileEvent]
    net_events: List[NetEvent]
    signal_feats: Dict[str, Any]
    qa: List[Dict[str, str]]

    def to_json(self) -> str:
        def _conv(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, "__dict__"):
                return asdict(obj)
            return obj

        return json.dumps(asdict(self), default=_conv)

# ──────────────────────────────────────────────────────────────────────────────
# Behaviour generators
# ──────────────────────────────────────────────────────────────────────────────

class BaseGenerator:
    """Yields event tuples (ts, event_obj)."""

    def __init__(self, start: datetime):
        self.ts = start

    def tick(self) -> Iterator[Any]:
        """Advance internal clock and yield events for this timestep."""
        raise NotImplementedError


class GoodwareGenerator(BaseGenerator):
    """Very simple benign behaviour: user editing docs and occasional backup."""

    def __init__(self, start: datetime, seed: int | None = None):
        super().__init__(start)
        self.rng = random.Random(seed)
        self.pid_ctr = 1000

    def _spawn_proc(self, name: str, parent: int):
        self.pid_ctr += 1
        return ProcessEvent(self.ts, self.pid_ctr, name, parent, cmd=name)

    def tick(self) -> Iterator[Any]:
        # Simulate light CPU/file I/O
        if self.rng.random() < 0.05:
            proc = self._spawn_proc("word.exe", 1)
            yield proc
            yield FileEvent(self.ts, f"C:/Users/A/report_{self.pid_ctr}.docx", "write", proc.pid)
        if self.rng.random() < 0.02:
            # backup compress
            proc = self._spawn_proc("7zip.exe", 1)
            yield proc
            yield FileEvent(self.ts, "C:/Backup/backup.zip", "write", proc.pid)
        self.ts += timedelta(seconds=1)


class RansomwareGenerator(BaseGenerator):
    """Very naive blitz encryptor."""

    def __init__(self, start: datetime, seed: int | None = None):
        super().__init__(start)
        self.rng = random.Random(seed)
        self.pid = 9001
        self.files_encrypted = 0

    def tick(self) -> Iterator[Any]:
        # Every second encrypt ~20 files until 1000 done
        if self.files_encrypted == 0:
            yield ProcessEvent(self.ts, self.pid, "evil.exe", 1, cmd="evil.exe /run")
        for _ in range(20):
            self.files_encrypted += 1
            yield FileEvent(
                self.ts,
                f"C:/Users/A/doc_{self.files_encrypted}.txt",
                "encrypt",
                self.pid,
            )
        if self.rng.random() < 0.3:
            yield NetEvent(self.ts, self.pid, "198.51.100.7:443", bytes=self.rng.randint(1_000, 10_000))
        self.ts += timedelta(seconds=1)


# ──────────────────────────────────────────────────────────────────────────────
# Simulator orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def simulate_episode(label: str, start: datetime, length: timedelta, seed: int | None = None) -> List[Any]:
    gen_cls = GoodwareGenerator if label == "benign" else RansomwareGenerator
    gen = gen_cls(start, seed)
    events = []
    end = start + length
    while gen.ts < end:
        events.extend(gen.tick())
    return events


def window_events(events: List[Any], window_start: datetime) -> TelemetryWindow:
    """Aggregate events inside [window_start, window_start+WINDOW_LEN)."""
    w_end = window_start + WINDOW_LEN
    p, f, n = [], [], []
    for ev in events:
        if not (window_start <= ev.ts < w_end):
            continue
        if isinstance(ev, ProcessEvent):
            p.append(ev)
        elif isinstance(ev, FileEvent):
            f.append(ev)
        elif isinstance(ev, NetEvent):
            n.append(ev)

    # basic signal features
    entropy_delta = sum(1 for fe in f if fe.action == "encrypt") * 2.7  # placeholder
    rename_rate = sum(1 for fe in f if "rename" in fe.action)

    truth = "ransomware" if any(fe.action == "encrypt" for fe in f) else "benign"
    stage = "encryption" if truth == "ransomware" else None

    qa_pairs = [
        {
            "q": "Is there any ransomware activity?",
            "a": "Yes, encryption detected." if truth == "ransomware" else "No suspicious activity detected."
        }
    ]

    return TelemetryWindow(
        episode_id=str(uuid.uuid4())[:8],
        window_start=window_start,
        label=truth,
        stage=stage,
        process_tree=p,
        file_events=f,
        net_events=n,
        signal_feats={
            "entropy_delta_mean": entropy_delta,
            "rename_rate": rename_rate
        },
        qa=qa_pairs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry
# ──────────────────────────────────────────────────────────────────────────────

def main(out_path: str, n_episodes: int = 100, seed: int = 1):
    rng = random.Random(seed)
    start = datetime.utcnow()
    out = Path(out_path)
    out.write_text("")  # truncate

    for epi in range(n_episodes):
        label = rng.choice(["benign", "ransomware"])
        length = timedelta(minutes=rng.randint(3, 8))
        events = simulate_episode(label, start, length, seed=rng.randint(0, 9999))

        # slide windows
        w_start = start
        end_time = start + length
        while w_start + WINDOW_LEN <= end_time:
            w = window_events(events, w_start)
            with out.open("a", encoding="utf-8") as fh:
                fh.write(w.to_json() + "\n")
            w_start += STRIDE

        start += length + timedelta(minutes=1)  # gap before next episode


if __name__ == "__main__":
    main("dataset.jsonl", n_episodes=10)
