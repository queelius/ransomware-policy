#!/usr/bin/env python
"""
telemetry_sim.py  (v0.2)
========================
Generate synthetic host -telemetry windows for benign vs. ransomware
scenarios.  CLI lets you control episode count, window length, stride,
and ransomware ratio.

Example
-------
python telemetry_sim.py \
    --out dataset.jsonl \
    --episodes 5_000 \
    --min-minutes 2 \
    --max-minutes 10 \
    --window 120 \
    --stride 30 \
    --ransomware-p   0.35 \
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Dict, Any

from tqdm import tqdm

# ----------------------------------------- event primitives ------------------
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
    action: str          # read | write | rename | encrypt
    pid: int

@dataclass
class NetEvent:
    ts: datetime
    pid: int
    dst: str
    bytes: int
    proto: str = "tcp"

# ------------------------------------- telemetry window ----------------------
@dataclass
class TelemetryWindow:
    episode_id: str
    window_start: datetime
    label: str          # benign | ransomware
    stage: str | None   # encryption, exfiltration, …
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

# ------------------------------- behaviour generators ------------------------
class BaseGenerator:
    def __init__(self, start: datetime, seed: int | None = None):
        self.ts   = start
        self.rng  = random.Random(seed)

class GoodwareGenerator(BaseGenerator):
    """User edits docs + periodic 7 -Zip backup."""
    def __init__(self, start: datetime, seed: int | None = None):
        super().__init__(start, seed)
        self.pid_ctr = 1000

    def _spawn_proc(self, name: str, parent: int):
        self.pid_ctr += 1
        return ProcessEvent(self.ts, self.pid_ctr, name, parent, cmd=name)

    def tick(self) -> Iterator[Any]:
        if self.rng.random() < 0.05:            # doc edit
            proc = self._spawn_proc("word.exe", 1)
            yield proc
            yield FileEvent(self.ts, f"C:/Users/A/report_{self.pid_ctr}.docx",
                            "write", proc.pid)
        if self.rng.random() < 0.02:            # backup
            proc = self._spawn_proc("7zip.exe", 1)
            yield proc
            yield FileEvent(self.ts, "C:/Backup/backup.zip", "write", proc.pid)
        self.ts += timedelta(seconds=1)

class RansomwareGenerator(BaseGenerator):
    """Very naive blitz encryptor (20 files/s)."""
    def __init__(self, start: datetime, seed: int | None = None):
        super().__init__(start, seed)
        self.pid = 9001
        self.files_encrypted = 0

    def tick(self) -> Iterator[Any]:
        if self.files_encrypted == 0:
            yield ProcessEvent(self.ts, self.pid, "evil.exe", 1,
                               cmd="evil.exe /run")
        for _ in range(20):
            self.files_encrypted += 1
            yield FileEvent(self.ts,
                            f"C:/Users/A/doc_{self.files_encrypted}.txt",
                            "encrypt", self.pid)
        if self.rng.random() < 0.30:  # 30 % chance outbound beacon
            yield NetEvent(self.ts, self.pid, "198.51.100.7:443",
                           bytes=self.rng.randint(1_000, 10_000))
        self.ts += timedelta(seconds=1)

# ----------------------------------- helpers ---------------------------------
def simulate_episode(label: str, start: datetime, length: timedelta,
                     seed: int | None = None) -> List[Any]:
    gen_cls = GoodwareGenerator if label == "benign" else RansomwareGenerator
    gen = gen_cls(start, seed)
    events, end_ts = [], start + length
    while gen.ts < end_ts:
        events.extend(gen.tick())
    return events

def window_events(events: List[Any], window_start: datetime,
                  window_len: timedelta) -> TelemetryWindow:
    w_end = window_start + window_len
    p, f, n = [], [], []
    for ev in events:
        if not (window_start <= ev.ts < w_end):
            continue
        (p if isinstance(ev, ProcessEvent) else
         f if isinstance(ev, FileEvent)  else
         n).append(ev)

    entropy_delta = sum(1 for fe in f if fe.action == "encrypt") * 2.7
    rename_rate   = sum(1 for fe in f if "rename" in fe.action)
    truth = "ransomware" if any(fe.action == "encrypt" for fe in f) else "benign"
    stage = "encryption" if truth == "ransomware" else None
    qa_pairs = [{
        "q": "Is there any ransomware activity?",
        "a": "Yes, encryption detected." if truth == "ransomware"
             else "No suspicious activity detected."
    }]

    return TelemetryWindow(
        episode_id=str(uuid.uuid4())[:8],
        window_start=window_start,
        label=truth,
        stage=stage,
        process_tree=p,
        file_events=f,
        net_events=n,
        signal_feats={"entropy_delta_mean": entropy_delta,
                      "rename_rate": rename_rate},
        qa=qa_pairs,
    )

# ------------------------------------ CLI ------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Synthetic ransomware vs benign telemetry generator")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--episodes", type=int, default=1000,
                    help="Number of episodes to simulate")
    ap.add_argument("--min-minutes", type=int, default=3,
                    help="Min episode length (minutes)")
    ap.add_argument("--max-minutes", type=int, default=8,
                    help="Max episode length (minutes)")
    ap.add_argument("--window", type=int, default=120,
                    help="Window length in seconds")
    ap.add_argument("--stride", type=int, default=30,
                    help="Stride in seconds")
    ap.add_argument("--ransomware-p", type=float, default=0.4,
                    help="Probability an episode is ransomware")
    ap.add_argument("--seed", type=int, default=1,
                    help="Master RNG seed")
    return ap.parse_args()

# ---------------------------------- main -------------------------------------
def main():
    args = parse_args()
    rng = random.Random(args.seed)

    window_len = timedelta(seconds=args.window)
    stride     = timedelta(seconds=args.stride)
    out_path   = Path(args.out)
    out_path.write_text("")        # truncate

    current_ts = datetime.utcnow()

    for _ in tqdm(range(args.episodes), desc="Episodes"):
        label   = ("ransomware" if rng.random() < args.ransomware_p
                   else "benign")
        minutes = rng.randint(args.min_minutes, args.max_minutes)
        length  = timedelta(minutes=minutes)

        events = simulate_episode(label, current_ts, length,
                                  seed=rng.randint(0, 1_000_000))

        w_start, end_ts = current_ts, current_ts + length

        while w_start + window_len <= end_ts:
            w = window_events(events, w_start, window_len)
            with out_path.open("a", encoding="utf‑8") as fh:
                fh.write(w.to_json() + "\n")

            w_start += stride

        current_ts = end_ts + timedelta(minutes=1)  # episode gap

if __name__ == "__main__":
    main()
