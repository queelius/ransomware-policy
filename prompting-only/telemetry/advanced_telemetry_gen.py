#!/usr/bin/env python3
"""
Advanced telemetry generation with realistic attack patterns and partial observability.

This module generates telemetry that reflects the key insight: we can only
partially observe the true system state through telemetry emissions.
"""

import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

# System State (Latent/Hidden)
class SystemState(Enum):
    CLEAN = "clean"
    RECONNAISSANCE = "reconnaissance"  # Attacker exploring
    STAGING = "staging"                # Preparing attack
    EXECUTION = "execution"            # Active ransomware
    PERSISTENCE = "persistence"        # Maintaining access
    EXFILTRATION = "exfiltration"      # Data theft
    IMPACT = "impact"                  # Encryption/destruction

# Observable Events
@dataclass
class TelemetryEvent:
    timestamp: datetime
    event_type: str

    def to_dict(self):
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

@dataclass
class ProcessEvent(TelemetryEvent):
    event_type: str = "process"
    pid: int = 0
    ppid: int = 0
    process_name: str = ""
    command_line: str = ""
    user: str = ""
    integrity_level: str = "medium"
    parent_name: str = ""
    process_hash: Optional[str] = None

@dataclass
class FileEvent(TelemetryEvent):
    event_type: str = "file"
    pid: int = 0
    file_path: str = ""
    operation: str = ""  # read, write, delete, rename
    bytes_affected: int = 0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    extension_before: str = ""
    extension_after: str = ""

@dataclass
class NetworkEvent(TelemetryEvent):
    event_type: str = "network"
    pid: int = 0
    direction: str = ""  # inbound, outbound
    src_ip: str = ""
    src_port: int = 0
    dest_ip: str = ""
    dest_port: int = 0
    protocol: str = ""
    bytes_sent: int = 0
    bytes_received: int = 0

@dataclass
class RegistryEvent(TelemetryEvent):
    event_type: str = "registry"
    pid: int = 0
    key_path: str = ""
    value_name: str = ""
    operation: str = ""  # create, modify, delete
    data: str = ""

class TelemetryGenerator:
    """Generates telemetry based on hidden system state transitions."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.current_time = datetime.utcnow()
        self.pid_counter = 1000
        self.system_state = SystemState.CLEAN
        self.state_duration = timedelta(seconds=0)
        self.processes: Dict[int, str] = {1: "System", 4: "smss.exe"}

        # Ransomware-specific tracking
        self.encrypted_files = set()
        self.c2_servers = ["185.220.101.45", "192.168.56.101", "10.0.0.45"]
        self.attacker_tools = ["powershell.exe", "cmd.exe", "wmic.exe", "net.exe"]

    def transition_state(self) -> SystemState:
        """Hidden state transitions - not directly observable."""
        transitions = {
            SystemState.CLEAN: [
                (SystemState.CLEAN, 0.95),
                (SystemState.RECONNAISSANCE, 0.05)
            ],
            SystemState.RECONNAISSANCE: [
                (SystemState.RECONNAISSANCE, 0.7),
                (SystemState.STAGING, 0.25),
                (SystemState.CLEAN, 0.05)
            ],
            SystemState.STAGING: [
                (SystemState.STAGING, 0.6),
                (SystemState.EXECUTION, 0.35),
                (SystemState.EXFILTRATION, 0.05)
            ],
            SystemState.EXECUTION: [
                (SystemState.EXECUTION, 0.4),
                (SystemState.IMPACT, 0.4),
                (SystemState.PERSISTENCE, 0.2)
            ],
            SystemState.EXFILTRATION: [
                (SystemState.EXFILTRATION, 0.6),
                (SystemState.IMPACT, 0.4)
            ],
            SystemState.PERSISTENCE: [
                (SystemState.PERSISTENCE, 0.5),
                (SystemState.EXECUTION, 0.3),
                (SystemState.CLEAN, 0.2)
            ],
            SystemState.IMPACT: [
                (SystemState.IMPACT, 0.9),
                (SystemState.CLEAN, 0.1)
            ]
        }

        choices, weights = zip(*transitions[self.system_state])
        return self.rng.choices(choices, weights=weights)[0]

    def emit_telemetry(self, state: SystemState) -> List[TelemetryEvent]:
        """Generate observable events based on hidden state - P(Z|X)."""
        events = []

        # Add noise - normal system activity happens regardless of state
        events.extend(self._generate_benign_noise())

        # State-specific emissions (may or may not be observable)
        if state == SystemState.RECONNAISSANCE:
            # Often uses legitimate tools - hard to distinguish
            if self.rng.random() < 0.3:  # Only 30% observable
                events.extend(self._generate_recon_events())

        elif state == SystemState.STAGING:
            # Dropping tools, creating scheduled tasks
            if self.rng.random() < 0.4:
                events.extend(self._generate_staging_events())

        elif state == SystemState.EXECUTION:
            # Running ransomware - more observable
            if self.rng.random() < 0.7:
                events.extend(self._generate_execution_events())

        elif state == SystemState.EXFILTRATION:
            # Network traffic - partially observable
            if self.rng.random() < 0.5:
                events.extend(self._generate_exfiltration_events())

        elif state == SystemState.IMPACT:
            # Encryption - highly observable
            if self.rng.random() < 0.9:
                events.extend(self._generate_impact_events())

        elif state == SystemState.PERSISTENCE:
            # Registry/service changes - moderately observable
            if self.rng.random() < 0.5:
                events.extend(self._generate_persistence_events())

        return events

    def _generate_benign_noise(self) -> List[TelemetryEvent]:
        """Normal system activity that happens continuously."""
        events = []

        # Browser activity
        if self.rng.random() < 0.3:
            pid = self._spawn_process("chrome.exe", "explorer.exe")
            events.append(ProcessEvent(
                timestamp=self.current_time,
                pid=pid,
                ppid=self.processes.get(100, 1),
                process_name="chrome.exe",
                command_line="chrome.exe --restore-last-session",
                user="DESKTOP\\user"
            ))

            # Browser making connections
            events.append(NetworkEvent(
                timestamp=self.current_time,
                pid=pid,
                direction="outbound",
                src_ip="192.168.1.100",
                dest_ip="142.250.185.78",  # google.com
                dest_port=443,
                protocol="https",
                bytes_sent=self.rng.randint(100, 5000)
            ))

        # Office applications
        if self.rng.random() < 0.2:
            office_apps = ["WINWORD.EXE", "EXCEL.EXE", "OUTLOOK.EXE"]
            app = self.rng.choice(office_apps)
            pid = self._spawn_process(app, "explorer.exe")

            events.append(ProcessEvent(
                timestamp=self.current_time,
                pid=pid,
                ppid=100,
                process_name=app,
                command_line=f"{app} /automation",
                user="DESKTOP\\user"
            ))

            # Document access
            events.append(FileEvent(
                timestamp=self.current_time,
                pid=pid,
                file_path=f"C:\\Users\\user\\Documents\\report_{self.rng.randint(1,100)}.docx",
                operation="read",
                bytes_affected=self.rng.randint(10000, 500000),
                entropy_before=3.2,
                entropy_after=3.2
            ))

        return events

    def _generate_recon_events(self) -> List[TelemetryEvent]:
        """Reconnaissance phase - often using legitimate tools."""
        events = []

        # Whoami, net user, systeminfo - living off the land
        recon_commands = [
            ("cmd.exe", "cmd.exe /c whoami /all"),
            ("net.exe", "net user"),
            ("net.exe", "net localgroup administrators"),
            ("powershell.exe", "powershell.exe -Command Get-Process"),
            ("wmic.exe", "wmic process list brief")
        ]

        proc, cmd = self.rng.choice(recon_commands)
        pid = self._spawn_process(proc, "explorer.exe")

        events.append(ProcessEvent(
            timestamp=self.current_time,
            pid=pid,
            ppid=100,
            process_name=proc,
            command_line=cmd,
            user="DESKTOP\\user",
            parent_name="explorer.exe"
        ))

        return events

    def _generate_staging_events(self) -> List[TelemetryEvent]:
        """Staging phase - dropping tools, creating persistence."""
        events = []

        # Download and execute pattern
        pid = self._spawn_process("powershell.exe", "explorer.exe")

        # Encoded PowerShell is suspicious but common
        encoded_cmd = "SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAIABOAGUAdAAuAFcAZQBiAEMAbABpAGUA"
        events.append(ProcessEvent(
            timestamp=self.current_time,
            pid=pid,
            ppid=100,
            process_name="powershell.exe",
            command_line=f"powershell.exe -enc {encoded_cmd}",
            user="DESKTOP\\user",
            integrity_level="medium"
        ))

        # Drops executable
        events.append(FileEvent(
            timestamp=self.current_time,
            pid=pid,
            file_path="C:\\Users\\user\\AppData\\Local\\Temp\\update.exe",
            operation="write",
            bytes_affected=self.rng.randint(100000, 500000),
            entropy_before=0.0,
            entropy_after=6.8  # Compressed/encrypted binary
        ))

        # Network connection to C2
        events.append(NetworkEvent(
            timestamp=self.current_time,
            pid=pid,
            direction="outbound",
            src_ip="192.168.1.100",
            dest_ip=self.rng.choice(self.c2_servers),
            dest_port=443,
            protocol="https",
            bytes_sent=self.rng.randint(1000, 10000),
            bytes_received=self.rng.randint(100000, 500000)
        ))

        return events

    def _generate_execution_events(self) -> List[TelemetryEvent]:
        """Execution phase - ransomware starting up."""
        events = []

        # Suspicious process spawn
        pid = self._spawn_process("update.exe", "powershell.exe")
        events.append(ProcessEvent(
            timestamp=self.current_time,
            pid=pid,
            ppid=self.processes.get(200, 1),
            process_name="update.exe",
            command_line="C:\\Users\\user\\AppData\\Local\\Temp\\update.exe",
            user="DESKTOP\\user",
            process_hash="a3f5b2c1d4e6f7g8h9i0"  # Unknown hash
        ))

        # Delete shadow copies - major red flag
        vss_pid = self._spawn_process("vssadmin.exe", "update.exe")
        events.append(ProcessEvent(
            timestamp=self.current_time,
            pid=vss_pid,
            ppid=pid,
            process_name="vssadmin.exe",
            command_line="vssadmin.exe delete shadows /all /quiet",
            user="DESKTOP\\user",
            parent_name="update.exe"
        ))

        # Disable Windows Defender
        events.append(RegistryEvent(
            timestamp=self.current_time,
            pid=pid,
            key_path="HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows Defender",
            value_name="DisableAntiSpyware",
            operation="modify",
            data="1"
        ))

        return events

    def _generate_impact_events(self) -> List[TelemetryEvent]:
        """Impact phase - active encryption."""
        events = []

        # Rapid file encryption pattern
        extensions = ['.docx', '.xlsx', '.pdf', '.jpg', '.png', '.txt']
        ransomware_pid = self.processes.get(9999, 9999)

        for i in range(self.rng.randint(10, 50)):  # Encrypt many files quickly
            ext = self.rng.choice(extensions)
            original_file = f"C:\\Users\\user\\Documents\\file_{i}{ext}"

            # File read (original)
            events.append(FileEvent(
                timestamp=self.current_time,
                pid=ransomware_pid,
                file_path=original_file,
                operation="read",
                bytes_affected=self.rng.randint(1000, 1000000),
                entropy_before=self.rng.uniform(2.0, 4.0),
                entropy_after=self.rng.uniform(2.0, 4.0),
                extension_before=ext,
                extension_after=ext
            ))

            # File write (encrypted)
            events.append(FileEvent(
                timestamp=self.current_time + timedelta(milliseconds=10),
                pid=ransomware_pid,
                file_path=f"{original_file}.locked",
                operation="write",
                bytes_affected=self.rng.randint(1000, 1000000),
                entropy_before=0.0,
                entropy_after=self.rng.uniform(7.8, 8.0),  # High entropy = encrypted
                extension_before=ext,
                extension_after=".locked"
            ))

            # Delete original
            events.append(FileEvent(
                timestamp=self.current_time + timedelta(milliseconds=20),
                pid=ransomware_pid,
                file_path=original_file,
                operation="delete",
                bytes_affected=0
            ))

        # Drop ransom note
        events.append(FileEvent(
            timestamp=self.current_time,
            pid=ransomware_pid,
            file_path="C:\\Users\\user\\Desktop\\READ_ME_URGENT.txt",
            operation="write",
            bytes_affected=2048,
            entropy_before=0.0,
            entropy_after=3.5  # Text file entropy
        ))

        return events

    def _generate_exfiltration_events(self) -> List[TelemetryEvent]:
        """Data exfiltration - often subtle."""
        events = []

        # Large outbound transfers
        exfil_pid = self.processes.get(8888, 8888)

        for _ in range(self.rng.randint(3, 10)):
            events.append(NetworkEvent(
                timestamp=self.current_time,
                pid=exfil_pid,
                direction="outbound",
                src_ip="192.168.1.100",
                dest_ip=self.rng.choice(self.c2_servers),
                dest_port=443,
                protocol="https",
                bytes_sent=self.rng.randint(1000000, 50000000),  # Large uploads
                bytes_received=self.rng.randint(100, 1000)
            ))

            # Reading sensitive files
            sensitive_paths = [
                "C:\\Users\\user\\Documents\\passwords.xlsx",
                "C:\\Users\\user\\Documents\\financial_report.pdf",
                "C:\\Users\\user\\Desktop\\confidential.docx"
            ]

            events.append(FileEvent(
                timestamp=self.current_time,
                pid=exfil_pid,
                file_path=self.rng.choice(sensitive_paths),
                operation="read",
                bytes_affected=self.rng.randint(10000, 1000000)
            ))

        return events

    def _generate_persistence_events(self) -> List[TelemetryEvent]:
        """Establishing persistence mechanisms."""
        events = []

        pid = self.processes.get(7777, 7777)

        # Registry run key
        events.append(RegistryEvent(
            timestamp=self.current_time,
            pid=pid,
            key_path="HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
            value_name="WindowsUpdate",
            operation="create",
            data="C:\\Users\\user\\AppData\\Local\\Temp\\update.exe"
        ))

        # Scheduled task creation
        schtask_pid = self._spawn_process("schtasks.exe", "cmd.exe")
        events.append(ProcessEvent(
            timestamp=self.current_time,
            pid=schtask_pid,
            ppid=pid,
            process_name="schtasks.exe",
            command_line='schtasks.exe /create /tn "WindowsUpdate" /tr "C:\\Users\\user\\AppData\\Local\\Temp\\update.exe" /sc onlogon',
            user="DESKTOP\\user"
        ))

        return events

    def _spawn_process(self, name: str, parent: str) -> int:
        """Create a new process and track it."""
        self.pid_counter += 1
        self.processes[self.pid_counter] = name
        return self.pid_counter

    def generate_window(self, duration: timedelta) -> Tuple[List[TelemetryEvent], SystemState]:
        """Generate a telemetry window with state transitions."""
        events = []
        end_time = self.current_time + duration

        while self.current_time < end_time:
            # State transition check
            if self.state_duration >= timedelta(seconds=self.rng.randint(10, 60)):
                self.system_state = self.transition_state()
                self.state_duration = timedelta(seconds=0)

            # Emit telemetry based on current hidden state
            window_events = self.emit_telemetry(self.system_state)
            events.extend(window_events)

            # Advance time
            self.current_time += timedelta(seconds=self.rng.uniform(0.1, 2.0))
            self.state_duration += timedelta(seconds=1)

        return events, self.system_state


def generate_dataset(
    output_file: str,
    num_windows: int = 1000,
    window_duration: int = 120,  # seconds
    seed: Optional[int] = None
):
    """Generate a dataset of telemetry windows with ground truth."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        generator = TelemetryGenerator(seed=seed)

        for i in range(num_windows):
            events, true_state = generator.generate_window(
                duration=timedelta(seconds=window_duration)
            )

            # Convert to JSON-serializable format
            window_data = {
                "window_id": i,
                "start_time": (generator.current_time - timedelta(seconds=window_duration)).isoformat(),
                "end_time": generator.current_time.isoformat(),
                "true_state": true_state.value,  # Ground truth - not observable in reality
                "events": [e.to_dict() for e in events],
                "event_count": len(events),
                "event_types": {
                    "process": sum(1 for e in events if e.event_type == "process"),
                    "file": sum(1 for e in events if e.event_type == "file"),
                    "network": sum(1 for e in events if e.event_type == "network"),
                    "registry": sum(1 for e in events if e.event_type == "registry")
                }
            }

            f.write(json.dumps(window_data) + '\n')

    print(f"Generated {num_windows} telemetry windows in {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate realistic telemetry data")
    parser.add_argument("--output", default="telemetry_dataset.jsonl", help="Output file path")
    parser.add_argument("--windows", type=int, default=1000, help="Number of windows to generate")
    parser.add_argument("--duration", type=int, default=120, help="Window duration in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    generate_dataset(
        output_file=args.output,
        num_windows=args.windows,
        window_duration=args.duration,
        seed=args.seed
    )