#!/usr/bin/env python3
"""
Integration with Atomic Red Team tests to generate realistic ransomware telemetry.
Maps ATT&CK techniques to telemetry events based on real-world ransomware behaviors.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Based on Atomic Red Team's ransomware-related tests
ATOMIC_RANSOMWARE_TECHNIQUES = {
    "T1486": {  # Data Encrypted for Impact
        "name": "Data Encrypted for Impact",
        "description": "Ransomware encrypting files",
        "tests": [
            {
                "name": "Encrypt files with GPG (Linux)",
                "telemetry_pattern": {
                    "processes": ["gpg", "find", "xargs"],
                    "file_patterns": ["mass_read", "mass_write", "extension_change"],
                    "entropy_change": "low_to_high"
                }
            },
            {
                "name": "Encrypt files with 7z (Windows)",
                "telemetry_pattern": {
                    "processes": ["7z.exe", "cmd.exe"],
                    "file_patterns": ["mass_read", "mass_write", "archive_creation"],
                    "entropy_change": "low_to_high"
                }
            },
            {
                "name": "BCryptEncrypt (Windows)",
                "telemetry_pattern": {
                    "processes": ["powershell.exe"],
                    "api_calls": ["BCryptEncrypt", "BCryptDecrypt"],
                    "file_patterns": ["mass_modification"],
                    "entropy_change": "low_to_high"
                }
            }
        ]
    },
    "T1490": {  # Inhibit System Recovery
        "name": "Inhibit System Recovery",
        "description": "Delete shadow copies and disable recovery",
        "tests": [
            {
                "name": "Delete Volume Shadow Copies",
                "telemetry_pattern": {
                    "processes": ["vssadmin.exe", "wmic.exe"],
                    "commands": [
                        "vssadmin.exe delete shadows /all /quiet",
                        "wmic shadowcopy delete"
                    ],
                    "registry_keys": []
                }
            },
            {
                "name": "Disable Windows Recovery Console",
                "telemetry_pattern": {
                    "processes": ["bcdedit.exe", "reagentc.exe"],
                    "commands": [
                        "bcdedit /set {default} recoveryenabled no",
                        "bcdedit /set {default} bootstatuspolicy ignoreallfailures",
                        "reagentc /disable"
                    ]
                }
            }
        ]
    },
    "T1055": {  # Process Injection
        "name": "Process Injection",
        "description": "Inject code into legitimate processes",
        "tests": [
            {
                "name": "Process Hollowing",
                "telemetry_pattern": {
                    "processes": ["svchost.exe", "explorer.exe"],
                    "api_calls": ["CreateProcess", "NtUnmapViewOfSection", "WriteProcessMemory"],
                    "suspicious_behavior": "legitimate_process_unusual_network"
                }
            }
        ]
    },
    "T1059.001": {  # PowerShell
        "name": "Command and Scripting: PowerShell",
        "description": "PowerShell execution for ransomware deployment",
        "tests": [
            {
                "name": "Base64 Encoded Commands",
                "telemetry_pattern": {
                    "processes": ["powershell.exe"],
                    "commands": ["powershell.exe -enc", "-ExecutionPolicy Bypass", "-NoProfile"],
                    "parent_processes": ["cmd.exe", "explorer.exe", "winword.exe"]
                }
            }
        ]
    },
    "T1083": {  # File and Directory Discovery
        "name": "File and Directory Discovery",
        "description": "Enumerate files for encryption",
        "tests": [
            {
                "name": "Recursive File Listing",
                "telemetry_pattern": {
                    "processes": ["cmd.exe", "powershell.exe"],
                    "commands": [
                        "dir /s /b C:\\Users",
                        "Get-ChildItem -Recurse -Force"
                    ],
                    "file_patterns": ["directory_enumeration"]
                }
            }
        ]
    },
    "T1222": {  # File and Directory Permissions Modification
        "name": "File Permissions Modification",
        "description": "Change file permissions before encryption",
        "tests": [
            {
                "name": "Take Ownership of Files",
                "telemetry_pattern": {
                    "processes": ["takeown.exe", "icacls.exe"],
                    "commands": [
                        "takeown /f * /r",
                        "icacls * /grant Everyone:F /t"
                    ]
                }
            }
        ]
    },
    "T1112": {  # Modify Registry
        "name": "Modify Registry",
        "description": "Registry changes for persistence/defense evasion",
        "tests": [
            {
                "name": "Disable Windows Defender",
                "telemetry_pattern": {
                    "processes": ["reg.exe", "powershell.exe"],
                    "registry_keys": [
                        "HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows Defender\\DisableAntiSpyware",
                        "HKLM\\SOFTWARE\\Microsoft\\Windows Defender\\Real-Time Protection"
                    ]
                }
            }
        ]
    }
}

# Real ransomware family behaviors (based on public analysis)
RANSOMWARE_FAMILIES = {
    "LockBit3": {
        "techniques": ["T1486", "T1490", "T1055", "T1059.001", "T1083"],
        "characteristics": {
            "speed": "fast",  # Encrypts very quickly
            "exfiltration": True,
            "double_extortion": True,
            "propagation": "aggressive",
            "ransom_note": "README.txt",
            "encrypted_extension": ".lockbit"
        }
    },
    "BlackCat/ALPHV": {
        "techniques": ["T1486", "T1490", "T1059.001", "T1083", "T1112"],
        "characteristics": {
            "speed": "medium",
            "exfiltration": True,
            "double_extortion": True,
            "language": "rust",  # Written in Rust
            "propagation": "selective",
            "ransom_note": "RECOVER-FILES.txt",
            "encrypted_extension": ".alphv"
        }
    },
    "Conti": {
        "techniques": ["T1486", "T1490", "T1055", "T1059.001", "T1083", "T1222"],
        "characteristics": {
            "speed": "fast",
            "exfiltration": True,
            "double_extortion": True,
            "propagation": "aggressive",
            "ransom_note": "readme.txt",
            "encrypted_extension": ".CONTI"
        }
    },
    "REvil/Sodinokibi": {
        "techniques": ["T1486", "T1490", "T1059.001", "T1083", "T1112"],
        "characteristics": {
            "speed": "medium",
            "exfiltration": True,
            "double_extortion": True,
            "propagation": "moderate",
            "ransom_note": "[random]-readme.txt",
            "encrypted_extension": ".[random]"
        }
    }
}

class AtomicTelemetryGenerator:
    """Generate realistic telemetry based on Atomic Red Team patterns."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.current_time = datetime.now()  # Fixed deprecation warning
        self.pid_counter = 5000

    def generate_ransomware_campaign(
        self,
        family: str = None,
        duration_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """Generate a complete ransomware campaign telemetry."""

        if family is None:
            family = self.rng.choice(list(RANSOMWARE_FAMILIES.keys()))

        campaign = RANSOMWARE_FAMILIES.get(family, RANSOMWARE_FAMILIES["LockBit3"])
        events = []

        # Phase 1: Initial Compromise (0-5 minutes)
        events.extend(self._generate_initial_compromise())
        self.current_time += timedelta(minutes=self.rng.randint(1, 5))

        # Phase 2: Discovery (5-10 minutes)
        for technique in ["T1083"]:  # File discovery
            if technique in campaign["techniques"]:
                events.extend(self._generate_technique_telemetry(technique))
        self.current_time += timedelta(minutes=self.rng.randint(3, 5))

        # Phase 3: Defense Evasion (10-15 minutes)
        for technique in ["T1112", "T1490"]:  # Registry mod, inhibit recovery
            if technique in campaign["techniques"]:
                events.extend(self._generate_technique_telemetry(technique))
        self.current_time += timedelta(minutes=self.rng.randint(3, 5))

        # Phase 4: Exfiltration (optional, 15-20 minutes)
        if campaign["characteristics"].get("exfiltration"):
            events.extend(self._generate_exfiltration())
        self.current_time += timedelta(minutes=self.rng.randint(3, 5))

        # Phase 5: Impact (20+ minutes)
        events.extend(self._generate_encryption_phase(campaign["characteristics"]))

        return events

    def _generate_initial_compromise(self) -> List[Dict[str, Any]]:
        """Generate initial compromise telemetry."""
        events = []

        # Common initial vectors
        vectors = [
            {
                "type": "phishing",
                "process": "WINWORD.EXE",
                "child": "powershell.exe",
                "command": "powershell.exe -nop -w hidden -enc [base64]"
            },
            {
                "type": "rdp_brute",
                "process": "svchost.exe",
                "network": {"port": 3389, "failed_logins": 50}
            },
            {
                "type": "exploit",
                "process": "w3wp.exe",  # IIS worker
                "child": "cmd.exe",
                "command": "cmd.exe /c whoami"
            }
        ]

        vector = self.rng.choice(vectors)

        if vector["type"] == "phishing":
            # Office spawning PowerShell
            events.append({
                "timestamp": self.current_time.isoformat(),
                "event_type": "process_create",
                "process_name": vector["process"],
                "pid": self._get_pid(),
                "child_process": vector["child"],
                "command_line": vector["command"],
                "integrity_level": "medium"
            })

        return events

    def _generate_technique_telemetry(self, technique_id: str) -> List[Dict[str, Any]]:
        """Generate telemetry for a specific ATT&CK technique."""
        events = []
        technique = ATOMIC_RANSOMWARE_TECHNIQUES.get(technique_id)

        if not technique:
            return events

        # Pick a random test from this technique
        test = self.rng.choice(technique["tests"])
        pattern = test["telemetry_pattern"]

        # Generate process events
        if "processes" in pattern:
            for proc in pattern["processes"]:
                pid = self._get_pid()
                events.append({
                    "timestamp": self.current_time.isoformat(),
                    "event_type": "process_create",
                    "process_name": proc,
                    "pid": pid,
                    "ppid": pid - 1
                })

        # Generate command events
        if "commands" in pattern:
            for cmd in pattern["commands"]:
                events.append({
                    "timestamp": (self.current_time + timedelta(seconds=self.rng.randint(1, 10))).isoformat(),
                    "event_type": "process_create",
                    "command_line": cmd,
                    "pid": self._get_pid()
                })

        # Generate registry events
        if "registry_keys" in pattern:
            for key in pattern["registry_keys"]:
                events.append({
                    "timestamp": self.current_time.isoformat(),
                    "event_type": "registry_modify",
                    "key_path": key,
                    "operation": "set_value",
                    "value": "1"
                })

        # Generate file events
        if "file_patterns" in pattern:
            for fp in pattern["file_patterns"]:
                if fp == "mass_read":
                    for i in range(self.rng.randint(50, 200)):
                        events.append({
                            "timestamp": (self.current_time + timedelta(milliseconds=i*10)).isoformat(),
                            "event_type": "file_read",
                            "file_path": f"C:\\Users\\user\\Documents\\file_{i}.docx",
                            "pid": self._get_pid()
                        })
                elif fp == "mass_write":
                    for i in range(self.rng.randint(50, 200)):
                        events.append({
                            "timestamp": (self.current_time + timedelta(milliseconds=i*10)).isoformat(),
                            "event_type": "file_write",
                            "file_path": f"C:\\Users\\user\\Documents\\file_{i}.docx.encrypted",
                            "pid": self._get_pid(),
                            "entropy_after": self.rng.uniform(7.8, 8.0)
                        })

        return events

    def _generate_exfiltration(self) -> List[Dict[str, Any]]:
        """Generate data exfiltration telemetry."""
        events = []

        # Large outbound transfers
        c2_servers = ["185.220.101.45", "192.168.1.100", "10.10.10.10"]

        for i in range(self.rng.randint(5, 20)):
            events.append({
                "timestamp": (self.current_time + timedelta(seconds=i*30)).isoformat(),
                "event_type": "network_connection",
                "direction": "outbound",
                "dest_ip": self.rng.choice(c2_servers),
                "dest_port": 443,
                "bytes_sent": self.rng.randint(1000000, 50000000),
                "pid": self._get_pid(),
                "process_name": "svchost.exe"  # Often uses legitimate process
            })

        return events

    def _generate_encryption_phase(self, characteristics: Dict) -> List[Dict[str, Any]]:
        """Generate file encryption telemetry."""
        events = []

        # Encryption speed varies by family
        speed_map = {
            "fast": 100,  # files per second
            "medium": 50,
            "slow": 10
        }

        files_per_second = speed_map.get(characteristics.get("speed", "medium"), 50)
        total_files = self.rng.randint(500, 5000)

        for i in range(total_files):
            timestamp = self.current_time + timedelta(seconds=i/files_per_second)

            # Original file read
            events.append({
                "timestamp": timestamp.isoformat(),
                "event_type": "file_read",
                "file_path": f"C:\\Users\\user\\Documents\\important_{i}.docx",
                "pid": self._get_pid(),
                "bytes_read": self.rng.randint(10000, 1000000)
            })

            # Encrypted file write
            events.append({
                "timestamp": (timestamp + timedelta(milliseconds=50)).isoformat(),
                "event_type": "file_write",
                "file_path": f"C:\\Users\\user\\Documents\\important_{i}.docx{characteristics.get('encrypted_extension', '.encrypted')}",
                "pid": self._get_pid(),
                "entropy_before": self.rng.uniform(2.0, 4.0),
                "entropy_after": self.rng.uniform(7.8, 8.0),
                "bytes_written": self.rng.randint(10000, 1000000)
            })

            # Original file deletion
            events.append({
                "timestamp": (timestamp + timedelta(milliseconds=100)).isoformat(),
                "event_type": "file_delete",
                "file_path": f"C:\\Users\\user\\Documents\\important_{i}.docx",
                "pid": self._get_pid()
            })

        # Drop ransom note
        events.append({
            "timestamp": (self.current_time + timedelta(seconds=total_files/files_per_second)).isoformat(),
            "event_type": "file_create",
            "file_path": f"C:\\Users\\user\\Desktop\\{characteristics.get('ransom_note', 'README.txt')}",
            "pid": self._get_pid(),
            "content_snippet": "Your files have been encrypted. Contact us at..."
        })

        return events

    def _get_pid(self) -> int:
        """Get a new PID."""
        self.pid_counter += 1
        return self.pid_counter


def generate_atomic_dataset(
    output_file: str,
    num_campaigns: int = 100,
    include_benign: bool = True,
    seed: Optional[int] = None
):
    """Generate a dataset of ransomware campaigns based on Atomic Red Team."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = AtomicTelemetryGenerator(seed=seed)

    with open(output_path, 'w') as f:
        for i in range(num_campaigns):
            # Mix ransomware campaigns with benign windows
            if include_benign and i % 3 == 0:
                # Generate benign window
                window_data = {
                    "window_id": i,
                    "label": "benign",
                    "family": "none",
                    "events": []  # Would add benign events here
                }
            else:
                # Generate ransomware campaign
                family = generator.rng.choice(list(RANSOMWARE_FAMILIES.keys()))
                events = generator.generate_ransomware_campaign(family=family)

                window_data = {
                    "window_id": i,
                    "label": "ransomware",
                    "family": family,
                    "start_time": events[0]["timestamp"] if events else generator.current_time.isoformat(),
                    "end_time": events[-1]["timestamp"] if events else generator.current_time.isoformat(),
                    "events": events,
                    "event_count": len(events),
                    "techniques_used": RANSOMWARE_FAMILIES[family]["techniques"],
                    "characteristics": RANSOMWARE_FAMILIES[family]["characteristics"]
                }

            f.write(json.dumps(window_data) + '\n')

            # Advance time for next campaign
            generator.current_time += timedelta(hours=generator.rng.randint(1, 24))

    print(f"Generated {num_campaigns} campaigns in {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Atomic Red Team based telemetry")
    parser.add_argument("--output", default="atomic_telemetry.jsonl", help="Output file")
    parser.add_argument("--campaigns", type=int, default=100, help="Number of campaigns")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    generate_atomic_dataset(
        output_file=args.output,
        num_campaigns=args.campaigns,
        seed=args.seed
    )