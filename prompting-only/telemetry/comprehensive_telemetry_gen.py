#!/usr/bin/env python3
"""
Comprehensive telemetry generator with latent state tracking and expert annotations.
Designed for fine-tuning LLMs to understand and explain ransomware behavior.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Latent states (hidden but true)
class LatentState(Enum):
    BENIGN = "benign"
    INITIAL_COMPROMISE = "initial_compromise"
    ESTABLISHING_FOOTHOLD = "establishing_foothold"
    RECONNAISSANCE = "reconnaissance"
    CREDENTIAL_THEFT = "credential_theft"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_STAGING = "data_staging"
    EXFILTRATION = "exfiltration"
    DEFENSE_EVASION = "defense_evasion"
    PRE_ENCRYPTION = "pre_encryption"
    ENCRYPTION_ACTIVE = "encryption_active"
    POST_ENCRYPTION = "post_encryption"

# Observable indicators (what telemetry can see)
@dataclass
class ObservableEvent:
    """What we can actually see in telemetry."""
    timestamp: str
    event_type: str  # process, file, network, registry
    details: Dict[str, Any]

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            **self.details
        }

@dataclass
class LatentContext:
    """Hidden context about what's really happening."""
    true_state: LatentState
    attacker_intent: str
    attack_phase: str
    threat_actor: str
    confidence: float
    next_likely_action: str
    time_to_impact: Optional[int]  # minutes
    kill_chain_position: str  # early, middle, late

    def to_dict(self):
        return {
            "true_state": self.true_state.value,
            "attacker_intent": self.attacker_intent,
            "attack_phase": self.attack_phase,
            "threat_actor": self.threat_actor,
            "confidence": self.confidence,
            "next_likely_action": self.next_likely_action,
            "time_to_impact": self.time_to_impact,
            "kill_chain_position": self.kill_chain_position
        }

@dataclass
class ExpertAnnotation:
    """Expert analysis and Q&A about the telemetry."""
    plain_english_summary: str
    risk_assessment: str  # low, medium, high, critical
    key_indicators: List[str]
    missing_context: List[str]  # What we wish we could see
    recommended_actions: List[str]
    qa_pairs: List[Dict[str, str]]
    confidence_explanation: str

    def to_dict(self):
        return asdict(self)

@dataclass
class TelemetryWindow:
    """Complete telemetry window with all information."""
    window_id: str
    start_time: str
    end_time: str

    # Observable (what the LLM sees)
    observable_events: List[ObservableEvent]

    # Latent (ground truth for training)
    latent_context: LatentContext

    # Expert annotations (what we want LLM to predict)
    expert_annotation: ExpertAnnotation

    # Metadata
    is_malicious: bool
    ransomware_family: Optional[str]

    def to_training_format(self) -> Dict[str, Any]:
        """Format for fine-tuning."""
        return {
            "window_id": self.window_id,
            "timeframe": {
                "start": self.start_time,
                "end": self.end_time
            },
            "observable_telemetry": [e.to_dict() for e in self.observable_events],
            "latent_state": self.latent_context.to_dict(),
            "expert_analysis": self.expert_annotation.to_dict(),
            "labels": {
                "is_malicious": self.is_malicious,
                "ransomware_family": self.ransomware_family
            }
        }

class ComprehensiveTelemetryGenerator:
    """Generate rich telemetry with latent states and expert annotations."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.current_time = datetime.now()
        self.pid_counter = 10000

        # Expert knowledge base
        self.expert_knowledge = self._load_expert_knowledge()

    def _load_expert_knowledge(self) -> Dict:
        """Load expert patterns and explanations."""
        return {
            "process_patterns": {
                "powershell.exe -enc": {
                    "risk": "high",
                    "explanation": "Base64 encoded PowerShell is often used to hide malicious commands",
                    "context": "Check parent process and network connections"
                },
                "vssadmin delete shadows": {
                    "risk": "critical",
                    "explanation": "Shadow copy deletion is a strong ransomware indicator",
                    "context": "Ransomware deletes backups to prevent recovery"
                },
                "whoami /all": {
                    "risk": "medium",
                    "explanation": "System enumeration, could be legitimate admin or attacker recon",
                    "context": "Look for other discovery commands in sequence"
                }
            },
            "file_patterns": {
                "mass_encryption": {
                    "indicator": "Multiple files changing to high entropy",
                    "risk": "critical",
                    "explanation": "Files being encrypted in bulk"
                },
                "ransom_note": {
                    "indicator": "README.txt or similar created in multiple directories",
                    "risk": "critical",
                    "explanation": "Ransom note deployment indicates encryption complete"
                }
            },
            "network_patterns": {
                "c2_beacon": {
                    "indicator": "Periodic outbound HTTPS to uncommon IP",
                    "risk": "high",
                    "explanation": "Command and control communication"
                },
                "data_exfil": {
                    "indicator": "Large outbound transfer (>100MB) to external IP",
                    "risk": "high",
                    "explanation": "Potential data exfiltration before encryption"
                }
            }
        }

    def generate_telemetry_sequence(
        self,
        num_windows: int = 10,
        window_duration_seconds: int = 120,
        scenario: str = "ransomware_attack"
    ) -> List[TelemetryWindow]:
        """Generate a sequence of telemetry windows."""

        windows = []

        if scenario == "ransomware_attack":
            # Generate a complete ransomware attack sequence
            attack_stages = [
                (LatentState.BENIGN, 2),  # Normal activity
                (LatentState.INITIAL_COMPROMISE, 1),
                (LatentState.RECONNAISSANCE, 2),
                (LatentState.CREDENTIAL_THEFT, 1),
                (LatentState.DEFENSE_EVASION, 1),
                (LatentState.DATA_STAGING, 1),
                (LatentState.EXFILTRATION, 1),
                (LatentState.ENCRYPTION_ACTIVE, 1)
            ]

            for stage, num_windows_in_stage in attack_stages:
                for _ in range(num_windows_in_stage):
                    window = self._generate_window(
                        latent_state=stage,
                        window_duration=timedelta(seconds=window_duration_seconds)
                    )
                    windows.append(window)
                    if len(windows) >= num_windows:
                        break

        elif scenario == "benign":
            # Generate normal activity
            for _ in range(num_windows):
                window = self._generate_window(
                    latent_state=LatentState.BENIGN,
                    window_duration=timedelta(seconds=window_duration_seconds)
                )
                windows.append(window)

        return windows

    def _generate_window(
        self,
        latent_state: LatentState,
        window_duration: timedelta
    ) -> TelemetryWindow:
        """Generate a single telemetry window."""

        window_start = self.current_time
        window_end = self.current_time + window_duration

        # Generate observable events based on latent state
        observable_events = self._generate_observables(latent_state, window_duration)

        # Create latent context (ground truth)
        latent_context = self._create_latent_context(latent_state)

        # Generate expert annotation
        expert_annotation = self._generate_expert_annotation(
            observable_events,
            latent_context
        )

        # Determine if malicious
        is_malicious = latent_state != LatentState.BENIGN
        ransomware_family = "LockBit3" if is_malicious else None

        window = TelemetryWindow(
            window_id=f"window_{self.rng.randint(10000, 99999)}",
            start_time=window_start.isoformat(),
            end_time=window_end.isoformat(),
            observable_events=observable_events,
            latent_context=latent_context,
            expert_annotation=expert_annotation,
            is_malicious=is_malicious,
            ransomware_family=ransomware_family
        )

        self.current_time = window_end
        return window

    def _generate_observables(
        self,
        state: LatentState,
        duration: timedelta
    ) -> List[ObservableEvent]:
        """Generate observable events based on latent state."""

        events = []

        # Add benign noise regardless of state
        events.extend(self._generate_benign_noise())

        # State-specific observables (partially observable!)
        if state == LatentState.INITIAL_COMPROMISE:
            # Only 40% chance of observing initial compromise
            if self.rng.random() < 0.4:
                events.append(ObservableEvent(
                    timestamp=self.current_time.isoformat(),
                    event_type="process",
                    details={
                        "process_name": "powershell.exe",
                        "parent_process": "winword.exe",
                        "command_line": "powershell.exe -enc SGVsbG8gV29ybGQ=",
                        "pid": self._get_pid()
                    }
                ))

        elif state == LatentState.RECONNAISSANCE:
            # 60% observable
            if self.rng.random() < 0.6:
                recon_commands = [
                    "whoami /all",
                    "net user",
                    "net localgroup administrators",
                    "systeminfo"
                ]
                for cmd in self.rng.sample(recon_commands, k=2):
                    events.append(ObservableEvent(
                        timestamp=(self.current_time + timedelta(seconds=self.rng.randint(1, 30))).isoformat(),
                        event_type="process",
                        details={
                            "process_name": "cmd.exe",
                            "command_line": cmd,
                            "pid": self._get_pid()
                        }
                    ))

        elif state == LatentState.DEFENSE_EVASION:
            # 70% observable
            if self.rng.random() < 0.7:
                events.append(ObservableEvent(
                    timestamp=self.current_time.isoformat(),
                    event_type="process",
                    details={
                        "process_name": "vssadmin.exe",
                        "command_line": "vssadmin delete shadows /all /quiet",
                        "pid": self._get_pid()
                    }
                ))
                events.append(ObservableEvent(
                    timestamp=self.current_time.isoformat(),
                    event_type="registry",
                    details={
                        "key": "HKLM\\SOFTWARE\\Microsoft\\Windows Defender",
                        "value": "DisableRealtimeMonitoring",
                        "data": "1",
                        "operation": "set"
                    }
                ))

        elif state == LatentState.EXFILTRATION:
            # 50% observable (encrypted traffic)
            if self.rng.random() < 0.5:
                events.append(ObservableEvent(
                    timestamp=self.current_time.isoformat(),
                    event_type="network",
                    details={
                        "direction": "outbound",
                        "destination_ip": "185.220.101.45",
                        "destination_port": 443,
                        "bytes_sent": self.rng.randint(100000000, 500000000),
                        "protocol": "HTTPS",
                        "pid": self._get_pid()
                    }
                ))

        elif state == LatentState.ENCRYPTION_ACTIVE:
            # 90% observable (very noisy)
            if self.rng.random() < 0.9:
                for i in range(self.rng.randint(50, 200)):
                    filename = f"document_{i}.docx"
                    events.append(ObservableEvent(
                        timestamp=(self.current_time + timedelta(milliseconds=i*100)).isoformat(),
                        event_type="file",
                        details={
                            "operation": "read",
                            "path": f"C:\\Users\\user\\Documents\\{filename}",
                            "pid": self._get_pid()
                        }
                    ))
                    events.append(ObservableEvent(
                        timestamp=(self.current_time + timedelta(milliseconds=i*100+50)).isoformat(),
                        event_type="file",
                        details={
                            "operation": "write",
                            "path": f"C:\\Users\\user\\Documents\\{filename}.locked",
                            "entropy_before": 3.2,
                            "entropy_after": 7.95,
                            "pid": self._get_pid()
                        }
                    ))

        return events

    def _generate_benign_noise(self) -> List[ObservableEvent]:
        """Generate normal system activity."""
        events = []

        # Browser activity
        if self.rng.random() < 0.3:
            events.append(ObservableEvent(
                timestamp=self.current_time.isoformat(),
                event_type="process",
                details={
                    "process_name": "chrome.exe",
                    "parent_process": "explorer.exe",
                    "pid": self._get_pid()
                }
            ))

        # Office activity
        if self.rng.random() < 0.2:
            events.append(ObservableEvent(
                timestamp=self.current_time.isoformat(),
                event_type="file",
                details={
                    "operation": "read",
                    "path": f"C:\\Users\\user\\Documents\\report.docx",
                    "pid": self._get_pid()
                }
            ))

        return events

    def _create_latent_context(self, state: LatentState) -> LatentContext:
        """Create the hidden ground truth context."""

        context_map = {
            LatentState.BENIGN: {
                "intent": "Normal user activity",
                "phase": "No attack",
                "actor": "Legitimate user",
                "next_action": "Continue normal work",
                "time_to_impact": None,
                "kill_chain": "none"
            },
            LatentState.INITIAL_COMPROMISE: {
                "intent": "Establish initial foothold",
                "phase": "Initial Access",
                "actor": "LockBit affiliate",
                "next_action": "System reconnaissance",
                "time_to_impact": 30,
                "kill_chain": "early"
            },
            LatentState.RECONNAISSANCE: {
                "intent": "Map network and identify targets",
                "phase": "Discovery",
                "actor": "LockBit affiliate",
                "next_action": "Credential theft",
                "time_to_impact": 25,
                "kill_chain": "early"
            },
            LatentState.DEFENSE_EVASION: {
                "intent": "Disable security controls",
                "phase": "Defense Evasion",
                "actor": "LockBit affiliate",
                "next_action": "Data exfiltration",
                "time_to_impact": 15,
                "kill_chain": "middle"
            },
            LatentState.EXFILTRATION: {
                "intent": "Steal data for double extortion",
                "phase": "Exfiltration",
                "actor": "LockBit affiliate",
                "next_action": "Deploy ransomware",
                "time_to_impact": 10,
                "kill_chain": "late"
            },
            LatentState.ENCRYPTION_ACTIVE: {
                "intent": "Encrypt files for ransom",
                "phase": "Impact",
                "actor": "LockBit ransomware",
                "next_action": "Drop ransom note",
                "time_to_impact": 0,
                "kill_chain": "late"
            }
        }

        ctx = context_map.get(state, context_map[LatentState.BENIGN])

        return LatentContext(
            true_state=state,
            attacker_intent=ctx["intent"],
            attack_phase=ctx["phase"],
            threat_actor=ctx["actor"],
            confidence=self.rng.uniform(0.7, 0.95),
            next_likely_action=ctx["next_action"],
            time_to_impact=ctx["time_to_impact"],
            kill_chain_position=ctx["kill_chain"]
        )

    def _generate_expert_annotation(
        self,
        events: List[ObservableEvent],
        context: LatentContext
    ) -> ExpertAnnotation:
        """Generate expert analysis of the telemetry."""

        # Analyze events for key indicators
        key_indicators = []
        risk = "low"

        for event in events:
            if event.event_type == "process":
                if "vssadmin" in str(event.details.get("command_line", "")):
                    key_indicators.append("Shadow copy deletion detected")
                    risk = "critical"
                elif "powershell.exe -enc" in str(event.details.get("command_line", "")):
                    key_indicators.append("Encoded PowerShell execution")
                    risk = "high" if risk != "critical" else risk
                elif "whoami" in str(event.details.get("command_line", "")):
                    key_indicators.append("System enumeration activity")
                    risk = "medium" if risk == "low" else risk

            elif event.event_type == "file":
                if event.details.get("entropy_after", 0) > 7.5:
                    key_indicators.append("File encryption detected (high entropy)")
                    risk = "critical"

            elif event.event_type == "network":
                if event.details.get("bytes_sent", 0) > 100000000:
                    key_indicators.append("Large data transfer detected")
                    risk = "high" if risk != "critical" else risk

        # Generate plain English summary
        if context.true_state == LatentState.BENIGN:
            summary = "Normal system activity observed with routine user operations."
        elif context.true_state == LatentState.ENCRYPTION_ACTIVE:
            summary = "CRITICAL: Active ransomware encryption detected. Multiple files being encrypted with high entropy changes."
        elif context.true_state == LatentState.DEFENSE_EVASION:
            summary = "Security controls being disabled. Shadow copies deleted. Likely ransomware preparation."
        else:
            summary = f"Suspicious activity detected consistent with {context.attack_phase} phase of attack."

        # Generate Q&A pairs
        qa_pairs = [
            {
                "question": "What is happening in this telemetry window?",
                "answer": summary
            },
            {
                "question": "What stage of attack is this?",
                "answer": f"This appears to be the {context.attack_phase} stage. {context.attacker_intent}."
            },
            {
                "question": "What should we do next?",
                "answer": self._get_recommended_action(context.true_state)
            },
            {
                "question": "How confident are you in this assessment?",
                "answer": f"Confidence: {context.confidence:.0%}. Based on {len(key_indicators)} key indicators observed."
            },
            {
                "question": "What is likely to happen next?",
                "answer": f"Next likely action: {context.next_likely_action}. Estimated time to impact: {context.time_to_impact or 'Unknown'} minutes."
            }
        ]

        # Recommended actions
        actions = self._get_recommended_actions(context.true_state, risk)

        # What we wish we could see
        missing_context = self._get_missing_context(context.true_state)

        return ExpertAnnotation(
            plain_english_summary=summary,
            risk_assessment=risk,
            key_indicators=key_indicators if key_indicators else ["No suspicious indicators detected"],
            missing_context=missing_context,
            recommended_actions=actions,
            qa_pairs=qa_pairs,
            confidence_explanation=f"Assessment based on {len(events)} observed events with {len(key_indicators)} suspicious indicators"
        )

    def _get_recommended_action(self, state: LatentState) -> str:
        """Get single recommended action for state."""
        actions = {
            LatentState.BENIGN: "Continue normal monitoring.",
            LatentState.INITIAL_COMPROMISE: "Investigate the suspicious process execution and check for persistence mechanisms.",
            LatentState.RECONNAISSANCE: "Alert SOC team. Begin containment planning. Check for lateral movement.",
            LatentState.DEFENSE_EVASION: "URGENT: Isolate affected systems immediately. Security controls are being disabled.",
            LatentState.EXFILTRATION: "Block outbound traffic to suspicious IPs. Check what data was accessed.",
            LatentState.ENCRYPTION_ACTIVE: "CRITICAL: Disconnect from network NOW. Initiate incident response plan."
        }
        return actions.get(state, "Investigate further.")

    def _get_recommended_actions(self, state: LatentState, risk: str) -> List[str]:
        """Get list of recommended actions."""
        if state == LatentState.BENIGN:
            return ["Continue routine monitoring", "No immediate action required"]
        elif state == LatentState.ENCRYPTION_ACTIVE:
            return [
                "Immediately isolate affected systems",
                "Disconnect from network",
                "Activate incident response team",
                "Begin recovery procedures",
                "Preserve forensic evidence"
            ]
        elif risk == "high" or risk == "critical":
            return [
                "Alert security team",
                "Isolate suspicious processes",
                "Block suspicious network connections",
                "Take memory dump for analysis",
                "Check for persistence mechanisms"
            ]
        else:
            return [
                "Increase monitoring",
                "Collect additional telemetry",
                "Review user activity"
            ]

    def _get_missing_context(self, state: LatentState) -> List[str]:
        """What additional telemetry would help?"""
        missing = {
            LatentState.BENIGN: [],
            LatentState.INITIAL_COMPROMISE: [
                "Email logs to identify phishing vector",
                "DNS queries before compromise",
                "User authentication logs"
            ],
            LatentState.RECONNAISSANCE: [
                "Full command history",
                "Network scan attempts",
                "Credential access attempts"
            ],
            LatentState.DEFENSE_EVASION: [
                "Service modification events",
                "Scheduled task creation",
                "WMI activity"
            ],
            LatentState.EXFILTRATION: [
                "List of files accessed",
                "Data staging locations",
                "Encryption keys used"
            ],
            LatentState.ENCRYPTION_ACTIVE: [
                "List of all affected files",
                "Encryption algorithm details",
                "Ransom note contents"
            ]
        }
        return missing.get(state, ["Additional context needed"])

    def _get_pid(self) -> int:
        """Get new PID."""
        self.pid_counter += 1
        return self.pid_counter


def generate_training_dataset(
    output_file: str,
    num_sequences: int = 100,
    windows_per_sequence: int = 10
):
    """Generate training dataset for fine-tuning."""

    generator = ComprehensiveTelemetryGenerator(seed=42)

    with open(output_file, 'w') as f:
        for i in range(num_sequences):
            # Mix attack and benign sequences
            if i % 3 == 0:
                scenario = "benign"
            else:
                scenario = "ransomware_attack"

            sequence = generator.generate_telemetry_sequence(
                num_windows=windows_per_sequence,
                scenario=scenario
            )

            # Write each window as training example
            for window in sequence:
                training_data = window.to_training_format()
                f.write(json.dumps(training_data) + '\n')

    print(f"Generated {num_sequences * windows_per_sequence} training windows")


if __name__ == "__main__":
    # Test generation
    generator = ComprehensiveTelemetryGenerator()

    # Generate one attack sequence
    sequence = generator.generate_telemetry_sequence(
        num_windows=5,
        scenario="ransomware_attack"
    )

    # Display first window
    print(json.dumps(sequence[0].to_training_format(), indent=2))