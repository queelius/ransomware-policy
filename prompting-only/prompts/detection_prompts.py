#!/usr/bin/env python3
"""
Prompting strategies for ransomware detection using LLMs.
No fine-tuning - just careful prompt engineering.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class DetectionAction(Enum):
    IGNORE = "ignore"
    MONITOR = "monitor"
    ALERT = "alert"
    QUARANTINE = "quarantine"
    BLOCK = "block"

@dataclass
class DetectionResult:
    action: DetectionAction
    confidence: float
    reasoning: str
    evidence: List[str]
    predicted_state: str
    recommended_next_steps: List[str]

class RansomwareDetectionPrompts:
    """Collection of prompting strategies for ransomware detection."""

    @staticmethod
    def zero_shot_detection(telemetry_window: Dict[str, Any]) -> str:
        """Basic zero-shot prompt for detection."""

        events_summary = RansomwareDetectionPrompts._summarize_events(telemetry_window['events'])

        prompt = f"""You are a cybersecurity expert analyzing system telemetry for ransomware detection.

TELEMETRY WINDOW:
Time Range: {telemetry_window.get('start_time')} to {telemetry_window.get('end_time')}
Total Events: {telemetry_window.get('event_count', 0)}

EVENT SUMMARY:
{events_summary}

TASK:
Analyze this telemetry window for signs of ransomware activity. Consider:
1. Process behavior patterns
2. File system activity (especially encryption indicators)
3. Network communications
4. System modifications (registry, services)

Provide your analysis in the following format:
ACTION: [ignore/monitor/alert/quarantine/block]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation]
EVIDENCE: [List key suspicious events]
PREDICTED_STATE: [clean/reconnaissance/staging/execution/impact]
NEXT_STEPS: [Recommended actions]
"""
        return prompt

    @staticmethod
    def few_shot_detection(telemetry_window: Dict[str, Any]) -> str:
        """Few-shot prompt with examples of ransomware patterns."""

        events_summary = RansomwareDetectionPrompts._summarize_events(telemetry_window['events'])

        prompt = f"""You are a cybersecurity expert analyzing system telemetry for ransomware detection.

EXAMPLE RANSOMWARE PATTERNS:

Example 1 - RECONNAISSANCE:
- Process: powershell.exe running "whoami /all" and "net user"
- Process: wmic.exe querying system information
- Network: Minimal outbound connections
- ACTION: monitor
- REASONING: System enumeration suggests early attack stage

Example 2 - STAGING:
- Process: powershell.exe with encoded command (-enc parameter)
- File: New executable written to AppData\\Local\\Temp\\
- Network: Large download from suspicious IP
- ACTION: alert
- REASONING: Payload delivery and staging detected

Example 3 - EXECUTION/IMPACT:
- Process: vssadmin.exe deleting shadow copies
- Files: Multiple files changing from low to high entropy (>7.8)
- Files: Original files deleted, new files with .locked extension
- Registry: Windows Defender disabled
- ACTION: block
- REASONING: Active ransomware encryption detected

CURRENT TELEMETRY WINDOW:
Time Range: {telemetry_window.get('start_time')} to {telemetry_window.get('end_time')}
Total Events: {telemetry_window.get('event_count', 0)}

EVENT SUMMARY:
{events_summary}

Analyze this telemetry and provide:
ACTION: [ignore/monitor/alert/quarantine/block]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation]
EVIDENCE: [List key suspicious events]
PREDICTED_STATE: [clean/reconnaissance/staging/execution/impact]
NEXT_STEPS: [Recommended actions]
"""
        return prompt

    @staticmethod
    def chain_of_thought_detection(telemetry_window: Dict[str, Any]) -> str:
        """Chain-of-thought prompting for step-by-step analysis."""

        events_summary = RansomwareDetectionPrompts._summarize_events(telemetry_window['events'])

        prompt = f"""You are a cybersecurity expert analyzing system telemetry for ransomware detection.

TELEMETRY WINDOW:
Time Range: {telemetry_window.get('start_time')} to {telemetry_window.get('end_time')}
Total Events: {telemetry_window.get('event_count', 0)}

EVENT SUMMARY:
{events_summary}

Let's analyze this step-by-step:

Step 1: PROCESS ANALYSIS
- Are there any suspicious process spawns or process chains?
- Look for: powershell.exe, cmd.exe, wmic.exe, vssadmin.exe, bcdedit.exe
- Check for encoded commands, unusual parent-child relationships

Step 2: FILE SYSTEM ANALYSIS
- Are files being modified rapidly?
- Check entropy changes (high entropy >7.5 suggests encryption)
- Look for: mass file renames, deletion patterns, ransom notes

Step 3: NETWORK ANALYSIS
- Any connections to known malicious IPs or unusual ports?
- Large data transfers that could indicate exfiltration?
- C2 communication patterns?

Step 4: SYSTEM MODIFICATIONS
- Registry changes to disable security tools?
- Service modifications or new persistence mechanisms?
- Shadow copy deletions or backup disruptions?

Step 5: TEMPORAL PATTERNS
- How quickly are events occurring?
- Is there an escalation pattern?
- Does the sequence suggest a kill chain progression?

Based on this analysis, provide:
ACTION: [ignore/monitor/alert/quarantine/block]
CONFIDENCE: [0.0-1.0]
REASONING: [Synthesis of the above analysis]
EVIDENCE: [Most critical indicators]
PREDICTED_STATE: [clean/reconnaissance/staging/execution/impact]
NEXT_STEPS: [Specific recommended actions]
"""
        return prompt

    @staticmethod
    def behavioral_pattern_detection(telemetry_window: Dict[str, Any]) -> str:
        """Focus on behavioral patterns rather than individual events."""

        # Extract behavioral features
        features = RansomwareDetectionPrompts._extract_behavioral_features(telemetry_window['events'])

        prompt = f"""You are a cybersecurity expert analyzing system behavior for ransomware detection.

BEHAVIORAL FEATURES EXTRACTED:
{json.dumps(features, indent=2)}

KNOWN RANSOMWARE BEHAVIORS:
1. ENCRYPTION BEHAVIOR:
   - Rapid file modifications with entropy increase
   - Sequential file access patterns
   - File extension changes to .locked, .encrypted, etc.

2. ANTI-FORENSICS:
   - Shadow copy deletion (vssadmin, wmic shadowcopy)
   - Event log clearing
   - Defender/AV disabling

3. LATERAL MOVEMENT:
   - SMB/RDP connections to internal IPs
   - Credential dumping (lsass.exe access)
   - Network scanning patterns

4. DATA EXFILTRATION:
   - Large outbound transfers before encryption
   - Connections to uncommon external IPs
   - Staged data in temp directories

5. PERSISTENCE:
   - Registry Run key modifications
   - Scheduled task creation
   - Service installation

Analyze the behavioral features and determine:
ACTION: [ignore/monitor/alert/quarantine/block]
CONFIDENCE: [0.0-1.0]
REASONING: [Which behaviors were detected and their significance]
EVIDENCE: [Specific behavioral indicators]
PREDICTED_STATE: [clean/reconnaissance/staging/execution/impact]
NEXT_STEPS: [Recommended response actions]
"""
        return prompt

    @staticmethod
    def contextual_analysis(
        telemetry_window: Dict[str, Any],
        previous_windows: List[Dict[str, Any]] = None,
        system_baseline: Dict[str, Any] = None
    ) -> str:
        """Analyze with historical context and baseline comparison."""

        events_summary = RansomwareDetectionPrompts._summarize_events(telemetry_window['events'])

        context_info = ""
        if previous_windows:
            context_info += f"\nPREVIOUS ACTIVITY: {len(previous_windows)} windows analyzed"
            context_info += f"\nRecent states: {[w.get('predicted_state', 'unknown') for w in previous_windows[-3:]]}"

        if system_baseline:
            context_info += f"\nBASELINE COMPARISON:"
            context_info += f"\n- Normal process count: {system_baseline.get('avg_process_count', 'unknown')}"
            context_info += f"\n- Normal network connections: {system_baseline.get('avg_network_count', 'unknown')}"
            context_info += f"\n- Normal file operations/min: {system_baseline.get('avg_file_ops', 'unknown')}"

        prompt = f"""You are a cybersecurity expert analyzing system telemetry with historical context.

CURRENT TELEMETRY:
Time Range: {telemetry_window.get('start_time')} to {telemetry_window.get('end_time')}
{events_summary}

HISTORICAL CONTEXT:
{context_info}

ANALYSIS CONSIDERATIONS:
1. Has there been a progression from previous windows?
2. Are current metrics significantly different from baseline?
3. Is this consistent with a multi-stage attack?
4. Could this be legitimate activity given the context?

Provide analysis:
ACTION: [ignore/monitor/alert/quarantine/block]
CONFIDENCE: [0.0-1.0]
REASONING: [Include historical context in reasoning]
EVIDENCE: [Key indicators considering context]
PREDICTED_STATE: [clean/reconnaissance/staging/execution/impact]
NEXT_STEPS: [Actions considering the progression]
"""
        return prompt

    @staticmethod
    def _summarize_events(events: List[Dict[str, Any]]) -> str:
        """Summarize events for inclusion in prompts."""
        if not events:
            return "No events in this window"

        # Group events by type
        process_events = [e for e in events if e.get('event_type') == 'process']
        file_events = [e for e in events if e.get('event_type') == 'file']
        network_events = [e for e in events if e.get('event_type') == 'network']
        registry_events = [e for e in events if e.get('event_type') == 'registry']

        summary = []

        # Process summary
        if process_events:
            suspicious_procs = [p for p in process_events
                               if any(s in p.get('process_name', '').lower()
                                     for s in ['powershell', 'cmd', 'wmic', 'vssadmin'])]
            summary.append(f"PROCESSES: {len(process_events)} total, {len(suspicious_procs)} potentially suspicious")
            for proc in suspicious_procs[:5]:  # Show first 5
                summary.append(f"  - {proc.get('process_name')}: {proc.get('command_line', '')[:100]}")

        # File summary
        if file_events:
            high_entropy = [f for f in file_events if f.get('entropy_after', 0) > 7.5]
            deletions = [f for f in file_events if f.get('operation') == 'delete']
            summary.append(f"FILES: {len(file_events)} total, {len(high_entropy)} high entropy, {len(deletions)} deletions")
            if high_entropy:
                summary.append(f"  - High entropy files: {len(high_entropy)} detected")

        # Network summary
        if network_events:
            external = [n for n in network_events
                       if not n.get('dest_ip', '').startswith(('192.168.', '10.', '172.'))]
            summary.append(f"NETWORK: {len(network_events)} total, {len(external)} external connections")

        # Registry summary
        if registry_events:
            security_keys = [r for r in registry_events
                           if 'defender' in r.get('key_path', '').lower()
                           or 'run' in r.get('key_path', '').lower()]
            summary.append(f"REGISTRY: {len(registry_events)} total, {len(security_keys)} security-related")

        return '\n'.join(summary)

    @staticmethod
    def _extract_behavioral_features(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract behavioral features from raw events."""
        features = {
            'temporal_features': {},
            'process_features': {},
            'file_features': {},
            'network_features': {},
            'system_features': {}
        }

        if not events:
            return features

        # Temporal features
        timestamps = [datetime.fromisoformat(e['timestamp']) for e in events if 'timestamp' in e]
        if timestamps:
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            features['temporal_features'] = {
                'event_rate': len(events) / max(time_span, 1),
                'time_span_seconds': time_span
            }

        # Process features
        process_events = [e for e in events if e.get('event_type') == 'process']
        if process_events:
            process_names = [e.get('process_name', '') for e in process_events]
            features['process_features'] = {
                'total_processes': len(process_events),
                'unique_processes': len(set(process_names)),
                'suspicious_tools': sum(1 for p in process_names
                                       if any(s in p.lower() for s in
                                             ['powershell', 'cmd', 'wmic', 'vssadmin', 'bcdedit'])),
                'encoded_commands': sum(1 for e in process_events
                                       if '-enc' in e.get('command_line', '').lower())
            }

        # File features
        file_events = [e for e in events if e.get('event_type') == 'file']
        if file_events:
            entropy_changes = []
            for e in file_events:
                if e.get('entropy_before') and e.get('entropy_after'):
                    entropy_changes.append(e['entropy_after'] - e['entropy_before'])

            features['file_features'] = {
                'total_file_ops': len(file_events),
                'high_entropy_files': sum(1 for e in file_events if e.get('entropy_after', 0) > 7.5),
                'deletions': sum(1 for e in file_events if e.get('operation') == 'delete'),
                'avg_entropy_change': sum(entropy_changes) / len(entropy_changes) if entropy_changes else 0,
                'extension_changes': sum(1 for e in file_events
                                        if e.get('extension_before') != e.get('extension_after'))
            }

        # Network features
        network_events = [e for e in events if e.get('event_type') == 'network']
        if network_events:
            features['network_features'] = {
                'total_connections': len(network_events),
                'unique_destinations': len(set(e.get('dest_ip') for e in network_events if e.get('dest_ip'))),
                'data_exfiltrated_bytes': sum(e.get('bytes_sent', 0) for e in network_events),
                'suspicious_ports': sum(1 for e in network_events
                                       if e.get('dest_port') in [445, 3389, 22, 23])
            }

        # System modification features
        registry_events = [e for e in events if e.get('event_type') == 'registry']
        features['system_features'] = {
            'registry_modifications': len(registry_events),
            'security_key_changes': sum(1 for e in registry_events
                                       if any(k in e.get('key_path', '').lower()
                                             for k in ['defender', 'run', 'security']))
        }

        return features


def analyze_telemetry(
    telemetry_window: Dict[str, Any],
    strategy: str = "chain_of_thought",
    llm_client=None
) -> DetectionResult:
    """Analyze telemetry window using specified prompting strategy."""

    prompts = RansomwareDetectionPrompts()

    # Select prompting strategy
    if strategy == "zero_shot":
        prompt = prompts.zero_shot_detection(telemetry_window)
    elif strategy == "few_shot":
        prompt = prompts.few_shot_detection(telemetry_window)
    elif strategy == "chain_of_thought":
        prompt = prompts.chain_of_thought_detection(telemetry_window)
    elif strategy == "behavioral":
        prompt = prompts.behavioral_pattern_detection(telemetry_window)
    else:
        prompt = prompts.chain_of_thought_detection(telemetry_window)

    # Here you would call your LLM
    # response = llm_client.generate(prompt)

    # For now, return a mock result
    return DetectionResult(
        action=DetectionAction.MONITOR,
        confidence=0.75,
        reasoning="Mock detection result - implement LLM integration",
        evidence=["suspicious process activity detected"],
        predicted_state="reconnaissance",
        recommended_next_steps=["Continue monitoring", "Collect additional telemetry"]
    )


if __name__ == "__main__":
    # Example usage
    sample_window = {
        "start_time": "2024-01-01T10:00:00",
        "end_time": "2024-01-01T10:02:00",
        "event_count": 45,
        "events": [
            {
                "event_type": "process",
                "timestamp": "2024-01-01T10:00:15",
                "process_name": "powershell.exe",
                "command_line": "powershell.exe -enc SGVsbG8gV29ybGQ="
            }
        ]
    }

    # Test different strategies
    strategies = ["zero_shot", "few_shot", "chain_of_thought", "behavioral"]

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {strategy} strategy:")
        print('='*60)

        prompt = getattr(RansomwareDetectionPrompts, f"{strategy}_detection")(sample_window)
        print(prompt[:500] + "...")  # Show first 500 chars