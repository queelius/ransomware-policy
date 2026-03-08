#!/usr/bin/env python3
"""
Real telemetry collection from live systems.
Supports Windows (via Sysmon/ETW) and Linux (via auditd/eBPF).
"""

import json
import subprocess
import platform
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

class TelemetryCollector:
    """Base class for OS-specific telemetry collectors."""

    def __init__(self, output_dir: str = "./telemetry_raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system = platform.system()

    def collect(self) -> List[Dict[str, Any]]:
        """Collect telemetry from the system."""
        raise NotImplementedError


class WindowsTelemetryCollector(TelemetryCollector):
    """Collects telemetry from Windows using Sysmon and WMI."""

    def __init__(self, output_dir: str = "./telemetry_raw"):
        super().__init__(output_dir)
        self.sysmon_config = self._generate_sysmon_config()

    def _generate_sysmon_config(self) -> str:
        """Generate Sysmon configuration for ransomware detection."""
        config = """<Sysmon schemaversion="4.90">
  <EventFiltering>
    <!-- Process Creation -->
    <RuleGroup name="ProcessCreate" groupRelation="or">
      <ProcessCreate onmatch="include">
        <CommandLine condition="contains any">
          vssadmin;bcdedit;wbadmin;shadowcopy;diskshadow
        </CommandLine>
        <CommandLine condition="contains">powershell -enc</CommandLine>
        <CommandLine condition="contains">bypass -nop -w hidden</CommandLine>
        <Image condition="contains any">
          powershell.exe;cmd.exe;wscript.exe;cscript.exe;mshta.exe;rundll32.exe
        </Image>
        <ParentImage condition="contains any">
          winword.exe;excel.exe;powerpnt.exe;outlook.exe;chrome.exe;firefox.exe
        </ParentImage>
      </ProcessCreate>
    </RuleGroup>

    <!-- File Creation -->
    <RuleGroup name="FileCreate" groupRelation="or">
      <FileCreate onmatch="include">
        <TargetFilename condition="contains any">
          .exe;.dll;.bat;.ps1;.vbs;.js
        </TargetFilename>
        <TargetFilename condition="contains any">
          AppData\\Roaming;AppData\\Local\\Temp;ProgramData
        </TargetFilename>
        <TargetFilename condition="end with any">
          .encrypted;.locked;.crypto;README;RECOVER;DECRYPT
        </TargetFilename>
      </FileCreate>
    </RuleGroup>

    <!-- Network Connections -->
    <RuleGroup name="NetworkConnect" groupRelation="or">
      <NetworkConnect onmatch="include">
        <Image condition="contains any">
          powershell.exe;cmd.exe;rundll32.exe;regsvr32.exe
        </Image>
        <DestinationPort condition="is any">
          445;3389;22;23;21;1433;3306
        </DestinationPort>
        <Initiated>true</Initiated>
      </NetworkConnect>
    </RuleGroup>

    <!-- Registry Events -->
    <RuleGroup name="RegistryEvent" groupRelation="or">
      <RegistryEvent onmatch="include">
        <TargetObject condition="contains any">
          CurrentVersion\\Run;CurrentVersion\\RunOnce;
          Windows Defender\\DisableAntiSpyware;
          System\\CurrentControlSet\\Services
        </TargetObject>
      </RegistryEvent>
    </RuleGroup>
  </EventFiltering>
</Sysmon>"""

        config_path = self.output_dir / "sysmon_config.xml"
        config_path.write_text(config)
        return str(config_path)

    def collect_sysmon_events(self, last_n_minutes: int = 5) -> List[Dict[str, Any]]:
        """Collect recent Sysmon events from Windows Event Log."""
        events = []

        try:
            # PowerShell command to get Sysmon events
            ps_script = f"""
            $events = Get-WinEvent -FilterHashtable @{{
                LogName='Microsoft-Windows-Sysmon/Operational'
                StartTime=(Get-Date).AddMinutes(-{last_n_minutes})
            }} -ErrorAction SilentlyContinue

            foreach ($event in $events) {{
                $event | ConvertTo-Json -Depth 5
            }}
            """

            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            event_json = json.loads(line)
                            events.append(self._parse_sysmon_event(event_json))
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            print(f"Error collecting Sysmon events: {e}")

        return events

    def _parse_sysmon_event(self, raw_event: Dict) -> Dict[str, Any]:
        """Parse raw Sysmon event into normalized format."""
        event_id = raw_event.get('Id', 0)
        timestamp = raw_event.get('TimeCreated', datetime.now().isoformat())

        parsed = {
            'timestamp': timestamp,
            'event_id': event_id,
            'provider': 'Sysmon',
            'raw': raw_event
        }

        # Parse based on event ID
        if event_id == 1:  # Process Create
            parsed['event_type'] = 'process_create'
            parsed['process_name'] = self._extract_field(raw_event, 'Image')
            parsed['command_line'] = self._extract_field(raw_event, 'CommandLine')
            parsed['pid'] = self._extract_field(raw_event, 'ProcessId')
            parsed['ppid'] = self._extract_field(raw_event, 'ParentProcessId')

        elif event_id == 3:  # Network Connection
            parsed['event_type'] = 'network_connect'
            parsed['src_ip'] = self._extract_field(raw_event, 'SourceIp')
            parsed['dest_ip'] = self._extract_field(raw_event, 'DestinationIp')
            parsed['dest_port'] = self._extract_field(raw_event, 'DestinationPort')

        elif event_id == 11:  # File Create
            parsed['event_type'] = 'file_create'
            parsed['file_path'] = self._extract_field(raw_event, 'TargetFilename')
            parsed['process_name'] = self._extract_field(raw_event, 'Image')

        elif event_id == 13:  # Registry Value Set
            parsed['event_type'] = 'registry_set'
            parsed['key_path'] = self._extract_field(raw_event, 'TargetObject')
            parsed['value'] = self._extract_field(raw_event, 'Details')

        return parsed

    def _extract_field(self, event: Dict, field_name: str) -> Optional[str]:
        """Extract field from Sysmon event message."""
        message = event.get('Message', '')
        pattern = f"{field_name}: ([^\\r\\n]+)"
        match = re.search(pattern, message)
        return match.group(1) if match else None

    def collect_wmi_processes(self) -> List[Dict[str, Any]]:
        """Collect current process information via WMI."""
        processes = []

        try:
            ps_script = """
            Get-WmiObject Win32_Process | Select-Object Name, ProcessId, ParentProcessId,
                CommandLine, CreationDate, WorkingSetSize | ConvertTo-Json
            """

            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.stdout:
                wmi_data = json.loads(result.stdout)
                if isinstance(wmi_data, list):
                    for proc in wmi_data:
                        processes.append({
                            'event_type': 'process_snapshot',
                            'timestamp': datetime.now().isoformat(),
                            'process_name': proc.get('Name'),
                            'pid': proc.get('ProcessId'),
                            'ppid': proc.get('ParentProcessId'),
                            'command_line': proc.get('CommandLine'),
                            'memory_usage': proc.get('WorkingSetSize')
                        })

        except Exception as e:
            print(f"Error collecting WMI data: {e}")

        return processes

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all available Windows telemetry."""
        all_events = []

        # Collect Sysmon events
        print("Collecting Sysmon events...")
        all_events.extend(self.collect_sysmon_events())

        # Collect WMI process snapshots
        print("Collecting process snapshots...")
        all_events.extend(self.collect_wmi_processes())

        # Save to file
        output_file = self.output_dir / f"windows_telemetry_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        with open(output_file, 'w') as f:
            for event in all_events:
                f.write(json.dumps(event) + '\n')

        print(f"Collected {len(all_events)} events to {output_file}")
        return all_events


class LinuxTelemetryCollector(TelemetryCollector):
    """Collects telemetry from Linux using auditd and /proc."""

    def collect_auditd_events(self) -> List[Dict[str, Any]]:
        """Parse recent auditd logs."""
        events = []

        try:
            # Get recent audit logs
            result = subprocess.run(
                ["ausearch", "-ts", "recent", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            event = json.loads(line)
                            events.append(self._parse_audit_event(event))
                        except json.JSONDecodeError:
                            continue

        except FileNotFoundError:
            print("auditd not found. Install with: sudo apt install auditd")
        except Exception as e:
            print(f"Error collecting audit events: {e}")

        return events

    def _parse_audit_event(self, audit_event: Dict) -> Dict[str, Any]:
        """Parse auditd event into normalized format."""
        event_type = audit_event.get('type', 'unknown')

        parsed = {
            'timestamp': datetime.fromtimestamp(
                float(audit_event.get('timestamp', 0))
            ).isoformat(),
            'event_type': event_type,
            'provider': 'auditd'
        }

        if event_type == 'EXECVE':
            parsed['event_type'] = 'process_create'
            parsed['command'] = audit_event.get('exe', '')
            parsed['pid'] = audit_event.get('pid', 0)

        elif event_type == 'CONNECT':
            parsed['event_type'] = 'network_connect'
            parsed['src_ip'] = audit_event.get('saddr', '')
            parsed['dest_port'] = audit_event.get('port', 0)

        return parsed

    def collect_proc_info(self) -> List[Dict[str, Any]]:
        """Collect process information from /proc."""
        processes = []

        try:
            proc_dir = Path('/proc')
            for pid_dir in proc_dir.iterdir():
                if not pid_dir.name.isdigit():
                    continue

                try:
                    cmdline_file = pid_dir / 'cmdline'
                    stat_file = pid_dir / 'stat'

                    if cmdline_file.exists():
                        cmdline = cmdline_file.read_text().replace('\x00', ' ').strip()

                        stat_content = stat_file.read_text() if stat_file.exists() else ''
                        ppid = self._extract_ppid_from_stat(stat_content)

                        processes.append({
                            'event_type': 'process_snapshot',
                            'timestamp': datetime.now().isoformat(),
                            'pid': int(pid_dir.name),
                            'ppid': ppid,
                            'command_line': cmdline
                        })

                except (PermissionError, FileNotFoundError):
                    continue

        except Exception as e:
            print(f"Error reading /proc: {e}")

        return processes

    def _extract_ppid_from_stat(self, stat_content: str) -> int:
        """Extract PPID from /proc/[pid]/stat content."""
        try:
            # Format: pid (comm) state ppid ...
            parts = stat_content.split(')')
            if len(parts) > 1:
                fields = parts[1].split()
                if len(fields) > 1:
                    return int(fields[1])
        except:
            pass
        return 0

    def collect_netstat(self) -> List[Dict[str, Any]]:
        """Collect network connections using netstat."""
        connections = []

        try:
            result = subprocess.run(
                ["netstat", "-tupn"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[2:]:  # Skip headers
                    parts = line.split()
                    if len(parts) >= 6:
                        connections.append({
                            'event_type': 'network_connection',
                            'timestamp': datetime.now().isoformat(),
                            'protocol': parts[0],
                            'local_addr': parts[3],
                            'remote_addr': parts[4],
                            'state': parts[5] if len(parts) > 5 else 'unknown'
                        })

        except Exception as e:
            print(f"Error collecting netstat data: {e}")

        return connections

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all available Linux telemetry."""
        all_events = []

        # Collect auditd events
        print("Collecting audit events...")
        all_events.extend(self.collect_auditd_events())

        # Collect process info
        print("Collecting process snapshots...")
        all_events.extend(self.collect_proc_info())

        # Collect network connections
        print("Collecting network connections...")
        all_events.extend(self.collect_netstat())

        # Save to file
        output_file = self.output_dir / f"linux_telemetry_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        with open(output_file, 'w') as f:
            for event in all_events:
                f.write(json.dumps(event) + '\n')

        print(f"Collected {len(all_events)} events to {output_file}")
        return all_events


def main():
    """Main entry point for telemetry collection."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect real system telemetry")
    parser.add_argument("--output", default="./telemetry_raw", help="Output directory")
    parser.add_argument("--continuous", action="store_true", help="Continuous collection mode")
    parser.add_argument("--interval", type=int, default=60, help="Collection interval in seconds")

    args = parser.parse_args()

    # Detect OS and create appropriate collector
    system = platform.system()
    if system == "Windows":
        collector = WindowsTelemetryCollector(args.output)
    elif system == "Linux":
        collector = LinuxTelemetryCollector(args.output)
    else:
        print(f"Unsupported OS: {system}")
        return

    if args.continuous:
        import time
        print(f"Starting continuous collection (interval: {args.interval}s)")
        try:
            while True:
                collector.collect()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nCollection stopped")
    else:
        collector.collect()


if __name__ == "__main__":
    main()