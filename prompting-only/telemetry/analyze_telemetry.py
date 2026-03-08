#!/usr/bin/env python3
"""
Analyze generated telemetry to verify quality and realism.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

def analyze_telemetry(filepath: str):
    """Analyze telemetry file for quality metrics."""

    path = Path(filepath)
    if not path.exists():
        print(f"Error: {filepath} not found")
        return

    stats = {
        'total_windows': 0,
        'benign_windows': 0,
        'ransomware_windows': 0,
        'families': Counter(),
        'event_types': Counter(),
        'techniques': Counter(),
        'total_events': 0,
        'events_per_window': [],
        'high_entropy_files': 0,
        'shadow_deletions': 0,
        'c2_connections': 0,
        'suspicious_processes': Counter()
    }

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                window = json.loads(line)
                stats['total_windows'] += 1

                # Count labels
                label = window.get('label', 'unknown')
                if label == 'benign':
                    stats['benign_windows'] += 1
                elif label == 'ransomware':
                    stats['ransomware_windows'] += 1
                    stats['families'][window.get('family', 'unknown')] += 1

                    # Count techniques
                    for technique in window.get('techniques_used', []):
                        stats['techniques'][technique] += 1

                # Analyze events
                events = window.get('events', [])
                num_events = len(events)
                stats['total_events'] += num_events
                stats['events_per_window'].append(num_events)

                for event in events:
                    event_type = event.get('event_type', 'unknown')
                    stats['event_types'][event_type] += 1

                    # Check for ransomware indicators
                    if 'vssadmin' in str(event.get('command_line', '')):
                        stats['shadow_deletions'] += 1

                    if event.get('entropy_after', 0) > 7.5:
                        stats['high_entropy_files'] += 1

                    if event_type == 'network_connection' and event.get('direction') == 'outbound':
                        stats['c2_connections'] += 1

                    # Track suspicious processes
                    process_name = event.get('process_name', '').lower()
                    if process_name in ['powershell.exe', 'cmd.exe', 'wmic.exe', 'vssadmin.exe']:
                        stats['suspicious_processes'][process_name] += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    # Calculate averages
    avg_events = sum(stats['events_per_window']) / len(stats['events_per_window']) if stats['events_per_window'] else 0

    # Print report
    print("\n" + "="*60)
    print("TELEMETRY QUALITY ANALYSIS REPORT")
    print("="*60)

    print(f"\nDataset: {filepath}")
    print(f"Total windows: {stats['total_windows']:,}")
    print(f"  - Benign: {stats['benign_windows']:,} ({stats['benign_windows']/stats['total_windows']*100:.1f}%)")
    print(f"  - Ransomware: {stats['ransomware_windows']:,} ({stats['ransomware_windows']/stats['total_windows']*100:.1f}%)")

    print(f"\nTotal events: {stats['total_events']:,}")
    print(f"Average events per window: {avg_events:.1f}")

    print("\nRansomware Families:")
    for family, count in stats['families'].most_common():
        print(f"  - {family}: {count} campaigns")

    print("\nATT&CK Techniques Used:")
    for technique, count in stats['techniques'].most_common(10):
        print(f"  - {technique}: {count} times")

    print("\nEvent Type Distribution:")
    for event_type, count in stats['event_types'].most_common():
        print(f"  - {event_type}: {count:,} events")

    print("\nRansomware Indicators Detected:")
    print(f"  - Shadow copy deletions: {stats['shadow_deletions']}")
    print(f"  - High entropy files: {stats['high_entropy_files']}")
    print(f"  - C2 connections: {stats['c2_connections']}")

    print("\nSuspicious Process Activity:")
    for process, count in stats['suspicious_processes'].most_common():
        print(f"  - {process}: {count:,} executions")

    # Quality assessment
    print("\n" + "="*60)
    print("QUALITY ASSESSMENT")
    print("="*60)

    quality_score = 0
    max_score = 0

    # Check for variety in families
    if len(stats['families']) >= 3:
        quality_score += 10
        print("✓ Good variety of ransomware families")
    else:
        print("✗ Limited ransomware family variety")
    max_score += 10

    # Check for realistic event counts
    if 100 <= avg_events <= 5000:
        quality_score += 10
        print("✓ Realistic event counts per window")
    else:
        print("✗ Unrealistic event counts")
    max_score += 10

    # Check for key indicators
    if stats['shadow_deletions'] > 0:
        quality_score += 10
        print("✓ Shadow deletion events present")
    else:
        print("✗ Missing shadow deletion events")
    max_score += 10

    if stats['high_entropy_files'] > 0:
        quality_score += 10
        print("✓ File encryption patterns detected")
    else:
        print("✗ Missing encryption indicators")
    max_score += 10

    if stats['c2_connections'] > 0:
        quality_score += 10
        print("✓ C2 communication patterns present")
    else:
        print("✗ Missing network indicators")
    max_score += 10

    # Check for balanced dataset
    ratio = stats['ransomware_windows'] / stats['total_windows'] if stats['total_windows'] > 0 else 0
    if 0.4 <= ratio <= 0.8:
        quality_score += 10
        print("✓ Good balance between benign and malicious")
    else:
        print("✗ Dataset imbalance detected")
    max_score += 10

    print(f"\nOverall Quality Score: {quality_score}/{max_score} ({quality_score/max_score*100:.1f}%)")

    if quality_score >= max_score * 0.8:
        print("Status: EXCELLENT - Ready for training")
    elif quality_score >= max_score * 0.6:
        print("Status: GOOD - Suitable for initial experiments")
    else:
        print("Status: NEEDS IMPROVEMENT - Review generation parameters")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_telemetry.py <telemetry_file.jsonl>")
        sys.exit(1)

    analyze_telemetry(sys.argv[1])