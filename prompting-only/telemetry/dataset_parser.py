#!/usr/bin/env python3
"""
Parser for public ransomware datasets to extract behavioral patterns.
Supports multiple dataset formats including RanSMAP, UGRansome, and others.
"""

import json
import csv
import struct
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class DatasetPattern:
    """Extracted pattern from ransomware dataset."""
    family: str
    behavior_type: str
    pattern_data: Dict[str, Any]
    confidence: float

class RanSMAPParser:
    """Parser for RanSMAP (Ransomware Storage and Memory Access Patterns) dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def parse(self) -> List[DatasetPattern]:
        """Parse RanSMAP dataset for storage/memory patterns."""
        patterns = []

        # RanSMAP contains low-level I/O patterns
        # Format: JSON lines with file access sequences

        if not self.dataset_path.exists():
            print(f"Warning: RanSMAP dataset not found at {self.dataset_path}")
            return patterns

        with open(self.dataset_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    pattern = self._extract_io_pattern(entry)
                    if pattern:
                        patterns.append(pattern)
                except json.JSONDecodeError:
                    continue

        return patterns

    def _extract_io_pattern(self, entry: Dict) -> Optional[DatasetPattern]:
        """Extract I/O pattern from RanSMAP entry."""

        # RanSMAP tracks:
        # - File access sequences (read/write patterns)
        # - Memory allocation patterns
        # - Entropy changes over time

        if "ransomware_family" not in entry:
            return None

        # Calculate access pattern metrics
        access_sequence = entry.get("file_access_sequence", [])

        pattern_data = {
            "read_write_ratio": self._calc_read_write_ratio(access_sequence),
            "sequential_access": self._is_sequential_access(access_sequence),
            "file_types_targeted": entry.get("file_extensions", []),
            "avg_entropy_change": entry.get("entropy_delta", 0),
            "files_per_second": entry.get("encryption_rate", 0),
            "memory_pattern": entry.get("memory_allocation_pattern", "unknown")
        }

        return DatasetPattern(
            family=entry["ransomware_family"],
            behavior_type="io_pattern",
            pattern_data=pattern_data,
            confidence=entry.get("confidence", 0.8)
        )

    def _calc_read_write_ratio(self, sequence: List[Dict]) -> float:
        """Calculate read/write ratio from access sequence."""
        if not sequence:
            return 1.0

        reads = sum(1 for op in sequence if op.get("operation") == "read")
        writes = sum(1 for op in sequence if op.get("operation") == "write")

        if writes == 0:
            return float('inf')
        return reads / writes

    def _is_sequential_access(self, sequence: List[Dict]) -> bool:
        """Detect if file access is sequential or random."""
        if len(sequence) < 2:
            return True

        # Check if file offsets are increasing
        offsets = [op.get("offset", 0) for op in sequence]
        sorted_offsets = sorted(offsets)
        return offsets == sorted_offsets


class UGRansomeParser:
    """Parser for UGRansome network traffic dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def parse(self) -> List[DatasetPattern]:
        """Parse UGRansome dataset for network patterns."""
        patterns = []

        if not self.dataset_path.exists():
            print(f"Warning: UGRansome dataset not found at {self.dataset_path}")
            return patterns

        # UGRansome uses CSV format with network flows
        with open(self.dataset_path, 'r') as f:
            reader = csv.DictReader(f)

            # Group by ransomware family
            family_flows = {}
            for row in reader:
                family = row.get("ransomware_family", "unknown")
                if family not in family_flows:
                    family_flows[family] = []
                family_flows[family].append(row)

            # Extract patterns per family
            for family, flows in family_flows.items():
                pattern = self._extract_network_pattern(family, flows)
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _extract_network_pattern(self, family: str, flows: List[Dict]) -> Optional[DatasetPattern]:
        """Extract network pattern from flow data."""

        if not flows:
            return None

        # Calculate network behavior metrics
        c2_ips = set()
        total_bytes_out = 0
        total_bytes_in = 0
        unique_ports = set()

        for flow in flows:
            if flow.get("direction") == "outbound":
                c2_ips.add(flow.get("dest_ip", ""))
                total_bytes_out += int(flow.get("bytes", 0))
            else:
                total_bytes_in += int(flow.get("bytes", 0))

            unique_ports.add(int(flow.get("dest_port", 0)))

        pattern_data = {
            "c2_servers": list(c2_ips),
            "num_c2_servers": len(c2_ips),
            "bytes_exfiltrated": total_bytes_out,
            "bytes_received": total_bytes_in,
            "exfil_ratio": total_bytes_out / max(total_bytes_in, 1),
            "unique_ports": list(unique_ports),
            "uses_tor": any(".onion" in ip or "tor" in ip.lower() for ip in c2_ips),
            "uses_standard_ports": 443 in unique_ports or 80 in unique_ports
        }

        return DatasetPattern(
            family=family,
            behavior_type="network_pattern",
            pattern_data=pattern_data,
            confidence=0.85
        )


class PEHeaderParser:
    """Parser for PE header ransomware datasets."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def parse(self) -> List[DatasetPattern]:
        """Parse PE header characteristics."""
        patterns = []

        if not self.dataset_path.exists():
            print(f"Warning: PE dataset not found at {self.dataset_path}")
            return patterns

        # Parse PE headers (first 1024 bytes)
        for pe_file in self.dataset_path.glob("*.pe"):
            with open(pe_file, 'rb') as f:
                header_bytes = f.read(1024)
                pattern = self._extract_pe_pattern(pe_file.stem, header_bytes)
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _extract_pe_pattern(self, family: str, header: bytes) -> Optional[DatasetPattern]:
        """Extract patterns from PE header."""

        if len(header) < 64:
            return None

        # Parse DOS header
        if header[:2] != b'MZ':
            return None

        # Get PE offset
        pe_offset = struct.unpack('<I', header[60:64])[0]

        if pe_offset + 24 > len(header):
            return None

        # Check PE signature
        if header[pe_offset:pe_offset+4] != b'PE\x00\x00':
            return None

        # Extract characteristics
        pattern_data = {
            "is_dll": False,  # Would check characteristics field
            "imports_crypto": self._check_crypto_imports(header),
            "has_manifest": b'manifest' in header.lower(),
            "is_packed": self._is_likely_packed(header),
            "entry_point_section": ".text",  # Would parse section headers
            "compile_timestamp": 0  # Would extract from header
        }

        return DatasetPattern(
            family=family,
            behavior_type="binary_characteristics",
            pattern_data=pattern_data,
            confidence=0.7
        )

    def _check_crypto_imports(self, header: bytes) -> bool:
        """Check if PE imports cryptographic APIs."""
        crypto_indicators = [
            b'crypt32', b'bcrypt', b'CryptEncrypt',
            b'CryptDecrypt', b'CryptGenKey', b'advapi32'
        ]
        return any(ind in header.lower() for ind in crypto_indicators)

    def _is_likely_packed(self, header: bytes) -> bool:
        """Simple entropy check for packing."""
        if len(header) < 256:
            return False

        # Calculate entropy of first 256 bytes after PE header
        entropy = self._calculate_entropy(header[128:384])
        return entropy > 7.0  # High entropy suggests packing/encryption

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy."""
        if not data:
            return 0.0

        # Count byte frequencies
        frequencies = np.bincount(np.frombuffer(data, dtype=np.uint8))
        probabilities = frequencies[frequencies > 0] / len(data)

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy


class BehaviorPatternLibrary:
    """Library of extracted ransomware behavioral patterns."""

    def __init__(self):
        self.patterns: Dict[str, List[DatasetPattern]] = {}
        self.telemetry_mappings: Dict[str, Dict] = {}

    def add_patterns(self, patterns: List[DatasetPattern]):
        """Add patterns to library."""
        for pattern in patterns:
            if pattern.family not in self.patterns:
                self.patterns[pattern.family] = []
            self.patterns[pattern.family].append(pattern)

    def map_to_telemetry(self, pattern: DatasetPattern) -> Dict[str, Any]:
        """Map dataset pattern to telemetry events."""

        telemetry = {
            "family": pattern.family,
            "events": []
        }

        if pattern.behavior_type == "io_pattern":
            # Convert I/O patterns to file events
            io_data = pattern.pattern_data

            # Generate file events based on pattern
            for i in range(int(io_data.get("files_per_second", 10))):
                telemetry["events"].append({
                    "event_type": "file_read",
                    "pattern": "sequential" if io_data.get("sequential_access") else "random"
                })
                telemetry["events"].append({
                    "event_type": "file_write",
                    "entropy_delta": io_data.get("avg_entropy_change", 5.0)
                })

        elif pattern.behavior_type == "network_pattern":
            # Convert network patterns to network events
            net_data = pattern.pattern_data

            for c2 in net_data.get("c2_servers", [])[:5]:  # Limit to 5 C2s
                telemetry["events"].append({
                    "event_type": "network_connect",
                    "dest_ip": c2,
                    "dest_port": net_data.get("unique_ports", [443])[0],
                    "bytes_sent": net_data.get("bytes_exfiltrated", 0) // max(len(net_data.get("c2_servers", [1])), 1)
                })

        elif pattern.behavior_type == "binary_characteristics":
            # Convert binary characteristics to process events
            bin_data = pattern.pattern_data

            telemetry["events"].append({
                "event_type": "process_create",
                "is_packed": bin_data.get("is_packed", False),
                "imports_crypto": bin_data.get("imports_crypto", False)
            })

        return telemetry

    def get_family_profile(self, family: str) -> Dict[str, Any]:
        """Get complete behavioral profile for a ransomware family."""

        if family not in self.patterns:
            return {}

        profile = {
            "family": family,
            "io_patterns": [],
            "network_patterns": [],
            "binary_characteristics": []
        }

        for pattern in self.patterns[family]:
            if pattern.behavior_type == "io_pattern":
                profile["io_patterns"].append(pattern.pattern_data)
            elif pattern.behavior_type == "network_pattern":
                profile["network_patterns"].append(pattern.pattern_data)
            elif pattern.behavior_type == "binary_characteristics":
                profile["binary_characteristics"].append(pattern.pattern_data)

        return profile

    def save_library(self, output_path: str):
        """Save pattern library to file."""
        library_data = {
            "patterns": {},
            "metadata": {
                "total_families": len(self.patterns),
                "total_patterns": sum(len(p) for p in self.patterns.values())
            }
        }

        for family, patterns in self.patterns.items():
            library_data["patterns"][family] = [
                {
                    "behavior_type": p.behavior_type,
                    "pattern_data": p.pattern_data,
                    "confidence": p.confidence
                }
                for p in patterns
            ]

        with open(output_path, 'w') as f:
            json.dump(library_data, f, indent=2)


def parse_all_datasets(config: Dict[str, str]) -> BehaviorPatternLibrary:
    """Parse all configured datasets and build pattern library."""

    library = BehaviorPatternLibrary()

    # Parse RanSMAP if available
    if "ransmap_path" in config:
        parser = RanSMAPParser(config["ransmap_path"])
        patterns = parser.parse()
        library.add_patterns(patterns)
        print(f"Parsed {len(patterns)} patterns from RanSMAP")

    # Parse UGRansome if available
    if "ugransome_path" in config:
        parser = UGRansomeParser(config["ugransome_path"])
        patterns = parser.parse()
        library.add_patterns(patterns)
        print(f"Parsed {len(patterns)} patterns from UGRansome")

    # Parse PE headers if available
    if "pe_dataset_path" in config:
        parser = PEHeaderParser(config["pe_dataset_path"])
        patterns = parser.parse()
        library.add_patterns(patterns)
        print(f"Parsed {len(patterns)} patterns from PE headers")

    return library


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse ransomware datasets")
    parser.add_argument("--ransmap", help="Path to RanSMAP dataset")
    parser.add_argument("--ugransome", help="Path to UGRansome dataset")
    parser.add_argument("--pe", help="Path to PE header dataset")
    parser.add_argument("--output", default="pattern_library.json", help="Output library file")

    args = parser.parse_args()

    config = {}
    if args.ransmap:
        config["ransmap_path"] = args.ransmap
    if args.ugransome:
        config["ugransome_path"] = args.ugransome
    if args.pe:
        config["pe_dataset_path"] = args.pe

    if not config:
        print("Note: No dataset paths provided. Using example patterns.")
        # Create example patterns for demonstration
        library = BehaviorPatternLibrary()

        example_patterns = [
            DatasetPattern(
                family="LockBit",
                behavior_type="io_pattern",
                pattern_data={
                    "read_write_ratio": 1.0,
                    "sequential_access": True,
                    "files_per_second": 100,
                    "avg_entropy_change": 5.5
                },
                confidence=0.9
            ),
            DatasetPattern(
                family="BlackCat",
                behavior_type="network_pattern",
                pattern_data={
                    "c2_servers": ["185.220.101.45", "192.168.1.100"],
                    "bytes_exfiltrated": 50000000,
                    "uses_tor": True
                },
                confidence=0.85
            )
        ]

        library.add_patterns(example_patterns)
    else:
        library = parse_all_datasets(config)

    # Save library
    library.save_library(args.output)
    print(f"Pattern library saved to {args.output}")