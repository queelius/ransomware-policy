"""Forensic investigation tools: query_registry, list_process_handles,
query_event_log, read_file_sample.
"""

from __future__ import annotations

import math

from simulator.event_log import EventLog
from simulator.registry import FileRegistry, ProcessTable
from simulator.registry_hive import RegistryHive


def query_registry(
    registry_hive: RegistryHive,
    key_path: str,
) -> dict:
    """Query a registry key and its values. Cost: -0.03.

    Returns all values under the key, or error if not found.
    """
    key = registry_hive.get_key(key_path)
    if key is None:
        # Try enumerating sub-keys
        sub_keys = registry_hive.enumerate_keys(key_path)
        if sub_keys:
            return {
                "key_path": key_path,
                "values": {},
                "sub_keys": sub_keys,
            }
        return {"error": f"Registry key not found: {key_path}"}

    return {
        "key_path": key.path,
        "modified_at": key.modified_at.isoformat(),
        "values": {
            name: {"data": v.data, "type": v.value_type}
            for name, v in key.values.items()
        },
    }


def list_process_handles(
    ptable: ProcessTable,
    pid: int,
) -> dict:
    """List a process's open handles, modules, and security context. Cost: -0.03.

    Returns file handles, network sockets, loaded modules,
    user, integrity level, and elevation status.
    """
    record = ptable.get_process(pid)
    if record is None:
        return {"error": f"Process not found: pid={pid}"}

    return {
        "pid": record.pid,
        "name": record.name,
        "open_file_handles": record.open_file_handles,
        "open_network_sockets": record.open_network_sockets,
        "loaded_modules": record.loaded_modules,
        "user": record.user,
        "integrity_level": record.integrity_level,
        "is_elevated": record.is_elevated,
    }


def query_event_log(
    event_log: EventLog,
    source: str | None = None,
    event_id: int | None = None,
    since: str | None = None,
) -> dict:
    """Query Windows Event Log entries. Cost: -0.04.

    Returns matching entries filtered by source, event_id, and/or timestamp.
    """
    from datetime import datetime

    since_dt = None
    if since is not None:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            return {"error": f"Invalid timestamp format: {since}"}

    # Convert event_id from possible string
    eid = None
    if event_id is not None:
        try:
            eid = int(event_id)
        except (TypeError, ValueError):
            return {"error": f"Invalid event_id: {event_id}"}

    entries = event_log.query_events(source=source, event_id=eid, since=since_dt)

    return {
        "entries": [
            {
                "timestamp": e.timestamp.isoformat(),
                "source": e.source,
                "event_id": e.event_id,
                "level": e.level,
                "message": e.message,
            }
            for e in entries
        ],
        "count": len(entries),
    }


def read_file_sample(
    file_registry: FileRegistry,
    path: str,
    offset: int = 0,
    length: int = 256,
) -> dict:
    """Read a raw byte sample from a file. Cost: -0.04.

    Returns hex-encoded bytes, computed entropy, and magic bytes.
    Returns error if the file has no contents.
    """
    record = file_registry.get_file(path)
    if record is None:
        return {"error": f"File not found: {path}"}

    sample = file_registry.read_file_sample(path, offset, length)
    if sample is None:
        return {
            "error": "File contents not available for sampling",
            "path": path,
            "metadata_only": {
                "entropy": round(record.entropy, 2),
                "size": record.size,
                "extension": record.extension,
            },
        }

    # Compute sample entropy
    sample_entropy = _compute_entropy(sample)

    # First 4 bytes as magic number
    magic = sample[:4].hex() if len(sample) >= 4 else sample.hex()

    return {
        "path": path,
        "offset": offset,
        "length": len(sample),
        "hex": sample.hex(),
        "entropy": round(sample_entropy, 2),
        "magic_bytes": magic,
    }


def _compute_entropy(data: bytes) -> float:
    """Compute Shannon entropy of byte data (bits per byte, 0-8)."""
    if not data:
        return 0.0

    counts = [0] * 256
    for b in data:
        counts[b] += 1

    total = len(data)
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy
