"""Benign and attack scenario generators.

Each generator mutates the FileRegistry/ProcessTable and returns
lists of TelemetryEvents. All generators accept an rng for reproducibility.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from simulator.models import (
    ContentType,
    FileEvent,
    NetEvent,
    ProcessEvent,
    TelemetryEvent,
)
from simulator.registry import FileRegistry, ProcessTable


# ── Benign activity generators ───────────────────────────────────────


def office_edits(
    registry: FileRegistry,
    ptable: ProcessTable,
    rng: np.random.RandomState,
    now: datetime,
) -> list[TelemetryEvent]:
    """Simulate Office application edits: small file modifications.

    Modifies 2-5 existing doc files with small size/entropy changes.
    Spawns office processes (winword.exe, excel.exe, etc.).
    """
    events: list[TelemetryEvent] = []
    doc_files = registry.files_by_content_type(ContentType.DOC)
    if not doc_files:
        return events

    apps = [
        ("WINWORD.EXE", "\"C:\\Program Files\\Microsoft Office\\WINWORD.EXE\""),
        ("EXCEL.EXE", "\"C:\\Program Files\\Microsoft Office\\EXCEL.EXE\""),
        ("POWERPNT.EXE", "\"C:\\Program Files\\Microsoft Office\\POWERPNT.EXE\""),
    ]

    n_edits = rng.randint(2, min(6, len(doc_files) + 1))
    chosen = rng.choice(len(doc_files), size=n_edits, replace=False)

    # Spawn an office process
    app_name, app_cmd = apps[rng.randint(0, len(apps))]
    explorer = _find_process(ptable, "explorer.exe")
    parent_pid = explorer.pid if explorer else 4
    proc = ptable.spawn_process(app_name, parent_pid, app_cmd, now)
    events.append(ProcessEvent(
        ts=now, pid=proc.pid, name=proc.name,
        parent_pid=proc.parent_pid, command_line=proc.command_line,
    ))

    for idx in chosen:
        f = doc_files[idx]
        offset = timedelta(seconds=int(rng.randint(1, 110)))
        ts = now + offset

        # Benign edits: small size change, tiny entropy change
        size_delta = int(rng.randint(-512, 2048))
        entropy_delta = rng.uniform(-0.3, 0.3)

        registry.modify_file(f.path, ts, size_delta=size_delta,
                             entropy_delta=entropy_delta)
        events.append(FileEvent(
            ts=ts, path=f.path, size_delta=size_delta,
            entropy_delta=entropy_delta, extension_change=None,
        ))

    return events


def browser_downloads(
    registry: FileRegistry,
    ptable: ProcessTable,
    rng: np.random.RandomState,
    now: datetime,
) -> list[TelemetryEvent]:
    """Simulate browser downloading files.

    Downloads 1-3 files to the Downloads directory. Spawns browser process.
    Generates network events for the downloads.
    """
    events: list[TelemetryEvent] = []

    browsers = [
        ("chrome.exe", "chrome.exe --type=renderer"),
        ("firefox.exe", "firefox.exe -contentproc"),
        ("msedge.exe", "msedge.exe --type=renderer"),
    ]
    browser_name, browser_cmd = browsers[rng.randint(0, len(browsers))]

    explorer = _find_process(ptable, "explorer.exe")
    parent_pid = explorer.pid if explorer else 4
    proc = ptable.spawn_process(browser_name, parent_pid, browser_cmd, now)
    events.append(ProcessEvent(
        ts=now, pid=proc.pid, name=proc.name,
        parent_pid=proc.parent_pid, command_line=proc.command_line,
    ))

    download_types = [
        (ContentType.DOC, ".pdf", (50_000, 5_000_000), (4.0, 5.5)),
        (ContentType.IMAGE, ".png", (100_000, 8_000_000), (5.5, 7.0)),
        (ContentType.ARCHIVE, ".zip", (200_000, 50_000_000), (6.5, 7.5)),
        (ContentType.DOC, ".docx", (10_000, 2_000_000), (3.5, 5.0)),
    ]

    n_downloads = rng.randint(1, 4)
    for i in range(n_downloads):
        ct, ext, (size_lo, size_hi), (ent_lo, ent_hi) = \
            download_types[rng.randint(0, len(download_types))]
        offset = timedelta(seconds=int(rng.randint(5, 100)))
        ts = now + offset

        size = int(rng.randint(size_lo, size_hi))
        entropy = rng.uniform(ent_lo, ent_hi)
        path = f"C:/Users/A/Downloads/download_{rng.randint(0, 10000):04d}{ext}"

        registry.add_file(path, size, entropy, ext, ct, ts)
        events.append(FileEvent(
            ts=ts, path=path, size_delta=size,
            entropy_delta=entropy, extension_change=None,
        ))

        # Network event for the download
        destinations = ["142.250.80.46", "151.101.1.69", "104.16.132.229"]
        events.append(NetEvent(
            ts=ts, destination=destinations[rng.randint(0, len(destinations))],
            bytes_transferred=size, protocol="tcp", direction="inbound",
        ))

    return events


def backup_operations(
    registry: FileRegistry,
    ptable: ProcessTable,
    rng: np.random.RandomState,
    now: datetime,
) -> list[TelemetryEvent]:
    """Simulate backup software scanning/copying files.

    Reads many files (high activity count), may create .bak/.zip files.
    Looks suspicious because it touches many files — a false positive trap.
    """
    events: list[TelemetryEvent] = []

    proc = ptable.spawn_process(
        "backup_service.exe", 4,
        "\"C:\\Program Files\\BackupSvc\\backup_service.exe\" --full-scan",
        now,
    )
    events.append(ProcessEvent(
        ts=now, pid=proc.pid, name=proc.name,
        parent_pid=proc.parent_pid, command_line=proc.command_line,
    ))

    # Touch many files with zero-delta reads
    all_files = registry.unencrypted_files()
    n_touch = min(rng.randint(8, 20), len(all_files))
    chosen = rng.choice(len(all_files), size=n_touch, replace=False)

    for idx in chosen:
        f = all_files[idx]
        offset = timedelta(seconds=int(rng.randint(1, 115)))
        ts = now + offset
        # Read-only: zero deltas
        events.append(FileEvent(
            ts=ts, path=f.path, size_delta=0,
            entropy_delta=0.0, extension_change=None,
        ))

    # Optionally create a backup archive
    if rng.random() < 0.5:
        ts = now + timedelta(seconds=int(rng.randint(60, 115)))
        bak_path = f"C:/Users/A/Documents/backup_{rng.randint(0, 100):03d}.zip"
        bak_size = int(rng.randint(1_000_000, 50_000_000))
        registry.add_file(bak_path, bak_size, rng.uniform(6.5, 7.5),
                          ".zip", ContentType.ARCHIVE, ts)
        events.append(FileEvent(
            ts=ts, path=bak_path, size_delta=bak_size,
            entropy_delta=rng.uniform(6.5, 7.5), extension_change=None,
        ))

    return events


def av_scan(
    registry: FileRegistry,
    ptable: ProcessTable,
    rng: np.random.RandomState,
    now: datetime,
) -> list[TelemetryEvent]:
    """Simulate antivirus scan reading many files.

    High read rate across directories, no modifications. Process: MsMpEng.exe.
    """
    events: list[TelemetryEvent] = []

    proc = ptable.spawn_process(
        "MsMpEng.exe", 4,
        "\"C:\\ProgramData\\Microsoft\\Windows Defender\\MsMpEng.exe\"",
        now,
    )
    events.append(ProcessEvent(
        ts=now, pid=proc.pid, name=proc.name,
        parent_pid=proc.parent_pid, command_line=proc.command_line,
    ))

    all_files = registry.unencrypted_files()
    n_scan = min(rng.randint(10, 30), len(all_files))
    chosen = rng.choice(len(all_files), size=n_scan, replace=False)

    for idx in chosen:
        f = all_files[idx]
        offset = timedelta(seconds=int(rng.randint(1, 115)))
        events.append(FileEvent(
            ts=now + offset, path=f.path, size_delta=0,
            entropy_delta=0.0, extension_change=None,
        ))

    return events


def system_maintenance(
    registry: FileRegistry,
    ptable: ProcessTable,
    rng: np.random.RandomState,
    now: datetime,
) -> list[TelemetryEvent]:
    """Simulate Windows system maintenance (updates, disk cleanup).

    Creates/deletes temp files, spawns system processes.
    """
    events: list[TelemetryEvent] = []

    maintenance_procs = [
        ("svchost.exe", "svchost.exe -k wusvcs"),
        ("TiWorker.exe", "TiWorker.exe -Embedding"),
        ("cleanmgr.exe", "cleanmgr.exe /autoclean"),
    ]
    name, cmd = maintenance_procs[rng.randint(0, len(maintenance_procs))]
    proc = ptable.spawn_process(name, 4, cmd, now)
    events.append(ProcessEvent(
        ts=now, pid=proc.pid, name=proc.name,
        parent_pid=proc.parent_pid, command_line=proc.command_line,
    ))

    # Create temp files
    n_temps = rng.randint(1, 5)
    for i in range(n_temps):
        offset = timedelta(seconds=int(rng.randint(5, 100)))
        ts = now + offset
        tmp_path = f"C:/Users/A/AppData/Local/Temp/tmp_{rng.randint(0, 100000):06d}.tmp"
        tmp_size = int(rng.randint(1024, 1_000_000))
        registry.add_file(tmp_path, tmp_size, rng.uniform(2.0, 4.0),
                          ".tmp", ContentType.CONFIG, ts)
        events.append(FileEvent(
            ts=ts, path=tmp_path, size_delta=tmp_size,
            entropy_delta=rng.uniform(2.0, 4.0), extension_change=None,
        ))

    return events


# ── Attack scenario generators ───────────────────────────────────────


def blitz_encryptor(
    registry: FileRegistry,
    ptable: ProcessTable,
    rng: np.random.RandomState,
    now: datetime,
    progress: float = 0.5,
) -> tuple[list[TelemetryEvent], str]:
    """Fast, noisy encryptor: 20+ files/sec.

    High entropy spikes, mass extension changes, spawns ransom note.
    Obvious if observed. Progress controls attack advancement.

    Returns (events, attack_phase).
    """
    events: list[TelemetryEvent] = []

    # Spawn malicious process
    proc = ptable.spawn_process(
        "svchost.exe", 4,  # masquerades as svchost
        "svchost.exe -k netsvcs -p -s Schedule",
        now,
    )
    events.append(ProcessEvent(
        ts=now, pid=proc.pid, name=proc.name,
        parent_pid=proc.parent_pid, command_line=proc.command_line,
    ))

    if progress < 0.1:
        phase = "reconnaissance"
        # Early stage: just process spawn, minimal file activity
        return events, phase

    phase = "encryption"
    targets = registry.unencrypted_files()
    # Scale encryption count with progress
    max_encrypt = max(1, int(len(targets) * progress))
    n_encrypt = min(rng.randint(max(1, max_encrypt // 2), max_encrypt + 1),
                    len(targets))
    chosen = rng.choice(len(targets), size=n_encrypt, replace=False)

    extensions = [".locked", ".encrypted", ".enc", ".cry"]
    ext = extensions[rng.randint(0, len(extensions))]

    for idx in chosen:
        f = targets[idx]
        offset = timedelta(seconds=int(rng.randint(0, 60)))
        ts = now + offset

        old_entropy = f.entropy
        old_ext = f.extension
        result = registry.encrypt_file(f.path, ts, new_extension=ext, rng=rng)
        if result is None:
            continue

        events.append(FileEvent(
            ts=ts, path=result.path,
            size_delta=result.size_delta,
            entropy_delta=result.entropy - old_entropy,
            extension_change=f"{old_ext} -> {result.extension}",
        ))

    # Drop ransom note late in attack
    if progress > 0.7:
        note_ts = now + timedelta(seconds=rng.randint(60, 115))
        note_path = "C:/Users/A/Desktop/README_DECRYPT.txt"
        registry.add_file(note_path, 2048, 3.0, ".txt", ContentType.DOC, note_ts)
        events.append(FileEvent(
            ts=note_ts, path=note_path, size_delta=2048,
            entropy_delta=3.0, extension_change=None,
        ))

    return events, phase


def slow_sleeper(
    registry: FileRegistry,
    ptable: ProcessTable,
    rng: np.random.RandomState,
    now: datetime,
    progress: float = 0.5,
) -> tuple[list[TelemetryEvent], str]:
    """Slow, stealthy encryptor: 1-2 files/min.

    Subtle entropy changes, legitimate-looking process names.
    Hard to detect in a single window — needs historical context.

    Returns (events, attack_phase).
    """
    events: list[TelemetryEvent] = []

    # Uses legitimate-looking process name
    legit_names = [
        ("SearchIndexer.exe", "SearchIndexer.exe /Embedding"),
        ("RuntimeBroker.exe", "RuntimeBroker.exe -Embedding"),
        ("taskhostw.exe", "taskhostw.exe Execute"),
    ]
    name, cmd = legit_names[rng.randint(0, len(legit_names))]
    proc = ptable.spawn_process(name, 4, cmd, now)
    events.append(ProcessEvent(
        ts=now, pid=proc.pid, name=proc.name,
        parent_pid=proc.parent_pid, command_line=proc.command_line,
    ))

    if progress < 0.15:
        return events, "reconnaissance"

    phase = "encryption"
    targets = registry.unencrypted_files()
    # Very few files per window — 1-2
    n_encrypt = min(rng.randint(1, 3), len(targets))
    if n_encrypt == 0:
        return events, phase

    chosen = rng.choice(len(targets), size=n_encrypt, replace=False)

    for idx in chosen:
        f = targets[idx]
        offset = timedelta(seconds=int(rng.randint(10, 110)))
        ts = now + offset

        old_entropy = f.entropy
        old_ext = f.extension
        result = registry.encrypt_file(f.path, ts, new_extension=".locked", rng=rng)
        if result is None:
            continue

        events.append(FileEvent(
            ts=ts, path=result.path,
            size_delta=result.size_delta,
            entropy_delta=result.entropy - old_entropy,
            extension_change=f"{old_ext} -> {result.extension}",
        ))

    return events, phase


def exfil_first(
    registry: FileRegistry,
    ptable: ProcessTable,
    rng: np.random.RandomState,
    now: datetime,
    progress: float = 0.5,
) -> tuple[list[TelemetryEvent], str]:
    """Data exfiltration phase followed by encryption.

    Progress < 0.6: exfiltration (network events only, no file modification).
    Progress >= 0.6: encryption phase (similar to blitz).

    Returns (events, attack_phase).
    """
    events: list[TelemetryEvent] = []

    proc = ptable.spawn_process(
        "svchost.exe", 4,
        "svchost.exe -k NetworkService -p",
        now,
    )
    events.append(ProcessEvent(
        ts=now, pid=proc.pid, name=proc.name,
        parent_pid=proc.parent_pid, command_line=proc.command_line,
    ))

    if progress < 0.6:
        # Exfiltration phase: large outbound transfers, no file changes
        phase = "exfiltration"
        c2_servers = ["185.142.98.12", "91.215.85.47", "45.153.241.81"]
        n_uploads = rng.randint(2, 6)

        for _ in range(n_uploads):
            offset = timedelta(seconds=int(rng.randint(5, 110)))
            ts = now + offset
            events.append(NetEvent(
                ts=ts,
                destination=c2_servers[rng.randint(0, len(c2_servers))],
                bytes_transferred=int(rng.randint(100_000, 10_000_000)),
                protocol="tcp",
                direction="outbound",
            ))

        # Also reads files (gathering data to exfil) but doesn't modify them
        targets = registry.unencrypted_files()
        n_reads = min(rng.randint(3, 8), len(targets))
        chosen = rng.choice(len(targets), size=n_reads, replace=False)
        for idx in chosen:
            f = targets[idx]
            offset = timedelta(seconds=int(rng.randint(1, 110)))
            events.append(FileEvent(
                ts=now + offset, path=f.path,
                size_delta=0, entropy_delta=0.0, extension_change=None,
            ))

        return events, phase

    # Encryption phase
    phase = "encryption"
    enc_progress = (progress - 0.6) / 0.4  # normalize to 0-1
    targets = registry.unencrypted_files()
    n_encrypt = min(
        rng.randint(5, max(6, int(len(targets) * enc_progress))),
        len(targets),
    )
    if n_encrypt > 0:
        chosen = rng.choice(len(targets), size=n_encrypt, replace=False)
        for idx in chosen:
            f = targets[idx]
            offset = timedelta(seconds=int(rng.randint(0, 60)))
            ts = now + offset

            old_entropy = f.entropy
            old_ext = f.extension
            result = registry.encrypt_file(f.path, ts, new_extension=".locked",
                                           rng=rng)
            if result is None:
                continue
            events.append(FileEvent(
                ts=ts, path=result.path,
                size_delta=result.size_delta,
                entropy_delta=result.entropy - old_entropy,
                extension_change=f"{old_ext} -> {result.extension}",
            ))

    return events, phase


def semantic_shuffle(
    registry: FileRegistry,
    ptable: ProcessTable,
    rng: np.random.RandomState,
    now: datetime,
    progress: float = 0.5,
) -> tuple[list[TelemetryEvent], str]:
    """Content manipulation without encryption indicators.

    Files are modified (content replaced, sizes change) but entropy stays
    moderate and extensions don't change. Designed to evade entropy heuristics.

    Returns (events, attack_phase).
    """
    events: list[TelemetryEvent] = []

    proc = ptable.spawn_process(
        "conhost.exe", 4,
        "conhost.exe 0x4",
        now,
    )
    events.append(ProcessEvent(
        ts=now, pid=proc.pid, name=proc.name,
        parent_pid=proc.parent_pid, command_line=proc.command_line,
    ))

    if progress < 0.1:
        return events, "reconnaissance"

    phase = "manipulation"
    targets = registry.unencrypted_files()
    n_modify = min(rng.randint(3, max(4, int(len(targets) * progress * 0.5))),
                   len(targets))
    if n_modify == 0:
        return events, phase

    chosen = rng.choice(len(targets), size=n_modify, replace=False)

    for idx in chosen:
        f = targets[idx]
        offset = timedelta(seconds=int(rng.randint(5, 110)))
        ts = now + offset

        # Modify content but keep entropy moderate (< 6.5)
        # This is the evasion: entropy doesn't spike like encryption
        size_delta = int(rng.randint(-1024, 4096))
        entropy_delta = rng.uniform(-0.5, 1.0)
        target_entropy = f.entropy + entropy_delta
        # Clamp to avoid looking like encryption
        if target_entropy > 6.5:
            entropy_delta = 6.5 - f.entropy

        registry.modify_file(f.path, ts, size_delta=size_delta,
                             entropy_delta=entropy_delta)

        # Mark as encrypted in ground truth even though it doesn't look like it
        f.is_encrypted = True

        events.append(FileEvent(
            ts=ts, path=f.path,
            size_delta=size_delta,
            entropy_delta=entropy_delta,
            extension_change=None,  # No extension change — evasion
        ))

    return events, phase


# ── Helpers ──────────────────────────────────────────────────────────


def _find_process(ptable: ProcessTable, name: str) -> ProcessRecord | None:
    """Find the first process with the given name."""
    for pid in ptable.all_pids():
        p = ptable.get_process(pid)
        if p and p.name.lower() == name.lower():
            return p
    return None
