"""Mutable host state: FileRegistry and ProcessTable.

These represent the ground-truth state of the simulated host.
Agent tools query these registries to gather evidence.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np

from simulator.content import encrypt_content, generate_content
from simulator.models import ContentType, FileRecord, ProcessRecord


# ── Filesystem seed data ─────────────────────────────────────────────

# Realistic directory → (content_types, extensions, count_range)
_DIRECTORY_TEMPLATES: list[tuple[str, list[tuple[ContentType, str]], tuple[int, int]]] = [
    ("C:/Users/A/Documents", [
        (ContentType.DOC, ".docx"),
        (ContentType.DOC, ".xlsx"),
        (ContentType.DOC, ".pdf"),
        (ContentType.DOC, ".pptx"),
    ], (8, 20)),
    ("C:/Users/A/Desktop", [
        (ContentType.DOC, ".docx"),
        (ContentType.DOC, ".txt"),
        (ContentType.IMAGE, ".png"),
        (ContentType.IMAGE, ".jpg"),
    ], (4, 10)),
    ("C:/Users/A/Downloads", [
        (ContentType.ARCHIVE, ".zip"),
        (ContentType.EXECUTABLE, ".exe"),
        (ContentType.DOC, ".pdf"),
        (ContentType.IMAGE, ".png"),
    ], (3, 8)),
    ("C:/Users/A/Pictures", [
        (ContentType.IMAGE, ".jpg"),
        (ContentType.IMAGE, ".png"),
        (ContentType.IMAGE, ".bmp"),
    ], (5, 15)),
    ("C:/Users/A/AppData/Local/Temp", [
        (ContentType.CONFIG, ".tmp"),
        (ContentType.CONFIG, ".log"),
    ], (2, 6)),
    ("C:/ProgramData", [
        (ContentType.CONFIG, ".ini"),
        (ContentType.CONFIG, ".xml"),
        (ContentType.DATABASE, ".db"),
    ], (2, 5)),
]

# Typical entropy ranges by content type (bits per byte, 0-8)
_ENTROPY_RANGES: dict[ContentType, tuple[float, float]] = {
    ContentType.DOC: (3.5, 5.5),        # structured text, some compression
    ContentType.CONFIG: (2.0, 4.0),      # low entropy, repetitive
    ContentType.IMAGE: (5.5, 7.2),       # compressed, higher entropy
    ContentType.EXECUTABLE: (5.0, 6.5),  # compiled code
    ContentType.ARCHIVE: (6.5, 7.5),     # compressed
    ContentType.DATABASE: (3.0, 5.0),    # structured data
}

# Typical file size ranges by content type (bytes)
_SIZE_RANGES: dict[ContentType, tuple[int, int]] = {
    ContentType.DOC: (4_096, 2_000_000),
    ContentType.CONFIG: (256, 32_768),
    ContentType.IMAGE: (50_000, 10_000_000),
    ContentType.EXECUTABLE: (100_000, 50_000_000),
    ContentType.ARCHIVE: (100_000, 100_000_000),
    ContentType.DATABASE: (8_192, 5_000_000),
}


class FileRegistry:
    """Mutable file registry tracking all files on the simulated host.

    The agent queries this via tools (inspect_file, scan_directory).
    Attack generators mutate it (encrypt_file, rename_file).
    """

    def __init__(self) -> None:
        self._files: dict[str, FileRecord] = {}

    def __len__(self) -> int:
        return len(self._files)

    def __contains__(self, path: str) -> bool:
        return path in self._files

    def all_paths(self) -> list[str]:
        return list(self._files.keys())

    def directories(self) -> list[str]:
        """Return unique directory prefixes."""
        dirs: set[str] = set()
        for path in self._files:
            parent = os.path.dirname(path)
            if parent:
                dirs.add(parent)
        return sorted(dirs)

    def add_file(
        self,
        path: str,
        size: int,
        entropy: float,
        extension: str,
        content_type: ContentType,
        modified_at: datetime,
        contents: bytes | None = None,
    ) -> FileRecord:
        record = FileRecord(
            path=path,
            size=size,
            entropy=entropy,
            extension=extension,
            modified_at=modified_at,
            content_type=content_type,
            contents=contents,
        )
        self._files[path] = record
        return record

    def get_file(self, path: str) -> FileRecord | None:
        return self._files.get(path)

    def encrypt_file(
        self,
        path: str,
        now: datetime,
        new_extension: str = ".locked",
        rng: np.random.RandomState | None = None,
    ) -> FileRecord | None:
        """Simulate encryption: high entropy, extension change, size change."""
        record = self._files.get(path)
        if record is None or record.is_encrypted:
            return None

        rng = rng or np.random.RandomState()

        record.is_encrypted = True
        record.entropy = 7.5 + rng.uniform(0.0, 0.49)  # near-random
        record.extension = record.extension + new_extension
        # Encrypted files are padded to block boundary
        record.size = int(math.ceil(record.size / 16) * 16) + rng.randint(0, 256)
        record.modified_at = now
        # Replace contents with encrypted (near-max entropy) bytes
        content_size = min(record.size, 1024)
        record.contents = encrypt_content(content_size, rng)

        # Update the registry with new path if extension changed
        old_path = path
        new_path = path + new_extension
        self._files[new_path] = record
        record.path = new_path
        del self._files[old_path]

        return record

    def modify_file(
        self,
        path: str,
        now: datetime,
        size_delta: int = 0,
        entropy_delta: float = 0.0,
    ) -> FileRecord | None:
        """Simulate a benign file modification."""
        record = self._files.get(path)
        if record is None:
            return None

        record.size = max(1, record.size + size_delta)
        record.entropy = max(0.0, min(8.0, record.entropy + entropy_delta))
        record.modified_at = now

        # Adjust contents to reflect size change
        if record.contents is not None and size_delta != 0:
            if size_delta > 0:
                record.contents = record.contents + b"\x00" * size_delta
            else:
                new_len = max(1, len(record.contents) + size_delta)
                record.contents = record.contents[:new_len]

        return record

    def rename_file(self, old_path: str, new_path: str) -> FileRecord | None:
        record = self._files.pop(old_path, None)
        if record is None:
            return None
        record.path = new_path
        new_ext = os.path.splitext(new_path)[1]
        if new_ext:
            record.extension = new_ext
        self._files[new_path] = record
        return record

    def list_directory(self, dir_path: str) -> list[FileRecord]:
        """List files whose path starts with dir_path."""
        # Normalize: ensure trailing separator
        prefix = dir_path.rstrip("/") + "/"
        return [r for p, r in self._files.items() if p.startswith(prefix)]

    def files_by_content_type(self, content_type: ContentType) -> list[FileRecord]:
        return [r for r in self._files.values() if r.content_type == content_type]

    def unencrypted_files(self) -> list[FileRecord]:
        return [r for r in self._files.values() if not r.is_encrypted]

    def read_file_sample(
        self,
        path: str,
        offset: int = 0,
        length: int = 256,
    ) -> bytes | None:
        """Read a sample of file contents (if available).

        Returns None if the file has no contents set.
        """
        record = self._files.get(path)
        if record is None or record.contents is None:
            return None
        return record.contents[offset:offset + length]

    def modify_contents(self, path: str, new_bytes: bytes) -> FileRecord | None:
        """Replace file contents."""
        record = self._files.get(path)
        if record is None:
            return None
        record.contents = new_bytes
        return record

    def seed_filesystem(
        self,
        rng: np.random.RandomState,
        now: datetime,
        templates: list | None = None,
    ) -> None:
        """Populate with realistic initial files."""
        templates = templates or _DIRECTORY_TEMPLATES

        file_counter = 0
        for dir_path, type_ext_pairs, (min_count, max_count) in templates:
            n_files = rng.randint(min_count, max_count + 1)
            for _ in range(n_files):
                content_type, ext = type_ext_pairs[rng.randint(0, len(type_ext_pairs))]
                entropy_lo, entropy_hi = _ENTROPY_RANGES[content_type]
                size_lo, size_hi = _SIZE_RANGES[content_type]

                filename = f"file_{file_counter:04d}{ext}"
                path = f"{dir_path}/{filename}"

                entropy = rng.uniform(entropy_lo, entropy_hi)
                contents = generate_content(ext, entropy, rng)

                self.add_file(
                    path=path,
                    size=rng.randint(size_lo, size_hi + 1),
                    entropy=entropy,
                    extension=ext,
                    content_type=content_type,
                    modified_at=now - _random_timedelta(rng, max_days=30),
                    contents=contents,
                )
                file_counter += 1


class ProcessTable:
    """Mutable process table tracking running processes.

    The agent queries this via the check_process tool.
    Generators spawn/kill processes here.
    """

    def __init__(self) -> None:
        self._processes: dict[int, ProcessRecord] = {}
        self._next_pid: int = 1000

    def __len__(self) -> int:
        return len(self._processes)

    def spawn_process(
        self,
        name: str,
        parent_pid: int,
        command_line: str,
        start_time: datetime,
    ) -> ProcessRecord:
        pid = self._next_pid
        self._next_pid += 1

        record = ProcessRecord(
            pid=pid,
            name=name,
            parent_pid=parent_pid,
            command_line=command_line,
            start_time=start_time,
        )
        self._processes[pid] = record

        # Register as child of parent
        parent = self._processes.get(parent_pid)
        if parent is not None:
            parent.child_pids.append(pid)

        return record

    def kill_process(self, pid: int) -> ProcessRecord | None:
        record = self._processes.pop(pid, None)
        if record is None:
            return None

        # Remove from parent's child list
        parent = self._processes.get(record.parent_pid)
        if parent is not None and pid in parent.child_pids:
            parent.child_pids.remove(pid)

        return record

    def get_process(self, pid: int) -> ProcessRecord | None:
        return self._processes.get(pid)

    def get_children(self, pid: int) -> list[int]:
        record = self._processes.get(pid)
        if record is None:
            return []
        return list(record.child_pids)

    def all_pids(self) -> list[int]:
        return list(self._processes.keys())

    def open_file_handle(self, pid: int, path: str) -> bool:
        """Record that a process has an open file handle."""
        record = self._processes.get(pid)
        if record is None:
            return False
        if path not in record.open_file_handles:
            record.open_file_handles.append(path)
        return True

    def close_file_handle(self, pid: int, path: str) -> bool:
        """Remove an open file handle from a process."""
        record = self._processes.get(pid)
        if record is None:
            return False
        if path in record.open_file_handles:
            record.open_file_handles.remove(path)
            return True
        return False

    def load_module(self, pid: int, module: str) -> bool:
        """Record that a process loaded a module/DLL."""
        record = self._processes.get(pid)
        if record is None:
            return False
        if module not in record.loaded_modules:
            record.loaded_modules.append(module)
        return True

    def inject_module(self, pid: int, module: str) -> bool:
        """Simulate DLL injection (same as load_module, distinct intent)."""
        return self.load_module(pid, module)

    def set_integrity(
        self,
        pid: int,
        level: str = "High",
        elevated: bool = False,
        user: str | None = None,
    ) -> bool:
        """Set a process's integrity level and elevation status."""
        record = self._processes.get(pid)
        if record is None:
            return False
        record.integrity_level = level
        record.is_elevated = elevated
        if user is not None:
            record.user = user
        return True

    def seed_processes(self, now: datetime) -> None:
        """Populate with typical background processes."""
        base_procs = [
            ("System", 0, "System"),
            ("smss.exe", 4, "\\SystemRoot\\System32\\smss.exe"),
            ("csrss.exe", 4, "csrss.exe ObjectDirectory=\\Windows"),
            ("wininit.exe", 4, "wininit.exe"),
            ("services.exe", 4, "services.exe"),
            ("svchost.exe", 4, "svchost.exe -k DcomLaunch"),
            ("svchost.exe", 4, "svchost.exe -k netsvcs"),
            ("lsass.exe", 4, "lsass.exe"),
            ("explorer.exe", 4, "C:\\Windows\\explorer.exe"),
            ("dwm.exe", 4, "dwm.exe"),
        ]
        # System process (pid 4) is the root
        system = ProcessRecord(
            pid=4, name="System", parent_pid=0,
            command_line="System", start_time=now,
            user="SYSTEM", integrity_level="System", is_elevated=True,
            loaded_modules=["ntdll.dll", "kernel32.dll"],
        )
        self._processes[4] = system

        for name, parent_pid, cmd in base_procs[1:]:
            proc = self.spawn_process(name, parent_pid, cmd, now)
            # Set typical forensic fields for system processes
            proc.user = "SYSTEM"
            proc.integrity_level = "System"
            proc.is_elevated = True
            proc.loaded_modules.append("ntdll.dll")
            proc.loaded_modules.append("kernel32.dll")


def _random_timedelta(
    rng: np.random.RandomState, max_days: int = 30
) -> timedelta:
    """Generate a random timedelta up to max_days."""
    seconds = rng.randint(0, max_days * 86400)
    return timedelta(seconds=int(seconds))
