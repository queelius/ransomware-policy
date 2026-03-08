# Active Detective: System Design Document

**Date:** 2026-03-05
**Status:** Comprehensive design specification (supersedes initial draft of 2026-03-03)

This document specifies every component of the Active Detective system:
the simulated host OS, synthetic telemetry generation, ransomware agent
implementations, the investigation environment, the RL training pipeline,
and the evaluation framework.

---

## 1. System Overview

Active Detective teaches a small LLM (Qwen3-8B, distilled to 1.7B-3B)
to actively investigate host telemetry for ransomware, rather than
passively classifying a fixed observation window. The agent operates in
a POMDP: host state is partially hidden, events are stochastically
dropped, and the agent must decide which evidence to seek before
rendering a verdict.

The system has five layers:

```
+-------------------------------------------------------+
|  5. TRAINING (GRPO via TRL)                           |
|     - multi-step rollouts via environment_factory     |
|     - reward = verdict + cost + efficiency + format   |
+-------------------------------------------------------+
|  4. EVALUATION                                        |
|     - baselines (random, exhaustive, heuristic)       |
|     - metrics (detection, efficiency, Pareto, phase)  |
|     - ablation (tool removal, budget sweep)           |
+-------------------------------------------------------+
|  3. ENVIRONMENT (RansomwareDetectionEnv)              |
|     - tool execution against simulated host state     |
|     - budget enforcement, cost accumulation           |
|     - deterministic reward computation (RLVR)         |
+-------------------------------------------------------+
|  2. AGENT TOOLS                                       |
|     - inspect_file, check_process, scan_directory     |
|     - list_connections, inspect_connection             |
|     - query_registry, list_process_handles            |
|     - query_event_log, read_file_sample               |
|     - recall_memory (episodic RAG), DECIDE            |
|     - tool-call parser (JSON + function-call syntax)  |
+-------------------------------------------------------+
|  1. SIMULATED HOST OS (HostState)                     |
|     - FileRegistry (mutable filesystem + contents)    |
|     - ProcessTable (processes + handles/modules)      |
|     - ConnectionTable (network connections)            |
|     - RegistryHive (Windows registry keys)            |
|     - EventLog (Windows event log)                    |
|     - SimClock (monotonic time)                       |
|     - Telemetry generators (benign + ransomware)      |
|     - Observability filter (stochastic event drop)    |
+-------------------------------------------------------+
```

---

## 2. Simulated Host OS

### 2.1 FileRegistry

A mutable dictionary of FileRecords representing the host filesystem.
Not a full filesystem -- only tracks files that are relevant to
ransomware detection (user documents, configs, executables, images).

```
FileRecord:
  path: str                    # e.g., "C:/Users/A/Documents/report.docx"
  size: int                    # bytes
  entropy: float               # bits per byte, 0-8
  extension: str               # e.g., ".docx"
  modified_at: datetime        # last modification timestamp
  content_type: ContentType    # doc|config|image|executable|archive|database
  is_encrypted: bool           # ground truth (HIDDEN from agent)
  contents: bytes | None       # actual file bytes (None = metadata-only mode)

  # Computed deltas (for telemetry events):
  entropy_delta: float         # current - original
  size_delta: int              # current - original
  extension_changed: bool      # current != original
```

**File contents model.** When `contents` is populated, entropy is *computed*
from the actual bytes via Shannon entropy rather than sampled from a range.
This enables realistic encryption simulation: AES-encrypting real bytes
produces near-random ciphertext (~7.99 bits/byte), and intermittent
encryption naturally produces moderate entropy because only some blocks
are encrypted.

```python
def shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy in bits per byte (0-8 scale)."""
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    return -sum((c / length) * log2(c / length) for c in counts.values())
```

**Content generators by type.** When seeding the filesystem, each
content type generates plausible byte content:

| Type | Content Generator | Typical Entropy |
|------|------------------|-----------------|
| doc | Lorem ipsum structured text with XML wrappers | 3.5-5.5 |
| config | Repetitive INI/XML key-value pairs | 2.0-4.0 |
| image | JFIF/PNG header + random compressed payload | 5.5-7.2 |
| executable | PE header + mixed code/data sections | 5.0-6.5 |
| archive | ZIP local file header + compressed stream | 6.5-7.5 |
| database | SQLite header + structured tabular rows | 3.0-5.0 |

Content sizes are typically 1KB-64KB (small enough for fast simulation,
large enough for meaningful entropy computation).

**Seeding.** `FileRegistry.seed_filesystem(rng, now)` populates the
registry from directory templates:

| Directory | Content Types | File Count |
|-----------|--------------|------------|
| `C:/Users/A/Documents` | .docx, .xlsx, .pdf, .pptx | 8-20 |
| `C:/Users/A/Desktop` | .docx, .txt, .png, .jpg | 4-10 |
| `C:/Users/A/Downloads` | .zip, .exe, .pdf, .png | 3-8 |
| `C:/Users/A/Pictures` | .jpg, .png, .bmp | 5-15 |
| `C:/Users/A/AppData/Local/Temp` | .tmp, .log | 2-6 |
| `C:/ProgramData` | .ini, .xml, .db | 2-5 |

**Entropy ranges by content type:**

| Type | Entropy Range | Rationale |
|------|--------------|-----------|
| doc | 3.5-5.5 | Structured text, some compression |
| config | 2.0-4.0 | Low entropy, repetitive patterns |
| image | 5.5-7.2 | Compressed formats |
| executable | 5.0-6.5 | Compiled code |
| archive | 6.5-7.5 | Already compressed |
| database | 3.0-5.0 | Structured tabular data |

**Mutation methods:**
- `encrypt_file(path, now, ext, rng)`: Metadata-only mode. Sets entropy to
  7.5+uniform(0, 0.49), appends extension (e.g., `.docx` -> `.docx.locked`),
  pads size to block boundary. Marks `is_encrypted=True`.
- `encrypt_file_real(path, key)`: Content-aware mode. AES-encrypts actual
  bytes in `contents`, recomputes entropy via `shannon_entropy()`. Extension
  and size changes follow from the real ciphertext. Requires `contents != None`.
- `partial_encrypt(path, ratio, key)`: Intermittent encryption (BlackCat
  style). Encrypts `ratio` fraction of 4KB blocks, leaving others as
  plaintext. Produces moderate entropy (5.5-6.5) rather than near-random.
- `modify_contents(path, new_bytes)`: Arbitrary content replacement.
  Recomputes entropy and size from new bytes.
- `modify_file(path, now, size_delta, entropy_delta)`: Benign edit. Clamps
  entropy to [0, 8]. If `contents` is populated, mutates bytes to match.
- `rename_file(old_path, new_path)`: Moves file in registry.
- `read_file_sample(path, offset, length) -> bytes`: Returns a byte window
  from `contents`. Used by the `read_file_sample` agent tool.

### 2.2 ProcessTable

A mutable dictionary of ProcessRecords representing the running process tree.

```
ProcessRecord:
  pid: int
  name: str                    # e.g., "svchost.exe"
  parent_pid: int
  command_line: str            # full command line
  start_time: datetime
  child_pids: list[int]

  # Forensic fields (NEW):
  open_file_handles: list[str]   # paths of open files
  open_network_sockets: list[int]  # conn_ids in ConnectionTable
  loaded_modules: list[str]      # DLL paths (e.g., "C:/Windows/System32/ntdll.dll")
  user: str                      # "NT AUTHORITY\\SYSTEM" | "DOMAIN\\user"
  integrity_level: str           # "Low" | "Medium" | "High" | "System"
  is_elevated: bool              # running with admin privileges
```

**Forensic field rationale.** These fields expose indicators that
real EDR tools surface and SOC analysts rely on:
- `open_file_handles`: svchost.exe holding handles on user documents
  is anomalous. Ransomware must open files to encrypt them.
- `loaded_modules`: DLL injection (T1055) adds unexpected modules.
  A legitimate svchost.exe loads a known set of DLLs.
- `integrity_level` + `is_elevated`: Privilege escalation (T1134) changes
  these. Ransomware often needs High/System to delete VSS.
- `user`: Processes running as SYSTEM that normally run as the user
  (or vice versa) are suspicious.

**Additional mutation methods:**
- `load_module(pid, dll_path)`: Adds a DLL to the process's loaded modules.
  Used to simulate DLL injection attacks.
- `open_file_handle(pid, file_path)`: Adds a file handle to the process.
  Auto-called when ransomware opens files for encryption.
- `close_file_handle(pid, file_path)`: Removes a file handle.
- `set_integrity(pid, level)`: Changes integrity level (simulates
  privilege escalation). Updates `is_elevated` accordingly.

**Seeding.** `ProcessTable.seed_processes(now)` populates with standard
Windows background processes:

```
pid=4    System         (root,    user=SYSTEM, integrity=System)
pid=1000 smss.exe       (parent=4, user=SYSTEM, integrity=System)
pid=1001 csrss.exe      (parent=4, user=SYSTEM, integrity=System)
pid=1002 wininit.exe    (parent=4, user=SYSTEM, integrity=System)
pid=1003 services.exe   (parent=4, user=SYSTEM, integrity=System)
pid=1004 svchost.exe    (parent=4, -k DcomLaunch, user=SYSTEM, integrity=System)
pid=1005 svchost.exe    (parent=4, -k netsvcs,    user=SYSTEM, integrity=System)
pid=1006 lsass.exe      (parent=4, user=SYSTEM, integrity=System)
pid=1007 explorer.exe   (parent=4, user=DOMAIN\\user, integrity=Medium)
pid=1008 dwm.exe        (parent=4, user=SYSTEM, integrity=System)
```

Seeded processes include standard loaded modules (ntdll.dll, kernel32.dll,
etc.) and background network sockets (svchost.exe connections to Windows
Update, DNS, etc.). Attack generators spawn new processes (often masquerading
as legitimate names like svchost.exe) and benign generators spawn application
processes (chrome.exe, WINWORD.EXE, backup_service.exe, etc.).

### 2.3 ConnectionTable

A mutable table of active network connections, analogous to what `netstat`
or `Get-NetTCPConnection` surfaces on a real Windows host. Generators
create and modify connections; the agent queries them via tools.

```
ConnectionRecord:
  conn_id: int                   # unique connection identifier
  pid: int                       # owning process
  local_addr: str                # "192.168.1.100"
  local_port: int
  remote_addr: str               # "185.142.98.12"
  remote_port: int
  protocol: str                  # "tcp" | "udp"
  state: str                     # "ESTABLISHED" | "LISTEN" | "SYN_SENT" | "CLOSE_WAIT" | ...
  bytes_sent: int
  bytes_received: int
  dns_name: str | None           # reverse DNS if available
  tls_enabled: bool
  started_at: datetime
```

**Methods:**
- `open_connection(pid, remote_addr, remote_port, protocol, now) -> ConnectionRecord`:
  Creates a new connection. Auto-assigns `conn_id`, sets `local_addr` to
  host IP (192.168.1.100), picks ephemeral `local_port`, state=ESTABLISHED.
- `close_connection(conn_id, now)`: Sets state to CLOSED, removes from
  active table after a short delay.
- `transfer_data(conn_id, bytes_sent, bytes_received)`: Increments
  cumulative byte counters. Used to simulate data exfiltration.
- `beacon(pid, remote_addr, remote_port, interval)`: Registers a
  periodic callback. Every `interval` seconds, opens a short-lived
  connection, transfers a small payload (128-512 bytes), then closes.
  Produces a characteristic periodic pattern in the connection log.
- `list_by_pid(pid) -> list[ConnectionRecord]`: All connections owned
  by a process.
- `list_by_state(state) -> list[ConnectionRecord]`: Filter by TCP state.
- `list_by_remote(addr) -> list[ConnectionRecord]`: All connections to
  a given remote address.

**Seeding.** `ConnectionTable.seed_connections(ptable, now)` creates
standard background connections:

| Process | Remote | Port | Purpose |
|---------|--------|------|---------|
| svchost.exe (pid=1005) | 13.107.4.50 | 443 | Windows Update |
| svchost.exe (pid=1004) | 192.168.1.1 | 53 | DNS resolver |
| lsass.exe (pid=1006) | 192.168.1.10 | 389 | Active Directory LDAP |

Benign generators add connections dynamically (browser to CDN endpoints,
Office to SharePoint, etc.). This creates a realistic background against
which C2 traffic must be distinguished.

### 2.4 RegistryHive

A mutable representation of security-relevant Windows registry keys.
Not a full registry -- only keys that ransomware commonly reads, writes,
or deletes (persistence, defense evasion, recovery inhibition).

```
RegistryKey:
  path: str                      # "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run"
  values: dict[str, RegistryValue]
  last_modified: datetime

RegistryValue:
  name: str
  data: str | int | bytes
  value_type: str                # REG_SZ, REG_DWORD, REG_BINARY
```

**Methods:**
- `get_value(key_path, value_name) -> RegistryValue | None`: Read a
  registry value. Returns None if key or value doesn't exist.
- `set_value(key_path, value_name, data, value_type, now)`: Write a
  registry value. Creates the key if it doesn't exist. Updates
  `last_modified`.
- `delete_key(key_path)`: Removes an entire key and all its values.
- `enumerate_keys(key_path) -> list[str]`: Lists subkey names under
  a given key path.
- **Convenience methods (sugar for common attack patterns):**
  - `add_run_key(name, command)`: Shortcut for writing to
    `HKCU\...\Run` or `HKLM\...\Run` (persistence T1547.001).
  - `disable_defender()`: Sets `DisableRealtimeMonitoring=1` under
    `HKLM\SOFTWARE\Microsoft\Windows Defender\Real-Time Protection`
    (T1562.001).
  - `disable_vss()`: Sets `Start=4` (disabled) under
    `HKLM\SYSTEM\CurrentControlSet\Services\VSS` (T1490).

**Pre-seeded keys (MITRE ATT&CK-relevant):**

| Key Path | Purpose | ATT&CK |
|----------|---------|--------|
| `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run` | Machine-level persistence | T1547.001 |
| `HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Run` | User-level persistence | T1547.001 |
| `HKLM\SOFTWARE\Microsoft\Windows Defender\Real-Time Protection` | Defender status | T1562.001 |
| `HKLM\SYSTEM\CurrentControlSet\Services\VSS` | Volume Shadow Copy service | T1490 |
| `HKLM\SYSTEM\CurrentControlSet\Services` | Service registration | T1543.003 |
| `HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion` | OS version info | — |

Default seeded values represent a healthy system: Defender enabled
(`DisableRealtimeMonitoring=0`), VSS running (`Start=2`), no
suspicious Run keys.

### 2.5 EventLog

A structured event log modeled after Windows Event Log. Generators
produce events as side effects of their actions; the agent can query
the log as a forensic tool.

```
EventLogEntry:
  event_id: int                  # Windows Event ID
  timestamp: datetime
  source: str                    # "Security", "System", "Application"
  level: str                     # "Information", "Warning", "Error", "Critical"
  pid: int | None                # originating process (if applicable)
  message: str                   # human-readable event description
  data: dict                     # structured event data (varies by event_id)
```

**Key event IDs to model:**

| Event ID | Source | Description | Forensic Value |
|----------|--------|-------------|----------------|
| 4688 | Security | Process creation (with command line) | Process genealogy |
| 4689 | Security | Process termination | Lifecycle tracking |
| 1102 | Security | Security log cleared | Critical anti-forensic indicator! |
| 7045 | System | New service installed | Persistence detection |
| 4663 | Security | File access attempt (with handle) | File access auditing |
| 5001 | Microsoft-Windows-Defender | Real-time protection disabled | Defense evasion |
| 1116 | Microsoft-Windows-Defender | Malware detection | AV alert |

**Methods:**
- `log_event(event_id, source, level, pid, message, data, now)`:
  Appends a new event. Many events are auto-generated: `ProcessTable.spawn_process()`
  auto-logs event 4688, `RegistryHive.disable_defender()` auto-logs event 5001.
- `query_events(source?, event_id?, since?, pid?) -> list[EventLogEntry]`:
  Filtered query. All parameters optional; returns events matching all
  specified criteria.
- `clear_log(source)`: Deletes all events for a source. Auto-logs event
  1102 ("Security log cleared") — the act of clearing the log is itself
  logged, as on real Windows. This is a strong anti-forensic indicator
  (T1070.001).
- `count_events(source?, event_id?, since?) -> int`: Count without
  returning full entries. Useful for quick checks.

**Seeding.** `EventLog.seed_events(now)` populates with normal background
events: svchost.exe process creation (4688), service starts (7045 for
standard services), periodic Defender scan results. This creates a
realistic baseline against which ransomware log patterns stand out.

### 2.6 SimClock

Simple monotonic clock providing consistent timestamps across all
registries. Generators advance the clock; tools read it.

```
SimClock:
  _current: datetime             # current simulation time
  _start: datetime               # episode start time

  tick(seconds) -> datetime      # advance clock, return new time
  now() -> datetime              # current time (no advance)
  elapsed() -> float             # seconds since episode start
```

SimClock ensures temporal consistency: if a generator spawns a process
at t=10s and that process encrypts a file at t=15s, all timestamps
across ProcessTable, FileRegistry, ConnectionTable, and EventLog
agree. Without a shared clock, each registry would need its own
time-tracking logic.

### 2.7 HostState Composition

All six components are composed into a single `HostState` dataclass.
This is the **sole interface** that generators mutate and tools query.

```python
@dataclass
class HostState:
    files: FileRegistry
    processes: ProcessTable
    connections: ConnectionTable
    registry: RegistryHive
    event_log: EventLog
    clock: SimClock

    @classmethod
    def create(cls, rng, now) -> "HostState":
        """Seed a fresh host with realistic baseline state."""
        clock = SimClock(now)
        files = FileRegistry.seed_filesystem(rng, now)
        processes = ProcessTable.seed_processes(now)
        connections = ConnectionTable.seed_connections(processes, now)
        registry = RegistryHive.seed_registry(now)
        event_log = EventLog.seed_events(now)
        return cls(files, processes, connections, registry, event_log, clock)
```

**Generator signature change.** All generators now take `HostState`
instead of individual registries:

```python
# Old signature:
def generate_blitz(registry, ptable, rng, now, progress) -> list[TelemetryEvent]

# New signature:
def generate_blitz(host: HostState, rng, progress) -> list[TelemetryEvent]
```

Generators call mutation methods on `host.files`, `host.processes`,
`host.connections`, `host.registry`, and `host.event_log`. The clock
is advanced via `host.clock.tick()`. This makes generators self-documenting:
reading the code shows exactly what state changes each attack produces.

### 2.8 Telemetry Event Types

Five event types flow through the telemetry pipeline:

**ProcessEvent:**
```
[t+0s] PROC pid=1009 name=chrome.exe parent=1007 cmd="chrome.exe --type=renderer"
```

**FileEvent:**
```
[t+15s] FILE path=C:/Users/A/Documents/file_0005.xlsx size_delta=+4096 entropy_delta=+3.82 ext_change=.xlsx->.xlsx.locked
```

**NetEvent:**
```
[t+22s] NET dest=185.220.101.42 bytes=8192 proto=tcp dir=outbound
```

**RegistryEvent (NEW):**
```
[t+3s] REG op=set key=HKLM\SOFTWARE\Microsoft\Windows Defender\Real-Time Protection val=DisableRealtimeMonitoring old=0 new=1
```
Fields: `ts`, `key_path`, `value_name`, `old_data`, `new_data`,
`operation` (set | delete). Generated automatically when
`RegistryHive.set_value()` or `delete_key()` is called.

**EventLogEvent (NEW):**
```
[t+3s] EVTLOG id=5001 source=Microsoft-Windows-Defender level=Warning msg="Real-time protection disabled"
```
Fields: `ts`, `event_id`, `source`, `level`, `message` (summarized).
Surfaced from `EventLog` entries. Note: the EventLog *stores* full
structured entries, but the telemetry stream shows a summarized
one-line version -- the agent must use `query_event_log` to get details.

Each event carries a `drop_probability` field. The observability filter
retains each event independently with probability `observability`
(or `1 - drop_probability` if per-event override is set). All five
event types flow through the same filter.

### 2.9 Observability Model

The agent never sees ground truth. It receives a filtered subset of
events. The `observability` parameter (0.0 to 1.0) controls what
fraction survives:

- **0.9**: Near-complete visibility. Easy mode.
- **0.7**: Most events visible, some gaps. Realistic EDR.
- **0.5**: Half of events dropped. Moderate challenge.
- **0.3**: Severe information loss. Agent must investigate actively.
- **0.1**: Near-blind. Extreme test case.

This is the independent variable for our central claim: the active agent
recovers accuracy at low observability where passive classifiers fail.

---

## 3. Synthetic Data: Ransomware Agent Implementations

Each generator mutates `HostState` (FileRegistry, ProcessTable,
ConnectionTable, RegistryHive, EventLog via the mutation API), producing
`TelemetryEvent` lists. All generators are deterministic given an rng seed.

### 3.1 Benign Activity Generators

These produce false-positive traps -- activity that looks suspicious
but is legitimate.

**`office_edits`**: Spawns Office process (WINWORD.EXE, EXCEL.EXE,
POWERPNT.EXE), modifies 2-5 doc files with small size/entropy deltas
(-512 to +2048 bytes, -0.3 to +0.3 entropy). Looks like normal
document editing.

**`browser_downloads`**: Spawns browser (chrome, firefox, msedge),
downloads 1-3 files to `Downloads/`. Creates new files with network
events (inbound TCP). Could look like C2 communication to a naive
detector.

**`backup_operations`**: Spawns `backup_service.exe`, touches 8-20
files with zero deltas (read-only scan). Optionally creates a `.zip`
archive. This is a deliberate false-positive trap: high file-access
count mimics encryption, but deltas are zero.

**`av_scan`**: Spawns `MsMpEng.exe` (Windows Defender), reads 10-30
files with zero deltas. Another false-positive trap: high read rate
across directories.

**`system_maintenance`**: Spawns maintenance processes (svchost.exe,
TiWorker.exe, cleanmgr.exe), creates 1-5 temp files. Routine OS noise.

### 3.2 Ransomware Agent Implementations

Each ransomware generator models a distinct attack strategy, controlled
by a `progress` parameter (0.0 to 1.0) that determines how far the
attack has advanced when the agent observes it.

#### 3.2.1 Blitz Encryptor

**Model:** WannaCry / NotPetya style -- fast, noisy, mass encryption.

```
Behavior:
  Process: masquerades as svchost.exe
  Speed: 20+ files/window (60 sec)
  Extension: .locked | .encrypted | .enc | .cry
  Entropy: jumps to 7.5+ (near-random)
  Phases:
    progress < 0.1  -> reconnaissance (process spawn only)
    progress >= 0.1 -> encryption (scales with progress)
    progress > 0.7  -> drops README_DECRYPT.txt on Desktop
```

**Telemetry signature:** Massive FILE events with `entropy_delta > +2.0`,
extension changes, and a suspicious process. Easy to detect if events
are observed. Challenge comes from low observability.

#### 3.2.2 Slow Sleeper

**Model:** Ryuk-style -- patient, 1-2 files per observation window.

```
Behavior:
  Process: masquerades as SearchIndexer.exe, RuntimeBroker.exe, or taskhostw.exe
  Speed: 1-2 files/window
  Extension: .locked only
  Entropy: same spike as blitz, but fewer events
  Phases:
    progress < 0.15 -> reconnaissance
    progress >= 0.15 -> encryption (always 1-2 files)
```

**Telemetry signature:** Subtle. One or two files with entropy spikes
in a window full of normal activity. Requires the agent to notice
the anomaly and investigate, or to use `recall_memory` to correlate
across windows.

#### 3.2.3 Exfil-First

**Model:** Maze / REvil-style -- data exfiltration before encryption.

```
Behavior:
  Process: masquerades as svchost.exe (NetworkService)
  Phases:
    progress < 0.6 -> exfiltration:
      - 2-6 large outbound network transfers to C2 IPs
      - reads 3-8 files (zero-delta, data gathering)
      - NO file modification
    progress >= 0.6 -> encryption:
      - same as blitz, scaled by progress
```

**Telemetry signature:** Exfiltration phase has NO file entropy changes.
Only suspicious outbound network events to unusual IPs with large byte
counts. The agent must recognize C2 traffic patterns. Encryption phase
(if reached) looks like blitz.

**C2 IP pool:** 185.142.98.12, 91.215.85.47, 45.153.241.81
(known malicious ranges, hardcoded for reproducibility).

#### 3.2.4 Semantic Shuffle

**Model:** Novel evasion strategy -- content manipulation without
classic encryption indicators.

```
Behavior:
  Process: masquerades as conhost.exe
  Strategy:
    - Modifies file content (size changes, entropy increases)
    - BUT keeps entropy < 6.5 (below encryption threshold)
    - NO extension changes
    - Designed to evade entropy heuristics
  Phases:
    progress < 0.1 -> reconnaissance
    progress >= 0.1 -> manipulation
```

**Telemetry signature:** File modifications with moderate entropy deltas
(-0.5 to +1.0), no extension changes. Looks like normal editing to
threshold-based detectors. Requires the agent to notice the pattern
(many files modified in a short window) and investigate file metadata.

### 3.3 Planned Ransomware Agents (Future)

To strengthen evaluation, implement simulator-level replicas of known
ransomware families with documented TTPs. Each is specified as pseudocode
using the mutation API (Section 3.5), making the exact state changes
and observable evidence explicit.

#### 3.3.1 LockBit Simulator

LockBit is the most active ransomware strain (2022-2025). Key behaviors:
- Self-propagating across network shares
- Disables Windows Defender via PowerShell (`Set-MpPreference`)
- Deletes shadow copies (`vssadmin delete shadows`)
- Multi-threaded encryption (extremely fast)
- Drops `Restore-My-Files.txt` ransom note per directory

```python
def lockbit_ransomware(host: HostState, rng, progress):
    t = host.clock.now()
    explorer_pid = host.processes.find_by_name("explorer.exe").pid

    # Phase 1: Defense evasion (progress >= 0.05)
    ps = host.processes.spawn_process("powershell.exe", explorer_pid,
        'powershell.exe -ep bypass -c "Set-MpPreference '
        '-DisableRealtimeMonitoring $true"', t)
    host.registry.disable_defender()           # → RegistryEvent + EventLog 5001

    # Phase 2: Inhibit recovery (progress >= 0.1)
    t = host.clock.tick(5)
    vss = host.processes.spawn_process("vssadmin.exe", ps.pid,
        "vssadmin.exe delete shadows /all /quiet", t)
    host.registry.disable_vss()                # → RegistryEvent
    host.processes.terminate(vss.pid, host.clock.tick(2))

    # Phase 3: C2 beacon (continuous)
    conn = host.connections.open_connection(ps.pid, "185.142.98.12", 443, "tcp")
    host.connections.beacon(ps.pid, "185.142.98.12", 443, interval=30)

    # Phase 4: Multi-threaded encryption (progress >= 0.2)
    workers = []
    for i in range(4):
        t = host.clock.tick(1)
        w = host.processes.spawn_process("svchost.exe", ps.pid,
            f"svchost.exe -k netsvcs -worker {i}", t)
        workers.append(w)

    targets = host.files.unencrypted_files()
    for i, file in enumerate(targets):
        worker = workers[i % 4]
        host.processes.open_file_handle(worker.pid, file.path)
        host.files.encrypt_file(file.path, host.clock.tick(0.3), ".lockbit", rng)
        host.processes.close_file_handle(worker.pid, file.path)

    # Phase 5: Ransom note (progress >= 0.8)
    for directory in host.files.list_directories():
        host.files.add_file(f"{directory}/Restore-My-Files.txt",
            content_type="doc", contents=b"YOUR FILES HAVE BEEN ENCRYPTED...")
```

**Observable evidence:** PowerShell spawn → Defender registry change →
EventLog 5001 → vssadmin spawn → VSS registry change → C2 beacon every
30s → 4 svchost.exe workers with file handles on user documents →
mass FileEvents with `.lockbit` extension → ransom notes.

#### 3.3.2 Conti Simulator

Conti uses a two-phase approach with manual lateral movement:
- Cobalt Strike beacon for C2 (modeled as rundll32.exe loader)
- Extended reconnaissance before encryption
- Selective targeting (high-value files first)
- Skip executables and system files

```python
def conti_ransomware(host: HostState, rng, progress):
    t = host.clock.now()

    # Phase 1: Cobalt Strike loader (progress >= 0.05)
    loader = host.processes.spawn_process("rundll32.exe", 1007,
        "rundll32.exe shell32.dll,ShellExec_RunDLL", t)
    host.processes.inject_module(loader.pid,
        "C:/Windows/Temp/beacon_x64.dll")         # silent DLL injection

    # Phase 2: C2 + persistence (progress >= 0.1)
    host.connections.open_connection(loader.pid, "91.215.85.47", 443, "tcp")
    host.connections.beacon(loader.pid, "91.215.85.47", 443, interval=60)
    host.registry.add_run_key("WindowsUpdate",
        "rundll32.exe shell32.dll,ShellExec_RunDLL")  # persistence

    # Phase 3: Reconnaissance (progress >= 0.2, extended)
    for file in host.files.list_all():
        host.processes.open_file_handle(loader.pid, file.path)
        host.clock.tick(0.5)                       # slow read scan
        host.processes.close_file_handle(loader.pid, file.path)
        host.connections.transfer_data(                # exfil file list
            host.connections.list_by_pid(loader.pid)[0].conn_id,
            bytes_sent=256, bytes_received=0)

    # Phase 4: Selective encryption (progress >= 0.6)
    high_value = [f for f in host.files.unencrypted_files()
                  if f.extension in (".docx", ".xlsx", ".pdf", ".pptx", ".db")]
    for file in high_value:
        host.files.encrypt_file(file.path, host.clock.tick(0.5), ".CONTI", rng)
```

**Observable evidence:** rundll32.exe with unexpected child behavior →
injected DLL (only visible via `list_process_handles`) → C2 beacon →
Run key persistence → slow file scanning (many opens, zero deltas) →
small outbound transfers → selective encryption (only high-value exts).

#### 3.3.3 BlackCat/ALPHV Simulator

BlackCat is notable for using intermittent encryption (encrypts only
parts of files), which specifically evades entropy-based detectors:

```python
def blackcat_ransomware(host: HostState, rng, progress):
    t = host.clock.now()

    # Phase 1: Unusual process (non-Windows-native name)
    proc = host.processes.spawn_process("svchostx.exe", 1007,
        "svchostx.exe --access-token ABC123 --config enc.json", t)
    host.processes.set_integrity(proc.pid, "High")    # privilege escalation

    # Phase 2: Disable recovery tools (progress >= 0.1)
    t = host.clock.tick(3)
    bcdedit = host.processes.spawn_process("bcdedit.exe", proc.pid,
        "bcdedit.exe /set {default} recoveryenabled no", t)
    host.processes.terminate(bcdedit.pid, host.clock.tick(1))
    wbadmin = host.processes.spawn_process("wbadmin.exe", proc.pid,
        "wbadmin.exe delete catalog -quiet", host.clock.tick(1))
    host.processes.terminate(wbadmin.pid, host.clock.tick(1))

    # Phase 3: Intermittent encryption (progress >= 0.2)
    ext = "." + "".join(rng.choices("abcdefghijklmnop", k=6))  # random ext
    for file in host.files.unencrypted_files():
        # Encrypt only 30% of 4KB blocks — entropy stays moderate (5.5-6.5)
        host.files.partial_encrypt(file.path, ratio=0.3,
            key=rng.randbytes(32))
        host.files.rename_file(file.path, file.path + ext)
        host.clock.tick(0.4)
```

**Observable evidence:** Non-standard process name (svchostx.exe, not
svchost.exe) → integrity escalation → bcdedit + wbadmin spawns →
file modifications with *moderate* entropy deltas (NOT the 7.5+ spike
of full encryption) → random extension. The agent must notice the
pattern (many files modified + unusual process) rather than relying
on entropy thresholds. This is the hardest scenario for passive
classifiers.

#### 3.3.4 Royal/BlackSuit Simulator

Royal uses Windows built-in tools (LOLBins) for execution and actively
clears forensic evidence:

```python
def royal_ransomware(host: HostState, rng, progress):
    t = host.clock.now()

    # Phase 1: LOLBin execution chain (progress >= 0.05)
    cmd = host.processes.spawn_process("cmd.exe", 1007,
        "cmd.exe /c wmic process call create", t)
    wmic = host.processes.spawn_process("wmic.exe", cmd.pid,
        "wmic.exe process call create svchost.exe", host.clock.tick(1))
    worker = host.processes.spawn_process("svchost.exe", wmic.pid,
        "svchost.exe -k netsvcs", host.clock.tick(1))

    # Phase 2: Anti-forensics — clear event logs (progress >= 0.1)
    wevt = host.processes.spawn_process("wevtutil.exe", cmd.pid,
        "wevtutil.exe cl Security", host.clock.tick(2))
    host.event_log.clear_log("Security")           # → EventLog 1102
    host.processes.terminate(wevt.pid, host.clock.tick(1))
    wevt2 = host.processes.spawn_process("wevtutil.exe", cmd.pid,
        "wevtutil.exe cl System", host.clock.tick(1))
    host.event_log.clear_log("System")             # → EventLog 1102
    host.processes.terminate(wevt2.pid, host.clock.tick(1))

    # Phase 3: Encryption (progress >= 0.3)
    for file in host.files.unencrypted_files():
        host.files.encrypt_file_real(file.path, key=rng.randbytes(32))
        host.files.rename_file(file.path, file.path + ".royal")
        host.clock.tick(0.3)

    # Phase 4: C2 check-in
    host.connections.open_connection(worker.pid, "45.153.241.81", 8443, "tcp")
    host.connections.transfer_data(
        host.connections.list_by_pid(worker.pid)[0].conn_id,
        bytes_sent=4096, bytes_received=512)
```

**Observable evidence:** cmd.exe → wmic.exe → svchost.exe chain
(LOLBin pattern) → wevtutil.exe spawns → event logs cleared (1102
events survive because they're logged *before* the clear) → standard
encryption with `.royal` extension → late C2 connection. The log
clearing is a critical signal — the agent should learn that 1102
events are strongly indicative of attack.

### 3.4 Scenario Mix

Default training distribution:

| Scenario | Percentage | Rationale |
|----------|-----------|-----------|
| benign | 40% | Class balance; FP pressure |
| blitz | 20% | Most common, easy to detect |
| sleeper | 15% | Hard; tests temporal reasoning |
| exfil_first | 15% | Tests network-level detection |
| semantic_shuffle | 10% | Tests evasion resistance |

Each scenario is generated at a randomly sampled observability level
from {0.3, 0.5, 0.7, 0.9} and attack progress from uniform(0.2, 0.9)
for ransomware scenarios.

### 3.5 Mutation API

The mutation API is the interface between ransomware generator scripts
and `HostState`. Each mutation method performs the state change *and*
auto-generates the appropriate telemetry events and cross-registry
side effects. This makes generators self-documenting and ensures
consistency across all six registries.

#### File Mutations

| Method | Effect | Telemetry |
|--------|--------|-----------|
| `host.files.encrypt_file(path, now, ext, rng)` | Metadata-only encryption (existing) | FileEvent with entropy_delta, ext_change |
| `host.files.encrypt_file_real(path, key)` | AES-encrypt actual bytes, recompute entropy | FileEvent with computed deltas |
| `host.files.partial_encrypt(path, ratio, key)` | Intermittent encryption (BlackCat) | FileEvent with moderate entropy_delta |
| `host.files.modify_contents(path, new_bytes)` | Arbitrary content change | FileEvent with size/entropy deltas |
| `host.files.add_file(path, content_type, ...)` | Create new file (ransom note) | FileEvent (creation) |

#### Process Mutations

| Method | Effect | Side Effects |
|--------|--------|-------------|
| `host.processes.spawn_process(name, parent, cmd, now)` | Create process (existing) | EventLog 4688, ProcessEvent |
| `host.processes.inject_module(pid, dll_path)` | Add DLL to loaded_modules (T1055) | — (silent, no telemetry) |
| `host.processes.set_integrity(pid, level)` | Privilege escalation (T1134) | Updates is_elevated |
| `host.processes.open_file_handle(pid, path)` | Track open file handle | — |
| `host.processes.close_file_handle(pid, path)` | Release file handle | — |
| `host.processes.terminate(pid, now)` | Kill process | EventLog 4689, ProcessEvent |

Note: `inject_module` is deliberately silent (no telemetry event). DLL
injection is a stealth technique — the agent can only detect it by
actively calling `list_process_handles(pid)`. This rewards investigation.

#### Network Mutations

| Method | Effect | Telemetry |
|--------|--------|-----------|
| `host.connections.open_connection(pid, remote, port, proto)` | New connection | NetEvent (outbound) |
| `host.connections.transfer_data(conn_id, sent, received)` | Increment byte counters | NetEvent (transfer) |
| `host.connections.beacon(pid, remote, port, interval)` | Periodic C2 callback | Repeated NetEvents at interval |
| `host.connections.close_connection(conn_id)` | Close connection | — |

#### Registry Mutations

| Method | Effect | Telemetry |
|--------|--------|-----------|
| `host.registry.set_value(key, name, data, type)` | Write registry value | RegistryEvent (set) |
| `host.registry.delete_key(key)` | Remove key and values | RegistryEvent (delete) |
| `host.registry.add_run_key(name, command)` | Persistence (T1547.001) | RegistryEvent |
| `host.registry.disable_defender()` | Defense evasion (T1562.001) | RegistryEvent + EventLog 5001 |
| `host.registry.disable_vss()` | Recovery inhibition (T1490) | RegistryEvent |

Note: `disable_defender()` produces *two* telemetry events — a
RegistryEvent for the key change and an EventLogEvent (5001) for
Defender reporting its own disablement. This mirrors real Windows
behavior.

#### Event Log Mutations

| Method | Effect | Telemetry |
|--------|--------|-----------|
| `host.event_log.log_event(id, source, ...)` | Append event entry | EventLogEvent |
| `host.event_log.clear_log(source)` | Delete all events for source (T1070.001) | EventLogEvent 1102 |

Note: `clear_log` auto-logs event 1102 ("Security log cleared") because
on real Windows, the act of clearing the security log is itself logged
before the clear takes effect. This is a critical forensic indicator.

#### Cross-Registry Side Effects

Some mutations trigger automatic side effects across registries:

```
host.processes.spawn_process(...)
  → host.event_log.log_event(4688, ...)     # process creation event
  → telemetry: ProcessEvent

host.registry.disable_defender()
  → host.registry.set_value(...)            # DisableRealtimeMonitoring=1
  → host.event_log.log_event(5001, ...)     # Defender disabled event
  → telemetry: RegistryEvent + EventLogEvent

host.files.encrypt_file_real(path, key)
  → host.processes.open_file_handle(pid, path)  # if pid context provided
  → telemetry: FileEvent
```

This ensures that generators don't need to manually produce every
side effect — the mutation API handles consistency automatically.

---

## 4. Agent Tools

### 4.1 Tool Specifications

Each tool executes against the simulated host state and returns a JSON
response. Tools have explicit costs that are subtracted from the
agent's reward.

**inspect_file(path) -> JSON**
Cost: -0.02. Returns file metadata from `FileRegistry`:
```json
{
  "entropy": 7.92,
  "size": 2048,
  "extension": ".docx.locked",
  "modified_ts": "2025-06-15T10:00:30",
  "content_type": "doc"
}
```
Note: `is_encrypted` is NOT returned. The agent must infer encryption
from entropy + extension + size changes.

**check_process(pid) -> JSON**
Cost: -0.02. Returns process details from `ProcessTable`:
```json
{
  "name": "svchost.exe",
  "command_line": "svchost.exe -k netsvcs -p -s Schedule",
  "parent": "System",
  "child_pids": [1020, 1021, 1022]
}
```

**scan_directory(path) -> JSON**
Cost: -0.05 (most expensive). Returns all files in directory:
```json
{
  "files": [
    {"path": "C:/Users/A/Documents/file_0001.xlsx", "size": 573130, "entropy": 4.48, "extension": ".xlsx"},
    {"path": "C:/Users/A/Documents/file_0002.docx.locked", "size": 156000, "entropy": 7.89, "extension": ".docx.locked"}
  ]
}
```
Expensive because it reveals the most information. The agent should
learn to use it selectively.

**recall_memory(query) -> JSON**
Cost: -0.03. Searches past telemetry windows by keyword overlap:
```json
{
  "matches": [
    {"text": "[t+5s] FILE path=... entropy_delta=+3.8 ext_change=.docx->.docx.locked", "metadata": {"window_id": "hist-0"}, "score": 0.67}
  ]
}
```
Implementation: `MemoryStore` with keyword-overlap similarity (not
embeddings -- keeps it fast and deterministic). Stores up to N past
windows, returns top-k matches for a given query.

**list_connections(filter?) -> JSON**
Cost: -0.04. Returns active network connections (netstat-like) from
`ConnectionTable`. Optional filter by pid, state, or remote address:
```json
{
  "connections": [
    {"conn_id": 1, "pid": 1005, "remote": "13.107.4.50:443", "state": "ESTABLISHED",
     "bytes_sent": 2048, "bytes_received": 8192, "dns_name": "windowsupdate.com",
     "tls": true, "started_at": "2025-06-15T09:55:00"},
    {"conn_id": 5, "pid": 1020, "remote": "185.142.98.12:443", "state": "ESTABLISHED",
     "bytes_sent": 128, "bytes_received": 64, "dns_name": null,
     "tls": false, "started_at": "2025-06-15T10:00:15"}
  ]
}
```
Expensive because it returns the full connection table. The agent should
learn to use targeted filters when investigating specific processes.

**inspect_connection(conn_id) -> JSON**
Cost: -0.02. Returns full details of a single connection:
```json
{
  "conn_id": 5,
  "pid": 1020,
  "process_name": "svchost.exe",
  "local": "192.168.1.100:49152",
  "remote": "185.142.98.12:443",
  "protocol": "tcp",
  "state": "ESTABLISHED",
  "bytes_sent": 128,
  "bytes_received": 64,
  "dns_name": null,
  "tls_enabled": false,
  "started_at": "2025-06-15T10:00:15"
}
```
Cheap because it targets a single known connection. Useful as a
follow-up after `list_connections` reveals suspicious entries.

**query_registry(key_path) -> JSON**
Cost: -0.03. Returns all values under a registry key from `RegistryHive`:
```json
{
  "key": "HKLM\\SOFTWARE\\Microsoft\\Windows Defender\\Real-Time Protection",
  "values": [
    {"name": "DisableRealtimeMonitoring", "data": 1, "type": "REG_DWORD"},
    {"name": "DisableBehaviorMonitoring", "data": 0, "type": "REG_DWORD"}
  ],
  "last_modified": "2025-06-15T10:00:05"
}
```
Medium cost. The agent must know *which* keys to check — investigating
the right key (e.g., Defender status after seeing EventLog 5001) is
the skill the RL policy should learn.

**list_process_handles(pid) -> JSON**
Cost: -0.04. Returns open file handles, network sockets, and loaded
modules for a process from `ProcessTable`:
```json
{
  "pid": 1020,
  "name": "svchost.exe",
  "open_file_handles": ["C:/Users/A/Documents/report.docx", "C:/Users/A/Documents/budget.xlsx"],
  "open_network_sockets": [5],
  "loaded_modules": ["ntdll.dll", "kernel32.dll", "beacon_x64.dll"],
  "user": "NT AUTHORITY\\SYSTEM",
  "integrity_level": "System",
  "is_elevated": true
}
```
Expensive because it reveals deep forensic information. The key signals:
svchost.exe with open handles on user documents is anomalous; an
unexpected DLL (beacon_x64.dll) indicates injection; System-level
integrity for a user-spawned process indicates escalation.

**query_event_log(source?, event_id?, since?) -> JSON**
Cost: -0.03. Returns filtered event log entries from `EventLog`:
```json
{
  "events": [
    {"event_id": 5001, "timestamp": "2025-06-15T10:00:05",
     "source": "Microsoft-Windows-Defender", "level": "Warning",
     "pid": 1020, "message": "Real-time protection disabled"},
    {"event_id": 1102, "timestamp": "2025-06-15T10:00:12",
     "source": "Security", "level": "Information",
     "pid": 1025, "message": "The audit log was cleared"}
  ],
  "total_count": 2
}
```
Medium cost. All parameters are optional filters. The agent should
learn to query for specific high-value event IDs (5001, 1102, 7045)
rather than dumping the entire log.

**read_file_sample(path, offset?, length?) -> JSON**
Cost: -0.04. Returns a hex-encoded byte sample from file contents
plus computed entropy:
```json
{
  "path": "C:/Users/A/Documents/report.docx",
  "offset": 0,
  "length": 256,
  "hex": "504b030414000600080000002100...",
  "entropy": 4.32,
  "magic_bytes": "PK (ZIP/DOCX)",
  "total_size": 45056
}
```
Expensive. Returns raw bytes (hex-encoded) so the agent can observe
file magic bytes (PK for ZIP/DOCX, JFIF for JPEG, random bytes for
encrypted). Default: first 256 bytes. The `magic_bytes` field provides
a human-readable interpretation of the first few bytes. Only available
when `FileRecord.contents` is populated.

**DECIDE(verdict, explanation) -> JSON**
Cost: 0.00. Terminal action that ends the investigation:
```json
{
  "verdict": "quarantine",
  "explanation": "High entropy deltas + extension changes...",
  "status": "Investigation complete."
}
```

### 4.2 Tool-Call Parsing

The parser supports two formats for model-generated tool calls:

**Qwen3 native JSON format:**
```
<tool_call>{"name": "inspect_file", "arguments": {"path": "C:/test.docx"}}</tool_call>
```

**Function-call syntax (fallback):**
```
<tool_call>inspect_file("C:/test.docx")</tool_call>
```

Positional arguments are mapped to parameter names by tool:
- `inspect_file` -> `["path"]`
- `check_process` -> `["pid"]`
- `scan_directory` -> `["path"]`
- `list_connections` -> `["filter"]`
- `inspect_connection` -> `["conn_id"]`
- `query_registry` -> `["key_path"]`
- `list_process_handles` -> `["pid"]`
- `query_event_log` -> `["source", "event_id", "since"]`
- `read_file_sample` -> `["path", "offset", "length"]`
- `recall_memory` -> `["query"]`
- `DECIDE` -> `["verdict", "explanation"]`

### 4.3 Tool Cost Design

Tools are organized into cost tiers based on the breadth of information
they reveal:

| Tier | Cost | Tools |
|------|------|-------|
| Free | 0.00 | DECIDE |
| Cheap probes | -0.02 | inspect_file, check_process, inspect_connection |
| Medium queries | -0.03 | recall_memory, query_registry, query_event_log |
| Expensive scans | -0.04 | list_connections, list_process_handles, read_file_sample |
| Bulk scans | -0.05 | scan_directory |

Costs create an efficiency-accuracy tradeoff. The agent learns that:
- Cheap probes (-0.02) are low-risk: use them freely to spot-check
  individual files, processes, or connections
- Medium queries (-0.03) require knowing *what* to look for: the right
  registry key, the right event ID, the right memory query
- Expensive scans (-0.04) reveal deep information but at high cost:
  full connection tables, process forensics, raw file bytes
- Bulk scans (-0.05) should be reserved for when other evidence suggests
  a specific directory is compromised
- DECIDE (free) should be called as soon as the agent is confident

The optimal strategy is NOT to use all tools. An agent that recognizes
obvious ransomware from telemetry alone and calls DECIDE immediately
earns +1.0 (correct) + 0.04 (efficiency bonus for 4 unused steps) +
0.10 (format bonus) = +1.14 with zero tool cost.

---

## 5. Environment

### 5.1 RansomwareDetectionEnv (Evaluation/Inference)

The evaluation environment manages the full episode lifecycle:

```python
env = RansomwareDetectionEnv(max_steps=5)
telemetry_text = env.reset(scenario_type, observability, rng)

# Multi-step loop (agent or inference code drives this):
for model_output in generate_completions(model, prompt):
    step = env.step_from_text(model_output)  # parse + execute
    if step is None:
        env.mark_format_error()
        continue
    inject(step.result_text)  # feed <tool_result> back to model
    if env.is_done:
        break

rollout = env.finish()  # computes reward, returns RolloutResult
```

**State managed by env:**
- `_episode`: Generated episode with ground truth
- `_host`: HostState (FileRegistry, ProcessTable, ConnectionTable,
  RegistryHive, EventLog, SimClock) for tool execution
- `_memory`: MemoryStore for recall_memory
- `_steps`: List of StepResults
- `_cumulative_cost`: Running total of tool costs
- `_well_formatted`: Whether all model outputs parsed correctly
- `_verdict`, `_explanation`: Final decision
- `_done`: Whether episode has terminated

**Termination conditions:**
1. Agent calls DECIDE -> verdict recorded, episode done
2. Budget exhausted (steps_taken >= max_steps) -> default to "ignore"
3. Both conditions trigger `finish()` which computes reward

### 5.2 DetectionEnv (TRL Training)

For GRPO training, TRL's `environment_factory` creates one instance per
rollout. Public methods become tools. TRL handles the full multi-step
loop natively:

```python
class DetectionEnv:
    def reset(self, **kwargs) -> str:
        # Receives scenario_data from dataset row
        # Creates HostState, runs generators, returns telemetry text

    # Original tools
    def inspect_file(self, path: str) -> str:                    # tool
    def check_process(self, pid: int) -> str:                    # tool
    def scan_directory(self, path: str) -> str:                  # tool
    def recall_memory(self, query: str) -> str:                  # tool

    # New tools (query expanded HostState)
    def list_connections(self, filter: str = "") -> str:         # tool
    def inspect_connection(self, conn_id: int) -> str:           # tool
    def query_registry(self, key_path: str) -> str:              # tool
    def list_process_handles(self, pid: int) -> str:             # tool
    def query_event_log(self, source: str = "",
                        event_id: int = 0, since: str = "") -> str:  # tool
    def read_file_sample(self, path: str,
                         offset: int = 0, length: int = 256) -> str: # tool

    # Terminal action
    def decide(self, verdict: str, explanation: str) -> str:     # tool

# GRPOTrainer setup:
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[detection_reward, format_reward],
    environment_factory=DetectionEnv,
    ...
)
```

The key insight: TRL's GRPOTrainer handles the generate-parse-execute
loop internally. We provide `DetectionEnv` as a class (not instance),
and TRL instantiates one per rollout. Public methods with type hints
and docstrings are exposed as tools.

### 5.3 Reward Function (RLVR)

Rewards are deterministic and verifiable -- the simulator provides
ground truth, so no learned reward model is needed.

```
RewardBreakdown:
  verdict_reward:    +1.0 (correct) | -1.0 (FP/wrong) | -2.0 (FN)
  action_cost:       sum of per-tool costs (negative)
  efficiency_bonus:  +0.01 per unused budget step
  format_reward:     +0.10 if output follows tool-call format

  total = verdict_reward + action_cost + efficiency_bonus + format_reward
```

**Asymmetric penalty rationale:** Missing active ransomware (-2.0) is
strictly worse than a false alarm (-1.0). This mirrors real-world SOC
priorities: a false negative means data loss; a false positive means
an analyst spends 10 minutes investigating.

**Efficiency bonus:** +0.01 per unused step rewards early stopping.
An agent that correctly detects ransomware in 1 step instead of 5
gets +0.04 extra. This teaches the agent not to investigate when
already confident.

**Format reward:** +0.10 for well-formed output (valid tool-call syntax
throughout the rollout). This provides a stable learning signal even
when verdicts are wrong.

**Reward range:** Approximately [-2.25, +1.14]. (The floor is slightly
lower than before due to the availability of more expensive tools, but
the budget cap of k_max=5 steps limits total cost to at most 5 × -0.05
= -0.25.)

---

## 6. Evaluation Framework

### 6.1 Metrics

**Detection metrics (binary: ransomware vs benign):**
- True positives (TP), False positives (FP), True negatives (TN), False negatives (FN)
- Precision, Recall, F1, Accuracy
- Verdict mapping: {alert, quarantine, block} -> ransomware prediction;
  {ignore, monitor} -> benign prediction

**Efficiency metrics:**
- Mean steps per episode
- Mean tool cost per episode
- Early stop rate (fraction of episodes using < k_max steps)

**Phase accuracy:**
- Per-attack-phase detection accuracy (encryption, exfiltration,
  reconnaissance, manipulation, benign)

**Tool usage:**
- Per-tool call frequency across all episodes
- Tool usage shift under ablation

**Tool appropriateness (NEW):**
- Per-tool-type accuracy breakdown: did the agent use the *right category*
  of tool for each scenario? Measures whether the agent learns to:
  - Use network tools (list_connections, inspect_connection) for exfil scenarios
  - Use registry tools (query_registry) when Defender/VSS changes are present
  - Use event log tools (query_event_log) when anti-forensic indicators appear
  - Use process handles (list_process_handles) when DLL injection is suspected
  - Use file sampling (read_file_sample) for intermittent encryption detection
- Tool appropriateness score: fraction of tool calls that target the
  evidence type most relevant to the ground-truth scenario

**Pareto frontier:**
- Plot accuracy vs. cost across configurations
- Identify Pareto-optimal (config, budget) pairs

### 6.2 Baselines

All baselines use the same `RansomwareDetectionEnv` interface for fair
comparison:

**RandomAgent:** Random tool selection + random verdict.
Sanity check -- any trained agent should beat this.

**ExhaustiveAgent:** Uses all k-1 budget steps with round-robin tools,
then decides via keyword heuristic. Tests whether the RL agent learns
meaningful early stopping and tool selection.

**HeuristicAgent:** Extracts entropy_delta and rename_rate directly from
telemetry text (no tools, no LLM). Applies thresholds:
- entropy > 2.0 OR renames >= 2 -> "alert"
- entropy > 1.0 OR renames >= 1 -> "monitor"
- otherwise -> "ignore"
This is the passive classifier baseline. Any LLM agent should beat it,
and the gap should widen at low observability.

**PassiveLLM (planned):** Same model, but forced to DECIDE immediately
(no investigation tools). Measures how much the active investigation
adds vs. the LLM's reasoning on raw telemetry alone.

**Distilled agent (planned):** Same architecture trained via supervised
distillation on Claude-generated traces. Tests whether RL learns a
better policy than imitation.

### 6.3 Ablation Studies

#### Tool ablation

Remove one tool at a time from the trained agent (no retraining) to
measure each tool's marginal contribution:

| Variant | Removed Tool | Expected Impact |
|---------|-------------|-----------------|
| full | — (baseline) | — |
| no_recall_memory | recall_memory | Sleeper detection (temporal correlation) |
| no_scan_directory | scan_directory | Blitz detection (bulk file view) |
| no_check_process | check_process | Process masquerading detection |
| no_list_connections | list_connections | Exfil-first detection (C2 traffic) |
| no_query_registry | query_registry | Defense evasion detection (Defender/VSS) |
| no_event_log | query_event_log | Anti-forensics detection (log clearing) |
| no_process_handles | list_process_handles | DLL injection detection (Conti) |
| no_file_sample | read_file_sample | Intermittent encryption detection (BlackCat) |
| file_tools_only | all network/registry/log tools | File-centric investigation only |
| network_tools_only | all file/registry/log tools | Network-centric investigation only |
| inspect_only | all except inspect_file | Minimal investigation capability |

Track `attempted_removed_tools`: how often the agent tries to call a
tool that's been removed. This reveals tool dependency in the learned
policy.

**Per-scenario ablation analysis (NEW).** For each tool ablation variant,
report per-scenario F1 to identify which tools are critical for which
attack types. Expected pattern: removing `list_connections` should
disproportionately hurt exfil_first detection; removing `query_registry`
should hurt LockBit/Royal detection (defense evasion indicators).

#### Budget ablation

Evaluate with k_max in {1, 3, 5, 10}:
- k=1: DECIDE-only (equivalent to passive LLM)
- k=3: Constrained investigation
- k=5: Default budget
- k=10: Extended budget (does more help?)

#### Observability ablation

Evaluate at observability in {0.1, 0.3, 0.5, 0.7, 0.9}. The key
experiment: plot F1 vs. observability for active agent vs. heuristic
vs. passive LLM. The active agent's advantage should increase as
observability decreases.

### 6.4 Evaluation Hypotheses

| ID | Hypothesis | Measurement |
|----|-----------|-------------|
| H1 | Active agent > passive LLM at obs=0.3-0.5 | F1 gap at low observability |
| H2 | RL-trained agent > distilled agent | F1 comparison, same architecture |
| H3 | Agent learns to stop early on easy cases | Early-stop rate on benign |
| H4 | Agent generalizes to unseen attack scenarios | F1 on held-out scenario type |
| H5 | Distilled 1.7B retains most 8B accuracy | F1 vs model size curve |
| H6 | recall_memory provides measurable gain | Full vs no_recall_memory F1 |
| H7 | Agent learns scenario-specific investigation strategies | Tool usage patterns per scenario |
| H8 | Network/registry/log tools improve detection of evasive attacks | F1 on exfil/BlackCat/Royal: expanded vs. file-only tools |
| H9 | Agent learns tool appropriateness (right tool for right scenario) | Tool appropriateness score across scenarios |

### 6.5 Generalization Tests

- **Held-out scenario:** Train on {benign, blitz, sleeper, semantic_shuffle},
  evaluate on exfil_first. Tests zero-shot transfer to novel attack patterns.
- **Cross-observability:** Train at obs=0.7, evaluate at obs=0.3.
  Tests whether the investigation policy generalizes to harder conditions.

---

## 7. Training Pipeline

### 7.1 Dataset Preparation

```python
# Generate scenario batch
batch = generate_training_scenarios(
    n_episodes=500,
    scenario_mix=DEFAULT_SCENARIO_MIX,  # 40% benign, 20% blitz, ...
    observability_levels=[0.3, 0.5, 0.7, 0.9],
    seed=42,
)

# Convert to TRL conversational format
dataset = [
    {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Analyze the following telemetry window."},
        ],
        "scenario_data": json.dumps({
            "scenario_type": "blitz",
            "observability": 0.7,
            "attack_progress": 0.65,
            "seed": 12345,
            "history_windows": [],
        }),
    }
    ...
]
```

The `scenario_data` JSON string is passed to `DetectionEnv.reset(**kwargs)`
by TRL. The environment reconstructs the full episode from this seed,
ensuring reproducibility.

### 7.2 Model Configuration

```
Base model:  Qwen/Qwen3-8B
Quantization: 4-bit NF4 (BitsAndBytes or Unsloth)
LoRA config:
  r = 16
  lora_alpha = 32
  target_modules = [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
  dropout = 0.0
  bias = "none"
```

### 7.3 GRPO Training Loop

```
GRPOConfig:
  num_generations = 4          # group size (rollouts per prompt)
  max_completion_length = 2048 # max tokens per rollout
  learning_rate = 1e-5
  gradient_accumulation_steps = 4
  max_steps = 200              # training iterations

reward_funcs = [detection_reward, format_reward]
environment_factory = DetectionEnv
```

Each training step:
1. Sample a batch of scenarios from the dataset
2. For each scenario, TRL generates G=4 rollouts:
   - Model generates tokens
   - On `<tool_call>`, TRL parses and calls the corresponding
     DetectionEnv method
   - Tool result is injected as text, generation continues
   - Continues until DECIDE, EOS, or max_completion_length
3. `detection_reward` reads completed environments' accumulated state
   (verdict, ground truth, costs) and returns scalar rewards
4. `format_reward` checks if output has `<think>` and tool-call tags
5. GRPO ranks the G rollouts by total reward and updates policy
   toward higher-reward rollouts

### 7.4 Training Phases (Curriculum)

**Phase 1: Binary (k=1, 2 scenarios)**
Train with only DECIDE available, only benign + blitz scenarios.
Goal: learn basic verdict mapping (is this ransomware?).

**Phase 2: Investigation (k=3, 4 scenarios)**
Add inspect_file + check_process + query_event_log. Add sleeper + exfil_first.
Goal: learn that investigation helps when telemetry is ambiguous.

**Phase 3: Full (k=5, all scenarios)**
Full tool set (11 tools), all scenarios, varied observability.
Goal: learn the complete investigation policy including network tools,
registry queries, process forensics, recall_memory, and efficient
tool selection across evidence types.

---

## 8. Running the System

### 8.1 Demo Mode (No GPU Required)

```bash
cd active-detective
python -m scripts.run_untrained --mode demo
```

Shows the full pipeline with simulated model output: telemetry generation,
virtual filesystem, tool execution, reward computation.

### 8.2 Untrained Model Inference

```bash
python -m scripts.run_untrained --mode live --model Qwen/Qwen3-0.6B
```

Loads a small untrained model, runs it through 3 episodes, shows the
multi-step tool-call loop with real model generations.

### 8.3 GRPO Training

```bash
accelerate launch -m training.train_grpo \
    --model Qwen/Qwen3-8B \
    --output-dir ./checkpoints \
    --n-episodes 500 \
    --group-size 4 \
    --max-steps 200 \
    --lr 1e-5
```

### 8.4 Baseline Evaluation

```python
from evaluation.baselines import RandomAgent, HeuristicAgent, evaluate_baseline
from training.scenarios import generate_training_scenarios

scenarios = generate_training_scenarios(n_episodes=100, seed=42)
scenario_dicts = [{"scenario_type": ep.scenario_type.value, "observability": ep.observability} for ep in scenarios.episodes]

results = evaluate_baseline(HeuristicAgent(), scenario_dicts, k_max=5, seed=42)
```

### 8.5 Ablation Sweep

```python
from evaluation.ablation import run_ablation_sweep, format_ablation_table

results = run_ablation_sweep(eval_fn, scenarios)
print(format_ablation_table(results))
```

### 8.6 Tests

```bash
python -m pytest tests/ -q          # all 276 tests
python -m pytest tests/ -k "eval"   # evaluation tests only
python -m pytest tests/ -k "train"  # training tests only
```

---

## 9. Project Structure

```
active-detective/
  simulator/
    models.py         # FileRecord, ProcessRecord, ConnectionRecord, RegistryKey,
                      # EventLogEntry, ScenarioType, Verdict, HostState
    registry.py       # FileRegistry, ProcessTable (mutable host state)
    network.py        # ConnectionTable (network connections)
    registry_hive.py  # RegistryHive (Windows registry)
    event_log.py      # EventLog (Windows event log)
    clock.py          # SimClock (monotonic time)
    host.py           # HostState composition, seed_host()
    generators.py     # Benign + ransomware generators (Actor protocol)
    telemetry.py      # Episode generation, observability filter, text formatting

  tools/
    inspection.py     # inspect_file, check_process, scan_directory, execute_tool
    network_tools.py  # list_connections, inspect_connection
    forensic_tools.py # query_registry, list_process_handles, query_event_log,
                      # read_file_sample
    parser.py         # Tool-call parser (JSON + function-call syntax)
    memory.py         # MemoryStore for recall_memory (keyword similarity)

  environment/
    env.py            # RansomwareDetectionEnv (eval/inference harness)
    reward.py         # RLVR reward function (RewardBreakdown)

  training/
    prompts.py        # System prompt templates, tool descriptions
    scenarios.py      # Batch scenario generator, save/load
    train_grpo.py     # DetectionEnv (TRL), GRPO training script

  evaluation/
    metrics.py        # DetectionMetrics, EfficiencyMetrics, Pareto, phase accuracy,
                      # tool appropriateness
    baselines.py      # RandomAgent, ExhaustiveAgent, HeuristicAgent
    ablation.py       # Tool ablation sweep (expanded), serialization

  scripts/
    run_untrained.py  # Demo and live inference (no training)

  tests/
    test_*.py         # Tests covering all modules
```

---

## 10. Design Decisions and Rationale

**Why tool-call-as-text, not discrete actions?**
The model generates `<tool_call>` tokens just like it generates any
other text. This means: (a) no separate policy head needed, (b) Qwen3's
native tool-use priors transfer directly, (c) the same model format
works for training and deployment, (d) the action space scales naturally
with new tools.

**Why GRPO over PPO?**
GRPO needs no value model (saves ~50% VRAM), is what Qwen3 was natively
trained with, and is more stable for LLM RL. PPOTrainer is deprecated
in TRL.

**Why a simulator, not real telemetry?**
We need ground truth labels with known attack phases, controllable
observability, and unlimited data. The simulator provides all three.
Real telemetry validation (Phase 4 in the roadmap) is planned but
not required for the initial paper.

**Why asymmetric rewards?**
A missed ransomware attack (FN) causes data loss. A false alarm (FP)
causes a 10-minute analyst investigation. The 2:1 penalty ratio
(-2.0 vs -1.0) reflects this asymmetry and pushes the agent toward
higher recall.

**Why keyword similarity for recall_memory?**
Embedding-based similarity requires a separate model and adds compute
cost. Keyword overlap is deterministic, fast, and sufficient for the
structured telemetry format (entropy_delta, ext_change, etc. are
keywords the model can learn to query for). Upgrading to embeddings
is straightforward if needed.

**Why HostState composition over individual registries?**
Passing `(registry, ptable, rng, now)` to generators doesn't scale:
adding ConnectionTable, RegistryHive, and EventLog would balloon the
argument list. HostState is a single facade that all generators mutate
and all tools query. This also enables automatic cross-registry side
effects (e.g., `spawn_process` auto-logs EventLog 4688) without
generators needing to know about every registry. The mutation API
documents exactly what state changes each ATT&CK technique produces.

**Why computed entropy over sampled entropy?**
Sampled entropy (7.5 + uniform(0, 0.49)) was adequate for the initial
prototype, but it can't model intermittent encryption (BlackCat), where
only a fraction of file blocks are encrypted. With actual byte contents
and Shannon entropy computed from ciphertext, intermittent encryption
*naturally* produces moderate entropy (5.5-6.5) — no hand-tuned ranges
needed. This makes the simulation more faithful and the agent's task
harder in the right way.
