# Telemetry Generation Strategy

## Core Formalism
**The system state is only partially observable through telemetry data.**

We observe telemetry events `Z_t` that are emissions from the true latent state `X_t` of the system:
- `X_t`: True system state (benign, compromised, under attack, etc.)
- `Z_t`: Observable telemetry (process events, file operations, network connections)
- `P(Z_t | X_t)`: Emission probability - how likely we observe certain telemetry given the true state

## Telemetry Data Generation Approaches

### 1. Synthetic Generation (Current Approach)
**Pros:**
- Full control over scenarios
- Can generate unlimited data
- Ground truth labels available

**Cons:**
- May not capture real-world complexity
- Risk of "toy problem" that doesn't generalize

### 2. Real System Instrumentation
**Options:**

#### A. Sysmon/ETW on Windows
- Rich process creation events with command lines
- File creation/modification events
- Network connections
- Registry modifications
- Process injection detection

#### B. Auditd/eBPF on Linux
- System call tracing
- File access patterns
- Network activity
- Process genealogy

#### C. EDR/XDR Data Export
- CrowdStrike, SentinelOne, Defender for Endpoint APIs
- Pre-processed telemetry with some enrichment
- Often includes behavioral indicators

### 3. Hybrid Approach (Recommended)

```
Real System Baseline + Synthetic Attack Injection
```

1. **Collect real baseline telemetry** from normal systems
2. **Inject synthetic attack patterns** into the baseline
3. **Maintain temporal consistency** (attacks that evolve over time)

## Key Telemetry Events for Ransomware Detection

### Process Events
```json
{
  "event_type": "process_create",
  "timestamp": "2024-01-01T10:00:00Z",
  "pid": 1234,
  "ppid": 1000,
  "process_name": "powershell.exe",
  "command_line": "powershell -enc [base64]",
  "user": "DOMAIN\\user",
  "integrity_level": "medium",
  "parent_name": "explorer.exe"
}
```

### File Events
```json
{
  "event_type": "file_write",
  "timestamp": "2024-01-01T10:00:01Z",
  "pid": 1234,
  "file_path": "C:\\Users\\user\\Documents\\file.docx",
  "operation": "write",
  "bytes_written": 1024,
  "entropy": 7.95  // Near 8.0 suggests encryption
}
```

### Network Events
```json
{
  "event_type": "network_connection",
  "timestamp": "2024-01-01T10:00:02Z",
  "pid": 1234,
  "direction": "outbound",
  "dest_ip": "192.168.1.100",
  "dest_port": 445,
  "protocol": "SMB",
  "bytes_sent": 5000
}
```

### Behavioral Signals (Derived)
- **Entropy changes**: Files going from low to high entropy
- **File extension changes**: .docx → .docx.locked
- **Ransom note creation**: Multiple "README.txt" files created
- **Shadow copy deletion**: vssadmin.exe delete shadows
- **Rapid file iteration**: Accessing many files quickly
- **Process injection**: Remote thread creation events

## Implementation Steps

### Phase 1: Enhanced Synthetic Generation
1. Expand `telemetry-sim.py` with more realistic patterns:
   - Living-off-the-land techniques (PowerShell, WMI, etc.)
   - Lateral movement patterns
   - Data staging before exfiltration
   - Anti-forensics (log clearing, shadow deletion)

### Phase 2: Real Data Collection Pipeline
1. Set up Sysmon/Auditd collection
2. Parse logs into normalized JSON format
3. Create baseline behavior profiles
4. Label known-good periods

### Phase 3: Attack Injection Framework
1. Define attack scenarios as state machines
2. Inject attack telemetry maintaining temporal consistency
3. Add noise and variations to avoid overfitting

## Critical Questions to Address

1. **Window Size**: How much history does the model need?
   - Current: 120-second windows
   - Consider: Variable-length windows based on activity density

2. **Temporal Resolution**: How fine-grained should events be?
   - Second-level? Millisecond-level?
   - Group related events into "bursts"?

3. **Feature Engineering vs Raw Events**:
   - Current: Some features (entropy_delta, rename_rate)
   - Alternative: Let model learn all features from raw events

4. **Partial Observability Challenges**:
   - Encrypted C2 channels (can't see content)
   - Fileless attacks (no file events)
   - Legitimate tools used maliciously (context matters)

## Next Steps

1. Implement richer synthetic scenarios in `telemetry-sim.py`
2. Create a Sysmon configuration for real data collection
3. Build telemetry normalization pipeline
4. Design attack injection state machines
5. Create evaluation framework for detection accuracy