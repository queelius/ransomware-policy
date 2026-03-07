# POMDP Environment Design: Frozen Snapshot with Temporal History

**Date:** 2026-03-07
**Status:** Validated design (refines the environment model from 2026-03-05 system design)

This document specifies how the Active Detective environment simulates
a POMDP for the RL-trained agent: what the agent observes, how it
interacts with the simulated host, how temporal context is provided,
and what information is passive vs. active.

---

## 1. Agent Role: Pure Classifier

The agent is a **pure classifier**. Its verdicts (ignore, monitor, alert,
quarantine, block) are classification labels signaled to external systems.
The agent does not mutate simulator state, and the adversary does not
react to the agent's actions.

Rationale: separation of detection and response. The agent's job is to
correctly identify whether ransomware is present and how severe it is.
Containment, quarantine execution, and incident response are out of scope
for the RL agent and are handled by external orchestration.

---

## 2. Temporal Model: Frozen Snapshot with Pre-Generated History

Each episode presents the agent with **1-3 prior telemetry windows**
(read-only history context) plus a **current window** whose underlying
host state the agent actively investigates.

```
Episode structure:
  [window t-2]: history (read-only, in prompt)
  [window t-1]: history (read-only, in prompt)
  [window t-0]: CURRENT (agent investigates this host)

  agent makes up to k_max tool calls against frozen host at t=0
  agent renders verdict
```

### 2.1 History Window Generation

**Attack episodes**: run the same attack generator at linearly-spaced
earlier `progress` values. For an episode with target `progress=0.6`
and `n_history=2`:

```python
history_progress = [0.2, 0.4]  # progress * (i+1) / (n_history+1)
current_progress = 0.6
```

Each window gets a fresh `HostState` at an offset timestamp
(`t - (n_history - i) * 120` seconds), so the host is independently
seeded per window. The agent only investigates the current window's host.

**Benign episodes**: each history window uses a different randomly-chosen
benign generator (office_edits, browser_downloads, backup_operations,
av_scan, system_maintenance). This produces natural variation in
background activity across windows.

### 2.2 Why Not Multi-Window Streaming?

Multi-window streaming (where the adversary advances during investigation)
requires the agent to learn two policies simultaneously: *what to
investigate* and *when to act*. This makes credit assignment harder,
rollouts longer (more tokens per episode), and convergence slower
(~5-10x). The frozen model with history provides temporal reasoning
signal without the training complexity. Streaming is a natural v2 once
the agent can classify reliably.

### 2.3 Why Not Progressive Reveal?

A "advance_time" tool that generates new windows on demand adds temporal
control but conflates investigation budget with temporal exploration.
The agent should spend its k_max budget on investigation quality, not
on deciding whether to wait. This is also a v2 consideration.

---

## 3. Observability: Scalar Per-Episode

A single `observability` float in [0.0, 1.0] filters events uniformly:

```python
filtered = [e for e in all_events if rng.random() < observability]
```

The training curriculum varies observability across episodes
(default levels: 0.3, 0.5, 0.7, 0.9). Per-event-type visibility rates
and per-phase matrices were considered but add tuning complexity without
clear benefit for v1.

---

## 4. Passive vs. Active Information

A key design decision: **what the agent sees for free in telemetry
vs. what it must actively investigate**.

### 4.1 Passive Telemetry (in prompt)

File events show:
- `path`, `operation` (write/rename/delete)
- `size_delta` (bytes changed)
- `extension_change` (e.g., `.docx -> .locked`)
- `pid` (process that performed the operation)
- `timestamp`

Process events show:
- `name`, `pid`, `parent_pid`
- `command_line`

Network events show:
- `direction` (inbound/outbound), `bytes_transferred`
- `remote_address`, `remote_port`

Registry events show:
- `key_path`, `value_name`, `action` (set/delete)

Event log events show:
- `source`, `event_id`, `message`, `level`

### 4.2 What Is NOT in Passive Telemetry

**`entropy_delta` is removed from passive telemetry events.**

Computing file entropy requires reading file contents and running
Shannon entropy calculation. This is an active, computationally
expensive operation in real EDR systems, not something that appears
in passive event streams from minifilter drivers or ETW.

The agent must use `inspect_file` or `read_file_sample` to discover
entropy. This creates a natural information asymmetry: the agent gets
*hints* from passive telemetry (extension changes, size deltas, rapid
writes) and must decide which hints are worth spending tool budget to
verify.

This design decision makes tool selection genuinely learnable. Without
it, the agent can classify ransomware from passive telemetry alone,
rendering the investigation tools optional.

### 4.3 Active Investigation (via tools)

The agent has 9 investigation tools + DECIDE:

| Tool | Returns | Cost |
|------|---------|------|
| `inspect_file(path)` | entropy, size, extension, is_encrypted, timestamps | -0.02 |
| `check_process(pid)` | name, cmd, children, modules, handles, user, integrity | -0.02 |
| `scan_directory(path)` | file listing with entropy, size, is_encrypted | -0.05 |
| `list_connections(filter?)` | active network connections by state | -0.03 |
| `inspect_connection(conn_id)` | connection details, bytes transferred | -0.02 |
| `query_registry(key_path)` | registry key values | -0.02 |
| `list_process_handles(pid)` | open file handles, sockets, loaded modules | -0.03 |
| `query_event_log(source?, id?)` | filtered event log entries | -0.03 |
| `read_file_sample(path)` | hex bytes, entropy, magic bytes | -0.03 |
| `DECIDE(verdict, explanation)` | terminal action | 0.00 |

**`recall_memory` is removed.** With 1-3 history windows directly in
the prompt context, keyword-similarity search over history is redundant.
The agent can read the history that's already there.

---

## 5. Reward Function

Unchanged from v1:

| Component | Value |
|-----------|-------|
| Correct detection (ransomware correctly identified) | +1.0 |
| False positive (benign classified as ransomware) | -1.0 |
| False negative (ransomware missed) | -2.0 |
| Correct benign (benign correctly ignored) | +1.0 |
| No verdict (budget exhausted without DECIDE) | -1.0 (benign) / -2.0 (ransomware) |
| Action cost | cumulative tool costs |
| Efficiency bonus | `0.1 * (1 - steps/k_max)` |
| Format: thinking tags | +0.05 |
| Format: tool call syntax | +0.05 |

The asymmetric penalty (FN=-2 vs. FP=-1) reflects that missing
ransomware is worse than a false alarm.

---

## 6. Implementation Changes Required

### 6.1 Changes

1. **`simulator/telemetry.py`**: `generate_episode()` produces a list
   of windows (multi-progress snapshots). `format_telemetry_window()`
   drops `entropy_delta` from file event output.

2. **`training/prompts.py`**: `build_chat_messages()` formats history
   windows as labeled prior context in the user message. Remove
   `recall_memory` from `TOOL_DESCRIPTIONS`.

3. **`environment/env.py`**: `reset()` accepts/stores multi-window data.
   Remove `recall_memory` execution branch.

4. **`training/train_grpo.py`**: `DetectionEnv` drops `recall_memory`
   method. `prepare_dataset()` includes history in scenario data.

5. **`tools/inspection.py`**: Remove `recall_memory` from `TOOL_COSTS`
   and `execute_tool`. Remove `tools/memory.py` if it exists as a
   separate module.

6. **`evaluation/ablation.py`**: Remove `recall_memory` from `ALL_TOOLS`
   and drop `no_recall_memory` variant.

7. **`evaluation/baselines.py`**: Remove `recall_memory` from agent
   tool pools.

### 6.2 Unchanged

- All 9 generators (benign + attack)
- HostState and all sub-registries
- Remaining 9 investigation tools
- GRPO training loop
- Reward computation
- Tool parser (JSON + function-call syntax)
- Evaluation metrics, Pareto analysis

---

## 7. Design Decisions Log

| Decision | Choice | Alternatives Considered |
|----------|--------|------------------------|
| Agent role | Pure classifier | Classifier + responder; full POMDP agent |
| Temporal model | Frozen snapshot + history | Multi-window streaming; progressive reveal |
| Observability | Scalar per-episode | Per-event-type rates; per-phase matrix |
| `entropy_delta` | Removed from telemetry | Keep in telemetry (less realistic) |
| `recall_memory` | Removed | Keep for ablation study |
| History generation | Multi-progress snapshots | Template-based summaries |
