# Prompted Evaluation & Data Sources Design

## Overview

Two workstreams that feed the four-stage research arc (prompting -> SFT -> GRPO -> distillation):

1. **Prompted evaluation via MCP** — Run Claude Opus 4.6 (1M context) as a zero-training baseline through the active-detective evaluation framework, using an MCP server to expose investigation tools
2. **Fine-tuning data sources YAML** — Curated list of training data with good provenance for the domain knowledge SFT stage

## 1. MCP Server for Prompted Evaluation

### 1.1 Purpose

Let Claude Code act as the investigation agent — no Anthropic SDK wrapper, no separate script. Claude Code already knows how to use MCP tools. We build an MCP server backed by the active-detective simulator, and Claude Code naturally investigates.

### 1.2 Architecture

```
Claude Code (Opus 4.6, 1M context)
    │
    ├── reads telemetry window(s) from prompt
    ├── calls MCP tools to investigate
    │       │
    │       ▼
    │   MCP Server (Python, stdio)
    │       │
    │       ├── HostState (frozen snapshot from scenario)
    │       ├── 9 investigation tools (backed by simulator)
    │       ├── DECIDE tool (terminal action)
    │       └── writes EvalResult to results.jsonl
    │
    └── renders verdict via DECIDE tool
```

### 1.3 MCP Server Implementation

**File**: `active-detective/mcp_server.py`

**Protocol**: stdio (simplest, works with `claude` CLI)

**Scenario loading**: Environment variable `SCENARIO_SEED` controls which scenario is generated. On startup, the server:
1. Reads `SCENARIO_SEED` and `SCENARIO_TYPE` (required — one of: `benign`, `blitz`, `sleeper`, `exfil_first`, `semantic_shuffle`), plus optional `OBSERVABILITY` (default 0.7) and `ATTACK_PROGRESS` (default 0.5, ignored for benign)
2. Generates an `Episode` via `generate_episode()`
3. Stores the `HostState` for tool execution
4. Exposes a `get_telemetry()` MCP tool that returns the formatted telemetry text (current window + history windows with `--- Window t-N (prior) ---` labels, same format as `training/prompts.py` lines 170-177). The evaluation prompt instructs the agent to call this first.

**Tools exposed** (10 total):

| MCP Tool | Maps to | Cost |
|----------|---------|------|
| `inspect_file(path)` | `tools.inspection.inspect_file` | -0.02 |
| `check_process(pid)` | `tools.inspection.check_process` | -0.02 |
| `scan_directory(path)` | `tools.inspection.scan_directory` | -0.05 |
| `list_connections(filter_state?)` | `tools.network_tools.list_connections` | -0.03 | Note: codebase has `filter`/`filter_state` split — MCP canonicalizes to `filter_state` |
| `inspect_connection(conn_id)` | `tools.network_tools.inspect_connection` | -0.03 |
| `query_registry(key_path)` | `tools.forensic_tools.query_registry` | -0.03 |
| `list_process_handles(pid)` | `tools.forensic_tools.list_process_handles` | -0.03 |
| `query_event_log(source?, event_id?, since?)` | `tools.forensic_tools.query_event_log` | -0.04 |
| `read_file_sample(path, offset?, length?)` | `tools.forensic_tools.read_file_sample` | -0.04 |
| `DECIDE(verdict, explanation)` | Terminal action | 0.0 |

**State tracking**: The server maintains per-session:
- `steps`: number of tool calls made
- `cumulative_cost`: sum of tool costs
- `tool_usage`: list of tools called (for ablation analysis)
- `verdict`: set when `DECIDE()` is called

**Result recording**: When `DECIDE()` is called, the server computes the `RewardBreakdown` (using existing `environment/reward.py`) and appends to a results JSONL file. The JSONL schema maps to `EvalResult` fields:

```json
{
  "seed": 42,
  "scenario_type": "sleeper",
  "ground_truth_is_ransomware": true,
  "attack_phase": "encryption",
  "observability": 0.7,
  "verdict": "quarantine",
  "explanation": "...",
  "steps": 3,
  "cumulative_cost": -0.09,
  "tools_used": ["inspect_file", "read_file_sample", "query_registry"],
  "reward": 0.86,
  "correct": true,
  "is_ransomware_gt": true,
  "k_max": 5
}
```

**Budget enforcement**: The server allows up to `k_max` tool calls (default 5, configurable via `K_MAX` env var). After the budget is exhausted, all tools except `DECIDE` return an error message prompting the agent to render a verdict. This is the sole mechanism for budget control (no CLI-level turn limits).

### 1.4 MCP Configuration

**File**: `active-detective/.mcp.json` (or passed via `--mcp-config`)

```json
{
  "mcpServers": {
    "detective": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/active-detective",
      "env": {
        "SCENARIO_SEED": "42",
        "RESULTS_FILE": "results.jsonl"
      }
    }
  }
}
```

### 1.5 Batch Evaluation Script

**File**: `active-detective/scripts/eval_prompted.sh`

A Python helper (`scripts/generate_eval_manifest.py`) pre-generates a manifest JSONL of scenario configs (seed, type, observability, progress) using the same `DEFAULT_SCENARIO_MIX` from `training/scenarios.py`. The batch script iterates over this manifest:

```bash
#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:-"eval_manifest.jsonl"}
MODEL=${2:-claude-opus-4-6}
RESULTS="results/prompted_${MODEL}_$(date +%Y%m%d_%H%M%S).jsonl"

# Generate manifest if not provided
if [ ! -f "$MANIFEST" ]; then
    python scripts/generate_eval_manifest.py --n 200 --output "$MANIFEST"
fi

while IFS= read -r line; do
    seed=$(echo "$line" | python -c "import sys,json; print(json.load(sys.stdin)['seed'])")
    stype=$(echo "$line" | python -c "import sys,json; print(json.load(sys.stdin)['scenario_type'])")
    obs=$(echo "$line" | python -c "import sys,json; print(json.load(sys.stdin)['observability'])")
    prog=$(echo "$line" | python -c "import sys,json; print(json.load(sys.stdin)['attack_progress'])")

    SCENARIO_SEED=$seed SCENARIO_TYPE=$stype OBSERVABILITY=$obs \
    ATTACK_PROGRESS=$prog RESULTS_FILE="$RESULTS" \
        claude -p "$(cat prompts/eval_prompt.md)" \
        --model "$MODEL" \
        --mcp-config detective.json
done < "$MANIFEST"

python scripts/compute_metrics.py "$RESULTS"
```

**Note on permissions**: The user must configure tool trust for the MCP server's tools before batch runs (or use `--permission-mode` if supported). Each `claude` invocation spawns a fresh MCP server instance.

### 1.6 Metrics Computation

**File**: `active-detective/scripts/compute_metrics.py`

Reads the results JSONL and computes:
- `DetectionMetrics` (precision, recall, F1, accuracy) via existing `evaluation/metrics.py`
- `EfficiencyMetrics` (mean steps, mean cost, early stop rate)
- Per-scenario-type breakdown
- Per-observability-level breakdown
- Comparison table (if multiple result files provided)

### 1.7 Prompt Design

**File**: `active-detective/prompts/eval_prompt.md`

For Opus with 1M context, the prompt includes:

1. **Task description**: You are investigating a host for ransomware. Call `get_telemetry()` first to see the host telemetry, then use investigation tools, and render a verdict via `DECIDE`. You have up to 5 investigation steps — unnecessary tool calls reduce your score.

2. **Domain knowledge section** (what SFT gives the small model, Opus gets in-context):
   - Ransomware kill chain phases (reconnaissance, credential access, lateral movement, exfiltration, encryption)
   - Key Windows registry indicators (Defender disabled, VSS deleted, Run keys)
   - File entropy significance (normal ranges by type, encryption = 7.5+)
   - Common evasion techniques (process masquerading, no extension change, moderate entropy)

3. **Verdict definitions**:
   - `ignore`: No indicators of compromise
   - `monitor`: Suspicious but inconclusive
   - `alert`: Likely malicious, needs human review
   - `quarantine`: High confidence ransomware, isolate host
   - `block`: Active encryption in progress, immediate action

4. **Few-shot examples** (2-3 complete investigations):
   - Benign case: backup_operations (high file reads, zero deltas — false positive trap)
   - Attack case: slow_sleeper (subtle entropy changes, needs temporal reasoning)
   - Each shows: telemetry -> tool calls -> reasoning -> verdict

5. **Telemetry window(s)**: Current + history windows, same format as RL agent sees

For the Qwen prompted baseline, a minimal version (just sections 1, 3, 5) to fit in 4K-8K context.

### 1.8 Tool Ablation

The MCP server accepts a `TOOL_SET` env var that maps to the `ABLATION_VARIANTS` dict in `evaluation/ablation.py` (12 existing variants: `full`, `no_scan_directory`, `no_check_process`, `inspect_only`, `file_tools_only`, `network_tools_only`, `v1_tools`, etc.). An additional `no_tools` variant is added for MCP evaluation — exposes only `get_telemetry` and `DECIDE` (passive analysis).

For MCP batch evaluation, a focused subset is recommended to manage API cost: `full`, `file_tools_only`, `no_tools`, plus 1-2 others of interest.

## 2. Local Model Evaluation (Qwen Prompted Baseline)

For the apples-to-apples comparison with the untrained Qwen3-4B/1.7B, we use the existing Python-based evaluation with a new `LocalModelAgent` class (to be implemented):

- **New class**: `LocalModelAgent` in `evaluation/baselines.py` — loads a local checkpoint (or HuggingFace model ID) via transformers, generates `<tool_call>` responses
- Uses existing dual-format parser (`tools/parser.py`) and `RansomwareDetectionEnv`
- Same scenarios, same metrics, different agent
- Minimal prompt (fits in 4K-8K context)
- Extends the existing baseline pattern (`RandomAgent`, `ExhaustiveAgent`, `HeuristicAgent`)

Note: `PassiveLLM` (mentioned in baselines.py docstring) also needs implementation — it's a `LocalModelAgent` that is forced to `DECIDE` immediately without tools (the "no tools" ablation for local models).

This runs locally on the user's RTX 3060.

## 3. Fine-Tuning Data Sources YAML

### 3.1 Purpose

Curated list of training data sources for the domain knowledge SFT stage. Good provenance — every source has a URL, license, and clear description of what it provides.

### 3.2 File Location

`fine-tuning/data/sources.yaml`

### 3.3 Source Categories

**Knowledge bases** (for SFT training data generation):
- MITRE ATT&CK ransomware techniques
- Windows Internals documentation (registry, processes, services)
- DFIR forensic analysis guides
- Ransomware family behavior reports (public threat intel)

**Real traces** (for evaluation, not training):
- Malware Bazaar ransomware samples (abuse.ch)
- EMBER dataset (endpoint malware behavioral)
- Public CTI ransomware incident reports
- Atomic Red Team test cases

**Synthetic** (from our generators):
- active-detective simulator output
- prompting-only telemetry generators

### 3.4 Schema

```yaml
sources:
  - name: "Source Name"
    url: "https://..."
    type: knowledge_base | real_traces | synthetic
    use: training | evaluation | both
    license: "SPDX identifier or description"
    topics:
      - ransomware_ttps
      - windows_internals
      - forensic_analysis
    format: "json | csv | text | pcap | evtx"
    notes: "Any caveats, preprocessing needed, etc."
```

### 3.5 Populating the YAML

Initial population via web research for well-known, freely available sources. The YAML is a living document — updated as new sources are found or evaluated for quality.

## 4. Evaluation Matrix

All agents evaluated on the same scenario set (default 200 scenarios, same seeds):

| Agent | Context | Tools | Training | Hardware |
|-------|---------|-------|----------|----------|
| Claude Opus 4.6 | 1M (full prompt) | MCP (9 tools) | none | Claude Code (free) |
| Qwen3-4B prompted | 4-8K (minimal) | Python (9 tools) | none | RTX 3060 |
| Qwen3-4B + SFT | 4-8K | Python (9 tools) | domain SFT | RTX 3060 |
| Qwen3-4B + SFT + GRPO | 4-8K | Python (9 tools) | domain SFT + RL | RTX 3060 |
| Qwen3-1.7B distilled | 4-8K | Python (9 tools) | distilled | RTX 3060 |
| HeuristicAgent | — | none | algorithmic | CPU |
| ExhaustiveAgent | — | round-robin | algorithmic | CPU |
| RandomAgent | — | random | algorithmic | CPU |

Metrics per agent:
- Detection: precision, recall, F1, accuracy (binary: ransomware vs benign)
- Efficiency: mean steps, mean cost, early stop rate
- Per-scenario breakdown: blitz, sleeper, exfil-first, semantic_shuffle, benign
- Per-observability breakdown: 0.3, 0.5, 0.7, 0.9
- Tool usage patterns: which tools, how often, in what order

## 5. Deliverables

### Phase 1 (Prompting — current)
1. `active-detective/mcp_server.py` — MCP server backed by simulator (9 tools + `get_telemetry` + `DECIDE`)
2. `active-detective/prompts/eval_prompt.md` — Full prompt for Opus (domain knowledge + few-shot)
3. `active-detective/prompts/eval_prompt_minimal.md` — Minimal prompt for Qwen baseline
4. `active-detective/scripts/eval_prompted.sh` — Batch evaluation runner
5. `active-detective/scripts/generate_eval_manifest.py` — Scenario manifest generator
6. `active-detective/scripts/compute_metrics.py` — Results aggregation
7. `active-detective/.mcp.json` — MCP configuration
8. `active-detective/evaluation/baselines.py` — Add `LocalModelAgent` and `PassiveLLM` classes

### Phase 2 (Data Sources — next)
7. `fine-tuning/data/sources.yaml` — Curated data source list

### Later Phases (not in this spec)
- SFT training pipeline updates (for domain knowledge, not telemetry)
- GRPO training on SFT checkpoint
- Distillation pipeline
- Full evaluation comparison paper
