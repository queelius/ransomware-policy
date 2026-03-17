# SFT Domain Knowledge Data Pipeline Design

## Overview

A principled, re-runnable pipeline for generating ~2,000 supervised fine-tuning examples from curated domain knowledge sources. Every example is traceable to its source document, every generation step is logged with the prompt used, and the entire pipeline can be re-run from cached content.

**Note on reproducibility**: LLM-generated data is inherently non-deterministic. The pipeline is *re-runnable* (all inputs and prompts are captured), not *deterministic*. The provenance log is the ground truth record of a specific generation run. Each run produces a `run_id` (ISO timestamp) to distinguish outputs. Re-running replaces the dataset (old versions should be committed or archived first).

## Goals

1. Generate ~2,000 high-quality SFT examples covering ransomware TTPs, Windows internals, and forensic analysis
2. Full provenance — every example links back to source document, section, and generation prompt
3. Re-runnability — all inputs, prompts, and content cached; each run tagged with `run_id`
4. Observability — generation logs, quality metrics, and source coverage statistics

## 1. Data Format

### 1.1 Training Examples (JSONL)

Each line in `sft_domain_knowledge.jsonl`:

```json
{
  "messages": [
    {"role": "system", "content": "You are a ransomware forensic analyst with deep expertise in Windows internals."},
    {"role": "user", "content": "What does Windows Event ID 5001 from Windows Defender indicate?"},
    {"role": "assistant", "content": "Event ID 5001 indicates that Windows Defender real-time protection has been disabled..."}
  ],
  "metadata": {
    "source": "windows_event_log_reference",
    "source_url": "https://learn.microsoft.com/en-us/windows/win32/wes/windows-event-log-reference",
    "source_section": "Security Event IDs",
    "example_type": "qa",
    "topics": ["windows_internals"],
    "generation_prompt_id": "event_id_qa_v1",
    "generated_at": "2026-03-17T14:30:00Z",
    "generator_model": "claude-opus-4-6",
    "quality_score": null
  }
}
```

### 1.2 Metadata Fields

| Field | Purpose |
|-------|---------|
| `source` | Explicit `key` field from sources.yaml entry (machine-readable, slug-style) |
| `source_url` | Original URL for verification |
| `source_section` | Section/page/technique ID within the source |
| `example_type` | `qa`, `instruction`, or `passage` |
| `topics` | From sources.yaml topic taxonomy |
| `generation_prompt_id` | References the prompt template used |
| `generated_at` | ISO timestamp |
| `generator_model` | Model that generated the example |
| `quality_score` | Optional post-hoc quality rating (null until reviewed) |

### 1.3 Example Types

- **qa**: Direct question → factual answer. Tests knowledge recall.
- **instruction**: "Analyze/explain/describe X" → structured reasoning. Tests applied knowledge.
- **passage**: System prompt + "Continue this analysis" → domain-rich completion. Bakes in knowledge patterns.

## 2. Source Processing Pipeline

### 2.1 Architecture

```
sources.yaml (curated list)
    ↓
source_fetcher.py — downloads/caches raw content per source
    ↓
raw_content/ (cached, gitignored)
    ↓
generate_sft_data.py — orchestrates Claude Code subagents
    ↓
  ┌─ For each source:
  │   1. Load raw content
  │   2. Load prompt templates for this source type
  │   3. Dispatch Claude Code subagent with content + prompts
  │   4. Subagent generates examples, writes to per-source JSONL
  │   5. Log generation metadata
  └─
    ↓
per-source JSONL files in data/raw/
    ↓
merge_and_validate.py — dedup, format check, stats
    ↓
data/sft_domain_knowledge.jsonl (final dataset)
data/generation_log.jsonl (full provenance log)
data/coverage_report.md (statistics)
```

### 2.2 Source Fetcher

`fine-tuning/scripts/source_fetcher.py`

For each source in sources.yaml:
- **GitHub repos** (MITRE ATT&CK STIX, Atomic Red Team): Clone or download specific files via GitHub API / raw URLs
- **Web pages** (CISA, Kaspersky, DFIR Report, Microsoft Learn): Fetch HTML, extract text content
- **PDFs** (SANS cheat sheets): Download and extract text via pdfplumber
- **Local** (simulator, prompting-only): Read directly from repo

All fetched content cached to `fine-tuning/data/raw_content/` (gitignored). A manifest file `raw_content/manifest.json` records what was fetched, when, and content hashes for re-runnability.

**Content extraction policies:**
- **MITRE ATT&CK STIX**: Filter Enterprise techniques by tactic (Impact, Defense Evasion, Execution, Persistence, Lateral Movement) — ~150 techniques, not the full 600+
- **Atomic Red Team**: All atomics under `atomics/T1486/`, `atomics/T1490/`, plus techniques tagged with ransomware in the index
- **DFIR Report / Kaspersky**: Fetch article index, filter by keyword (ransomware, encryption, exfiltration, lateral movement), limit to 20 most recent matching articles per source
- **SANS PDF**: Manual download (requires free registration), then automated text extraction via pdfplumber
- **Windows Event Log**: Fetch key reference pages (Security, System, Application event IDs relevant to forensics — curated list in the prompt template)
- **Local sources**: Read directly from repo paths

### 2.3 Prompt Templates

`fine-tuning/prompts/` directory with one YAML file per source type:

```yaml
# fine-tuning/prompts/mitre_attack.yaml
source: mitre_attack_stix
description: "Generate QA and instruction examples from ATT&CK technique descriptions"

templates:
  - id: technique_qa_v1
    type: qa
    system: "You are a ransomware forensic analyst."
    prompt: |
      Given this ATT&CK technique description, generate 3-5 QA pairs that would
      help a small language model understand this technique. Cover: what it is,
      how to detect it, what artifacts it leaves, and how ransomware groups use it.

      Technique: {technique_id} — {technique_name}
      Description: {description}
      Detection: {detection}
      Mitigations: {mitigations}

      Output as JSON array of {question, answer} objects.
    target_count: 3-5 per technique

  - id: technique_instruction_v1
    type: instruction
    system: "You are a ransomware forensic analyst."
    prompt: |
      Write an instruction-following example where an analyst asks about
      detecting {technique_name} on a Windows host, and you provide a detailed
      response covering registry keys, event log IDs, and file system artifacts.
    target_count: 1 per technique
```

Each prompt template has a stable `id` that's recorded in the example metadata for traceability.

### 2.4 Generation Orchestration

`fine-tuning/scripts/generate_sft_data.py`

The main orchestration script:
1. Reads `sources.yaml` and `prompts/*.yaml`
2. Loads cached raw content from `raw_content/`
3. For each source × prompt template combination:
   - Chunks content into manageable pieces (e.g., one ATT&CK technique at a time)
   - Dispatches Claude Code subagent with the content chunk + prompt template
   - Subagent returns generated examples as JSON
   - Script validates format, adds metadata, writes to `data/raw/{source_name}.jsonl`
   - Logs the generation to `data/generation_log.jsonl`
4. Reports progress and coverage statistics

**Subagent invocation**: `generate_sft_data.py` is run inside a Claude Code session. It uses `subprocess` to invoke `claude -p "<prompt>" --output-format json` for each content chunk. The prompt includes the content chunk + prompt template + output format instructions. The script parses the JSON response, validates example format, and writes to the per-source JSONL. If a subagent returns malformed JSON, the script logs the error and retries once; on second failure it skips the chunk and records it in the generation log.

**Concurrency**: Serial processing (one chunk at a time) to keep the pipeline simple and avoid rate issues. A `--dry-run` flag prints prompts and estimated token counts without dispatching.

**Content ordering**: Chunks are deterministically ordered (sorted by technique ID, article title, or filename) so that the generation log is stable across runs. Each run is tagged with a `run_id` (ISO timestamp). Generation log records the full prompt sent and response received for each chunk.

### 2.5 Merge and Validate

`fine-tuning/scripts/merge_and_validate.py`

1. Reads all `data/raw/*.jsonl` files
2. Validates each example: correct message format, required metadata fields, non-empty content
3. Deduplicates: exact match on user+assistant content (keeps first occurrence), plus fuzzy dedup via Jaccard similarity on tokenized user content (threshold 0.85 — flags near-duplicates for review)
4. Computes statistics:
   - Total examples by source, type, topic
   - Average message lengths
   - Topic coverage heatmap
   - Source balance (examples per source)
5. Writes:
   - `data/sft_domain_knowledge.jsonl` — final clean dataset
   - `data/coverage_report.md` — human-readable statistics
   - `data/validation_log.jsonl` — dropped examples with reasons

## 3. Source Coverage Targets

| Source | Key | Types | Target |
|--------|-----|-------|--------|
| MITRE ATT&CK STIX | `mitre_attack_stix` | QA + instruction | ~400 |
| Atomic Red Team | `atomic_red_team` | QA + instruction | ~300 |
| CISA StopRansomware | `cisa_stopransomware` | QA + instruction + passage | ~200 |
| DFIR Report | `dfir_report` | instruction + passage | ~300 |
| Windows Event Log | `windows_event_log` | QA | ~200 |
| Kaspersky SecureList | `kaspersky_securelist` | instruction + passage | ~200 |
| SANS DFIR Cheat Sheets | `sans_dfir` | QA | ~150 |
| Prompting-only generator | `prompting_only_gen` | instruction | ~250 |
| **Total** | | | **~2,000** |

**Excluded sources:**
- `mitre_attack_website` — redundant with STIX data (same content, unstructured HTML vs. structured JSON)
- `active_detective_sim` — used for GRPO stage 2, not domain knowledge SFT
- All `use: evaluation` sources (MalwareBazaar, Malware Traffic Analysis, CISA IOC feeds) — eval only, not training

## 4. Quality Controls

### 4.1 Automated Checks (merge_and_validate.py)
- Message format valid (system/user/assistant roles)
- No empty messages
- Assistant response > 50 characters (no trivial answers)
- No duplicate user+assistant pairs
- Metadata fields present and valid
- Source key matches sources.yaml

### 4.2 Coverage Checks
- Every source in sources.yaml with `use: training` has generated examples
- Topic distribution roughly matches source weights
- Example type distribution: ~50% QA, ~35% instruction, ~15% passage

### 4.3 Manual Spot-Check
- After generation, sample 20 examples from each source
- Review for accuracy, relevance, and quality
- Flag low-quality examples for regeneration

## 5. Provenance Chain

For any example in the final dataset, you can trace:

```
sft_domain_knowledge.jsonl line N
  → metadata.source = "mitre_attack_stix"
  → metadata.source_section = "T1486"
  → metadata.generation_prompt_id = "technique_qa_v1"
    → prompts/mitre_attack.yaml (exact prompt template)
  → metadata.generated_at = "2026-03-17T14:30:00Z"
    → generation_log.jsonl (full request/response)
  → raw_content/mitre_attack_stix/ (cached source data)
    → raw_content/manifest.json (fetch timestamp, content hash)
  → sources.yaml (canonical URL, license)
```

## 6. File Structure

```
fine-tuning/
├── data/
│   ├── sources.yaml                    # Source catalog (exists)
│   ├── sft_domain_knowledge.jsonl      # Final dataset (generated)
│   ├── generation_log.jsonl            # Full provenance log (generated)
│   ├── coverage_report.md             # Statistics (generated)
│   ├── validation_log.jsonl           # Dropped examples (generated)
│   ├── raw/                           # Per-source generated examples (gitignored)
│   │   ├── mitre_attack_stix.jsonl
│   │   ├── atomic_red_team.jsonl
│   │   └── ...
│   └── raw_content/                   # Cached source documents (gitignored)
│       ├── manifest.json
│       ├── mitre_attack_stix/
│       ├── atomic_red_team/
│       └── ...
├── prompts/                           # Generation prompt templates
│   ├── mitre_attack.yaml
│   ├── atomic_red_team.yaml
│   ├── cisa_guide.yaml
│   ├── dfir_report.yaml
│   ├── windows_event_log.yaml
│   ├── kaspersky_securelist.yaml
│   ├── sans_dfir.yaml
│   └── prompting_only.yaml
├── scripts/
│   ├── source_fetcher.py              # Download and cache raw content
│   ├── generate_sft_data.py           # Orchestrate generation via Claude Code
│   ├── merge_and_validate.py          # Deduplicate, validate, produce final dataset
│   └── prepare_training_data.py       # (exists) Older telemetry-based pipeline
└── .gitignore                         # Ignore raw_content/ and raw/
```

## 7. Train/Eval Split and Format Compatibility

### 7.1 Split Strategy
`merge_and_validate.py` produces a 90/10 train/eval split stratified by source and example type. Output:
- `data/sft_domain_knowledge_train.jsonl` (~1,800 examples)
- `data/sft_domain_knowledge_eval.jsonl` (~200 examples)

The eval split is used for before/after comparison when measuring SFT impact.

### 7.2 Format Compatibility
This pipeline outputs chat-format `messages` (compatible with TRL `SFTTrainer` and Qwen chat templates). The existing `finetune_ransomware_llm.py` expects `{prompt, completion}` format — it will need a new training script or an adapter. The deliverables include a new `train_sft.py` that accepts the chat-format JSONL directly. The old `finetune_ransomware_llm.py` is preserved for the legacy telemetry-based pipeline.

## 8. Deliverables

1. `fine-tuning/scripts/source_fetcher.py` — fetch and cache source content
2. `fine-tuning/prompts/*.yaml` — prompt templates per source (8 files)
3. `fine-tuning/scripts/generate_sft_data.py` — generation orchestrator
4. `fine-tuning/scripts/merge_and_validate.py` — dedup, validate, stats
5. `fine-tuning/data/sft_domain_knowledge.jsonl` — the dataset (~2K examples)
6. `fine-tuning/data/generation_log.jsonl` — provenance log
7. `fine-tuning/data/coverage_report.md` — statistics and coverage
8. `fine-tuning/scripts/train_sft.py` — new SFT training script for chat-format JSONL (QLoRA, Qwen3-4B/1.7B)
9. `fine-tuning/.gitignore` — ignore `data/raw_content/` and `data/raw/`
