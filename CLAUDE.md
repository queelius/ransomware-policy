# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project exploring ransomware detection using LLMs, structured as three approaches:

1. **Prompting-only** (`prompting-only/`): Zero-shot, few-shot, and chain-of-thought prompting against pre-trained LLMs (baseline, no training needed)
2. **Fine-tuning** (`fine-tuning/`): QLoRA fine-tuning of small LLMs (TinyLlama, Phi-2, Mistral) on synthetic telemetry with expert annotations
3. **Active Detective Agent** (`active-detective/`): RL-trained agent (GRPO) that actively investigates via tool calls — the main research contribution. See below.

The core theoretical idea: treat detection as an implicit Dynamic Bayes Net where the LLM's latent embedding serves as hidden state. The model learns to predict both observable telemetry and expert explanations via multi-task loss (telemetry MLM + QA generation + action classification).

## Architecture

### Telemetry generators

`prompting-only/telemetry/` contains the telemetry generation pipeline with latent state tracking (`LatentState` enum with 12 attack phases), partial observability simulation, expert annotations (`ExpertAnnotation`), and Atomic Red Team integration. `comprehensive_telemetry_gen.py` is the primary generator.

The `fine-tuning/scripts/prepare_training_data.py` imports from the prompting-only telemetry generators (via `sys.path.append('./telemetry')`), so run fine-tuning scripts from the `fine-tuning/` or `prompting-only/` directory context.

### Telemetry data model

- **TelemetryWindow**: 120-second snapshots with 30-second stride
- **Observable events**: process, file, network, registry events (what the LLM sees)
- **Latent context**: true attack state, attacker intent, kill chain position, time-to-impact (ground truth for training)
- **Expert annotations**: risk assessment, key indicators, recommended actions, QA pairs (what the LLM learns to predict)
- **Signal features**: `entropy_delta_mean`, `rename_rate` (quick heuristics)
- Output format: JSONL with `prompt`/`completion` pairs for fine-tuning

### Prompt format conventions

Training prompts use custom tokens: `<|system|>`, `<|telemetry|>`, `<|task|>`, `<|answer|>`, `<|analysis|>`. Check the `_format_prompt` function in `fine-tuning/scripts/finetune_ransomware_llm.py`.

## Development Commands

### Install dependencies
```bash
pip install -r fine-tuning/scripts/requirements_finetune.txt  # Fine-tuning
```

### Generate telemetry data

```bash
# Comprehensive generator with latent states (run from prompting-only/)
cd prompting-only
python telemetry/comprehensive_telemetry_gen.py

# Atomic Red Team patterns
python telemetry/atomic_red_team_integration.py --output telemetry.jsonl --campaigns 10
```

### Prepare training data and fine-tune

```bash
cd fine-tuning

# Simple hand-crafted examples (quick test, no GPU)
python scripts/prepare_training_data.py --simple --output train.jsonl

# Full telemetry-based examples
python scripts/prepare_training_data.py --sequences 100 --output train.jsonl

# Fine-tune (model sizes: tiny=4GB, small=6GB, medium=12GB)
python scripts/finetune_ransomware_llm.py --model tiny --train-data train.jsonl --epochs 3
```

### Test prompting strategies

```bash
cd prompting-only
python prompts/detection_prompts.py  # runs demo with sample telemetry window
```

## Key design notes

- **Partial observability**: The comprehensive generator simulates realistic detection difficulty — initial compromise is only 40% observable, reconnaissance 60%, encryption 90%. This is controlled per-state in `_generate_observables()`.
- **QLoRA config**: Both fine-tuning scripts use `r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]`, 4-bit NF4 quantization. The `fine-tuning/` version additionally targets MLP modules (`gate_proj`, `up_proj`, `down_proj`).
- **Multi-layer prediction**: The model is trained to output 7 layers — state, meaning, indicators, next events, timeline, risk, and recommended actions — all in a single structured completion.
- Detection actions vocabulary: `ignore`, `monitor`, `alert`, `quarantine`, `block`.

## Active Detective Agent (`active-detective/`)

The main research contribution. An LLM agent trained via GRPO to actively investigate ransomware by selecting which host evidence to examine.

### Architecture
- **Simulator** (`simulator/`): HostState with FileRegistry (mutable filesystem + synthetic file contents via `content.py`) + ProcessTable (processes + handles/modules), 5 benign generators, 4 attack generators (blitz, sleeper, exfil-first, semantic shuffle), observability filter
- **Tools** (`tools/`): 9 investigation tools (inspect_file, check_process, scan_directory, list_connections, inspect_connection, query_registry, list_process_handles, query_event_log, read_file_sample) + DECIDE verdict action; dual-format parser (Qwen3 JSON + function-call syntax)
- **Environment** (`environment/`): RansomwareDetectionEnv (frozen-snapshot HostState, multi-window history), RLVR reward (asymmetric: FN=-2, FP=-1, correct=+1), budget enforcement and cost accumulation
- **Training** (`training/`): GRPO via TRL's `environment_factory` for real multi-step rollouts, QLoRA (4-bit NF4, r=16), Qwen3-8B base
- **Evaluation** (`evaluation/`): Detection metrics, baselines (Random/Exhaustive/Heuristic), tool ablation sweep, Pareto analysis

### Commands
```bash
cd active-detective

# Run tests (408 tests)
python -m pytest tests/ -q

# Run specific test file
python -m pytest tests/test_env.py -v

# Generate training scenarios
python -c "from training.scenarios import generate_training_scenarios, save_scenarios; save_scenarios(generate_training_scenarios(1000), 'scenarios.jsonl')"

# Train (requires GPU + trl + transformers>=5.2.0)
accelerate launch -m training.train_grpo --model Qwen/Qwen3-8B --output-dir ./checkpoints --n-episodes 500 --group-size 4
```

### Temporal model
Each episode generates a frozen HostState snapshot at a given attack progress level. Prior telemetry windows (default 2) are generated at earlier progress values to provide temporal context. The agent sees passive telemetry (path, size_delta, ext_change, pid) but must use tools to discover entropy, file contents, and other forensic details.

### Host coupling
The Episode dataclass carries the HostState that generated its telemetry. Both environments (`RansomwareDetectionEnv` and training `DetectionEnv`) use `episode.host` for tool execution, ensuring the agent inspects the same host that produced the telemetry it sees. History windows are formatted into the training prompt via `--- Window t-N (prior) ---` labels.

### Design docs
- System design: `docs/plans/2026-03-05-active-detective-system-design.md`
- POMDP environment design: `docs/plans/2026-03-07-pomdp-environment-design.md`
- Pre-training audit: `docs/plans/2026-03-09-pre-training-audit-design.md`

## Current status and next priorities

- Active Detective: Full pipeline implemented (408 tests), POMDP environment with frozen snapshots + multi-window history, host-coupled tool execution, synthetic file contents for forensics, ready for GPU training
- Pre-training audit complete: fixed host state mismatch, dead history windows in training, weak efficiency signal (0.05/step), format_reward false positives, hardcoded k_max
- Next: run GRPO training on Vast.ai A100, evaluate against baselines, tool ablation study
- Fine-tuning pipeline (older approach) structurally complete but superseded by active-detective

## Papers and references

- `docs/prop-ransomeware.tex`: Summer research proposal for adaptive AI ransomware defense as MDP
- `docs/grad-report-alex-srdc.pdf`: Review of a ransomware paper using a small pretrained model (useful reference)
- `docs/early-research/`: Brainstorming notes, proposals, and theoretical groundwork from the early research phase. Key files:
  - `notes.md` — Most detailed exploration of LM, DNN, RL approaches and the "reverse problem" idea for synthetic data
  - `research-outline.md` — Comprehensive outline (sequence prediction, tools, experimental plan)
  - `prob_def_fuj.md` — Dr. Fujinoki's original problem definition
  - `report.md` — Report on predictive models for ransomware detection

## Related repositories

- `../on-beat-backup`: Published On-Beat Backups paper (backup-based defense with Bloom filters/fake fields — different research track, co-authored with Fujinoki)
- `../llm-policy`: LLM orchestration papers (MDP framework for goal-directed prompting + DINO-LLM hybrid cognitive architecture — general-purpose, not ransomware-specific)
