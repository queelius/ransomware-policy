# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project exploring ransomware detection using LLMs. The **Active Detective Agent** (`active-detective/`) is the main research contribution — an RL-trained LLM agent (GRPO) that actively investigates ransomware by choosing which host evidence to examine under partial observability. Two older approaches serve as comparison baselines:

1. **Active Detective Agent** (`active-detective/`): GRPO-trained agent with 9 investigation tools, POMDP environment, Qwen3.5-9B base. **See `active-detective/CLAUDE.md` for full architectural reference.**
2. **Prompting-only** (`prompting-only/`): Zero-shot, few-shot, and chain-of-thought prompting against pre-trained LLMs (no training)
3. **Fine-tuning** (`fine-tuning/`): QLoRA fine-tuning of small LLMs (TinyLlama, Phi-2, Mistral) — structurally complete but superseded by active-detective

## Development Commands

### Active Detective (primary)

```bash
cd active-detective

# Run all tests (408 tests, ~30s, pure Python — no GPU needed)
python -m pytest tests/ -q

# Run a single test file
python -m pytest tests/test_env.py -v

# Run a single test by name
python -m pytest tests/test_reward.py -k "test_false_negative"

# Generate training scenarios
python -c "from training.scenarios import generate_training_scenarios, save_scenarios; save_scenarios(generate_training_scenarios(1000), 'scenarios.jsonl')"

# Train (requires GPU — see version note below)
accelerate launch -m training.train_grpo \
    --model Qwen/Qwen3.5-9B \
    --output-dir ./checkpoints \
    --n-episodes 500 --group-size 4 --max-steps 200

# Smoke test (~15-30 min on A5000)
accelerate launch -m training.train_grpo \
    --model Qwen/Qwen3.5-9B \
    --output-dir ./checkpoints-smoke \
    --n-episodes 50 --group-size 2 --max-steps 10

# Run untrained agent (sanity check)
python scripts/run_untrained.py
```

**Version note**: `requirements.txt` lists `transformers>=4.46.0` (sufficient for tests and development). Actual GPU training with Qwen3.5-9B needs `transformers>=5.2.0`, `trl>=0.16.0`, `peft>=0.15.0` — see `scripts/vastai_setup.sh` for exact training-time versions.

### Prompting-only and fine-tuning (baselines)

```bash
# Prompting demo
cd prompting-only && python prompts/detection_prompts.py

# Fine-tuning (quick test, no GPU)
cd fine-tuning && python scripts/prepare_training_data.py --simple --output train.jsonl
python scripts/finetune_ransomware_llm.py --model tiny --train-data train.jsonl --epochs 3

# Install fine-tuning deps
pip install -r fine-tuning/scripts/requirements_finetune.txt
```

**Cross-module import**: `fine-tuning/scripts/prepare_training_data.py` imports from `prompting-only/telemetry/` via `sys.path.append('./telemetry')`. Run fine-tuning scripts from the `fine-tuning/` or `prompting-only/` directory context.

## Active Detective Architecture

Full details in `active-detective/CLAUDE.md`. Key concepts:

### Data flow
```
scenarios.py → generate_training_scenarios() → ScenarioBatch
                                                    ↓
telemetry.py → generate_episode()             → Episode (input_text, host, ground_truth, history_windows)
                                                    ↓
train_grpo.py → prepare_dataset()             → HF Dataset (prompt messages + scenario_data JSON)
                                                    ↓
               → DetectionEnv (TRL env)       → multi-step rollout (tool calls ↔ host queries)
                                                    ↓
reward.py    → compute_reward()               → RewardBreakdown (verdict + cost + efficiency + format)
```

### Critical design invariant: host coupling

The `Episode` dataclass carries the `HostState` that generated its telemetry. Both environments (`RansomwareDetectionEnv` and training `DetectionEnv`) use `episode.host` for tool execution. This ensures tools inspect the same host that produced the observed telemetry. Breaking this coupling causes state divergence bugs.

### Reward: asymmetric, deterministic

- Correct verdict: +1.0, false positive: -1.0, **false negative: -2.0** (missing ransomware is worse)
- Efficiency bonus: `unused_steps * 0.05` (rewards early decisions)
- Format bonus: +0.05 for `<think>` tags, +0.05 for `<tool_call>` tags
- Tool costs: subtracted per use (scan_directory most expensive at -0.05)

### Simulator

`HostState` composes 6 mutable registries: `FileRegistry`, `ProcessTable`, `ConnectionTable`, `RegistryHive`, `EventLog`, `SimClock`. Attack generators (blitz, sleeper, exfil-first, semantic_shuffle) and benign generators (office_edits, browser_downloads, backup_operations, av_scan, system_maintenance) mutate this state. `content.py` generates synthetic file contents with realistic magic bytes and controlled Shannon entropy.

### Partial observability

Events are stochastically dropped based on attack phase — initial compromise ~40% observable, reconnaissance ~60%, encryption ~90%. The agent sees passive telemetry (path, size_delta, ext_change, pid) but must use tools to discover entropy, file contents, and forensic details.

### Temporal context

`generate_episode()` creates prior telemetry windows (default 2) at earlier attack progress values, formatted as `--- Window t-N (prior) ---` in the prompt. Enables trend detection (e.g., slow_sleeper encrypts 1-2 files/window — only visible across windows).

## Current status and next priorities

- Active Detective: Full pipeline implemented (408 tests), POMDP environment with frozen snapshots + multi-window history, host-coupled tool execution, synthetic file contents for forensics, ready for GPU training
- Pre-training audit complete: fixed host state mismatch, dead history windows in training, weak efficiency signal (0.05/step), format_reward false positives, hardcoded k_max
- Next: run GRPO training on Vast.ai (RTX A5000 or similar, ~24GB VRAM sufficient), evaluate against baselines, tool ablation study

## Design docs

- System design: `docs/plans/2026-03-05-active-detective-system-design.md`
- POMDP environment: `docs/plans/2026-03-07-pomdp-environment-design.md`
- Pre-training audit: `docs/plans/2026-03-09-pre-training-audit-design.md`

## Papers and references

- `docs/prop-ransomeware.tex`: Summer research proposal for adaptive AI ransomware defense as MDP
- `docs/grad-report-alex-srdc.pdf`: Review of a ransomware paper using a small pretrained model
- `docs/early-research/notes.md`: Most detailed exploration of LM, DNN, RL approaches and the "reverse problem" idea
- `docs/early-research/prob_def_fuj.md`: Dr. Fujinoki's original problem definition

## Related repositories

- `../on-beat-backup`: Published On-Beat Backups paper (backup-based defense with Bloom filters/fake fields — co-authored with Fujinoki)
- `../llm-policy`: LLM orchestration papers (MDP framework for goal-directed prompting + DINO-LLM cognitive architecture)
