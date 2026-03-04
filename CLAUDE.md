# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project exploring ransomware detection using LLMs, structured as three progressively sophisticated approaches:

1. **Prompting-only** (`prompting-only/`): Zero-shot, few-shot, and chain-of-thought prompting against pre-trained LLMs (baseline, no training needed)
2. **Fine-tuning** (`fine-tuning/`): QLoRA fine-tuning of small LLMs (TinyLlama, Phi-2, Mistral) on synthetic telemetry with expert annotations
3. **RL policy** (`rl-policy/`): Reinforcement learning policy head for action selection (future work, not yet implemented)

The core theoretical idea: treat detection as an implicit Dynamic Bayes Net where the LLM's latent embedding serves as hidden state. The model learns to predict both observable telemetry and expert explanations via multi-task loss (telemetry MLM + QA generation + action classification).

## Architecture

### Two telemetry codebases (historical)

- `code/` — Original v0.1 scripts: `telemetry-sim.py` (simple benign/blitz-encryptor generators), `telemetry-concat.py` (k-stack window builder), `finetune_llm.py` (basic LoRA fine-tuning). These use simple dataclasses (`ProcessEvent`, `FileEvent`, `NetEvent`) and produce flat JSONL.
- `prompting-only/telemetry/` — More developed generators with latent state tracking (`LatentState` enum with 12 attack phases), partial observability simulation, expert annotations (`ExpertAnnotation`), and Atomic Red Team integration. `comprehensive_telemetry_gen.py` is the primary generator here.

The `fine-tuning/scripts/prepare_training_data.py` imports from the prompting-only telemetry generators (via `sys.path.append('./telemetry')`), so run fine-tuning scripts from the `fine-tuning/` or `prompting-only/` directory context.

### Telemetry data model

- **TelemetryWindow**: 120-second snapshots with 30-second stride
- **Observable events**: process, file, network, registry events (what the LLM sees)
- **Latent context**: true attack state, attacker intent, kill chain position, time-to-impact (ground truth for training)
- **Expert annotations**: risk assessment, key indicators, recommended actions, QA pairs (what the LLM learns to predict)
- **Signal features**: `entropy_delta_mean`, `rename_rate` (quick heuristics)
- Output format: JSONL with `prompt`/`completion` pairs for fine-tuning

### Prompt format conventions

Training prompts use custom tokens: `<|system|>`, `<|telemetry|>`, `<|task|>`, `<|answer|>`, `<|analysis|>`. The fine-tuning scripts in `code/finetune_llm.py` and `fine-tuning/scripts/finetune_ransomware_llm.py` use slightly different prompt templates — check the `build_prompt` / `_format_prompt` functions.

## Development Commands

### Install dependencies
```bash
pip install -r code/requirements.txt                    # Original pipeline (pinned versions)
pip install -r fine-tuning/scripts/requirements_finetune.txt  # Fine-tuning (flexible versions)
```

### Generate telemetry data

```bash
# Original simple generator
python code/telemetry-sim.py --out dataset.jsonl --episodes 5000 --window 120 --stride 30 --ransomware-p 0.35 --seed 42

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

# Original fine-tuning script (simpler, from code/)
python code/finetune_llm.py --data dataset.jsonl --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output ft_lora_out --epochs 3 --batch 2
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

## Current status and next priorities

- Telemetry simulator works but only covers simple benign (Office/backup) and blitz encryptor behaviors
- Fine-tuning pipeline is structurally complete but not extensively tested (needs GPU)
- Next priorities per `code/plan.md`: slow-sleeper ransomware, diversified goodware generators, retrieval-augmented memory (RAG) for long-horizon detection, parameter sweep for window sizes
- The `rl-policy/` approach is documented in `reward_signlas.md` but has no implementation yet

## Papers and references

- `llm-policy.tex` / `policy-model.tex`: LaTeX source for the theoretical framework paper
- `prop-ransomeware.tex`: Research proposal
- `code/plan.md`: Detailed Stage I implementation plan with milestones
