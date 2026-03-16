# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RL-trained LLM agent (GRPO) that actively investigates ransomware by selecting which host evidence to examine. Operates as a POMDP: host state is partially hidden, telemetry events are stochastically dropped, and the agent chooses tools before rendering a verdict.

Base model: Qwen3.5-9B with QLoRA 4-bit. Training via TRL's `environment_factory` for real multi-step rollouts.

## Commands

```bash
# Run all tests (408 tests, ~30s, pure Python — no GPU or ML deps needed)
python -m pytest tests/ -q

# Run a single test file
python -m pytest tests/test_env.py -v

# Run a single test by name
python -m pytest tests/test_reward.py -k "test_false_negative"

# Run untrained agent (sanity check)
python scripts/run_untrained.py

# Generate training scenarios
python -c "from training.scenarios import generate_training_scenarios, save_scenarios; save_scenarios(generate_training_scenarios(1000), 'scenarios.jsonl')"

# Train (requires GPU — see version note below)
accelerate launch -m training.train_grpo \
    --model Qwen/Qwen3.5-9B \
    --output-dir ./checkpoints \
    --n-episodes 500 --group-size 4 --max-steps 200

# Smoke test (50 episodes, ~15-30 min on A5000)
accelerate launch -m training.train_grpo \
    --model Qwen/Qwen3.5-9B \
    --output-dir ./checkpoints-smoke \
    --n-episodes 50 --group-size 2 --max-steps 10
```

**Version note**: `requirements.txt` lists `transformers>=4.46.0` (sufficient for tests/dev). GPU training with Qwen3.5-9B needs `transformers>=5.2.0`, `trl>=0.16.0`, `peft>=0.15.0` — see `scripts/vastai_setup.sh` for exact versions.

## Architecture: Data flow

```
scenarios.py          → generate_training_scenarios() → ScenarioBatch
                                                           ↓
telemetry.py          → generate_episode()            → Episode (input_text, host, ground_truth, history_windows)
                                                           ↓
train_grpo.py         → prepare_dataset()             → HF Dataset (prompt messages + scenario_data JSON)
                                                           ↓
                      → DetectionEnv (TRL env)        → multi-step rollout (tool calls ↔ host queries)
                                                           ↓
reward.py             → compute_reward()              → RewardBreakdown (verdict + cost + efficiency + format)
```

## Critical design invariant: host coupling

The `Episode` dataclass carries the `HostState` that generated its telemetry. Both environments (`RansomwareDetectionEnv` and training `DetectionEnv`) use `episode.host` for tool execution. This ensures tools inspect the same host that produced the observed telemetry. Breaking this coupling (e.g., creating a new HostState for tools) causes state divergence bugs.

## Simulator: HostState composition

`HostState` composes 6 mutable registries — `FileRegistry`, `ProcessTable`, `ConnectionTable`, `RegistryHive`, `EventLog`, `SimClock`. Generators mutate this state; tools query it. `content.py` generates synthetic file contents with realistic magic bytes and controlled entropy for the `read_file_sample` tool.

Attack generators: `blitz_encryptor`, `slow_sleeper`, `exfil_first`, `semantic_shuffle`. Benign: `office_edits`, `browser_downloads`, `backup_operations`, `av_scan`, `system_maintenance`.

## Tools: 9 investigation tools + DECIDE

Defined in `tools/inspection.py`, `tools/network_tools.py`, `tools/forensic_tools.py`. Each has a cost (negative reward penalty). `scan_directory` is most expensive (-0.05). `DECIDE(verdict, explanation)` is the terminal action (cost 0).

The dual-format parser (`tools/parser.py`) handles Qwen3 JSON (`<tool_call>{"name": ..., "arguments": ...}</tool_call>`) and function-call fallback (`<tool_call>inspect_file("path")</tool_call>`).

## Reward: asymmetric, deterministic

- Correct verdict: +1.0
- False positive: -1.0
- False negative: -2.0 (missing ransomware is worse)
- Efficiency bonus: `unused_steps * 0.05` (rewards early decisions)
- Format bonus: +0.05 for `<think>` tags, +0.05 for `<tool_call>` tags
- Tool costs: subtracted per use

## TRL training integration

`DetectionEnv` in `train_grpo.py` is the TRL-compatible environment. Its public methods become tools exposed by `GRPOTrainer`. `reset(**kwargs)` receives dataset row fields and reconstructs the episode from `scenario_data` JSON. The training loop: model generates → TRL parses tool call → calls env method → injects result → model continues → ... → DECIDE or max length.

Unsloth is used for 2x speedup. The `load_model` function tries Unsloth first, falls back to plain transformers + PEFT.

## Scenario mix (training balance)

Default: 40% benign, 20% blitz, 15% sleeper, 15% exfil-first, 10% semantic_shuffle. Observability levels: [0.3, 0.5, 0.7, 0.9]. Attack progress sampled from [0.2, 0.9] for attacks, 0.0 for benign.

## History windows (temporal context)

`generate_episode()` creates prior telemetry windows at earlier attack progress values (default 2 windows). Formatted as `--- Window t-N (prior) ---` labels in the prompt. Provides trend information for the agent.

## Evaluation baselines

`evaluation/baselines.py`: `RandomAgent`, `ExhaustiveAgent`, `HeuristicAgent`, `PassiveLLM` (no tools, immediate DECIDE). `evaluation/ablation.py`: 15 tool-subset variants for ablation studies.

## Dependencies

Tests run on pure Python (no GPU, no ML deps beyond numpy). For GPU training: `scripts/vastai_setup.sh` installs exact versions (`transformers>=5.2.0`, `trl>=0.16.0`, `unsloth[cu124-ampere]>=2025.3`). See `requirements.txt` for the full list.

## Design docs

- System design: `../docs/plans/2026-03-05-active-detective-system-design.md`
- POMDP environment: `../docs/plans/2026-03-07-pomdp-environment-design.md`
- Pre-training audit: `../docs/plans/2026-03-09-pre-training-audit-design.md`
