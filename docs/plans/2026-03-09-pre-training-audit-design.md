# Pre-Training Audit: Fix Critical Issues Before GPU Run

**Date**: 2026-03-09
**Status**: Approved

## Context

Full audit of the active-detective training pipeline before launching GRPO training on A100. Found 2 critical, 3 moderate issues that would waste GPU budget or produce poor training signal.

## Issues and Fixes

### 1. Host State Mismatch (Critical)

**Problem**: `generate_episode()` creates a HostState for telemetry, but `DetectionEnv.reset()` creates a *separate* HostState for tool queries. The agent's tools query a different host than the one that produced the telemetry — file paths match (same template) but values differ (different RNG state).

**Fix**: Return the HostState from `generate_episode()` on the Episode dataclass. Both environments (`DetectionEnv` in train_grpo.py and `RansomwareDetectionEnv` in env.py) use the episode's host directly for tool execution. No separate `HostState.create()`.

**Design choice**: Agent sees the full post-generation host, including attack artifacts (encrypted files with high entropy). This is correct — the agent should be able to inspect files mentioned in telemetry and get consistent results. Ground truth (is_ransomware) is not directly readable from any tool.

### 2. History Windows Dead in Training (Critical)

**Problem**: `generate_episode()` produces `history_windows` (prior telemetry) but `DetectionEnv.reset()` only returns `episode.input_text` (current window). History never reaches the model prompt. The temporal reasoning feature is dead code in training.

**Fix**: In `DetectionEnv.reset()`, format history + current window into the returned string:
```
--- Window t-2 (prior) ---
[telemetry from 240s ago]

--- Window t-1 (prior) ---
[telemetry from 120s ago]

--- Current window ---
[current telemetry]
```
Reuse formatting from `build_chat_messages()`.

### 3. Weak Efficiency Signal (Medium)

**Problem**: `EFFICIENCY_BONUS_PER_STEP = 0.01` — a 5-step agent loses only 0.04 vs a 1-step agent. Signal is drowned out by ±1.0 verdict reward. Agent has no meaningful incentive to be efficient.

**Fix**: Increase to `0.05/step`. A 1-step answer is worth +0.20 more than a 5-step answer — meaningful tiebreaker during GRPO ranking, but still dominated by verdict accuracy.

**Files**: `environment/reward.py` (constant) + `training/train_grpo.py` (`_compute_env_reward`)

### 4. format_reward False Positives (Low-Medium)

**Problem**: `format_reward()` uses `"decide" in content.lower()` to detect tool calls. This matches the word "decide" in natural language, giving false format rewards.

**Fix**: Check `"<tool_call>" in content` instead.

### 5. Hardcoded k_max (Low)

**Problem**: `_compute_env_reward()` hardcodes `k_max = 5` instead of reading from the env. Changing `--k-max` won't affect training reward.

**Fix**: Store `k_max` on `DetectionEnv.__init__()`, read from `env._k_max` in `_compute_env_reward()`.

## Files to Change

| File | Changes |
|------|---------|
| `simulator/telemetry.py` | Add `host: HostState` to Episode, return host from `_generate_window()` and `generate_episode()` |
| `training/train_grpo.py` | Use `episode.host`, format history in reset, fix format_reward, fix k_max |
| `environment/env.py` | Use `episode.host` instead of fresh HostState |
| `environment/reward.py` | `EFFICIENCY_BONUS_PER_STEP` 0.01 → 0.05 |
| `tests/test_env.py` | Update reward expectations for 0.05 bonus |
| `tests/test_reward.py` | Update reward expectations for 0.05 bonus |
| `tests/test_train_grpo.py` | Update for Episode.host field, history in prompt, reward values |
| `tests/test_telemetry.py` | Add host to Episode assertions |

## Verification

```bash
cd active-detective
python -m pytest tests/ -q                    # all tests pass
python -m scripts.run_untrained --mode demo   # verify end-to-end
```
