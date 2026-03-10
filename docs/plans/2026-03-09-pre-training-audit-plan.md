# Pre-Training Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 5 issues (2 critical, 3 moderate) in the active-detective training pipeline before launching GRPO training on A100.

**Architecture:** Episode generation returns the HostState it built, eliminating the telemetry-vs-tool divergence. History windows get formatted into the prompt returned by `env.reset()`. Efficiency bonus increases 5x to provide meaningful training signal.

**Tech Stack:** Python, numpy, pytest. All changes in `active-detective/`.

**Design doc:** `docs/plans/2026-03-09-pre-training-audit-design.md`

---

### Task 1: Add `host` field to Episode and return it from `generate_episode()`

**Files:**
- Modify: `active-detective/simulator/telemetry.py:29-38` (Episode dataclass)
- Modify: `active-detective/simulator/telemetry.py:128-167` (`_generate_window`)
- Modify: `active-detective/simulator/telemetry.py:170-254` (`generate_episode`)
- Test: `active-detective/tests/test_telemetry.py`

**Step 1: Write failing tests**

Add to `tests/test_telemetry.py`:

```python
def test_episode_has_host(self):
    rng = np.random.RandomState(42)
    episode = generate_episode(ScenarioType.BENIGN, 0.8, rng)
    assert episode.host is not None
    # Host should have files seeded
    assert len(episode.host.files) > 0

def test_episode_host_matches_telemetry_files(self):
    """Files mentioned in telemetry should exist in episode.host."""
    rng = np.random.RandomState(42)
    episode = generate_episode(ScenarioType.BLITZ, 0.9, rng, attack_progress=0.5)
    # Extract file paths from telemetry text
    import re
    paths = re.findall(r'path=(C:/[^\s]+)', episode.input_text)
    for path in paths:
        # File should exist in the host (possibly renamed by attack)
        # At minimum, the directory should exist
        dir_path = path.rsplit('/', 1)[0]
        assert dir_path in episode.host.files.directories()
```

**Step 2: Run tests to verify they fail**

Run: `cd active-detective && python -m pytest tests/test_telemetry.py -v -k "test_episode_has_host or test_episode_host_matches" 2>&1 | tail -5`
Expected: FAIL — Episode has no attribute `host`

**Step 3: Implement**

In `simulator/telemetry.py`:

1. Add `host` field to Episode (after `history_windows`):
```python
@dataclass
class Episode:
    input_text: str
    ground_truth: GroundTruth
    scenario_type: ScenarioType
    observability: float
    raw_event_count: int
    visible_event_count: int
    history_windows: list[str] = field(default_factory=list)
    host: HostState | None = None
```

Add `HostState` import at top (already imported).

2. In `_generate_window()`, change return type to include host:
```python
def _generate_window(...) -> tuple[str, str | None, HostState]:
```
Add `return window_text, attack_phase, host` at the end.

3. In `generate_episode()`:
- History window calls: capture but discard the host (we only need the current window's host).
  Change: `hist_text, _ = _generate_window(...)` → `hist_text, _, _ = _generate_window(...)`
- Current window: the existing inline code creates `host = HostState.create(rng, now)` on line 206. Just pass it to the Episode:
  Add `host=host` to the Episode constructor at the end.

**Step 4: Run tests**

Run: `cd active-detective && python -m pytest tests/test_telemetry.py -v 2>&1 | tail -10`
Expected: all PASS

**Step 5: Commit**

```bash
git add simulator/telemetry.py tests/test_telemetry.py
git commit -m "feat: return HostState on Episode for tool-telemetry coupling"
```

---

### Task 2: Wire `episode.host` into both environments

**Files:**
- Modify: `active-detective/environment/env.py:134-138`
- Modify: `active-detective/training/train_grpo.py:121-125`
- Test: `active-detective/tests/test_env.py`
- Test: `active-detective/tests/test_train_grpo.py`

**Step 1: Write failing test**

Add to `tests/test_env.py`:

```python
class TestHostCoupling:
    def test_tool_host_matches_episode_host(self, env, rng):
        """The host used for tools should be the same one that generated telemetry."""
        env.reset(ScenarioType.BENIGN, 0.9, rng)
        assert env._host is env._episode.host
```

Add to `tests/test_train_grpo.py` (TestDetectionEnv):

```python
def test_host_from_episode(self):
    """DetectionEnv should use episode's host, not a fresh one."""
    env = DetectionEnv()
    env.reset(scenario_data=self._make_scenario_data())
    # The host should be the episode's host
    assert env._host is not None
    # Inspect a file that exists — should get real data, not error
    paths = env._host.files.all_paths()
    assert len(paths) > 0
    result = json.loads(env.inspect_file(paths[0]))
    assert "error" not in result
    assert "entropy" in result
```

**Step 2: Run tests to verify they fail**

Run: `cd active-detective && python -m pytest tests/test_env.py::TestHostCoupling -v 2>&1 | tail -5`
Expected: FAIL — `env._host is env._episode.host` is False

**Step 3: Implement**

In `environment/env.py`, replace lines 134-138:
```python
        # Re-create host state for tool execution.
        from datetime import datetime
        now = datetime(2025, 6, 15, 10, 0, 0)
        host_rng = np.random.RandomState(rng.randint(0, 2**31))
        self._host = HostState.create(host_rng, now)
```
With:
```python
        # Use the episode's host for tool execution — ensures tools
        # query the same state that generated the telemetry.
        self._host = self._episode.host
```

In `training/train_grpo.py`, replace lines 121-125:
```python
        # Set up host state for tool execution
        from datetime import datetime
        now = datetime(2025, 6, 15, 10, 0, 0)
        host_rng = np.random.RandomState(rng.randint(0, 2**31))
        self._host = HostState.create(host_rng, now)
```
With:
```python
        # Use the episode's host for tool execution — ensures tools
        # query the same state that generated the telemetry.
        self._host = episode.host
```

Also remove the now-unused `HostState` import from `training/train_grpo.py` line 33.

**Step 4: Run tests**

Run: `cd active-detective && python -m pytest tests/test_env.py tests/test_train_grpo.py -v 2>&1 | tail -15`
Expected: all PASS

**Step 5: Commit**

```bash
git add environment/env.py training/train_grpo.py tests/test_env.py tests/test_train_grpo.py
git commit -m "fix: use episode host for tool execution, eliminating state divergence"
```

---

### Task 3: Wire history windows into `DetectionEnv.reset()` return value

**Files:**
- Modify: `active-detective/training/train_grpo.py:97-134`
- Test: `active-detective/tests/test_train_grpo.py`

**Step 1: Write failing test**

Add to `tests/test_train_grpo.py` (TestDetectionEnv):

```python
def test_reset_includes_history_windows(self):
    """reset() should return history windows + current window."""
    env = DetectionEnv()
    result = env.reset(scenario_data=self._make_scenario_data(
        "blitz", attack_progress=0.5))
    assert "Window t-2 (prior)" in result
    assert "Window t-1 (prior)" in result
    assert "Current window" in result

def test_reset_no_history_when_zero(self):
    env = DetectionEnv()
    result = env.reset(scenario_data=self._make_scenario_data(n_history=0))
    assert "Window t-" not in result
    assert "Current window" not in result  # no label when no history
```

**Step 2: Run to verify fail**

Run: `cd active-detective && python -m pytest tests/test_train_grpo.py -v -k "history_windows or no_history_when_zero" 2>&1 | tail -5`
Expected: FAIL — no "Window t-2" in result

**Step 3: Implement**

In `training/train_grpo.py`, `DetectionEnv.reset()`, replace the return statement (line 133-134):

```python
        # Return the telemetry window text — TRL appends it to the user message
        return episode.input_text
```

With:

```python
        # Format history + current window for the model
        if episode.history_windows:
            parts = []
            n = len(episode.history_windows)
            for i, hw in enumerate(episode.history_windows):
                label = f"Window t-{n - i} (prior)"
                parts.append(f"--- {label} ---\n{hw}")
            parts.append(f"--- Current window ---\n{episode.input_text}")
            return "\n\n".join(parts)

        return episode.input_text
```

**Step 4: Run tests**

Run: `cd active-detective && python -m pytest tests/test_train_grpo.py -v 2>&1 | tail -15`
Expected: all PASS

**Step 5: Commit**

```bash
git add training/train_grpo.py tests/test_train_grpo.py
git commit -m "feat: include history windows in training prompt"
```

---

### Task 4: Increase efficiency bonus from 0.01 to 0.05

**Files:**
- Modify: `active-detective/environment/reward.py:38`
- Modify: `active-detective/training/train_grpo.py:397`
- Test: `active-detective/tests/test_reward.py`
- Test: `active-detective/tests/test_train_grpo.py`

**Step 1: Change the constant**

In `environment/reward.py`, line 38:
```python
EFFICIENCY_BONUS_PER_STEP = 0.05
```

In `training/train_grpo.py`, line 397 (inside `_compute_env_reward`):
```python
    efficiency_bonus = unused * 0.05
```

**Step 2: Update test expectations**

In `tests/test_reward.py`, line 128 — the `RewardBreakdown` test uses a hardcoded `efficiency_bonus=0.03`. This is a literal value test for the dataclass (not computed), so leave it as-is — it tests the total formula, not the constant.

The tests at lines 93, 97, 101 use `EFFICIENCY_BONUS_PER_STEP` by reference, so they'll update automatically.

**Step 3: Run tests**

Run: `cd active-detective && python -m pytest tests/test_reward.py tests/test_train_grpo.py -v 2>&1 | tail -15`
Expected: all PASS (tests reference the constant, not hardcoded 0.01)

**Step 4: Commit**

```bash
git add environment/reward.py training/train_grpo.py
git commit -m "tune: increase efficiency bonus from 0.01 to 0.05 per step"
```

---

### Task 5: Fix `format_reward()` false positive detection

**Files:**
- Modify: `active-detective/training/train_grpo.py:417`
- Test: `active-detective/tests/test_train_grpo.py`

**Step 1: Write failing test**

Add to `tests/test_train_grpo.py` (TestFormatReward):

```python
def test_decide_in_prose_not_rewarded(self):
    """Natural language 'decide' shouldn't count as tool call."""
    completions = ["I need to decide what to do here."]
    rewards = format_reward(completions)
    assert rewards[0] == 0.0  # no format reward
```

**Step 2: Run to verify fail**

Run: `cd active-detective && python -m pytest tests/test_train_grpo.py::TestFormatReward::test_decide_in_prose_not_rewarded -v 2>&1 | tail -5`
Expected: FAIL — current code matches "decide" in prose

**Step 3: Implement**

In `training/train_grpo.py`, line 417, replace:
```python
        has_tool_call = "tool_call" in content.lower() or "decide" in content.lower()
```
With:
```python
        has_tool_call = "<tool_call>" in content
```

**Step 4: Update existing test**

The test at line 296 (`test_both_thinking_and_tool_call`) uses:
```python
completions = ["<think>Analysis</think> decision made tool_call"]
```
This has `tool_call` as a bare word, not inside `<tool_call>` tags. Update to:
```python
completions = ["<think>Analysis</think> <tool_call>DECIDE(\"ignore\")</tool_call>"]
```

**Step 5: Run tests**

Run: `cd active-detective && python -m pytest tests/test_train_grpo.py::TestFormatReward -v 2>&1 | tail -10`
Expected: all PASS

**Step 6: Commit**

```bash
git add training/train_grpo.py tests/test_train_grpo.py
git commit -m "fix: format_reward checks <tool_call> tag, not 'decide' substring"
```

---

### Task 6: Fix hardcoded `k_max` in `_compute_env_reward`

**Files:**
- Modify: `active-detective/training/train_grpo.py:82-96,370`
- Test: `active-detective/tests/test_train_grpo.py`

**Step 1: Write failing test**

Add to `tests/test_train_grpo.py` (TestComputeEnvReward):

```python
def test_k_max_from_env(self):
    """_compute_env_reward should respect env._k_max, not hardcode 5."""
    env = DetectionEnv(k_max=10)
    env._ground_truth = GroundTruth(
        scenario_type=ScenarioType.BENIGN,
        is_ransomware=False,
    )
    env._verdict = "ignore"
    env._steps = 1
    env._cumulative_cost = 0.0
    r = _compute_env_reward(env)
    # With k_max=10 and 1 step: 9 unused * 0.05 = 0.45 efficiency
    # With k_max=5 and 1 step: 4 unused * 0.05 = 0.20 efficiency
    # r should be > 1.0 + 0.20 = 1.20 (if it uses k_max=10)
    assert r > 1.3
```

**Step 2: Run to verify fail**

Run: `cd active-detective && python -m pytest tests/test_train_grpo.py::TestComputeEnvReward::test_k_max_from_env -v 2>&1 | tail -5`
Expected: FAIL — hardcoded k_max=5 gives only 0.20 efficiency

**Step 3: Implement**

1. Add `k_max` param to `DetectionEnv.__init__()`:
```python
def __init__(self, k_max: int = 5) -> None:
    self._host: HostState | None = None
    self._ground_truth: GroundTruth | None = None
    self._steps: int = 0
    self._cumulative_cost: float = 0.0
    self._verdict: str | None = None
    self._explanation: str = ""
    self._k_max: int = k_max
```

2. In `_compute_env_reward()`, replace line 370:
```python
    k_max = 5  # default, matches training config
```
With:
```python
    k_max = env._k_max
```

3. In `train()` function, pass `k_max` through the environment_factory. Since TRL's `environment_factory` takes a callable, wrap it:
```python
    env_factory = lambda: DetectionEnv(k_max=config.k_max)
    # ...
    environment_factory=env_factory,
```

**Step 4: Run tests**

Run: `cd active-detective && python -m pytest tests/test_train_grpo.py -v 2>&1 | tail -15`
Expected: all PASS

**Step 5: Commit**

```bash
git add training/train_grpo.py tests/test_train_grpo.py
git commit -m "fix: DetectionEnv reads k_max from config instead of hardcoding 5"
```

---

### Task 7: Full test suite + end-to-end verification

**Files:** None new — verification only.

**Step 1: Run full test suite**

Run: `cd active-detective && python -m pytest tests/ -q`
Expected: 400+ tests pass (original 400 + new tests from tasks 1-6)

**Step 2: Run demo**

Run: `cd active-detective && python -m scripts.run_untrained --mode demo 2>&1 | head -80`
Expected: Demo runs, shows telemetry + tools + rollout.

**Step 3: Verify host coupling end-to-end**

```python
cd active-detective && python3 -c "
from environment.env import RansomwareDetectionEnv
from simulator.models import ScenarioType
from tools.parser import parse_tool_call
import numpy as np, json, re

env = RansomwareDetectionEnv(max_steps=5)
text = env.reset(ScenarioType.BLITZ, 0.9, np.random.RandomState(42), attack_progress=0.5)

# Extract a file path from telemetry
paths = re.findall(r'path=(C:/[^\s]+)', text)
if paths:
    path = paths[0]
    tc = parse_tool_call(f'<tool_call>{{\"name\": \"inspect_file\", \"arguments\": {{\"path\": \"{path}\"}}}}</tool_call>')
    result = env.step(tc)
    print(f'Telemetry path: {path}')
    print(f'Tool result: {json.dumps(result.result, indent=2)}')
    # Should NOT say 'error: File not found'
    assert 'error' not in result.result, 'Host state mismatch: file from telemetry not found in tool host!'
    print('HOST COUPLING VERIFIED')
"
```
Expected: `HOST COUPLING VERIFIED`

**Step 4: Verify history in prompt**

```python
cd active-detective && python3 -c "
from training.train_grpo import DetectionEnv
import json

env = DetectionEnv()
result = env.reset(scenario_data=json.dumps({
    'scenario_type': 'blitz', 'observability': 0.8,
    'attack_progress': 0.6, 'seed': 42, 'n_history': 2,
}))
assert 'Window t-2 (prior)' in result, 'History not in prompt!'
assert 'Window t-1 (prior)' in result, 'History not in prompt!'
assert 'Current window' in result, 'Current window label missing!'
print('HISTORY WINDOWS VERIFIED')
"
```
Expected: `HISTORY WINDOWS VERIFIED`

**Step 5: Commit all test additions**

```bash
git add -A
git commit -m "test: verify host coupling and history windows end-to-end"
```

---

### Task 8: Update CLAUDE.md

**Files:**
- Modify: `active-detective/../../CLAUDE.md` (project root)

**Step 1:** Update the test count and current status section to reflect the fixes.

Change `368 tests` → actual count from test run.
Add note about host coupling fix and history windows.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with post-audit test count and status"
```
