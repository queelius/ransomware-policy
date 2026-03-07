# POMDP Environment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the frozen-snapshot-with-history POMDP model: multi-progress history windows, remove entropy_delta from passive telemetry, remove recall_memory tool.

**Architecture:** Three independent changes — (1) remove entropy_delta from telemetry formatting, (2) remove recall_memory from all layers, (3) add multi-window history generation — applied bottom-up from simulator through training/evaluation. Changes are mechanical and well-scoped: no new dataclasses, no new files, no architectural shifts.

**Tech Stack:** Python 3.11, numpy, pytest. No new dependencies.

---

### Task 1: Remove entropy_delta from passive telemetry formatting

`entropy_delta` should not appear in the formatted telemetry text the agent sees.
The `FileEvent` dataclass keeps the field (generators still set it), but
`format_telemetry_window()` stops rendering it. The agent must use
`inspect_file` or `read_file_sample` to discover entropy.

**Files:**
- Modify: `active-detective/simulator/telemetry.py` (lines 86-94)
- Modify: `active-detective/tests/test_telemetry.py`
- Modify: `active-detective/evaluation/baselines.py` (HeuristicAgent parses entropy_delta from telemetry text)

**Step 1: Update test to expect no entropy_delta in formatted output**

In `tests/test_telemetry.py`, the `TestFormatTelemetryWindow` class tests FILE event formatting.
Add a test that entropy_delta is NOT in the output, and update any existing test that asserts it IS present.

```python
# In TestFormatTelemetryWindow class
def test_file_event_excludes_entropy_delta(self):
    """entropy_delta must not appear in passive telemetry — agent must actively investigate."""
    events = [
        FileEvent(
            ts=datetime(2025, 6, 15, 10, 0, 5),
            path="C:/test.docx",
            size_delta=1024,
            entropy_delta=3.5,
            extension_change=".docx->.locked",
        )
    ]
    text = format_telemetry_window(events, datetime(2025, 6, 15, 10, 0, 0))
    assert "entropy_delta" not in text
    assert "FILE" in text
    assert "size_delta" in text
    assert "ext_change" in text
```

Run: `cd active-detective && python -m pytest tests/test_telemetry.py::TestFormatTelemetryWindow::test_file_event_excludes_entropy_delta -v`
Expected: FAIL (entropy_delta is currently rendered)

**Step 2: Remove entropy_delta from format_telemetry_window()**

In `simulator/telemetry.py` lines 86-94, the FILE event formatting block:

```python
# BEFORE (lines 86-94):
parts = [
    f"[{offset_str}] FILE path={event.path}",
    f"size_delta={event.size_delta:+d}",
    f"entropy_delta={event.entropy_delta:+.2f}",
]

# AFTER:
parts = [
    f"[{offset_str}] FILE path={event.path}",
    f"size_delta={event.size_delta:+d}",
]
```

Remove the `entropy_delta` line. Keep everything else.

Run: `cd active-detective && python -m pytest tests/test_telemetry.py -v`
Expected: All pass (including new test)

**Step 3: Update HeuristicAgent to not rely on entropy_delta**

The `HeuristicAgent` in `evaluation/baselines.py` uses `_extract_entropy_deltas()` to parse
`entropy_delta=+X.XX` from telemetry text for its heuristic. Since entropy_delta is no longer
in telemetry, the heuristic agent needs a different signal.

Update `HeuristicAgent.act()` to use extension changes and size deltas instead:
- Count `ext_change=` occurrences (strong encryption signal)
- Count large `size_delta` values
- Threshold: 2+ extension changes → quarantine, 1+ → alert, large size deltas → monitor

Also update `_extract_entropy_deltas()`: keep the function (it's tested and used by ExhaustiveAgent
for parsing tool results, not telemetry), but update the `HeuristicAgent` to not depend on it
for telemetry-based decisions.

The `ExhaustiveAgent` keyword check (line ~188) also looks for `entropy_delta=+` in accumulated
output — this is fine because ExhaustiveAgent calls tools like `inspect_file` which DO return
entropy in their results. The keyword matching happens on tool results, not raw telemetry.

Run: `cd active-detective && python -m pytest tests/test_evaluation.py -v`
Expected: All pass

**Step 4: Verify no other code parses entropy_delta from telemetry text**

Run: `grep -rn "entropy_delta" active-detective/ --include="*.py" | grep -v __pycache__ | grep -v ".pyc"`

Check that remaining references are:
- `simulator/models.py` (FileEvent dataclass field) — keep
- `simulator/generators.py` (generators set entropy_delta on FileEvent) — keep
- `tools/inspection.py` or tool modules (tools return entropy in results) — keep
- `evaluation/baselines.py` (`_extract_entropy_deltas` helper) — keep but only used on tool results
- Test files that test the dataclass or tool results — keep

None should parse entropy_delta from formatted telemetry text.

**Step 5: Commit**

```bash
git add simulator/telemetry.py tests/test_telemetry.py evaluation/baselines.py
git commit -m "remove entropy_delta from passive telemetry formatting

Agent must now use inspect_file or read_file_sample to discover
file entropy. Telemetry events show path, size_delta, ext_change,
and pid only. HeuristicAgent updated to use extension changes."
```

---

### Task 2: Remove recall_memory from tool layer

Remove `recall_memory` from `tools/inspection.py` (TOOL_COSTS + execute_tool dispatch),
`tools/parser.py` (positional arg map), and `training/prompts.py` (TOOL_DESCRIPTIONS).

**Files:**
- Modify: `active-detective/tools/inspection.py` (lines 17, 115-121)
- Modify: `active-detective/tools/parser.py` (line ~171)
- Modify: `active-detective/training/prompts.py` (lines ~25-28)
- Modify: `active-detective/tests/test_tools.py`
- Modify: `active-detective/tests/test_prompts.py`

**Step 1: Update tests to reflect recall_memory removal**

In `tests/test_tools.py`:
- Remove any test that calls `execute_tool("recall_memory", ...)` directly
- Remove recall_memory from `TestParseToolCallFunction` if there's a test for parsing
  `recall_memory("query")` positional syntax
- Update `TestExecuteTool` to verify recall_memory is NOT in TOOL_COSTS

Add a test:
```python
def test_recall_memory_not_in_tool_costs(self):
    from tools.inspection import TOOL_COSTS
    assert "recall_memory" not in TOOL_COSTS
```

In `tests/test_prompts.py`:
- Update test that checks all tools are in system prompt to exclude recall_memory
- Add test:
```python
def test_recall_memory_not_in_tool_descriptions(self):
    from training.prompts import TOOL_DESCRIPTIONS
    assert "recall_memory" not in TOOL_DESCRIPTIONS
```

Run: `cd active-detective && python -m pytest tests/test_tools.py tests/test_prompts.py -v`
Expected: FAIL (recall_memory still exists)

**Step 2: Remove recall_memory from tools/inspection.py**

In `TOOL_COSTS` dict: remove `"recall_memory": -0.03` entry.

In `execute_tool()` function: remove the `elif tool_name == "recall_memory":` branch (lines ~115-121).
Also remove the `memory_store` parameter from `execute_tool()` signature — it's no longer needed.

Update the function signature:
```python
# BEFORE:
def execute_tool(tool_name, args, host, memory_store=None):

# AFTER:
def execute_tool(tool_name, args, host):
```

Update ALL callers of `execute_tool()`:
- `environment/env.py` step() method: remove `self._memory` argument
- Any other callers

**Step 3: Remove recall_memory from tools/parser.py**

In the `param_maps` dict, remove:
```python
"recall_memory": ["query"],
```

**Step 4: Remove recall_memory from training/prompts.py**

In `TOOL_DESCRIPTIONS` dict, remove the `"recall_memory"` entry:
```python
"recall_memory": (
    "recall_memory(query) — Search past telemetry windows by similarity. "
    "Returns the most relevant historical observations."
),
```

**Step 5: Run tests**

Run: `cd active-detective && python -m pytest tests/test_tools.py tests/test_prompts.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add tools/inspection.py tools/parser.py training/prompts.py \
        tests/test_tools.py tests/test_prompts.py
git commit -m "remove recall_memory tool from tool layer

History windows are now provided directly in prompt context,
making keyword-similarity search redundant."
```

---

### Task 3: Remove recall_memory from environment layer

Remove MemoryStore usage from `environment/env.py` and `training/train_grpo.py`.

**Files:**
- Modify: `active-detective/environment/env.py` (reset, step, imports)
- Modify: `active-detective/training/train_grpo.py` (reset, recall_memory method, imports)
- Modify: `active-detective/tests/test_env.py`
- Modify: `active-detective/tests/test_train_grpo.py`
- Modify: `active-detective/tests/test_e2e.py` (if it references memory)

**Step 1: Update environment tests**

In `tests/test_env.py`:
- Remove any test that calls `env.step(ParsedToolCall("recall_memory", ...))` or tests
  memory-related behavior
- Remove `memory_top_k` from env constructor if tested
- Update the env fixture if it passes memory-related params

In `tests/test_train_grpo.py`:
- Remove `test_recall_memory_empty` and `test_recall_memory_with_history` tests
- Remove any test that calls `env.recall_memory(...)`

Run: `cd active-detective && python -m pytest tests/test_env.py tests/test_train_grpo.py -v`
Expected: FAIL (recall_memory still exists in source)

**Step 2: Remove MemoryStore from environment/env.py**

- Remove `from tools.memory import MemoryStore` import
- Remove `memory_top_k` constructor parameter
- Remove `self._memory` initialization in `__init__` and `reset()`
- Remove memory seeding with history_windows in `reset()` (lines ~143-146)
- Remove `self._memory` argument from `execute_tool()` call in `step()` — this was already
  handled in Task 2 when we changed the signature

Keep the `history_windows` parameter in `reset()` for now — it will be repurposed in Task 5
to store history window text for prompt context.

**Step 3: Remove recall_memory method from training/train_grpo.py DetectionEnv**

- Remove `from tools.memory import MemoryStore` import
- Remove `self._memory` from `reset()` method
- Delete the entire `recall_memory()` public method (lines ~221-234)
- Remove memory seeding with history_windows in `reset()`

**Step 4: Run all tests**

Run: `cd active-detective && python -m pytest tests/ -q`
Expected: All pass

**Step 5: Commit**

```bash
git add environment/env.py training/train_grpo.py \
        tests/test_env.py tests/test_train_grpo.py tests/test_e2e.py
git commit -m "remove recall_memory from environment and training layers

MemoryStore no longer needed — history provided in prompt context."
```

---

### Task 4: Remove recall_memory from evaluation layer

Remove from ablation variants and baseline agent tool pools.

**Files:**
- Modify: `active-detective/evaluation/ablation.py` (ALL_TOOLS, ABLATION_VARIANTS)
- Modify: `active-detective/evaluation/baselines.py` (RandomAgent, ExhaustiveAgent tool pools)
- Modify: `active-detective/tests/test_evaluation.py`

**Step 1: Update evaluation tests**

In `tests/test_evaluation.py`:
- `TestAblationVariants.test_full_has_all_tools`: remove `"recall_memory"` from expected set
- `TestAblationVariants.test_all_variants_defined`: remove `assert "no_recall_memory" in ABLATION_VARIANTS`
- `TestAblationVariants.test_no_recall_excludes_recall`: delete this test entirely
- `TestAblationVariants.test_each_removal_variant_excludes_exactly_one`: remove `"no_recall_memory"` from
  the `removal_variants` list
- `TestAblationVariants.test_v1_tools`: remove `"recall_memory"` from expected set (v1_tools was
  the original 4 tools which included recall_memory — update to 3 original tools)

Run: `cd active-detective && python -m pytest tests/test_evaluation.py -v`
Expected: FAIL (recall_memory still in source)

**Step 2: Update evaluation/ablation.py**

- Remove `"recall_memory"` from `ALL_TOOLS` list
- Remove `"recall_memory"` from `V1_TOOLS` list
- Delete `"no_recall_memory"` variant from `ABLATION_VARIANTS` dict
- Update `"full"` variant (it copies ALL_TOOLS, so this is automatic)

**Step 3: Update evaluation/baselines.py**

- `RandomAgent`: remove `"recall_memory"` from the tool pool and its argument construction
- `ExhaustiveAgent`: remove `("recall_memory", {"query": ...})` from `TOOL_SEQUENCE`

**Step 4: Run tests**

Run: `cd active-detective && python -m pytest tests/test_evaluation.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add evaluation/ablation.py evaluation/baselines.py tests/test_evaluation.py
git commit -m "remove recall_memory from evaluation: ablation variants and baselines"
```

---

### Task 5: Clean up tools/memory.py and remaining references

Delete the memory module and remove any remaining imports/references.

**Files:**
- Delete: `active-detective/tools/memory.py`
- Delete: `active-detective/tests/test_memory.py`
- Modify: `active-detective/tools/__init__.py` (if it exports memory)
- Modify: `active-detective/scripts/run_untrained.py` (if it references memory)

**Step 1: Verify no remaining imports of tools.memory**

Run: `grep -rn "tools.memory\|tools/memory\|from tools import.*memory\|MemoryStore" active-detective/ --include="*.py" | grep -v __pycache__`

Should return only `tools/memory.py` itself and `tests/test_memory.py`.

**Step 2: Delete files**

```bash
rm active-detective/tools/memory.py
rm active-detective/tests/test_memory.py
```

**Step 3: Update any __init__.py that exports memory**

Check `tools/__init__.py` — if it imports or exports `MemoryStore` or `recall_memory`, remove those lines.

**Step 4: Run full test suite**

Run: `cd active-detective && python -m pytest tests/ -q`
Expected: All pass (same count minus deleted memory tests)

**Step 5: Commit**

```bash
git add -A  # includes deletions
git commit -m "delete tools/memory.py and test_memory.py — recall_memory fully removed"
```

---

### Task 6: Add multi-window history to episode generation

Update `generate_episode()` in `simulator/telemetry.py` to produce a list of
telemetry windows at different progress levels.

**Files:**
- Modify: `active-detective/simulator/telemetry.py` (generate_episode, Episode dataclass)
- Modify: `active-detective/tests/test_telemetry.py`

**Step 1: Write failing tests for multi-window generation**

Add to `tests/test_telemetry.py`:

```python
class TestMultiWindowGeneration:
    def test_returns_history_windows(self):
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BLITZ, 0.8, rng, attack_progress=0.6, n_history=2)
        assert len(episode.history_windows) == 2
        assert isinstance(episode.history_windows[0], str)
        assert len(episode.history_windows[0]) > 0

    def test_benign_has_history(self):
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BENIGN, 0.8, rng, n_history=2)
        assert len(episode.history_windows) == 2

    def test_zero_history(self):
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BLITZ, 0.8, rng, attack_progress=0.6, n_history=0)
        assert episode.history_windows == []

    def test_default_history_is_two(self):
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BLITZ, 0.8, rng, attack_progress=0.6)
        assert len(episode.history_windows) == 2

    def test_history_progress_increases(self):
        """History windows should represent earlier attack stages."""
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BLITZ, 0.8, rng, attack_progress=0.6, n_history=2)
        # History windows exist and are different from each other
        assert episode.history_windows[0] != episode.history_windows[1]

    def test_reproducible_with_seed(self):
        ep1 = generate_episode(
            ScenarioType.BLITZ, 0.8, np.random.RandomState(42),
            attack_progress=0.6, n_history=2)
        ep2 = generate_episode(
            ScenarioType.BLITZ, 0.8, np.random.RandomState(42),
            attack_progress=0.6, n_history=2)
        assert ep1.history_windows == ep2.history_windows
        assert ep1.input_text == ep2.input_text
```

Run: `cd active-detective && python -m pytest tests/test_telemetry.py::TestMultiWindowGeneration -v`
Expected: FAIL (Episode has no history_windows attribute)

**Step 2: Add history_windows field to Episode dataclass**

In `simulator/telemetry.py`, the `Episode` dataclass (lines 28-37):

```python
# Add field:
@dataclass
class Episode:
    input_text: str
    ground_truth: GroundTruth
    scenario_type: ScenarioType
    observability: float
    raw_event_count: int
    visible_event_count: int
    history_windows: list[str] = field(default_factory=list)  # NEW
```

**Step 3: Implement multi-window generation in generate_episode()**

Add `n_history: int = 2` parameter to `generate_episode()`.

The implementation strategy:
- For attack episodes: generate history windows at progress values
  `progress * (i+1) / (n_history+1)` for `i in range(n_history)`
- For benign episodes: generate history windows using randomly-chosen benign generators
- Each history window gets a fresh HostState at an earlier timestamp
- The current window (already generated) uses the original `attack_progress`

```python
def generate_episode(
    scenario_type: ScenarioType,
    observability: float,
    rng: np.random.RandomState,
    now: datetime | None = None,
    attack_progress: float = 0.5,
    n_history: int = 2,                    # NEW
) -> Episode:
    if now is None:
        now = datetime(2025, 6, 15, 10, 0, 0)

    # ── Generate history windows ──
    history_windows: list[str] = []
    for i in range(n_history):
        hist_now = now - timedelta(seconds=(n_history - i) * 120)
        hist_rng = np.random.RandomState(rng.randint(0, 2**31))
        hist_host = HostState.create(hist_rng, hist_now)

        hist_events: list = []
        # Add background benign activity
        benign_gen = rng.choice([office_edits, browser_downloads, system_maintenance])
        hist_events.extend(benign_gen(hist_host, np.random.RandomState(rng.randint(0, 2**31))))

        # For attack scenarios, add attack events at earlier progress
        if scenario_type != ScenarioType.BENIGN:
            hist_progress = attack_progress * (i + 1) / (n_history + 1)
            attack_gen = _get_attack_generator(scenario_type)
            attack_events, _ = attack_gen(
                hist_host, np.random.RandomState(rng.randint(0, 2**31)),
                progress=hist_progress,
            )
            hist_events.extend(attack_events)

        hist_visible = apply_observability_filter(hist_events, observability,
                                                   np.random.RandomState(rng.randint(0, 2**31)))
        history_windows.append(format_telemetry_window(hist_visible, hist_now))

    # ── Generate current window (existing logic) ──
    host = HostState.create(rng, now)
    # ... rest of existing generate_episode code ...

    return Episode(
        input_text=...,
        ground_truth=...,
        scenario_type=...,
        observability=...,
        raw_event_count=...,
        visible_event_count=...,
        history_windows=history_windows,  # NEW
    )
```

You will need a helper to map scenario type to generator:

```python
def _get_attack_generator(scenario_type: ScenarioType):
    """Return the attack generator function for a scenario type."""
    return {
        ScenarioType.BLITZ: gen.blitz_encryptor,
        ScenarioType.SLEEPER: gen.slow_sleeper,
        ScenarioType.EXFIL_FIRST: gen.exfil_first,
        ScenarioType.SEMANTIC_SHUFFLE: gen.semantic_shuffle,
    }[scenario_type]
```

Important: consume RNG state for history BEFORE the existing current-window generation
so that `n_history=0` produces identical output to the old code with the same seed.
To preserve backward compatibility, extract the current-window RNG draws to happen after
history generation. OR accept that seeds produce different results (simpler, acceptable since
we're changing the episode format anyway).

**Step 4: Run tests**

Run: `cd active-detective && python -m pytest tests/test_telemetry.py -v`
Expected: All pass

Then run full suite to check nothing broke:
Run: `cd active-detective && python -m pytest tests/ -q`

**Step 5: Commit**

```bash
git add simulator/telemetry.py tests/test_telemetry.py
git commit -m "add multi-window history generation to generate_episode()

Each episode now produces n_history prior telemetry windows at
earlier attack progress levels. Benign episodes use different
benign generators per window."
```

---

### Task 7: Format history windows in prompt context

Update `build_chat_messages()` to include history windows in the user message.

**Files:**
- Modify: `active-detective/training/prompts.py` (build_chat_messages)
- Modify: `active-detective/tests/test_prompts.py`

**Step 1: Write failing tests**

Add to `tests/test_prompts.py`:

```python
class TestHistoryInPrompt:
    def test_history_windows_in_user_message(self):
        system = build_system_prompt()
        history_windows = [
            "[+0.0s] FILE path=C:/test.docx size_delta=+100",
            "[+0.0s] PROC pid=5 name=svchost.exe parent=1",
        ]
        current = "[+0.0s] FILE path=C:/test2.docx size_delta=+200 ext_change=.docx->.locked"
        messages = build_chat_messages(system, current, history_windows=history_windows)
        user_msg = messages[-1]["content"]
        assert "Window t-2" in user_msg or "window t-2" in user_msg.lower()
        assert "Window t-1" in user_msg or "window t-1" in user_msg.lower()
        assert "CURRENT" in user_msg or "current" in user_msg.lower()
        assert history_windows[0] in user_msg
        assert current in user_msg

    def test_no_history_just_current(self):
        system = build_system_prompt()
        current = "[+0.0s] FILE path=C:/test.docx size_delta=+100"
        messages = build_chat_messages(system, current)
        user_msg = messages[-1]["content"]
        assert current in user_msg
        # Should not have window labels when there's no history
        assert "t-2" not in user_msg

    def test_single_history_window(self):
        system = build_system_prompt()
        history_windows = ["[+0.0s] PROC pid=5 name=svchost.exe parent=1"]
        current = "[+0.0s] FILE path=C:/test.docx size_delta=+100"
        messages = build_chat_messages(system, current, history_windows=history_windows)
        user_msg = messages[-1]["content"]
        assert history_windows[0] in user_msg
        assert current in user_msg
```

Run: `cd active-detective && python -m pytest tests/test_prompts.py::TestHistoryInPrompt -v`
Expected: FAIL

**Step 2: Update build_chat_messages()**

Add `history_windows: list[str] | None = None` parameter.

When history windows are provided, format the user message as:

```
--- Window t-2 (prior) ---
[telemetry from 2 windows ago]

--- Window t-1 (prior) ---
[telemetry from 1 window ago]

--- Current window ---
[current telemetry]
```

When no history windows, just use the current telemetry as before (no labels).

```python
def build_chat_messages(
    system_prompt: str,
    telemetry_window: str,
    history: list[dict] | None = None,
    history_windows: list[str] | None = None,  # NEW
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)

    # Build user content with history context
    if history_windows:
        parts = []
        n = len(history_windows)
        for i, hw in enumerate(history_windows):
            label = f"Window t-{n - i} (prior)"
            parts.append(f"--- {label} ---\n{hw}")
        parts.append(f"--- Current window ---\n{telemetry_window}")
        user_content = "\n\n".join(parts)
    else:
        user_content = telemetry_window

    messages.append({"role": "user", "content": user_content})
    return messages
```

**Step 3: Run tests**

Run: `cd active-detective && python -m pytest tests/test_prompts.py -v`
Expected: All pass

**Step 4: Commit**

```bash
git add training/prompts.py tests/test_prompts.py
git commit -m "format history windows in prompt context

build_chat_messages() now accepts history_windows parameter
and formats them as labeled prior context in the user message."
```

---

### Task 8: Wire history through environment and training

Connect multi-window history from episode generation through to the prompt.

**Files:**
- Modify: `active-detective/environment/env.py` (reset stores history, passes to prompt)
- Modify: `active-detective/training/train_grpo.py` (reset, prepare_dataset)
- Modify: `active-detective/training/scenarios.py` (generate_training_scenarios)
- Modify: `active-detective/tests/test_env.py`
- Modify: `active-detective/tests/test_train_grpo.py`

**Step 1: Write failing tests**

In `tests/test_env.py`, add:

```python
def test_reset_stores_history_windows(self, env, rng):
    """Environment should store history windows from episode generation."""
    telemetry = env.reset(ScenarioType.BLITZ, 0.8, rng, attack_progress=0.6)
    assert hasattr(env._episode, 'history_windows')
    assert len(env._episode.history_windows) == 2  # default n_history
```

In `tests/test_train_grpo.py`, in `TestPrepareDataset`:

```python
def test_scenario_data_includes_n_history(self):
    config = TrainingConfig(n_episodes=3, seed=42)
    dataset = prepare_dataset(config)
    for item in dataset:
        data = json.loads(item["scenario_data"])
        assert "n_history" in data
```

Run: `cd active-detective && python -m pytest tests/test_env.py::test_reset_stores_history_windows tests/test_train_grpo.py::TestPrepareDataset::test_scenario_data_includes_n_history -v`
Expected: FAIL

**Step 2: Update environment/env.py**

In `reset()`:
- `generate_episode()` now returns `history_windows` on the Episode — no changes needed for that
- Pass history_windows to prompt construction when building messages

The environment doesn't directly build prompts (that's done in `scripts/run_untrained.py`
and `training/train_grpo.py`), so the main change is ensuring `self._episode.history_windows`
is accessible.

Since `generate_episode()` now defaults to `n_history=2`, this happens automatically.
Add `n_history` parameter to `reset()` to allow control:

```python
def reset(self, scenario_type, observability, rng,
          attack_progress=0.5, history_windows=None, n_history=2):
    self._episode = generate_episode(
        scenario_type, observability, rng,
        attack_progress=attack_progress,
        n_history=n_history,
    )
    # ... rest of reset ...
```

Remove the old `history_windows` parameter (it was for seeding MemoryStore, which is gone).

**Step 3: Update training/train_grpo.py**

In `DetectionEnv.reset()`:
- Parse `n_history` from scenario_data (default 2)
- Pass to `generate_episode()`
- Remove old history_windows/memory code

In `prepare_dataset()`:
- Add `n_history` to scenario_data JSON:
```python
scenario_data = {
    "scenario_type": ...,
    "observability": ...,
    "attack_progress": ...,
    "seed": ...,
    "n_history": config.n_history if hasattr(config, 'n_history') else 2,
}
```
- Remove `"history_windows": []` from scenario_data (no longer needed — history is generated
  at reset time from n_history + attack_progress)

Add `n_history: int = 2` to `TrainingConfig`.

**Step 4: Update training/scenarios.py**

`generate_training_scenarios()` calls `generate_episode()` — it now gets history_windows
for free since the default `n_history=2` kicks in. No changes needed unless we want
configurable n_history, which can wait.

**Step 5: Update scripts/run_untrained.py**

In `run_demo()` and `run_live()`:
- After `env.reset()`, access `env._episode.history_windows` and pass to `build_chat_messages()`
- Update the `build_chat_messages()` calls to include `history_windows=env._episode.history_windows`

**Step 6: Run full test suite**

Run: `cd active-detective && python -m pytest tests/ -q`
Expected: All pass

**Step 7: Commit**

```bash
git add environment/env.py training/train_grpo.py training/scenarios.py \
        scripts/run_untrained.py tests/test_env.py tests/test_train_grpo.py
git commit -m "wire multi-window history through environment and training

Episode history_windows now flow from generate_episode() through
environment reset to prompt construction. prepare_dataset()
includes n_history in scenario_data."
```

---

### Task 9: Update system prompt to mention history context

The system prompt should tell the agent that history windows are provided.

**Files:**
- Modify: `active-detective/training/prompts.py` (SYSTEM_PROMPT_TEMPLATE)
- Modify: `active-detective/tests/test_prompts.py`

**Step 1: Write failing test**

```python
def test_system_prompt_mentions_history(self):
    prompt = build_system_prompt()
    assert "prior" in prompt.lower() or "history" in prompt.lower() or "previous" in prompt.lower()
```

Run: `cd active-detective && python -m pytest tests/test_prompts.py -v`
Expected: FAIL (or pass if the template already mentions it — check first)

**Step 2: Update SYSTEM_PROMPT_TEMPLATE**

Add a brief note about history context. Something like:

```
You will receive telemetry from a Windows endpoint. Prior telemetry windows
(if any) show what happened earlier on this host. The current window shows
the latest activity. Investigate the current window using the tools below.
```

This replaces or augments the existing telemetry description in the template.

**Step 3: Run tests**

Run: `cd active-detective && python -m pytest tests/test_prompts.py -v`
Expected: All pass

**Step 4: Commit**

```bash
git add training/prompts.py tests/test_prompts.py
git commit -m "update system prompt to describe history window context"
```

---

### Task 10: Final verification and cleanup

**Step 1: Run full test suite**

```bash
cd active-detective && python -m pytest tests/ -v
```

Expected: All pass. Count should be roughly 365 minus deleted memory tests minus
deleted recall_memory tests, plus new history/entropy tests.

**Step 2: Run demo script**

```bash
cd active-detective && python -m scripts.run_untrained --mode demo
```

Expected: Runs without error. Telemetry output should NOT show entropy_delta.
History windows should appear in the prompt section.

**Step 3: Grep for stale references**

```bash
# Should find NO results:
grep -rn "recall_memory\|MemoryStore\|memory_store\|memory_top_k" \
    active-detective/ --include="*.py" | grep -v __pycache__

# entropy_delta should only appear in models.py, generators.py, tool results:
grep -rn "entropy_delta" active-detective/ --include="*.py" | grep -v __pycache__
```

**Step 4: Check test coverage of new code**

```bash
cd active-detective && python -m pytest tests/ -q --tb=short
```

Verify no warnings or unexpected test skips.

**Step 5: Final commit if any cleanup was needed**

```bash
git add -A
git commit -m "final cleanup: verify all recall_memory and entropy_delta references removed"
```

---

## Summary of Changes

| Change | Files Modified | Tests |
|--------|---------------|-------|
| Remove entropy_delta from telemetry | telemetry.py, baselines.py | test_telemetry |
| Remove recall_memory from tools | inspection.py, parser.py, prompts.py | test_tools, test_prompts |
| Remove recall_memory from env | env.py, train_grpo.py | test_env, test_train_grpo |
| Remove recall_memory from eval | ablation.py, baselines.py | test_evaluation |
| Delete memory module | memory.py, test_memory.py | — |
| Add history windows | telemetry.py, Episode dataclass | test_telemetry |
| Format history in prompt | prompts.py | test_prompts |
| Wire history through layers | env.py, train_grpo.py, scenarios.py, run_untrained.py | test_env, test_train_grpo |
| Update system prompt | prompts.py | test_prompts |

**Files created:** 0
**Files deleted:** 2 (tools/memory.py, tests/test_memory.py)
**Files modified:** ~15
**Estimated test count delta:** -10 (deleted memory tests) + 12 (new history tests) = +2 net
