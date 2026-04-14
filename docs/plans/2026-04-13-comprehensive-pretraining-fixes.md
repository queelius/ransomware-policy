# Comprehensive Pre-Training Fix Plan

Date: 2026-04-13
Status: In progress

## Context

Before spending GPU money on GRPO training, code review plus simplifier identified 4 critical issues, 10 important issues, 5 simplifications, plus 4 new issues from expanded scan. Many interact, so fixing them in isolation would waste work.

## Issue Inventory

### Critical (training would fail or produce garbage)

- **C1**: GRPO config `generation_batch_size % num_generations != 0` (crash at startup)
- **C2**: `detection_reward` iteration plus env state leaks between GRPO siblings
- **C3**: `DetectionEnv` has no `k_max` budget enforcement
- **C4**: History windows use independent HostState instances (not temporally coupled)
- **C5** (new): Pareto frontier sorted descending in `evaluation/metrics.py:210`, silently excludes cheap baselines

### Important

- **I1**: `format_reward` hackable, emit tokens once for +0.1 forever
- **I2**: Reward logic duplicated between train and eval with subtle drift
- **I3**: Train vs eval verdict handling diverges (casing, set-before-validate)
- **I4**: `prepare_dataset` generates episodes then discards; `save_scenarios` saves different episodes than trained on
- **I5**: RNG state leaks between history/current windows, so `n_history` changes the current episode
- **I6**: `scan_directory` does recursive prefix match, so `scan_directory("C:/")` dumps filesystem
- **I7**: Benign `backup_operations`/`av_scan` touch more files than sleeper, teaches inverted heuristic
- **I8** (new): `PassiveLLM` baseline documented but never implemented
- **I9** (new): Test bug in `tests/test_evaluation.py:228` asserts verdict in ABLATION_VARIANTS dict keys (wrong variable)
- **I10** (new): `HostState.create()` receives `rng` but does not thread it to processes/connections/registry/event_log seeders

### Simplifications (unblock/enable above fixes)

- **S1**: `DetectionEnv` duplicates `execute_tool()` with ~150 lines of copy-paste
- **S2**: Reward computed twice (fixes I2)
- **S3**: 10+ copies of "spawn process plus emit ProcessEvent" pattern
- **S4**: Tool names as bare strings in 5+ locations
- **S5**: `VALID_VERDICTS` set duplicates `Verdict` enum

### Minor

- **M1** (new): `read_file_sample` returns 3 different JSON schemas (error-only, error plus metadata, data-only)
- **M2** (new): `scripts/run_untrained.py` accesses `env._host` private attribute

## Implementation Phases

Each phase equals one coherent commit. Critical path to training-ready: Phase 1 through Phase 5.

### Phase 1: Unify tool dispatch
Addresses: S1, S4, C3
Risk: Safe refactor plus budget enforcement
Effort: 4 to 6 hours

- Introduce `ToolName` str-enum in `tools/inspection.py`
- Replace dispatch `if/elif` with a `_DISPATCH: dict[ToolName, Callable]`
- Rewrite each `DetectionEnv` tool method as a thin wrapper:
  - Enforce budget: `if self._steps >= self._k_max: return error` (C3)
  - Delegate to `execute_tool(name, args, self._host)`
  - Accumulate cost from returned cost
- `VALID_VERDICTS = {v.value for v in Verdict}` (S5 partial)
- Validate verdict before setting `_verdict` (I3)

Tests first:
- `test_budget_exhaustion_returns_error_no_side_effect`
- `test_decide_invalid_does_not_set_verdict`
- `test_tool_name_enum_matches_cost_keys`

### Phase 2: Single reward source
Addresses: S2, I2, I3 (casing)
Risk: Behavioral change
Effort: 2 hours

- Extend `compute_reward` to handle `verdict=None` uniformly
- Normalize verdict case before `Verdict(...)` coercion
- Add `MISSING_VERDICT_BENIGN_PENALTY = -0.5` constant
- Rewrite `_compute_env_reward` as thin adapter

### Phase 3: History temporal coupling
Addresses: C4, I5
Risk: Architectural change, redefines episode semantics
Effort: 4 hours

- One `HostState` per episode, not per window
- Replay generators against the same host at increasing `attack_progress`
- Derive per-window RNG: `window_rng = np.random.RandomState(parent.randint(...))`
- `n_history` should not change current-window episode at fixed seed

Tests first:
- `test_history_shares_single_host`
- `test_current_window_invariant_to_n_history`
- `test_files_encrypted_in_earlier_window_stay_encrypted`

### Phase 4: GRPO config plus env isolation
Addresses: C1, C2
Risk: Behavioral plus architectural
Effort: 3 hours

- `per_device_train_batch_size = config.group_size`
- `gradient_accumulation_steps = 1`
- Verify TRL `environment_factory` creates fresh env per rollout; test with `test_reset_produces_fresh_host`

### Phase 5: Dataset reproducibility
Addresses: I4
Risk: Behavioral (what gets saved)
Effort: 2 hours

- New `build_scenario_plan()` that returns `(scenario_type, observability, progress, seed, n_history)` tuples without generating episodes
- `save_scenario_plan` serializes exactly what the env uses at reset

### Phase 6: Parser and simulator hygiene
Addresses: I6, I7
Risk: Behavioral
Effort: 1.5 hours

- `FileRegistry.list_directory` becomes non-recursive; add `list_directory_recursive` for explicit recursive use
- Reduce `backup_operations` and `av_scan` file-touch counts below sleeper's minimum

### Phase 7: Harden format_reward
Addresses: I1
Risk: Behavioral
Effort: 1 hour

- Count well-formed `<think>...</think>` pairs, not just open tags
- Require parser-validated tool calls
- Reject empty thinking blocks

### Phase 8: Generator dedup
Addresses: S3
Risk: Safe refactor
Effort: 45 min

- Extract `_spawn_and_emit(host, name, parent_pid, cmd, now)` helper
- Replace ~10 call sites

### Phase 9: Evaluation fixes
Addresses: C5, I8, I9, M1, M2, I10
Risk: Bug fixes plus new baseline
Effort: 2 hours

- Fix Pareto sort direction in `metrics.py:210`
- Implement `PassiveLLM` baseline
- Fix `test_evaluation.py:228` assertion
- Unify `read_file_sample` return schema
- Drop `env._host` access in `run_untrained.py` or expose via property
- Thread `rng` through all `HostState.create` seeders (or document intentional determinism)

## Test Strategy

- Keep all 408 existing tests passing
- TDD: new tests fail first, then fix
- Target: ~435 tests after all phases
- Add `tests/test_reproducibility.py` pinning top-level hash of a 10-episode plan to prevent future regressions

## Execution Principles

1. One commit per phase
2. Run full test suite before each commit
3. No mocking in integration tests; use real HostState
4. Use `Verdict` enum, `ToolName` enum; no string constants at call sites
5. Document behavioral changes in commit message
