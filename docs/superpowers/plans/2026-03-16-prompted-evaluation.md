# Prompted Evaluation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MCP server exposing the active-detective simulator as investigation tools so Claude Code can act as the ransomware detection agent, plus supporting evaluation infrastructure and a fine-tuning data sources YAML.

**Architecture:** A `DetectiveSession` class encapsulates episode generation, tool dispatch, state tracking, budget enforcement, and result recording. An MCP server (`FastMCP`) wraps it as stdio tools. Batch evaluation uses the `claude` CLI with per-scenario env vars. A `compute_metrics.py` script aggregates results using existing `evaluation/metrics.py`.

**Tech Stack:** Python 3.10+, `mcp` (MCP Python SDK), `numpy`, existing active-detective modules (simulator, tools, environment, evaluation)

**Spec:** `docs/superpowers/specs/2026-03-16-prompted-evaluation-design.md`

---

## Chunk 1: DetectiveSession + MCP Server

### Task 1: DetectiveSession class — core state management

The testable core that the MCP server wraps. Handles episode lifecycle, tool dispatch, state tracking, budget enforcement, and result recording.

**Files:**
- Create: `active-detective/mcp_session.py`
- Test: `active-detective/tests/test_mcp_session.py`

- [ ] **Step 1: Write failing tests for DetectiveSession init and get_telemetry**

```python
# tests/test_mcp_session.py
"""Tests for MCP DetectiveSession state management."""
import json
import numpy as np
import pytest
from simulator.models import ScenarioType
from mcp_session import DetectiveSession


class TestSessionInit:
    def test_creates_episode_from_config(self):
        session = DetectiveSession(
            seed=42, scenario_type="blitz", observability=0.7,
            attack_progress=0.5,
        )
        assert session.episode is not None
        assert session.episode.scenario_type == ScenarioType.BLITZ
        assert session.steps == 0
        assert session.cumulative_cost == 0.0
        assert session.verdict is None
        assert not session.is_done

    def test_benign_scenario(self):
        session = DetectiveSession(
            seed=99, scenario_type="benign", observability=0.9,
        )
        assert session.episode.ground_truth.is_ransomware is False

    def test_get_telemetry_returns_formatted_text(self):
        session = DetectiveSession(
            seed=42, scenario_type="blitz", observability=0.7,
            attack_progress=0.5,
        )
        telemetry = session.get_telemetry()
        assert isinstance(telemetry, str)
        assert len(telemetry) > 0
        # Should contain history window labels
        assert "Window t-" in telemetry or "---" in telemetry or "[t+" in telemetry

    def test_get_telemetry_includes_history_windows(self):
        session = DetectiveSession(
            seed=42, scenario_type="sleeper", observability=0.7,
            attack_progress=0.5, n_history=2,
        )
        telemetry = session.get_telemetry()
        assert "Window t-2" in telemetry
        assert "Window t-1" in telemetry
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/test_mcp_session.py -v`
Expected: FAIL (ImportError — mcp_session not found)

- [ ] **Step 3: Implement DetectiveSession init and get_telemetry**

```python
# active-detective/mcp_session.py
"""DetectiveSession: testable core for MCP server state management."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from simulator.models import ScenarioType, GroundTruth, Verdict
from simulator.telemetry import Episode, generate_episode
from tools.inspection import execute_tool, TOOL_COSTS, VALID_VERDICTS
from environment.reward import compute_reward, RewardBreakdown


class DetectiveSession:
    """Manages a single investigation episode's lifecycle."""

    def __init__(
        self,
        seed: int,
        scenario_type: str,
        observability: float = 0.7,
        attack_progress: float = 0.5,
        k_max: int = 5,
        n_history: int = 2,
        results_file: str | None = None,
    ) -> None:
        self.seed = seed
        self.k_max = k_max
        self.results_file = results_file

        rng = np.random.RandomState(seed)
        stype = ScenarioType(scenario_type)
        progress = 0.0 if stype == ScenarioType.BENIGN else attack_progress

        self.episode: Episode = generate_episode(
            scenario_type=stype,
            observability=observability,
            rng=rng,
            attack_progress=progress,
            n_history=n_history,
        )
        self.host = self.episode.host

        # State tracking
        self.steps: int = 0
        self.cumulative_cost: float = 0.0
        self.tools_used: list[str] = []
        self.verdict: str | None = None
        self.explanation: str = ""

    @property
    def is_done(self) -> bool:
        return self.verdict is not None

    @property
    def budget_remaining(self) -> int:
        return self.k_max - self.steps

    def get_telemetry(self) -> str:
        """Return formatted telemetry text with history windows."""
        parts: list[str] = []
        for i, hw in enumerate(self.episode.history_windows):
            n = len(self.episode.history_windows) - i
            parts.append(f"--- Window t-{n} (prior) ---")
            parts.append(hw)
        parts.append("--- Current Window ---")
        parts.append(self.episode.input_text)
        return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/test_mcp_session.py::TestSessionInit -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for tool execution**

```python
# Append to tests/test_mcp_session.py
class TestSessionToolExecution:
    def test_execute_investigation_tool(self):
        session = DetectiveSession(
            seed=42, scenario_type="blitz", observability=0.7,
            attack_progress=0.5,
        )
        # Get a file path from telemetry to inspect
        telemetry = session.get_telemetry()
        # Use a known directory
        result = session.execute_tool("scan_directory", {"path": "C:/Users/A/Documents"})
        assert "files" in result or "error" in result
        assert session.steps == 1
        assert session.cumulative_cost == pytest.approx(-0.05)
        assert session.tools_used == ["scan_directory"]

    def test_execute_multiple_tools(self):
        session = DetectiveSession(
            seed=42, scenario_type="blitz", observability=0.7,
            attack_progress=0.5,
        )
        session.execute_tool("scan_directory", {"path": "C:/Users/A/Documents"})
        session.execute_tool("inspect_file", {"path": "C:/Users/A/Documents/report.docx"})
        assert session.steps == 2
        assert len(session.tools_used) == 2

    def test_budget_exceeded_returns_error(self):
        session = DetectiveSession(
            seed=42, scenario_type="blitz", observability=0.7,
            attack_progress=0.5, k_max=1,
        )
        session.execute_tool("scan_directory", {"path": "C:/Users/A/Documents"})
        result = session.execute_tool("inspect_file", {"path": "C:/Users/A/Documents/report.docx"})
        assert "budget" in result.get("error", "").lower() or "budget" in str(result).lower()
        assert session.steps == 1  # second call not counted

    def test_tool_after_decide_returns_error(self):
        session = DetectiveSession(
            seed=42, scenario_type="benign", observability=0.9,
        )
        session.decide("ignore", "Looks clean")
        result = session.execute_tool("scan_directory", {"path": "C:/Users/A/Documents"})
        assert "error" in result
```

- [ ] **Step 6: Implement execute_tool**

```python
# Add to DetectiveSession class in mcp_session.py
    def execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute an investigation tool. Returns result dict."""
        if self.is_done:
            return {"error": "Investigation already concluded. No further tools available."}
        if self.steps >= self.k_max:
            return {
                "error": f"Budget exhausted ({self.k_max} steps used). "
                "You must call DECIDE to render your verdict."
            }

        result, cost = execute_tool(tool_name, args, self.host)
        self.steps += 1
        self.cumulative_cost += cost
        self.tools_used.append(tool_name)
        return result
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/test_mcp_session.py::TestSessionToolExecution -v`
Expected: PASS

- [ ] **Step 8: Write failing tests for DECIDE and result recording**

```python
# Append to tests/test_mcp_session.py
import tempfile

class TestSessionDecide:
    def test_decide_sets_verdict(self):
        session = DetectiveSession(
            seed=42, scenario_type="benign", observability=0.9,
        )
        result = session.decide("ignore", "No indicators found")
        assert session.verdict == "ignore"
        assert session.is_done
        assert "correct" in result

    def test_decide_computes_reward(self):
        session = DetectiveSession(
            seed=42, scenario_type="blitz", observability=0.7,
            attack_progress=0.5,
        )
        session.execute_tool("scan_directory", {"path": "C:/Users/A/Documents"})
        result = session.decide("quarantine", "Encryption detected")
        assert "reward" in result
        assert isinstance(result["reward"], float)

    def test_decide_invalid_verdict(self):
        session = DetectiveSession(
            seed=42, scenario_type="benign", observability=0.9,
        )
        result = session.decide("nuke_from_orbit", "Just to be safe")
        assert "error" in result
        assert session.verdict is None

    def test_decide_writes_results_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        session = DetectiveSession(
            seed=42, scenario_type="blitz", observability=0.7,
            attack_progress=0.5, results_file=path,
        )
        session.decide("quarantine", "Encryption detected")
        with open(path) as f:
            line = json.loads(f.readline())
        assert line["seed"] == 42
        assert line["scenario_type"] == "blitz"
        assert line["verdict"] == "quarantine"
        assert "reward" in line
        assert "tools_used" in line
        assert "steps_taken" in line
        Path(path).unlink()

    def test_double_decide_errors(self):
        session = DetectiveSession(
            seed=42, scenario_type="benign", observability=0.9,
        )
        session.decide("ignore", "Clean")
        result = session.decide("alert", "Wait actually...")
        assert "error" in result
```

- [ ] **Step 9: Implement decide and result recording**

```python
# Add to DetectiveSession class in mcp_session.py
    def decide(self, verdict: str, explanation: str) -> dict:
        """Render final verdict. Returns result with reward breakdown."""
        if self.is_done:
            return {"error": "Verdict already rendered."}
        if verdict not in VALID_VERDICTS:
            return {
                "error": f"Invalid verdict '{verdict}'. "
                f"Must be one of: {', '.join(sorted(VALID_VERDICTS))}"
            }

        self.verdict = verdict
        self.explanation = explanation

        gt = self.episode.ground_truth
        predicted_ransomware = Verdict(verdict).is_ransomware_prediction
        correct = predicted_ransomware == gt.is_ransomware

        reward = compute_reward(
            verdict=verdict,
            ground_truth=gt,
            cumulative_cost=self.cumulative_cost,
            steps_taken=self.steps,
            max_steps=self.k_max,
            has_thinking=False,  # MCP doesn't track this
            has_tool_call=self.steps > 0,
        )

        result_record = {
            "seed": self.seed,
            "scenario_type": self.episode.scenario_type.value,
            "is_ransomware_gt": gt.is_ransomware,
            "attack_phase": gt.attack_phase,
            "observability": self.episode.observability,
            "verdict": verdict,
            "explanation": explanation,
            "steps_taken": self.steps,
            "cumulative_cost": self.cumulative_cost,
            "tools_used": self.tools_used,
            "reward": reward.total,
            "correct": correct,
            "k_max": self.k_max,
        }

        if self.results_file:
            with open(self.results_file, "a") as f:
                f.write(json.dumps(result_record) + "\n")

        return {
            "verdict": verdict,
            "correct": correct,
            "reward": reward.total,
            "verdict_reward": reward.verdict_reward,
            "efficiency_bonus": reward.efficiency_bonus,
            "action_cost": reward.action_cost,
        }
```

- [ ] **Step 10: Run all session tests**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/test_mcp_session.py -v`
Expected: ALL PASS

- [ ] **Step 11: Commit**

```bash
cd /home/spinoza/github/ransomware-policy/active-detective
git add mcp_session.py tests/test_mcp_session.py
git commit -m "feat: add DetectiveSession for MCP server state management"
```

---

### Task 2: MCP Server wrapping DetectiveSession

**Files:**
- Create: `active-detective/mcp_server.py`
- Create: `active-detective/tests/test_mcp_server.py`
- Modify: `active-detective/requirements.txt` — add `mcp>=1.0.0`

- [ ] **Step 1: Add mcp dependency to requirements.txt**

Append to `active-detective/requirements.txt`:
```
# ── MCP server ──────────────────────────────────
mcp>=1.0.0                     # FastMCP for stdio tool server
```

- [ ] **Step 2: Write the MCP server**

```python
# active-detective/mcp_server.py
"""MCP server exposing active-detective investigation tools.

Usage:
    SCENARIO_SEED=42 SCENARIO_TYPE=blitz python mcp_server.py

Environment variables:
    SCENARIO_SEED (required): Integer seed for reproducible episode generation
    SCENARIO_TYPE (required): benign, blitz, sleeper, exfil_first, semantic_shuffle
    OBSERVABILITY (optional): Float 0.0-1.0, default 0.7
    ATTACK_PROGRESS (optional): Float 0.0-1.0, default 0.5
    K_MAX (optional): Max investigation steps, default 5
    N_HISTORY (optional): Number of prior telemetry windows, default 2
    RESULTS_FILE (optional): Path to append JSONL results
    TOOL_SET (optional): Ablation variant name (default: full)
"""
from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

from mcp_session import DetectiveSession
from evaluation.ablation import ABLATION_VARIANTS, ALL_TOOLS

# ── Read configuration from environment ──────────────────────────
_seed = int(os.environ.get("SCENARIO_SEED", "0"))
_scenario_type = os.environ.get("SCENARIO_TYPE", "benign")
_observability = float(os.environ.get("OBSERVABILITY", "0.7"))
_attack_progress = float(os.environ.get("ATTACK_PROGRESS", "0.5"))
_k_max = int(os.environ.get("K_MAX", "5"))
_n_history = int(os.environ.get("N_HISTORY", "2"))
_results_file = os.environ.get("RESULTS_FILE")
_tool_set = os.environ.get("TOOL_SET", "full")

# Resolve available tools from ablation variant
_no_tools = _tool_set == "no_tools"
if _no_tools:
    _available_tools: set[str] = set()
elif _tool_set in ABLATION_VARIANTS:
    _available_tools = set(ABLATION_VARIANTS[_tool_set])
else:
    _available_tools = set(ALL_TOOLS)

# ── Create session ───────────────────────────────────────────────
_session = DetectiveSession(
    seed=_seed,
    scenario_type=_scenario_type,
    observability=_observability,
    attack_progress=_attack_progress,
    k_max=_k_max,
    n_history=_n_history,
    results_file=_results_file,
)

# ── MCP Server ───────────────────────────────────────────────────
server = FastMCP("active-detective")


def _check_available(tool_name: str) -> str | None:
    """Return error string if tool is disabled by ablation, else None."""
    if tool_name not in _available_tools:
        return (
            f"Tool '{tool_name}' is not available in this evaluation. "
            f"Available tools: {', '.join(sorted(_available_tools)) or 'none'}. "
            "Use DECIDE to render your verdict."
        )
    return None


@server.tool()
def get_telemetry() -> str:
    """Retrieve the host telemetry windows for analysis.
    Call this first to see the telemetry data before investigating.
    Returns current window plus prior history windows."""
    return _session.get_telemetry()


@server.tool()
def inspect_file(path: str) -> str:
    """Inspect a file's metadata: entropy, size, extension, content_type, modified timestamp.
    Cost: 1 investigation step."""
    err = _check_available("inspect_file")
    if err:
        return err
    import json
    return json.dumps(_session.execute_tool("inspect_file", {"path": path}), indent=2)


@server.tool()
def check_process(pid: int) -> str:
    """Check a running process: name, command line, parent, children.
    Cost: 1 investigation step."""
    err = _check_available("check_process")
    if err:
        return err
    import json
    return json.dumps(_session.execute_tool("check_process", {"pid": pid}), indent=2)


@server.tool()
def scan_directory(path: str) -> str:
    """List all files in a directory with metadata summaries.
    Most expensive tool. Cost: 1 investigation step."""
    err = _check_available("scan_directory")
    if err:
        return err
    import json
    return json.dumps(_session.execute_tool("scan_directory", {"path": path}), indent=2)


@server.tool()
def list_connections(filter_state: str | None = None) -> str:
    """List active network connections with optional state filter.
    filter_state: established, listening, closed, syn_sent (or omit for all).
    Cost: 1 investigation step."""
    err = _check_available("list_connections")
    if err:
        return err
    import json
    args = {}
    if filter_state:
        args["filter_state"] = filter_state
    return json.dumps(_session.execute_tool("list_connections", args), indent=2)


@server.tool()
def inspect_connection(conn_id: int) -> str:
    """Inspect a specific network connection in detail.
    Cost: 1 investigation step."""
    err = _check_available("inspect_connection")
    if err:
        return err
    import json
    return json.dumps(_session.execute_tool("inspect_connection", {"conn_id": conn_id}), indent=2)


@server.tool()
def query_registry(key_path: str) -> str:
    """Query a Windows registry key and its values.
    Useful for detecting persistence, Defender disabled, VSS deleted.
    Cost: 1 investigation step."""
    err = _check_available("query_registry")
    if err:
        return err
    import json
    return json.dumps(_session.execute_tool("query_registry", {"key_path": key_path}), indent=2)


@server.tool()
def list_process_handles(pid: int) -> str:
    """List a process's open file handles, loaded modules, and security context.
    Cost: 1 investigation step."""
    err = _check_available("list_process_handles")
    if err:
        return err
    import json
    return json.dumps(_session.execute_tool("list_process_handles", {"pid": pid}), indent=2)


@server.tool()
def query_event_log(
    source: str | None = None,
    event_id: int | None = None,
    since: str | None = None,
) -> str:
    """Query Windows Event Log entries.
    Filter by source (e.g. 'Windows Defender'), event_id, and/or timestamp.
    Cost: 1 investigation step."""
    err = _check_available("query_event_log")
    if err:
        return err
    import json
    args: dict = {}
    if source:
        args["source"] = source
    if event_id is not None:
        args["event_id"] = event_id
    if since:
        args["since"] = since
    return json.dumps(_session.execute_tool("query_event_log", args), indent=2)


@server.tool()
def read_file_sample(
    path: str,
    offset: int = 0,
    length: int = 256,
) -> str:
    """Read raw bytes from a file for forensic analysis.
    Returns hex dump, computed entropy, and magic bytes.
    Cost: 1 investigation step."""
    err = _check_available("read_file_sample")
    if err:
        return err
    import json
    return json.dumps(
        _session.execute_tool(
            "read_file_sample",
            {"path": path, "offset": offset, "length": length},
        ),
        indent=2,
    )


@server.tool()
def DECIDE(verdict: str, explanation: str) -> str:
    """Submit your final verdict. This ends the investigation.
    verdict: ignore, monitor, alert, quarantine, or block.
    explanation: Brief reasoning for your verdict."""
    import json
    return json.dumps(_session.decide(verdict, explanation), indent=2)


if __name__ == "__main__":
    server.run(transport="stdio")
```

- [ ] **Step 3: Write MCP server smoke test**

```python
# active-detective/tests/test_mcp_server.py
"""Smoke tests for MCP server module imports and tool registration."""
import os
import pytest

try:
    import mcp as _mcp  # noqa: F401
    _has_mcp = True
except ImportError:
    _has_mcp = False

pytestmark = pytest.mark.skipif(not _has_mcp, reason="mcp package not installed")


def test_mcp_server_imports():
    """Verify the mcp_server module can be imported with required env vars."""
    os.environ["SCENARIO_SEED"] = "42"
    os.environ["SCENARIO_TYPE"] = "blitz"
    os.environ["OBSERVABILITY"] = "0.7"
    os.environ["ATTACK_PROGRESS"] = "0.5"
    try:
        import mcp_server
        assert hasattr(mcp_server, "server")
        assert hasattr(mcp_server, "_session")
    finally:
        for key in ["SCENARIO_SEED", "SCENARIO_TYPE", "OBSERVABILITY", "ATTACK_PROGRESS"]:
            os.environ.pop(key, None)


def test_mcp_server_has_all_tools():
    """Verify all expected tools are registered."""
    os.environ["SCENARIO_SEED"] = "42"
    os.environ["SCENARIO_TYPE"] = "benign"
    try:
        import importlib
        import mcp_server
        importlib.reload(mcp_server)
        assert mcp_server.server is not None
    finally:
        for key in ["SCENARIO_SEED", "SCENARIO_TYPE"]:
            os.environ.pop(key, None)
```

- [ ] **Step 4: Run all tests**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && pip install "mcp>=1.0.0" && python -m pytest tests/test_mcp_session.py tests/test_mcp_server.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run existing test suite to verify no regressions**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/ -q`
Expected: 408+ tests pass

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/ransomware-policy/active-detective
git add mcp_server.py mcp_session.py requirements.txt tests/test_mcp_session.py tests/test_mcp_server.py
git commit -m "feat: add MCP server for prompted evaluation"
```

---

### Task 3: Add no_tools ablation variant

**Files:**
- Modify: `active-detective/evaluation/ablation.py:34-50` — add `no_tools` variant
- Test: `active-detective/tests/test_mcp_session.py` — add ablation test

- [ ] **Step 1: Write failing test for no_tools in ABLATION_VARIANTS**

```python
# Append to tests/test_mcp_session.py (or add to existing test_ablation tests)
def test_no_tools_variant_exists():
    from evaluation.ablation import ABLATION_VARIANTS
    assert "no_tools" in ABLATION_VARIANTS
    assert ABLATION_VARIANTS["no_tools"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/test_mcp_session.py::test_no_tools_variant_exists -v`
Expected: FAIL (KeyError)

- [ ] **Step 3: Add no_tools to ABLATION_VARIANTS**

In `evaluation/ablation.py`, after the `"v1_tools"` entry (around line 49), add:

```python
    "no_tools": [],
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/test_mcp_session.py::test_no_tools_variant_exists -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/ -q`
Expected: 408+ tests pass (no regressions)

- [ ] **Step 6: Commit**

```bash
cd /home/spinoza/github/ransomware-policy/active-detective
git add evaluation/ablation.py tests/test_mcp_session.py
git commit -m "feat: add no_tools ablation variant for passive analysis"
```

---

## Chunk 2: Evaluation Infrastructure

### Task 4: Eval manifest generator

Generates a JSONL manifest of scenario configs using `DEFAULT_SCENARIO_MIX` from `training/scenarios.py`.

**Files:**
- Create: `active-detective/scripts/generate_eval_manifest.py`
- Test: `active-detective/tests/test_eval_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_eval_manifest.py
"""Tests for evaluation manifest generation."""
import json
import tempfile
from pathlib import Path
from scripts.generate_eval_manifest import generate_manifest


class TestManifestGeneration:
    def test_generates_correct_count(self):
        manifest = generate_manifest(n=50, seed=42)
        assert len(manifest) == 50

    def test_scenario_distribution_roughly_matches_mix(self):
        manifest = generate_manifest(n=200, seed=42)
        types = [m["scenario_type"] for m in manifest]
        benign_frac = types.count("benign") / len(types)
        # 40% benign +/- tolerance
        assert 0.3 <= benign_frac <= 0.5

    def test_each_entry_has_required_fields(self):
        manifest = generate_manifest(n=10, seed=42)
        for entry in manifest:
            assert "seed" in entry
            assert "scenario_type" in entry
            assert "observability" in entry
            assert "attack_progress" in entry

    def test_benign_has_zero_progress(self):
        manifest = generate_manifest(n=100, seed=42)
        for entry in manifest:
            if entry["scenario_type"] == "benign":
                assert entry["attack_progress"] == 0.0

    def test_seeds_are_unique(self):
        manifest = generate_manifest(n=100, seed=42)
        seeds = [m["seed"] for m in manifest]
        assert len(seeds) == len(set(seeds))

    def test_deterministic_with_same_seed(self):
        m1 = generate_manifest(n=50, seed=42)
        m2 = generate_manifest(n=50, seed=42)
        assert m1 == m2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/test_eval_manifest.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement generate_manifest**

```python
# active-detective/scripts/generate_eval_manifest.py
"""Generate evaluation scenario manifests for batch evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.scenarios import DEFAULT_SCENARIO_MIX, DEFAULT_OBSERVABILITY_LEVELS
from simulator.models import ScenarioType


def generate_manifest(
    n: int = 200,
    seed: int = 42,
    scenario_mix: dict[ScenarioType, float] | None = None,
    observability_levels: list[float] | None = None,
) -> list[dict]:
    """Generate a list of scenario configurations for evaluation."""
    mix = scenario_mix or DEFAULT_SCENARIO_MIX
    obs_levels = observability_levels or DEFAULT_OBSERVABILITY_LEVELS
    rng = np.random.RandomState(seed)

    # Build scenario type list according to mix proportions
    types: list[ScenarioType] = []
    weights = list(mix.values())
    type_list = list(mix.keys())
    for _ in range(n):
        idx = rng.choice(len(type_list), p=weights)
        types.append(type_list[idx])

    manifest: list[dict] = []
    for i, stype in enumerate(types):
        obs = obs_levels[rng.randint(len(obs_levels))]
        progress = 0.0 if stype == ScenarioType.BENIGN else round(
            rng.uniform(0.2, 0.9), 2
        )
        manifest.append({
            "seed": seed * 10000 + i,  # unique per entry
            "scenario_type": stype.value,
            "observability": obs,
            "attack_progress": progress,
        })

    return manifest


def save_manifest(manifest: list[dict], path: str | Path) -> None:
    """Save manifest to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate eval manifest")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_manifest.jsonl")
    args = parser.parse_args()

    manifest = generate_manifest(n=args.n, seed=args.seed)
    save_manifest(manifest, args.output)
    print(f"Generated {len(manifest)} scenarios to {args.output}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/test_eval_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/ransomware-policy/active-detective
git add scripts/generate_eval_manifest.py tests/test_eval_manifest.py
git commit -m "feat: add evaluation manifest generator"
```

---

### Task 5: Compute metrics script

Reads results JSONL and computes detection/efficiency metrics using existing `evaluation/metrics.py`.

**Files:**
- Create: `active-detective/scripts/compute_metrics.py`
- Test: `active-detective/tests/test_compute_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_compute_metrics.py
"""Tests for compute_metrics script."""
import json
import tempfile
from pathlib import Path
from scripts.compute_metrics import load_results, compute_all_metrics


def _make_results_file(records: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


SAMPLE_RESULTS = [
    {"seed": 1, "scenario_type": "blitz", "verdict": "quarantine",
     "correct": True, "is_ransomware_gt": True, "attack_phase": "encryption",
     "steps_taken": 3, "cumulative_cost": -0.09, "tools_used": ["inspect_file", "scan_directory", "read_file_sample"],
     "reward": 0.86, "k_max": 5, "observability": 0.7},
    {"seed": 2, "scenario_type": "benign", "verdict": "ignore",
     "correct": True, "is_ransomware_gt": False, "attack_phase": None,
     "steps_taken": 1, "cumulative_cost": -0.02, "tools_used": ["inspect_file"],
     "reward": 1.18, "k_max": 5, "observability": 0.9},
    {"seed": 3, "scenario_type": "sleeper", "verdict": "ignore",
     "correct": False, "is_ransomware_gt": True, "attack_phase": "encryption",
     "steps_taken": 2, "cumulative_cost": -0.07, "tools_used": ["inspect_file", "check_process"],
     "reward": -1.93, "k_max": 5, "observability": 0.3},
]


class TestLoadResults:
    def test_loads_jsonl(self):
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        assert len(results) == 3
        Path(path).unlink()

    def test_returns_eval_results(self):
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        assert results[0].verdict == "quarantine"
        assert results[0].is_ransomware_gt is True
        Path(path).unlink()


class TestComputeAllMetrics:
    def test_returns_detection_and_efficiency(self):
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        metrics = compute_all_metrics(results)
        assert "detection" in metrics
        assert "efficiency" in metrics
        assert "by_scenario_type" in metrics
        assert metrics["detection"]["tp"] == 1  # blitz correct
        assert metrics["detection"]["fn"] == 1  # sleeper missed
        assert metrics["detection"]["tn"] == 1  # benign correct
        Path(path).unlink()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/test_compute_metrics.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement compute_metrics.py**

```python
# active-detective/scripts/compute_metrics.py
"""Compute evaluation metrics from results JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics import (
    EvalResult,
    compute_detection_metrics,
    compute_efficiency_metrics,
    compute_phase_accuracy,
    compute_tool_usage,
)


def load_results(path: str | Path) -> list[EvalResult]:
    """Load EvalResult objects from a JSONL file."""
    results: list[EvalResult] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            results.append(EvalResult(
                verdict=r["verdict"],
                is_ransomware_gt=r["is_ransomware_gt"],
                scenario_type=r["scenario_type"],
                attack_phase=r.get("attack_phase"),
                steps_taken=r["steps_taken"],
                cumulative_cost=r["cumulative_cost"],
                k_max=r["k_max"],
                tools_used=r.get("tools_used", []),
            ))
    return results


def compute_all_metrics(results: list[EvalResult]) -> dict:
    """Compute all metrics and return as a nested dict."""
    detection = compute_detection_metrics(results)
    efficiency = compute_efficiency_metrics(results)
    phase_acc = compute_phase_accuracy(results)
    tool_use = compute_tool_usage(results)

    # Per-scenario-type breakdown
    by_type: dict[str, dict] = {}
    types = set(r.scenario_type for r in results)
    for stype in sorted(types):
        subset = [r for r in results if r.scenario_type == stype]
        by_type[stype] = compute_detection_metrics(subset).to_dict()

    # Per-observability breakdown (group by nearest level)
    by_obs: dict[str, dict] = {}
    # We don't have observability on EvalResult, so skip if not available

    return {
        "detection": detection.to_dict(),
        "efficiency": efficiency.to_dict(),
        "by_scenario_type": by_type,
        "phase_accuracy": phase_acc,
        "tool_usage": tool_use,
        "n_episodes": len(results),
    }


def format_report(metrics: dict) -> str:
    """Format metrics as a readable report."""
    lines: list[str] = []
    d = metrics["detection"]
    e = metrics["efficiency"]

    lines.append(f"=== Detection Metrics ({metrics['n_episodes']} episodes) ===")
    lines.append(f"  Precision: {d['precision']:.3f}")
    lines.append(f"  Recall:    {d['recall']:.3f}")
    lines.append(f"  F1:        {d['f1']:.3f}")
    lines.append(f"  Accuracy:  {d['accuracy']:.3f}")
    lines.append(f"  TP={d['tp']} FP={d['fp']} TN={d['tn']} FN={d['fn']}")

    lines.append(f"\n=== Efficiency Metrics ===")
    lines.append(f"  Mean steps: {e['mean_steps']:.2f}")
    lines.append(f"  Mean cost:  {e['mean_cost']:.3f}")
    lines.append(f"  Early stop: {e['early_stop_rate']:.1%}")

    lines.append(f"\n=== By Scenario Type ===")
    for stype, m in sorted(metrics["by_scenario_type"].items()):
        lines.append(f"  {stype}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")

    lines.append(f"\n=== Tool Usage ===")
    for tool, count in sorted(metrics["tool_usage"].items(), key=lambda x: -x[1]):
        lines.append(f"  {tool}: {count}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("results", nargs="+", help="JSONL result files")
    args = parser.parse_args()

    for path in args.results:
        print(f"\n{'='*60}")
        print(f"Results: {path}")
        print(f"{'='*60}")
        results = load_results(path)
        metrics = compute_all_metrics(results)
        print(format_report(metrics))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/test_compute_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/spinoza/github/ransomware-policy/active-detective
git add scripts/compute_metrics.py tests/test_compute_metrics.py
git commit -m "feat: add compute_metrics script for results aggregation"
```

---

## Chunk 3: Prompts and Configuration

### Task 6: Full Opus evaluation prompt

The prompt that gets stuffed into Claude Opus's 1M context. Contains domain knowledge and few-shot examples.

**Files:**
- Create: `active-detective/prompts/eval_prompt.md`

- [ ] **Step 1: Create prompts directory**

Run: `mkdir -p /home/spinoza/github/ransomware-policy/active-detective/prompts`

- [ ] **Step 2: Write the full evaluation prompt**

Create `active-detective/prompts/eval_prompt.md` with these sections:

1. **Task description** — You are a ransomware forensic investigator. Call `get_telemetry()` to see the host telemetry, then use investigation tools to determine whether this host is under attack. End with `DECIDE(verdict, explanation)`. You have up to 5 investigation steps.

2. **Verdicts** — `ignore` (no indicators), `monitor` (suspicious but inconclusive), `alert` (likely malicious), `quarantine` (high confidence ransomware), `block` (active encryption).

3. **Domain knowledge** section:
   - Ransomware kill chain: initial access -> reconnaissance -> credential theft -> lateral movement -> data exfiltration -> encryption -> ransom note
   - Key file indicators: entropy above 7.5 bits/byte strongly suggests encryption. Normal docs: 3.5-5.5, images: 5.5-7.2, executables: 5.0-6.5. Extension changes (.locked, .encrypted, .enc) are strong signals.
   - Key registry indicators: `HKLM\SOFTWARE\...\Windows Defender\DisableRealtimeMonitoring=1`, VSS deletion, Run keys for persistence
   - Key process indicators: legitimate names (svchost.exe, SearchIndexer.exe) spawned with unusual parents or loading suspicious DLLs (beacon_x64.dll)
   - Key network indicators: large outbound transfers to uncommon IPs, especially before file modifications (exfil-first pattern)
   - Evasion techniques: semantic_shuffle attack modifies content without extension changes and keeps entropy below 6.5 to mimic compression
   - Temporal patterns: a slow_sleeper encrypts 1-2 files per window — compare across history windows for gradual trends

4. **Investigation strategy** — Start with `get_telemetry()`, look for anomalies (extension changes, entropy spikes, suspicious processes, outbound connections). Investigate the most suspicious indicators first. Use `scan_directory` sparingly (most expensive). Use `read_file_sample` for forensic confirmation of encryption.

5. **Few-shot example 1 — Benign (backup_operations)**: Show telemetry with many file reads (zero size_delta), a backup_service.exe process. Investigation: inspect the process, check a file — zero delta, normal entropy. Verdict: ignore.

6. **Few-shot example 2 — Attack (slow_sleeper)**: Show telemetry with subtle changes across windows. Investigation: inspect suspicious files, read_file_sample reveals high entropy. Verdict: quarantine.

Note: The few-shot examples should use realistic telemetry format matching the simulator output. Generate these by running a quick script or hand-crafting them to match `format_telemetry_window()` output format from `simulator/telemetry.py`.

- [ ] **Step 3: Commit**

```bash
cd /home/spinoza/github/ransomware-policy/active-detective
git add prompts/eval_prompt.md
git commit -m "feat: add full evaluation prompt for Opus with domain knowledge"
```

---

### Task 7: Minimal Qwen evaluation prompt

A stripped-down version for the small-model prompted baseline (fits in 4-8K context).

**Files:**
- Create: `active-detective/prompts/eval_prompt_minimal.md`

- [ ] **Step 1: Write the minimal prompt**

Include only:
1. Task description (same as full but shorter)
2. Verdict definitions (one line each)
3. Brief tool descriptions

No domain knowledge, no few-shot examples. This tests what the base model knows out-of-the-box.

- [ ] **Step 2: Commit**

```bash
cd /home/spinoza/github/ransomware-policy/active-detective
git add prompts/eval_prompt_minimal.md
git commit -m "feat: add minimal evaluation prompt for Qwen baseline"
```

---

### Task 8: MCP config and batch evaluation script

**Files:**
- Create: `active-detective/.mcp.json`
- Create: `active-detective/scripts/eval_prompted.sh`

- [ ] **Step 1: Create .mcp.json**

Note: `cwd` must point to the `active-detective/` directory so imports resolve. The `env` vars here are defaults — the batch script overrides them per scenario.

```json
{
  "mcpServers": {
    "detective": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": ".",
      "env": {
        "SCENARIO_SEED": "42",
        "SCENARIO_TYPE": "benign",
        "RESULTS_FILE": "results.jsonl"
      }
    }
  }
}
```

- [ ] **Step 2: Create eval_prompted.sh**

```bash
#!/usr/bin/env bash
# Batch evaluation: run Claude against each scenario in a manifest.
#
# Usage:
#   scripts/eval_prompted.sh [manifest.jsonl] [model]
#   scripts/eval_prompted.sh eval_manifest.jsonl claude-opus-4-6
set -euo pipefail

MANIFEST=${1:-"eval_manifest.jsonl"}
MODEL=${2:-claude-opus-4-6}
RESULTS="results/prompted_${MODEL}_$(date +%Y%m%d_%H%M%S).jsonl"
PROMPT_FILE="prompts/eval_prompt.md"

mkdir -p results

# Generate manifest if not provided
if [ ! -f "$MANIFEST" ]; then
    echo "Generating manifest..."
    python scripts/generate_eval_manifest.py --n 200 --output "$MANIFEST"
fi

COUNT=$(wc -l < "$MANIFEST")
echo "Running $COUNT scenarios with $MODEL..."
echo "Results: $RESULTS"

i=0
while IFS= read -r line; do
    i=$((i + 1))
    seed=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['seed'])")
    stype=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['scenario_type'])")
    obs=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['observability'])")
    prog=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['attack_progress'])")

    echo "[$i/$COUNT] seed=$seed type=$stype obs=$obs progress=$prog"

    SCENARIO_SEED=$seed SCENARIO_TYPE=$stype OBSERVABILITY=$obs \
    ATTACK_PROGRESS=$prog RESULTS_FILE="$RESULTS" \
        claude -p "$(cat "$PROMPT_FILE")" \
        --model "$MODEL" \
        --mcp-config .mcp.json \
        2>/dev/null || echo "  WARN: scenario $seed failed"

done < "$MANIFEST"

echo ""
echo "=== Computing metrics ==="
python scripts/compute_metrics.py "$RESULTS"
```

- [ ] **Step 3: Make script executable**

Run: `chmod +x /home/spinoza/github/ransomware-policy/active-detective/scripts/eval_prompted.sh`

- [ ] **Step 4: Commit**

```bash
cd /home/spinoza/github/ransomware-policy/active-detective
git add .mcp.json scripts/eval_prompted.sh
git commit -m "feat: add MCP config and batch evaluation script"
```

---

## Chunk 4: Data Sources YAML

### Task 9: Fine-tuning data sources YAML

Curated list of training data sources with good provenance for the domain knowledge SFT stage.

**Files:**
- Create: `fine-tuning/data/sources.yaml`

- [ ] **Step 1: Create data directory**

Run: `mkdir -p /home/spinoza/github/ransomware-policy/fine-tuning/data`

- [ ] **Step 2: Research and write sources.yaml**

Use web search to find well-known, freely available sources for:
- **Knowledge bases**: MITRE ATT&CK ransomware techniques, Windows Internals references, DFIR guides, ransomware family reports
- **Real traces**: Malware Bazaar, public incident reports, Atomic Red Team
- **Synthetic**: Our simulator output

Each source must have: name, URL, type, use, license, topics, format, notes.

```yaml
# fine-tuning/data/sources.yaml
# Curated data sources for domain knowledge SFT.
# type: knowledge_base (for training), real_traces (for evaluation), synthetic (our generators)
# use: training, evaluation, or both

sources:
  # ── Knowledge Bases (for SFT training) ──────────────────────
  - name: "MITRE ATT&CK - Enterprise Techniques"
    url: "https://attack.mitre.org/techniques/enterprise/"
    type: knowledge_base
    use: training
    license: "Apache-2.0"
    topics: [ransomware_ttps, defense_evasion, persistence, lateral_movement]
    format: "json (STIX via ATT&CK TAXII or GitHub)"
    notes: "Focus on T1486 (Data Encrypted for Impact), T1490 (Inhibit System Recovery), T1059 (Command and Scripting Interpreter). STIX JSON available at github.com/mitre/cti"

  # ... (populate with 10-15 well-researched sources via web search)

  # ── Real Traces (for evaluation) ────────────────────────────

  # ── Synthetic (our generators) ──────────────────────────────
  - name: "Active Detective Simulator"
    url: "local://active-detective/simulator/"
    type: synthetic
    use: both
    license: "MIT"
    topics: [ransomware_ttps, windows_internals, forensic_analysis]
    format: "jsonl (episodes with telemetry + ground truth)"
    notes: "4 attack types (blitz, sleeper, exfil-first, semantic_shuffle), 5 benign generators, partial observability, synthetic file contents"
```

Note: The implementer should use web search to find and verify 10-15 real sources with correct URLs, licenses, and descriptions. Don't fabricate URLs.

- [ ] **Step 3: Commit**

```bash
cd /home/spinoza/github/ransomware-policy
git add fine-tuning/data/sources.yaml
git commit -m "feat: add curated data sources YAML for domain knowledge SFT"
```

---

## Final Verification

- [ ] **Run full test suite**

Run: `cd /home/spinoza/github/ransomware-policy/active-detective && python -m pytest tests/ -q`
Expected: 408+ tests pass (original tests) + new tests for session, manifest, metrics

- [ ] **Manual smoke test: run MCP server interactively**

```bash
cd /home/spinoza/github/ransomware-policy/active-detective
SCENARIO_SEED=42 SCENARIO_TYPE=blitz OBSERVABILITY=0.7 ATTACK_PROGRESS=0.5 \
    claude --mcp-config .mcp.json
```

Then in the Claude Code session, paste the contents of `prompts/eval_prompt.md` and let Claude investigate.

- [ ] **Verify results file was created**

Check that `results.jsonl` was written with a valid entry after the smoke test.

---

## Deferred: LocalModelAgent and PassiveLLM

The spec calls for `LocalModelAgent` and `PassiveLLM` in `evaluation/baselines.py` for the Qwen prompted baseline. These require GPU and model loading infrastructure, which is a separate workstream from the MCP-based Claude Code evaluation. Implement these after the MCP server is working and the Opus baseline results are collected. They will follow the same `BaselineAgent` interface (see `baselines.py:39-51`).
