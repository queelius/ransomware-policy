# Active Detective Agent for Ransomware Detection — Design Document

**Date:** 2026-03-03
**Status:** Draft — approved for implementation planning

---

## 1. Thesis

A small fine-tuned LLM agent that actively selects which host evidence to inspect outperforms passive classifiers under partial observability, achieving accuracy approaching large prompted models at a fraction of the inference cost.

**Central novelty:** Detection framed as a sequential investigation problem — the agent chooses what to look at next, rather than passively classifying a fixed telemetry window. No prior work combines active evidence-seeking with fine-tuned LLMs for ransomware detection.

## 2. Motivation

Real EDR telemetry is noisy and incomplete. Events are dropped, logs rotate, and not all activity is instrumented. Passive classifiers degrade proportionally with observability. A human SOC analyst compensates by actively investigating — checking specific files, querying process trees, pulling historical context. This project teaches a small LLM to do the same.

**Key evaluation claim:** Under degraded observability (30-50% event visibility), the active agent recovers accuracy that passive methods lose, by strategically choosing which evidence to gather.

## 3. Architecture

### 3.1 Simulated Host Environment

A lightweight file registry (not a full filesystem) tracks files that can be mutated by ransomware:

```
FileRegistry:
  files: Dict[path, FileRecord]
    FileRecord:
      path: str
      size: int
      entropy: float
      extension: str
      modified_at: datetime
      content_type: str  # doc, config, image, executable
      is_encrypted: bool  # ground truth, hidden from agent
```

Background generators produce:
- **Benign activity:** Office edits, browser downloads, backup operations, AV scans
- **Ransomware activity:** Mutations to the file registry (encrypt, rename, exfiltrate)

**Attack scenarios:**
- Blitz encryptor (20+ files/sec, noisy, fast)
- Slow sleeper (1-2 files/min, low signal, extended)
- Exfil-first (data theft phase before encryption)
- Semantic shuffle (content manipulation without encryption indicators)

### 3.2 Telemetry Stream

Every window (120s default), the agent passively receives a telemetry snapshot containing a subset of events that occurred. Events are stochastically dropped to simulate partial observability.

Event types:
- **Process events:** pid, name, parent, command line
- **File metadata events:** path, size delta, entropy delta, extension change
- **Network events:** destination, bytes, protocol, direction

**Observability control:** Each event has a configurable drop probability. The independent variable in evaluation is the observability level (percentage of events that survive to the telemetry window).

### 3.3 Active Detective Agent

The agent operates in a ReAct-style loop (Yao et al. 2023):

```
Input: telemetry_window + accumulated_observations

Loop (up to k steps):
  LLM generates: Thought (reasoning about what's known/unknown)
  LLM generates: Action (from action space)
  Environment returns: Observation (result of action)
  Append observation to context

  If Action == DECIDE:
    Break — emit verdict + explanation
```

**Action space:**

| Action | Cost | Returns |
|---|---|---|
| `inspect_file(path)` | -0.02 | entropy, size, extension, modified_ts, content_type |
| `check_process(pid)` | -0.02 | full command line, parent process, child PIDs |
| `fetch_history(t)` | -0.03 | telemetry window from t steps ago |
| `scan_directory(path)` | -0.05 | list of files with metadata summaries |
| `DECIDE(verdict, explanation)` | 0 | terminal action — ends investigation |

**Verdicts:** ignore, monitor, alert, quarantine, block

**Budget:** k_max = 5 (default). Ablate with k = 1, 3, 5, 10.

**Stopping:** DECIDE is available at every step. Agent is rewarded for stopping early when confident (lower cumulative action cost).

### 3.4 Model

- Base: TinyLlama-1.1B (primary) or Phi-2-2.7B (if GPU allows)
- Fine-tuning: QLoRA (4-bit NF4, r=16, lora_alpha=32)
- Target modules: q_proj, v_proj, k_proj, o_proj + MLP (gate, up, down)
- Max context: 2048 tokens (telemetry window + investigation history must fit)

## 4. Training Pipeline

### Phase 1: Scenario generation
Generate 1000+ episodes with the simulator. Mix:
- 40% benign
- 20% blitz encryptor
- 15% slow sleeper
- 15% exfil-first
- 10% semantic shuffle

Each episode produces a sequence of telemetry windows at various observability levels.

### Phase 2: Expert trace generation (knowledge distillation)
For each scenario, prompt Claude (via Claude Code Max subscription) to produce a ReAct investigation trace:

```
Given this telemetry window and file registry state:
[telemetry JSON]

You are a SOC analyst. Investigate step by step.
Available actions: inspect_file, check_process, fetch_history, scan_directory, DECIDE
You have at most 5 steps. Each inspection has a cost.

Generate your investigation as:
Thought: [reasoning]
Action: [action with arguments]
Observation: [what you'd expect to see]
... repeat ...
Thought: [final reasoning]
Action: DECIDE(verdict, explanation)
```

The simulator fills in actual Observation values based on ground truth. ~5K tokens per trace, ~1000 traces, ~5M tokens total. Feasible on Max subscription.

### Phase 3: Fine-tuning
Fine-tune the small LLM on the expert traces using QLoRA. The model learns the Thought → Action pattern and when to DECIDE.

Training format:
```json
{
  "prompt": "<|telemetry|>\n[window]\n<|investigate|>",
  "completion": "Thought: I see an entropy spike...\nAction: inspect_file(doc_47.docx)\nObservation: {entropy: 7.9, ...}\nThought: Confirmed encryption...\nAction: DECIDE(alert, 'Active file encryption detected')"
}
```

## 5. Evaluation Design

### 5.1 Baselines
1. **Passive LR:** Logistic regression on handcrafted signal features (entropy_delta_mean, rename_rate)
2. **Passive LLM:** Same fine-tuned model, but no inspection actions — decides from telemetry window alone
3. **Large model zero-shot:** Claude/GPT-4 with chain-of-thought prompt on same telemetry (accuracy ceiling, impractical for real-time)
4. **Exhaustive inspector:** Agent that always uses all k steps (no learned stopping)

### 5.2 Independent variables
- **Observability level:** 10%, 30%, 50%, 70%, 90% event visibility
- **Inspection budget k:** 1, 3, 5, 10
- **Attack scenario:** per-scenario and aggregate

### 5.3 Dependent variables
- Detection accuracy: precision, recall, F1 (ransomware vs benign)
- Attack phase identification accuracy (12-class)
- Mean steps used before DECIDE (investigation efficiency)
- Correct verdict rate vs action cost (Pareto frontier)

### 5.4 Generalization tests
- **Unseen scenario:** Train on blitz + sleeper, evaluate on exfil-first
- **Public dataset:** Project RanSMAP or CICIDS telemetry into the JSONL schema, evaluate trained agent
- **Sim-to-real gap:** Acknowledged as limitation; partial real validation with sandboxed VM ransomware runs is a future extension

### 5.5 Key hypotheses to test
- H1: Active agent > passive LLM at low observability (30-50%)
- H2: Active agent approaches large-model accuracy at small-model cost
- H3: Agent learns to stop early on easy cases (efficiency)
- H4: Agent generalizes to unseen attack scenarios better than passive classifier

## 6. Prior Art Positioning

**Closest competing work:**
- LogBERT (Guo et al. 2021): Masked log-key prediction. Similar MLM objective but no active investigation, no attack-phase labels, no action selection.
- LLMLA (Ferrag et al. 2023): LLMs on security logs. Network-flow only, no POMDP framing, no active evidence gathering.

**Differentiation:** No prior work combines (a) fine-tuned small LLM, (b) active evidence-seeking actions, (c) host-behavioral telemetry, (d) partial observability evaluation. Each component exists; the combination and the "active detective" framing are novel.

**Must cite:** LogBERT, LLMLA, ReAct, QLoRA, MITRE ATT&CK, Kaelbling et al. 1998 (POMDP), Murphy 2002 (DBN).

## 7. Deferred to Future Work

- RL policy learning (replace supervised traces with learned policy via reward signal)
- RAG memory layer for long-horizon detection (slow sleeper over hours)
- GNN module for structural signals (process-file-network graph)
- Value-of-information estimation (explicit VoI scoring per action)
- Adversarial self-play (attacker adapts to detector)
- Real-world deployment validation

## 8. Risks

| Risk | Mitigation |
|---|---|
| Simulator bias (synthetic != real) | Cross-validate across scenarios; test on public datasets; acknowledge limitation |
| Expert traces too easy to mimic | Vary trace quality; compare against heuristic baseline |
| Context window overflow | Budget k and telemetry window size to stay under 2048 tokens |
| Small model can't learn ReAct | Start with Phi-2 (2.7B) if TinyLlama (1.1B) fails; this is an empirical question |
| Claude traces may be poor for this domain | Validate trace quality manually on 50 samples before scaling |
