# Active Detective Agent for Ransomware Detection — Design Document

**Date:** 2026-03-03 (revised 2026-03-04)
**Status:** Draft — approved for implementation planning (revised: RL-first)

---

## 1. Thesis

An LLM agent trained via reinforcement learning to actively select which host evidence to inspect outperforms passive classifiers under partial observability. The learned investigation policy distills to small models (1.7B-3B) for real-time edge deployment.

**Central novelty:** Detection framed as a sequential investigation problem — the agent chooses what to look at next, rather than passively classifying a fixed telemetry window. No prior work combines active evidence-seeking with RL-trained LLMs for ransomware detection.

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

**Strategy:** Start with a capable base model to prove the concept, then distill down to smaller models for edge deployment.

**Primary (proof of concept):**
- Base: **Qwen3-8B** — trained with GRPO natively, best-in-class tool use and structured action generation
- Fine-tuning: QLoRA (4-bit NF4, r=16, lora_alpha=32) or full LoRA via Unsloth
- Target modules: q_proj, v_proj, k_proj, o_proj + MLP (gate, up, down)
- Max context: 4096 tokens (Qwen3 supports 32K, but budget for training efficiency)
- Compute: A100 80GB on Vast.ai (~$0.50/hr)

**Distillation targets (post proof-of-concept):**
- Qwen3-1.7B — smallest Qwen3 variant, tests minimum viable size
- Qwen2.5-3B — well-studied for GRPO at smaller scale
- TinyLlama-1.1B — original target, tests extreme compression

**Distillation method:** Train the large model with RL first, then use its rollouts as supervised training data for the smaller models. This produces a natural ablation: accuracy vs. model size curve.

**Paper story:** "We train an 8B model via GRPO that learns an effective investigation policy. We then distill to 1.7B-3B models and show accuracy degrades gracefully, enabling real-time edge deployment."

**Alternatives considered:**
- Qwen2.5-7B-Instruct: most-studied GRPO baseline in literature, good fallback
- Llama-3.1-8B-Instruct: strong general reasoning, large community
- DeepSeek-R1-distill-Qwen-7B: already RL-trained, may converge faster

## 4. Training Pipeline (RL-First)

**Decision:** Start with RL rather than supervised distillation. The agent learning its own investigation policy IS the contribution — mimicking Claude traces would make it a distillation paper.

### Phase 1: Scenario generation
Generate 1000+ episodes with the simulator. Mix:
- 40% benign
- 20% blitz encryptor
- 15% slow sleeper
- 15% exfil-first
- 10% semantic shuffle

Each episode produces a sequence of telemetry windows at various observability levels.

### Phase 2: Environment (Gymnasium interface)

The environment exposes a standard `reset() → step(action) → (obs, reward, done, info)` interface.

```
RansomwareDetectionEnv(gymnasium.Env):
  observation_space: text (tokenized telemetry + accumulated observations)
  action_space: Discrete(N) or structured (action_type + arguments)

  reset():
    - sample a scenario (benign or attack variant)
    - generate telemetry window with configured observability
    - return initial observation (telemetry text)

  step(action):
    - if action == DECIDE(verdict):
        reward = verdict_reward(verdict, ground_truth) - cumulative_cost
        done = True
    - else:
        observation = execute_inspection(action, environment_state)
        reward = -action_cost
        done = (step_count >= k_max)
        append observation to context
    - return (observation, reward, done, info)
```

### Phase 3: RL training

**Algorithm:** GRPO (Group Relative Policy Optimization) via TRL's GRPOTrainer + Unsloth for efficiency.

GRPO is preferred over PPO because:
- No separate value model needed (saves ~50% VRAM)
- Qwen3-8B was natively trained with GRPO — compatible by design
- More stable than PPO for LLM RL (used by DeepSeek R1)
- TRL's PPOTrainer is being deprecated (removed in TRL 0.29.0)

**Reward structure:**
- Correct verdict: +1.0
- Wrong verdict: -1.0
- Wrong verdict on ransomware (false negative): -2.0 (asymmetric — missed ransomware is worse)
- Each inspection action: -0.02 to -0.05 (per action cost table)
- Unused budget bonus: +0.01 per remaining step (rewards efficiency)

**Key design choices:**
- Freeze base model weights; only train LoRA adapters with RL
- Start with small action space (inspect_file + DECIDE only), expand incrementally
- Warm-start from a short supervised fine-tune on ~50 hand-written traces (optional, for stability)
- Use Unsloth for 2x training speedup and reduced VRAM

### Phase 3b: Supervised baseline (for comparison)
Generate ~500 expert traces via Claude (knowledge distillation) and fine-tune a separate model. This becomes a baseline in the evaluation: "RL-learned policy vs. distilled policy vs. passive classifier."

### Training format (RL rollout)
```
<|telemetry|>
[window events]
<|investigate|>
Thought: [LLM generates freely]
Action: inspect_file(doc_47.docx)
<|observation|>
{entropy: 7.9, size: 2048, ext: .locked, modified: 2s ago}
Thought: [LLM generates freely]
Action: DECIDE(alert, "Active file encryption detected")
<|reward|> +0.96 (correct verdict +1.0, one inspection -0.02, 3 steps unused +0.02)
```

## 5. Evaluation Design

### 5.1 Baselines
1. **Passive LR:** Logistic regression on handcrafted signal features (entropy_delta_mean, rename_rate)
2. **Passive LLM:** Same fine-tuned model, but no inspection actions — decides from telemetry window alone
3. **Large model zero-shot:** Claude/GPT-4 with chain-of-thought prompt on same telemetry (accuracy ceiling, impractical for real-time)
4. **Distilled agent:** Same architecture, but trained via supervised distillation on Claude-generated traces (no RL)
5. **Exhaustive inspector:** Agent that always uses all k steps with random/heuristic inspection (no learned stopping)
6. **Random agent:** Randomly selects inspection actions, decides randomly (sanity check)

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
- H2: RL-trained agent > distilled agent (RL learns a better policy than imitation)
- H3: Agent learns to stop early on easy cases (efficiency)
- H4: Agent generalizes to unseen attack scenarios better than passive classifier
- H5: Distilled small model (1.7B-3B) retains most of the 8B agent's accuracy

## 6. Prior Art Positioning

**Closest competing work:**
- LogBERT (Guo et al. 2021): Masked log-key prediction. Similar MLM objective but no active investigation, no attack-phase labels, no action selection.
- LLMLA (Ferrag et al. 2023): LLMs on security logs. Network-flow only, no POMDP framing, no active evidence gathering.

**Differentiation:** No prior work combines (a) fine-tuned small LLM, (b) active evidence-seeking actions, (c) host-behavioral telemetry, (d) partial observability evaluation. Each component exists; the combination and the "active detective" framing are novel.

**Must cite:** LogBERT, LLMLA, ReAct, QLoRA, MITRE ATT&CK, Kaelbling et al. 1998 (POMDP), Murphy 2002 (DBN).

## 7. Deferred to Future Work

- Model distillation: compress RL-trained 8B agent to 1.7B-3B for edge deployment
- RAG memory layer for long-horizon detection (slow sleeper over hours)
- GNN module for structural signals (process-file-network graph)
- Value-of-information estimation (explicit VoI scoring per action)
- Adversarial self-play (attacker adapts to detector)
- Real-world deployment validation
- Multi-host correlation (agent investigates across hosts)

## 8. Risks

| Risk | Mitigation |
|---|---|
| Simulator bias (synthetic != real) | Cross-validate across scenarios; test on public datasets; acknowledge limitation |
| RL training instability | Start with small action space (inspect_file + DECIDE), expand incrementally; try GRPO if PPO fails; optional warm-start from ~50 hand-written traces |
| Reward hacking | Monitor for degenerate policies (e.g., always DECIDE immediately, or always exhaust budget); add diversity bonus if needed |
| Sparse reward signal | Asymmetric rewards (false negative penalized harder); small per-step cost provides intermediate signal; consider reward shaping with intermediate detection confidence |
| Context window overflow | Budget k and telemetry window size to stay under 2048 tokens |
| Distilled model too weak | Ablate across sizes (1.7B, 3B, 8B); report the accuracy-size Pareto frontier honestly |
| GRPO + QLoRA compute cost | LoRA keeps trainable params small; GRPO avoids separate value model; Unsloth for 2x speedup; A100 on Vast.ai ~$0.50/hr |
