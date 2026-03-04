# Active Detective Agent for Ransomware Detection — Design Document

**Date:** 2026-03-03 (revised 2026-03-04)
**Status:** Draft — approved for implementation planning (revised: RL-first, RLVR, tool-call-as-text)

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

The agent operates in a ReAct-style loop (Yao et al. 2023). Critically, **tools are invoked via text generation** — the model emits tool calls as structured tokens, and the environment executes them and injects results back into the context. No separate policy head is needed; the action space is the model's token vocabulary constrained to valid tool-call syntax.

This follows the same paradigm as DeepSeek R1 and Qwen3's native tool-use training. Qwen3 already understands tool-call formatting, so the base model starts with strong tool-use priors.

```
Input: system_prompt + telemetry_window

Loop (up to k steps):
  LLM generates:  <think>reasoning about what's known/unknown</think>
  LLM generates:  <tool_call>tool_name(args)</tool_call>
  Environment:    <tool_result>{structured JSON response}</tool_result>
  (result injected back into context)

  If tool_call == DECIDE:
    Break — emit verdict + explanation
```

**Tool set:**

| Tool | Cost | Returns |
|---|---|---|
| `inspect_file(path)` | -0.02 | entropy, size, extension, modified_ts, content_type |
| `check_process(pid)` | -0.02 | full command line, parent process, child PIDs |
| `scan_directory(path)` | -0.05 | list of files with metadata summaries |
| `recall_memory(query)` | -0.03 | top-k relevant historical telemetry windows (RAG retrieval) |
| `DECIDE(verdict, explanation)` | 0 | terminal action — ends investigation |

**Note on `recall_memory`:** Rather than hardcoding a RAG layer, memory retrieval is exposed as a tool the agent can learn to use. The RL training discovers *when* historical context is worth the cost. Backed by a simple embedding-similarity store over past telemetry windows.

**Verdicts:** ignore, monitor, alert, quarantine, block

**Budget:** k_max = 5 (default). Ablate with k = 1, 3, 5, 10.

**Stopping:** DECIDE is available at every step. Agent is rewarded for stopping early when confident (lower cumulative action cost).

**Implementation note:** During training, tool calls are parsed from the model's text output via string matching (or Qwen3's native tool-call format). During deployment, the same tools could be exposed as MCP servers for integration with real EDR systems, but for the paper the simulator handles tool execution directly.

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

## 4. Training Pipeline (RLVR — Reinforcement Learning with Verifiable Rewards)

**Decision:** Start with RL rather than supervised distillation. The agent learning its own investigation policy IS the contribution — mimicking Claude traces would make it a distillation paper.

**Paradigm:** RLVR (Reinforcement Learning with Verifiable Rewards), following the DeepSeek R1 approach. The simulator provides ground truth, so rewards are verifiable — no reward model needed. The model generates tool calls as text, the environment executes them, and the final verdict is checked against ground truth for a binary correct/incorrect signal.

### Phase 1: Scenario generation
Generate 1000+ episodes with the simulator. Mix:
- 40% benign
- 20% blitz encryptor
- 15% slow sleeper
- 15% exfil-first
- 10% semantic shuffle

Each episode produces a sequence of telemetry windows at various observability levels. Additionally, build a simple embedding-similarity memory store over past windows for the `recall_memory` tool.

### Phase 2: Environment (tool-execution loop)

The environment is a tool-execution harness that wraps the simulator. It is NOT a traditional Gymnasium env with discrete action spaces — instead, it operates on the model's text output, parsing tool calls and injecting tool results.

```
RansomwareDetectionEnv:

  reset():
    - sample a scenario (benign or attack variant)
    - generate telemetry window with configured observability
    - populate file registry and process table
    - initialize memory store with historical windows
    - return system_prompt + telemetry_window as text

  execute_rollout(model):
    - model generates text (thinking + tool calls)
    - on each <tool_call>:
        parse tool name and arguments
        execute against simulator state (file registry, process table, memory store)
        inject <tool_result> into context
        accumulate action cost
    - on <tool_call>DECIDE(verdict, explanation)</tool_call>:
        check verdict against ground truth
        compute final reward
        return rollout text + reward

  reward(verdict, ground_truth, costs):
    - correct verdict:    +1.0
    - wrong verdict:      -1.0
    - false negative:     -2.0 (missed ransomware is worse than false alarm)
    - action costs:       sum of per-tool costs (see tool table)
    - efficiency bonus:   +0.01 per unused budget step
    - format reward:      +0.1 if output follows expected tool-call format
```

This design follows the RLVR pattern: the environment is a verifier, not a reward model. The reward function is deterministic and rule-based.

### Phase 3: RL training (GRPO)

**Algorithm:** GRPO (Group Relative Policy Optimization) via TRL's GRPOTrainer + Unsloth for efficiency.

GRPO is preferred over PPO because:
- No separate value model needed (saves ~50% VRAM)
- Qwen3-8B was natively trained with GRPO — compatible by design
- More stable than PPO for LLM RL (used by DeepSeek R1)
- TRL's PPOTrainer is being deprecated (removed in TRL 0.29.0)

**GRPO training loop:**
1. For each batch, sample N scenarios from the simulator
2. For each scenario, generate G rollouts (group size, e.g., G=4) from the current policy
3. Each rollout: model generates thinking + tool calls → environment executes → reward computed
4. Rank rollouts within each group by reward
5. Update policy toward higher-reward rollouts using GRPO's relative advantage

**Key design choices:**
- Freeze base model weights; only train LoRA adapters with RL
- Start with small tool set (`inspect_file` + `DECIDE` only), expand incrementally to full set
- Warm-start from a short supervised fine-tune on ~50 hand-written traces (optional, for stability)
- Use Unsloth for 2x training speedup and reduced VRAM
- Group size G=4 (each scenario gets 4 rollouts for relative ranking)

### Phase 3b: Supervised baseline (for comparison)
Generate ~500 expert traces via Claude (knowledge distillation) and fine-tune a separate model. This becomes a baseline in the evaluation: "RL-learned policy vs. distilled policy vs. passive classifier."

### Phase 4: Validation on real ransomware (optional, strengthens paper)
Run 10-20 real ransomware samples in a sandboxed VM, capture Sysmon logs, project into JSONL schema, and evaluate the trained agent. Train on synthetic, validate on real — best of both worlds.

### Training format (GRPO rollout)
```
<|system|>
You are a ransomware detection agent. You will receive host telemetry
and can use tools to investigate. When ready, call DECIDE with your verdict.
Available tools: inspect_file, check_process, scan_directory, recall_memory, DECIDE

<|telemetry|>
[window events — partial observability applied]

<think>I see 3 file metadata events with entropy increases on .docx files
in C:/Users/A/Documents/. This could be encryption. Let me inspect one.</think>
<tool_call>inspect_file("C:/Users/A/Documents/report_12.docx")</tool_call>
<tool_result>{"entropy": 7.92, "size": 2048, "extension": ".docx.locked",
"modified_ts": "2s ago", "content_type": "doc"}</tool_result>

<think>Entropy 7.92 and extension changed to .locked confirms encryption.
Let me check if this pattern extends to other directories.</think>
<tool_call>recall_memory("entropy spike file encryption")</tool_call>
<tool_result>{"matches": [{"window": "t-2", "summary": "3 files in C:/Users/A/Desktop
showed entropy increase from 3.1 to 7.8"}]}</tool_result>

<think>Historical pattern confirms this is not isolated. Active encryption
across multiple directories. High confidence — deciding now.</think>
<tool_call>DECIDE("quarantine", "Active ransomware encryption detected across
multiple directories. File entropy >7.8 with .locked extension changes.
Historical pattern confirms spread over last 2 windows.")</tool_call>

Reward: +0.93 (correct: +1.0, 2 tools: -0.05, 2 steps unused: +0.02, format: -0.04)
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
- DeepSeek R1 (DeepSeek 2025): RLVR paradigm — RL with verifiable rewards, no reward model. We adopt this training paradigm but apply it to a domain-specific tool-use agent rather than general reasoning.
- Qwen3 (Qwen Team 2025): Native GRPO training with tool-call generation. We build on Qwen3's tool-use priors rather than training tool-call formatting from scratch.

**Differentiation:** No prior work combines (a) RL-trained LLM agent, (b) active evidence-seeking via tool calls, (c) host-behavioral telemetry, (d) partial observability evaluation, (e) verifiable rewards from a simulator. Each component exists; the combination and the "active detective" framing are novel.

**Must cite:** LogBERT, LLMLA, DeepSeek R1, GRPO, ReAct (Yao et al. 2023), QLoRA, MITRE ATT&CK, Kaelbling et al. 1998 (POMDP), Murphy 2002 (DBN), Qwen3.

## 7. Deferred to Future Work

- Model distillation: compress RL-trained 8B agent to 1.7B-3B for edge deployment
- GNN module for structural signals (process-file-network graph)
- Value-of-information estimation (explicit VoI scoring per action)
- Adversarial self-play (attacker adapts to detector)
- Real-world deployment validation
- Multi-host correlation (agent investigates across hosts)
- MCP server integration for real EDR tool execution (production deployment)

## 8. Risks

| Risk | Mitigation |
|---|---|
| Simulator bias (synthetic != real) | Cross-validate across scenarios; test on public datasets; optional Phase 4 real ransomware validation; acknowledge limitation |
| RL training instability | Start with small action space (inspect_file + DECIDE), expand incrementally; optional warm-start from ~50 hand-written traces; GRPO is more stable than PPO for LLM RL |
| Reward hacking | Monitor for degenerate policies (e.g., always DECIDE immediately, or always exhaust budget); add diversity bonus if needed |
| Sparse reward signal | Asymmetric rewards (false negative penalized harder); small per-step cost provides intermediate signal; format reward for well-formed tool calls |
| Tool-call parsing failures | Qwen3 has native tool-call format priors; format reward (+0.1) incentivizes correct syntax; fallback to string matching if structured parsing fails |
| Context window overflow | Budget k and telemetry window size to stay under 4096 tokens; Qwen3 supports 32K natively |
| Distilled model too weak | Ablate across sizes (1.7B, 3B, 8B); report the accuracy-size Pareto frontier honestly |
| GRPO + QLoRA compute cost | LoRA keeps trainable params small; GRPO avoids separate value model; Unsloth for 2x speedup; A100 on Vast.ai ~$0.50/hr |
