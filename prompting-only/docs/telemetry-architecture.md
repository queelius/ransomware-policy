# Telemetry Architecture for Ransomware Detection

## Overview

This document describes our comprehensive telemetry system designed for training LLMs to detect and explain ransomware attacks through partial observations.

## Core Concept: Partial Observability

The fundamental insight is that ransomware detection is a **partially observable Markov decision process (POMDP)**:

- **Latent State (X)**: The true system state (benign, compromised, under attack)
- **Observable Telemetry (Z)**: What we can actually see (process events, file operations, network traffic)
- **Emission Probability P(Z|X)**: Not all malicious activity is observable

## Data Structure

### 1. Observable Telemetry

What the detection system can actually see:

```json
{
  "timestamp": "2025-09-26T16:00:00",
  "event_type": "process",
  "details": {
    "process_name": "powershell.exe",
    "command_line": "powershell.exe -enc SGVsbG8gV29ybGQ=",
    "parent_process": "winword.exe",
    "pid": 1234
  }
}
```

### 2. Latent Context (Ground Truth)

The hidden state used for training:

```json
{
  "true_state": "initial_compromise",
  "attacker_intent": "Establish initial foothold",
  "attack_phase": "Initial Access",
  "threat_actor": "LockBit affiliate",
  "confidence": 0.85,
  "next_likely_action": "System reconnaissance",
  "time_to_impact": 30,
  "kill_chain_position": "early"
}
```

### 3. Expert Annotations

What we want the LLM to learn to generate:

```json
{
  "plain_english_summary": "Suspicious PowerShell execution from Office application detected. Likely initial compromise vector.",
  "risk_assessment": "high",
  "key_indicators": [
    "Encoded PowerShell execution",
    "Suspicious parent-child process relationship"
  ],
  "recommended_actions": [
    "Isolate the affected system",
    "Check for persistence mechanisms",
    "Review email logs for phishing"
  ],
  "qa_pairs": [
    {
      "question": "What stage of attack is this?",
      "answer": "Initial compromise phase. The attacker has just gained access."
    }
  ]
}
```

## Training Approaches

### 1. Prompt-Only (Baseline)

Zero-shot and few-shot prompting without fine-tuning:

```python
prompt = """
Analyze this telemetry:
[telemetry events]

Is this ransomware? Explain your reasoning.
"""
```

### 2. Fine-Tuning for Prediction

Train the LLM to predict both:
- **Next telemetry events** (learning patterns)
- **Expert explanations** (learning to reason)

Training format:
```
Input: [3 windows of telemetry]
Output: [Next window events + expert analysis]
```

### 3. Policy Learning (Future)

Train a lightweight policy head for actions:
```python
class PolicyHead(nn.Module):
    def forward(self, llm_embeddings):
        return action_probabilities  # [ignore, monitor, alert, block]
```

## Key Features of Our Telemetry

### Realistic Attack Progression

Attacks unfold over realistic timeframes:
- **Initial Access**: 0-5 minutes
- **Discovery**: 5-10 minutes
- **Defense Evasion**: 10-15 minutes
- **Exfiltration**: 15-30 minutes
- **Encryption**: 30+ minutes

### Partial Observability

Not everything is visible:
```python
if state == LatentState.RECONNAISSANCE:
    # Only 60% chance of observing recon
    if random() < 0.6:
        emit_telemetry()
```

### Benign Noise

Normal activity mixed with attacks:
- Browser activity
- Office applications
- System updates
- User file access

### Family-Specific Behaviors

Different ransomware behaves differently:
- **LockBit**: Fast encryption (100-200 files/sec)
- **BlackCat**: Uses Rust, kills VMs
- **Conti**: Human-operated, uses Cobalt Strike
- **REvil**: Safe mode encryption

## Training Data Generation

### Comprehensive Generator
```python
generator = ComprehensiveTelemetryGenerator()
sequence = generator.generate_telemetry_sequence(
    num_windows=10,
    scenario="ransomware_attack"
)
```

### Fine-Tuning Format
```python
formatter = FineTuningFormatter()
training_example = formatter.create_training_example(
    telemetry_windows=sequence,
    context_windows=3  # See 3, predict 1
)
```

## Quality Metrics

Our telemetry is validated for:
1. **Realistic event rates** (not too many/few events)
2. **Proper indicators** (shadow deletion, encryption, C2)
3. **Attack variety** (multiple ransomware families)
4. **Temporal accuracy** (realistic timeframes)
5. **Balanced dataset** (mix of benign and malicious)

## Usage Examples

### Generate Training Data
```bash
# Generate realistic telemetry
python atomic_red_team_integration.py --output telemetry.jsonl --campaigns 1000

# Create fine-tuning dataset
python finetune_format.py --output finetune.jsonl --sequences 100
```

### Analyze Quality
```bash
python analyze_telemetry.py telemetry.jsonl
```

## Why This Approach Works

1. **Learns from Partial Observations**: The LLM learns to infer hidden states from limited telemetry
2. **Predicts and Explains**: By predicting both events and explanations, the model learns deep patterns
3. **Incorporates Expert Knowledge**: Training includes expert annotations, teaching reasoning
4. **Realistic Scenarios**: Based on real ransomware behaviors, not toy examples
5. **Progressive Complexity**: Start with prompting, then fine-tuning, then policy learning

## Next Steps

1. **Collect Real Telemetry**: Run Atomic Red Team tests in sandbox
2. **Fine-Tune Small LLM**: Start with 1-3B parameter models
3. **Evaluate Detection**: Test on unseen ransomware families
4. **Add Memory/RAG**: Handle long-term attack patterns
5. **Policy Head**: Learn optimal response actions