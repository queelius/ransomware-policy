# Prompting-Only Approach (No Training Required)

This directory contains the **baseline approach** using zero-shot and few-shot prompting without any fine-tuning.

## What This Is

A collection of prompt engineering strategies to detect ransomware using pre-trained LLMs. No training required - just clever prompting!

## Directory Structure

```
prompting-only/
├── telemetry/              # Telemetry generation and datasets
│   ├── advanced_telemetry_gen.py     # Partial observability simulation
│   ├── atomic_red_team_integration.py # Real attack patterns
│   ├── comprehensive_telemetry_gen.py # Full pipeline with latent states
│   └── dataset_parser.py             # Parse public datasets
│
├── prompts/                # Detection strategies
│   └── detection_prompts.py          # Various prompting approaches
│
├── evaluation/             # Evaluation without training
│   └── evaluate_prompts.py           # Test prompt effectiveness
│
└── docs/                   # Documentation
    ├── telemetry-architecture.md     # System design
    └── prediction-breakdown.md        # What we're predicting
```

## Prompting Strategies

### 1. Zero-Shot Detection
```python
prompt = """
Analyze this telemetry for ransomware:
[telemetry events]
Is this ransomware? Explain your reasoning.
"""
```

### 2. Few-Shot with Examples
```python
prompt = """
Example 1: powershell -enc → ransomware
Example 2: chrome.exe → benign

Current: vssadmin delete shadows → ?
"""
```

### 3. Chain-of-Thought
```python
prompt = """
Let's analyze step by step:
1. Check processes...
2. Check file operations...
3. Check network...
Based on this analysis: [decision]
"""
```

### 4. Behavioral Pattern Matching
```python
prompt = """
Known ransomware behaviors:
- Shadow deletion
- Mass encryption
- C2 communication

Current telemetry: [events]
Which behaviors are present?
"""
```

## Quick Start

### Generate Test Data
```bash
# Simple synthetic data
python telemetry/advanced_telemetry_gen.py --output test.jsonl --windows 10

# Realistic attack patterns
python telemetry/atomic_red_team_integration.py --output realistic.jsonl --campaigns 10
```

### Test Detection
```bash
# Test different prompting strategies
python prompts/detection_prompts.py --input test.jsonl --strategy zero_shot
python prompts/detection_prompts.py --input test.jsonl --strategy few_shot
python prompts/detection_prompts.py --input test.jsonl --strategy chain_of_thought
```

## Advantages of Prompting-Only

1. **No Training Required**: Works immediately with any LLM
2. **No GPU Needed**: Just API calls or CPU inference
3. **Interpretable**: Can see exactly what the prompt says
4. **Flexible**: Easy to add new detection rules
5. **Good Baseline**: Establishes performance floor

## Limitations

1. **General Knowledge Only**: Can't learn specific patterns
2. **Context Length**: Limited telemetry window size
3. **Inference Cost**: Each detection requires full LLM call
4. **Consistency**: Results may vary between calls
5. **No Adaptation**: Can't improve from feedback

## When to Use This Approach

- **Quick Prototyping**: Test ideas without training
- **Baseline Comparison**: Measure improvement from fine-tuning
- **Limited Resources**: No GPU available
- **Interpretability Required**: Need explainable decisions
- **Real-time Detection**: Can't wait for training

## Example Results

```
Input: vssadmin delete shadows
Zero-shot: "Possibly ransomware" (60% confidence)
Few-shot: "Ransomware - shadow deletion" (85% confidence)
Chain-of-thought: "Ransomware - critical indicator" (95% confidence)
```

## Next Steps

If prompting-only doesn't provide sufficient accuracy:
1. Move to **fine-tuning** (../fine-tuning/) for better pattern recognition
2. Consider **RL approach** (../rl-policy/) for optimal action selection

## Key Files

- `detection_prompts.py`: Main detection interface
- `atomic_red_team_integration.py`: Realistic attack generation
- `comprehensive_telemetry_gen.py`: Full telemetry with hidden states