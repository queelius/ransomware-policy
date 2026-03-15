# Ransomware Detection using LLMs

Research project exploring ransomware detection using Large Language Models, with the **Active Detective Agent** as the primary contribution.

## Project Structure

```
ransomware-policy/
├── active-detective/     # Main contribution: RL-trained investigation agent
│   ├── simulator/       # HostState (FileRegistry + ProcessTable), attack/benign generators
│   ├── tools/           # 11 investigation tools + dual-format parser
│   ├── environment/     # RansomwareDetectionEnv, RLVR reward
│   ├── training/        # GRPO via TRL, scenario generation
│   ├── evaluation/      # Metrics, baselines, ablation
│   └── tests/           # 408 tests
│
├── prompting-only/       # Comparison: zero-shot/few-shot/CoT prompting
│   ├── telemetry/       # Telemetry generation (latent states, Atomic Red Team)
│   ├── prompts/         # Detection prompt templates
│   └── evaluation/      # Prompt-based evaluation
│
├── fine-tuning/          # Comparison: QLoRA fine-tuning (TinyLlama, Phi-2, Mistral)
│   ├── scripts/         # Training and evaluation scripts
│   └── data/            # Training datasets
│
└── docs/                 # Research documents and design specs
    ├── plans/           # Active Detective system design
    └── early-research/  # Brainstorming notes and proposals
```

## Three Approaches

### 1. Active Detective Agent (Main Contribution)

An LLM agent trained via GRPO to actively investigate ransomware by deciding which host evidence to examine. Operates in a POMDP: host state is partially hidden, events are stochastically dropped, and the agent must choose which evidence to seek before rendering a verdict.

```bash
cd active-detective

# Run tests
python -m pytest tests/ -q

# Generate training scenarios
python -c "from training.scenarios import generate_training_scenarios, save_scenarios; save_scenarios(generate_training_scenarios(1000), 'scenarios.jsonl')"

# Train (requires GPU + trl + transformers>=5.2.0)
accelerate launch -m training.train_grpo --model Qwen/Qwen3.5-9B --output-dir ./checkpoints --n-episodes 500 --group-size 4
```

See `docs/plans/2026-03-05-active-detective-system-design.md` for the full design specification.

### 2. Prompting-Only (Baseline)

Zero-shot, few-shot, and chain-of-thought prompting against pre-trained LLMs. No training required.

```bash
cd prompting-only
python prompts/detection_prompts.py
```

### 3. Fine-Tuning (Comparison)

QLoRA fine-tuning of small LLMs on synthetic telemetry with expert annotations.

```bash
cd fine-tuning
python scripts/prepare_training_data.py --simple --output train.jsonl
python scripts/finetune_ransomware_llm.py --model tiny --train-data train.jsonl --epochs 3
```

## Key Innovation: Multi-Layer Prediction

The model learns to predict:

1. **Latent State**: What's really happening (hidden)
2. **Observable Events**: What telemetry comes next
3. **Causal Understanding**: Why events occur
4. **Risk Assessment**: Current threat level
5. **Temporal Reasoning**: Time to impact
6. **Recommended Actions**: What to do
7. **Uncertainty**: Confidence and alternatives

## Requirements

```bash
pip install -r fine-tuning/scripts/requirements_finetune.txt
```

For Active Detective training, see `active-detective/` for GPU requirements.

## License

MIT License — see LICENSE file.

## Citation

```bibtex
@software{towell_ransomware_llm_2025,
  title={Ransomware Detection using LLMs},
  author={Towell, Alexander},
  year={2025},
  url={https://github.com/queelius/ransomware-policy}
}
```
