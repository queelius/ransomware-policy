"""finetune_llm.py

Fine‑tunes a small open‑source LLM (TinyLlama‑1.1B by default) on the
JSONL dataset produced by `telemetry_sim.py` or the k‑stack builder.

Requirements:
  pip install transformers==4.41.0 peft bitsandbytes datasets accelerate icetk
  # (icetk only for TinyLlama; remove if using phi‑2 / gemma‑2b)

Assumes a single NVIDIA GPU with ~12 GB VRAM.  Uses 4‑bit QLoRA + small batch.

Usage:
  python finetune_llm.py \
      --data path/to/dataset.jsonl \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

Outputs LoRA adapters in ./ft_lora_out/  — merge with base model for inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL telemetry dataset")
    ap.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF model ID (<=3B params recommended)",
    )
    ap.add_argument("--output", default="ft_lora_out", help="Output dir")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    return ap.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Dataset utils
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl_dataset(path: str):
    # quick wrapper — HF will treat entire JSON per line as one field
    return load_dataset("json", data_files=path, split="train")

def build_prompt(example):
    """
    Convert raw telemetry dict into a text prompt + answer.

    The implicit contract is:
      - example has process_tree, file_events, net_events, signal_feats
      - example['qa'] is list[{q,a}]  (optional)
      - example['label']  is 'benign' | 'ransomware'
    """
    def pretty_block(title, lst):
        if not lst:
            return f"{title}: none"
        lines = [f"{title}:"]
        for item in lst:
            lines.append("  " + json.dumps(item))
        return "\n".join(lines)

    context_sections = [
        f"EPISODE: {example['episode_id']}   START: {example['window_start']}",
        pretty_block("PROCESS_TREE", example["process_tree"]),
        pretty_block("FILE_EVENTS",  example["file_events"]),
        pretty_block("NET_EVENTS",   example["net_events"]),
        f"SIGNAL_FEATS: {json.dumps(example['signal_feats'])}",
    ]
    context = "\n".join(context_sections)

    qa_block = ""
    if "qa" in example:
        qa_block = "\n".join([f"Q: {qa['q']}\nA: {qa['a']}"
                              for qa in example["qa"]])

    label = example["label"]

    prompt = (
        "<|system|>You are an intelligent EDR assistant.\n"
        f"<|telemetry|>\n{context}\n"
        f"{qa_block}\n"
        "<|task|>What action should be taken (ignore/alert/quarantine) and why?\n"
        f"<|answer|> Action: {label}\nExplanation:"
    )
    return {"text": prompt}

# ──────────────────────────────────────────────────────────────────────────────
# Main fine‑tune routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. dataset
    raw_ds = load_jsonl_dataset(args.data)
    ds = raw_ds.map(build_prompt, remove_columns=raw_ds.column_names)

    # 2. tokenizer & model (4‑bit quantised)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. LoRA config (target q_proj/v_proj for llama‑based, adjust for others)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 4. data collator (standard LM)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 5. training args
    steps_per_device = 2  # adjust if OOM
    args_train = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=max(1, 16 // args.batch),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=False,
        fp16=True,
        logging_steps=20,
        save_total_limit=2,
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args_train,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

if __name__ == "__main__":
    main()
