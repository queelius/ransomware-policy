"""finetune_llm.py - LoRA fine-tuning on a 12 GB GPU
=====================================================
Fine-tunes a small open-source LLM (<= 3B params) on the synthetic
telemetry JSONL file produced by `telemetry_sim.py` **or** the k-stack
builder.

*   QLoRA 4-bit quantisation keeps VRAM use under 12 GB.
*   Prompts are generated on-the-fly; tokenisation maps to
    `input_ids`, `attention_mask`, and `labels`.
*   Objective: causal-LM next-token prediction (labels = input_ids).

Usage example
-------------
```bash
pip install transformers==4.41 peft bitsandbytes datasets accelerate icetk
python finetune_llm.py \
       --data dataset.jsonl \
       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --output ft_lora_out \
       --epochs 3 --batch 2
```
Output directory contains LoRA adapters + tokenizer; merge with base
model for inference or keep in PEFT form.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSONL dataset")
    ap.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF model ID (≤3 B params recommended)",
    )
    ap.add_argument("--output", default="ft_lora_out", help="Output dir")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2, help="per-GPU batch")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=1024)
    return ap.parse_args()

# ────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ────────────────────────────────────────────────────────────────────────────

def build_prompt(example: dict[str, any]) -> dict[str, str]:
    """Convert raw telemetry dict → human-readable prompt string."""

    def pretty_block(title: str, lst):
        if not lst:
            return f"{title}: none"
        lines = [f"{title}:"]
        for item in lst:
            lines.append("  " + json.dumps(item))
        return "\n".join(lines)

    context = "\n".join(
        [
            f"EPISODE: {example['episode_id']}  START: {example['window_start']}",
            pretty_block("PROCESS_TREE", example["process_tree"]),
            pretty_block("FILE_EVENTS", example["file_events"]),
            pretty_block("NET_EVENTS", example["net_events"]),
            f"SIGNAL_FEATS: {json.dumps(example['signal_feats'])}",
        ]
    )

    qa_block = ""
    if "qa" in example:
        qa_block = "\n".join([f"Q: {qa['q']}\nA: {qa['a']}" for qa in example["qa"]])

    label = example["label"]  # benign | ransomware

    prompt = (
        "<|system|>You are an intelligent EDR assistant.\n"
        f"<|telemetry|>\n{context}\n"
        f"{qa_block}\n"
        "<|task|>What action should be taken (ignore/alert/quarantine) and why?\n"
        f"<|answer|> Action: {label}\nExplanation:"
    )
    return {"text": prompt}


def tokenize_fn(tokenizer, max_len: int):
    def _inner(batch):
        out = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return _inner

# ────────────────────────────────────────────────────────────────────────────
# Main fine-tune
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1  dataset
    raw_ds = load_dataset("json", data_files=args.data, split="train")
    prompt_ds = raw_ds.map(build_prompt, remove_columns=raw_ds.column_names)

    # 2  tokeniser & 4-bit model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3  LoRA config
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

    # 4  Tokenise dataset
    ds = prompt_ds.map(
        tokenize_fn(tokenizer, args.max_len),
        batched=True,
        remove_columns=["text"],
    )

    # 5  training args
    steps_per_device = 16 // args.batch
    train_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=steps_per_device,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        remove_unused_columns=False,
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
