#!/usr/bin/env python3
"""
SFT training script for chat-format domain knowledge data.

Fine-tunes Qwen3-4B or Qwen3-1.7B on chat-format JSONL using QLoRA via
TRL's SFTTrainer.  Targets RTX 3060 12GB VRAM (Qwen3-4B in 4-bit ~2.5GB).

Usage:
    python train_sft.py --model Qwen/Qwen3-4B --data ../data/sft_domain_knowledge_train.jsonl
    python train_sft.py --model Qwen/Qwen3-1.7B \
        --data ../data/sft_domain_knowledge_train.jsonl \
        --eval-data ../data/sft_domain_knowledge_eval.jsonl \
        --epochs 5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_jsonl_messages(path: str) -> Dataset:
    """Load JSONL file and extract the ``messages`` field into a HF Dataset.

    Each line must contain a ``messages`` key with a list of
    ``{"role": ..., "content": ...}`` dicts.  Any other fields (e.g.
    ``metadata``) are ignored.
    """
    records = []
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed JSON at {path}:{lineno}: {exc}")
                continue
            if "messages" not in obj:
                print(f"WARNING: skipping line {lineno} (no 'messages' key)")
                continue
            records.append({"messages": obj["messages"]})

    if not records:
        print(f"ERROR: no valid records found in {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} conversations from {path}")
    return Dataset.from_list(records)


def build_model_and_tokenizer(model_id: str, max_seq_length: int):
    """Load a 4-bit quantized model and its tokenizer."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        model_max_length=max_seq_length,
    )
    # Ensure pad token is set (Qwen models typically have one, but be safe)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Required for gradient checkpointing with quantized models
    model.config.use_cache = False

    return model, tokenizer


def build_lora_config(lora_r: int, lora_alpha: int) -> LoraConfig:
    """Create LoRA config targeting all linear projections in Qwen."""
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3 on chat-format domain knowledge JSONL via QLoRA + SFTTrainer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model ID (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training JSONL (required)",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation JSONL (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints-sft",
        help="Checkpoint output directory (default: ./checkpoints-sft)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train batch size (default: 4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_dataset = load_jsonl_messages(args.data)
    eval_dataset = None
    if args.eval_data:
        eval_dataset = load_jsonl_messages(args.eval_data)

    # ------------------------------------------------------------------
    # 2. Load model + tokenizer with 4-bit quantization
    # ------------------------------------------------------------------
    print(f"Loading model {args.model} with 4-bit NF4 quantization ...")
    model, tokenizer = build_model_and_tokenizer(args.model, args.max_seq_length)

    # ------------------------------------------------------------------
    # 3. LoRA config
    # ------------------------------------------------------------------
    lora_config = build_lora_config(args.lora_r, args.lora_alpha)

    # ------------------------------------------------------------------
    # 4. SFTTrainer configuration
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_eval = eval_dataset is not None

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if has_eval else "no",
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        report_to="none",
        optim="paged_adamw_8bit",
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print("Starting SFT training ...")
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Save model + tokenizer
    # ------------------------------------------------------------------
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Model and tokenizer saved to {final_dir}")


if __name__ == "__main__":
    main()
