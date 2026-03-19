"""Evaluate a local SFT model on ransomware detection scenarios.

Loads a fine-tuned model (base + LoRA adapter), shows it telemetry,
and asks for a verdict. No tool use — tests pure domain knowledge.

Usage:
    PYTHONPATH=. python scripts/run_eval_local.py \
        --model Qwen/Qwen3-1.7B \
        --adapter ../fine-tuning/checkpoints-sft-1.7b/final \
        --n 200 --seed 42
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, ".")
from mcp_session import DetectiveSession
from scripts.generate_eval_manifest import generate_manifest

SYSTEM_PROMPT = """You are a ransomware forensic analyst. Analyze the telemetry below and classify the host.

Verdicts (pick exactly one):
- ignore: No indicators of compromise
- monitor: Suspicious but inconclusive
- alert: Likely malicious, needs review
- quarantine: High confidence ransomware
- block: Active encryption in progress

Key indicators:
- File extension changes to .locked/.encrypted/.enc/.cry = ransomware
- File entropy > 7.5 bits/byte = encrypted content
- Windows Defender disabled (Event ID 5001, DisableAntiSpyware registry) = defense evasion
- Mass file modifications across directories = encryption sweep
- Large outbound transfers to unusual IPs = data exfiltration
- VSS/shadow copy deletion = inhibit recovery

Respond with ONLY a JSON object: {"verdict": "...", "explanation": "..."}"""


def load_model(model_name: str, adapter_path: str | None = None):
    """Load base model with optional LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_verdict(model, tokenizer, telemetry: str, max_new_tokens: int = 256) -> dict:
    """Generate a verdict from the model given telemetry."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Telemetry:\n{telemetry}\n\nProvide your verdict as JSON."},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Parse verdict from response
    verdict = "ignore"
    explanation = response.strip()

    # Try JSON parse first
    json_match = re.search(r'\{[^}]*"verdict"\s*:\s*"([^"]+)"[^}]*\}', response)
    if json_match:
        verdict = json_match.group(1).lower().strip()
        # Try to extract explanation too
        exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', response)
        if exp_match:
            explanation = exp_match.group(1)
    else:
        # Fallback: look for verdict keywords
        response_lower = response.lower()
        for v in ["block", "quarantine", "alert", "monitor", "ignore"]:
            if v in response_lower:
                verdict = v
                break

    # Normalize verdict
    valid_verdicts = {"ignore", "monitor", "alert", "quarantine", "block"}
    if verdict not in valid_verdicts:
        verdict = "ignore"

    return {"verdict": verdict, "explanation": explanation}


def main():
    parser = argparse.ArgumentParser(description="Evaluate local SFT model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    label = "sft" if args.adapter else "base"
    results_file = args.output or f"results/{label}_eval_{args.n}_{args.seed}.jsonl"
    Path("results").mkdir(exist_ok=True)

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model, args.adapter)
    print("Model loaded.\n")

    manifest = generate_manifest(n=args.n, seed=args.seed)
    print(f"Evaluating {args.n} scenarios ({label} model)...\n")

    for i, entry in enumerate(manifest):
        session = DetectiveSession(
            seed=entry["seed"],
            scenario_type=entry["scenario_type"],
            observability=entry["observability"],
            attack_progress=entry["attack_progress"],
            results_file=results_file,
        )

        telemetry = session.get_telemetry()

        try:
            result = generate_verdict(model, tokenizer, telemetry)
            session.decide(result["verdict"], result["explanation"])

            gt = session.episode.ground_truth
            correct = (
                (result["verdict"] in ("quarantine", "block", "alert") and gt.is_ransomware)
                or (result["verdict"] in ("ignore", "monitor") and not gt.is_ransomware)
            )

            print(
                f"  [{i+1}/{args.n}] seed={entry['seed']} "
                f"type={entry['scenario_type']:20s} "
                f"verdict={result['verdict']:12s} "
                f"correct={correct}"
            )
        except Exception as e:
            print(f"  [{i+1}/{args.n}] seed={entry['seed']} type={entry['scenario_type']:20s} ERROR: {e}")
            if not session.is_done:
                session.decide("ignore", f"Error: {e}")

    print(f"\nResults written to {results_file}")

    from scripts.compute_metrics import load_results, compute_all_metrics, format_report
    results = load_results(results_file)
    metrics = compute_all_metrics(results)
    print(f"\n{format_report(metrics)}")


if __name__ == "__main__":
    main()
