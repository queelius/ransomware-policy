"""Evaluate a local model WITH tool access via RansomwareDetectionEnv.

Loads the model, generates text, parses tool calls, executes them
against the simulator, feeds results back, and loops until DECIDE
or budget exhausted. This matches the GRPO training setup.

Usage:
    PYTHONPATH=. python scripts/run_eval_with_tools.py \
        --model Qwen/Qwen3-1.7B \
        --adapter ./checkpoints-grpo-sft-1.7b \
        --n 200 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, ".")

from environment.env import RansomwareDetectionEnv
from simulator.models import ScenarioType
from tools.parser import parse_tool_call, is_decide_call, format_tool_result
from tools.inspection import TOOL_COSTS
from training.prompts import build_system_prompt, TOOL_DESCRIPTIONS, DECIDE_DESCRIPTION
from scripts.generate_eval_manifest import generate_manifest
import numpy as np


# Tool definitions in Qwen3 chat template format
QWEN_TOOLS = []
for tool_name, desc in TOOL_DESCRIPTIONS.items():
    # Parse params from the description string
    params = {}
    if "path" in desc.lower():
        params["path"] = {"type": "string", "description": "File or directory path"}
    if "pid" in desc.lower():
        params["pid"] = {"type": "integer", "description": "Process ID"}
    if "conn_id" in desc.lower():
        params["conn_id"] = {"type": "integer", "description": "Connection ID"}
    if "key_path" in desc.lower():
        params["key_path"] = {"type": "string", "description": "Registry key path"}
    if "filter" in desc.lower():
        params["filter_state"] = {"type": "string", "description": "Connection state filter"}
    if "source" in desc.lower() and "event" in desc.lower():
        params["source"] = {"type": "string", "description": "Event source"}
        params["event_id"] = {"type": "integer", "description": "Event ID"}

    QWEN_TOOLS.append({
        "type": "function",
        "function": {
            "name": tool_name,
            "description": desc.split(" — ")[1] if " — " in desc else desc,
            "parameters": {"type": "object", "properties": params},
        },
    })

# Add DECIDE tool
QWEN_TOOLS.append({
    "type": "function",
    "function": {
        "name": "DECIDE",
        "description": "Submit your final verdict. This ends the investigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "description": "One of: ignore, monitor, alert, quarantine, block"},
                "explanation": {"type": "string", "description": "Brief reasoning"},
            },
            "required": ["verdict", "explanation"],
        },
    },
})


def load_model(model_name: str, adapter_path: str | None = None):
    """Load base model with optional LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_episode(model, tokenizer, env: RansomwareDetectionEnv,
                system_prompt: str, telemetry: str,
                k_max: int = 5, max_new_tokens: int = 512) -> dict:
    """Run one episode: model generates, we parse tool calls, execute, loop."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze the following telemetry window.\n\n{telemetry}"},
    ]

    total_steps = 0
    tools_used = []

    for turn in range(k_max + 2):  # +2 for safety margin
        # Format messages with tool definitions (same as TRL does during training)
        text = tokenizer.apply_chat_template(
            messages, tools=QWEN_TOOLS, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Parse tool call from response
        tool_call = parse_tool_call(response)

        if tool_call is None:
            # No tool call found — force ignore verdict
            break

        if is_decide_call(tool_call.tool_name):
            # Terminal action
            verdict = tool_call.args.get("verdict", "ignore")
            explanation = tool_call.args.get("explanation", response[:200])
            return {
                "verdict": verdict,
                "explanation": explanation,
                "steps": total_steps,
                "tools_used": tools_used,
            }

        # Execute investigation tool
        step_result = env.step(tool_call)
        total_steps += 1
        tools_used.append(tool_call.tool_name)

        # Add assistant response and tool result to messages
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": step_result.result_text})

        if env.is_done:
            break

    # Never called DECIDE — default to ignore
    return {
        "verdict": "ignore",
        "explanation": "Budget exhausted without verdict",
        "steps": total_steps,
        "tools_used": tools_used,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate local model WITH tools")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    label = "grpo_tools" if args.adapter else "base_tools"
    results_file = args.output or f"results/{label}_eval_{args.n}_{args.seed}.jsonl"
    Path("results").mkdir(exist_ok=True)

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model, args.adapter)
    print("Model loaded.\n")

    system_prompt = build_system_prompt(k_max=args.k_max)
    manifest = generate_manifest(n=args.n, seed=args.seed)

    print(f"Evaluating {args.n} scenarios WITH tools ({label})...\n")

    correct_count = 0
    total_tools = 0

    for i, entry in enumerate(manifest):
        env = RansomwareDetectionEnv(max_steps=args.k_max)
        rng = np.random.RandomState(entry["seed"])
        stype = ScenarioType(entry["scenario_type"])
        progress = entry["attack_progress"]

        telemetry = env.reset(
            scenario_type=stype,
            observability=entry["observability"],
            rng=rng,
            attack_progress=progress,
        )

        try:
            result = run_episode(model, tokenizer, env, system_prompt, telemetry, k_max=args.k_max)

            gt = env._episode.ground_truth
            verdict = result["verdict"]
            predicted_malicious = verdict in ("alert", "quarantine", "block")
            correct = predicted_malicious == gt.is_ransomware

            if correct:
                correct_count += 1
            total_tools += result["steps"]

            # Write result
            record = {
                "seed": entry["seed"],
                "scenario_type": entry["scenario_type"],
                "is_ransomware_gt": gt.is_ransomware,
                "attack_phase": gt.attack_phase,
                "observability": entry["observability"],
                "verdict": verdict,
                "explanation": result["explanation"][:200],
                "steps_taken": result["steps"],
                "cumulative_cost": sum(-0.03 for _ in result["tools_used"]),  # approx
                "tools_used": result["tools_used"],
                "reward": 0.0,
                "correct": correct,
                "k_max": args.k_max,
            }
            with open(results_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            print(
                f"  [{i+1}/{args.n}] seed={entry['seed']} "
                f"type={entry['scenario_type']:20s} "
                f"verdict={verdict:12s} "
                f"correct={correct} "
                f"tools={result['steps']}"
            )
        except Exception as e:
            print(f"  [{i+1}/{args.n}] seed={entry['seed']} type={entry['scenario_type']:20s} ERROR: {e}")
            with open(results_file, "a") as f:
                f.write(json.dumps({
                    "seed": entry["seed"], "scenario_type": entry["scenario_type"],
                    "is_ransomware_gt": stype != ScenarioType.BENIGN,
                    "attack_phase": None, "observability": entry["observability"],
                    "verdict": "ignore", "explanation": f"Error: {e}",
                    "steps_taken": 0, "cumulative_cost": 0, "tools_used": [],
                    "reward": 0, "correct": stype == ScenarioType.BENIGN, "k_max": args.k_max,
                }) + "\n")

    print(f"\nResults: {results_file}")
    print(f"Accuracy: {correct_count}/{args.n} = {correct_count/args.n:.1%}")
    print(f"Avg tools/episode: {total_tools/args.n:.1f}")

    from scripts.compute_metrics import load_results, compute_all_metrics, format_report
    results = load_results(results_file)
    metrics = compute_all_metrics(results)
    print(f"\n{format_report(metrics)}")


if __name__ == "__main__":
    main()
