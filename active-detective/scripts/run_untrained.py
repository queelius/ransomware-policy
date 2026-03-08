#!/usr/bin/env python3
"""Run an untrained model through the ransomware detection environment.

Demonstrates the full pipeline: telemetry generation -> prompt construction ->
multi-step model inference with tool execution -> reward computation.

Modes:
    --mode demo     Show what each component looks like (no GPU needed)
    --mode live     Load a real model and run inference (needs GPU + model)

Usage:
    # Demo mode -- see the full pipeline with simulated model output
    python -m scripts.run_untrained --mode demo

    # Live mode -- run a real model (downloads ~1.5GB for Qwen3-0.6B)
    python -m scripts.run_untrained --mode live --model Qwen/Qwen3-0.6B

    # Live mode with more episodes
    python -m scripts.run_untrained --mode live --model Qwen/Qwen3-0.6B --episodes 5
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

from environment.env import RansomwareDetectionEnv
from environment.reward import compute_reward
from simulator.models import ScenarioType
from tools.parser import ParsedToolCall, parse_tool_call, extract_thinking
from training.prompts import build_system_prompt, build_chat_messages


# ── Colors for terminal output ──────────────────────────────────────

class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def section(title: str) -> None:
    print(f"\n{C.BOLD}{C.HEADER}{'=' * 60}{C.END}")
    print(f"{C.BOLD}{C.HEADER}  {title}{C.END}")
    print(f"{C.BOLD}{C.HEADER}{'=' * 60}{C.END}\n")


def subsection(title: str) -> None:
    print(f"\n{C.BOLD}{C.CYAN}  >> {title}{C.END}\n")


# ── Demo mode: walk through each component ──────────────────────────


def run_demo(seed: int = 42) -> None:
    """Show what each piece of the pipeline looks like, no GPU needed."""

    rng = np.random.RandomState(seed)

    # ── 1. Telemetry generation ──────────────────────────────────────
    section("1. TELEMETRY DATA -- What the agent sees")

    scenarios = [
        ("benign", ScenarioType.BENIGN, 0.8),
        ("blitz ransomware", ScenarioType.BLITZ, 0.8),
        ("sleeper ransomware", ScenarioType.SLEEPER, 0.7),
    ]

    for label, stype, obs in scenarios:
        env = RansomwareDetectionEnv(max_steps=5)
        ep_rng = np.random.RandomState(rng.randint(0, 2**31))
        telemetry = env.reset(stype, obs, ep_rng, attack_progress=0.6)

        subsection(f"Scenario: {label} (observability={obs})")
        print(f"{C.DIM}{telemetry}{C.END}")
        print(f"\n  Ground truth: is_ransomware={env._episode.ground_truth.is_ransomware}"
              f"  phase={env._episode.ground_truth.attack_phase}")

    # ── 2. Virtual filesystem ────────────────────────────────────────
    section("2. VIRTUAL FILESYSTEM -- What tools query against")

    env = RansomwareDetectionEnv(max_steps=5)
    ep_rng = np.random.RandomState(42)
    env.reset(ScenarioType.BLITZ, 0.8, ep_rng, attack_progress=0.6)

    print(f"  Files in registry: {len(env._registry)}")
    print(f"  Directories: {env._registry.directories()}")
    print(f"  Processes: {env._ptable.all_pids()}")
    print()

    # Show a few files
    subsection("Sample files in C:/Users/A/Documents")
    files = env._registry.list_directory("C:/Users/A/Documents")[:5]
    for f in files:
        encrypted_tag = f" {C.RED}[ENCRYPTED]{C.END}" if f.is_encrypted else ""
        print(f"  {f.path}  entropy={f.entropy:.2f}  size={f.size:,}  "
              f"ext={f.extension}{encrypted_tag}")

    # Show a process
    subsection("Sample process (pid=4)")
    proc = env._ptable.get_process(4)
    print(f"  name={proc.name}  cmd={proc.command_line}  "
          f"children={proc.child_pids[:5]}")

    # ── 3. Available tools + costs ───────────────────────────────────
    section("3. AVAILABLE TOOLS -- What the agent can do")

    from tools.inspection import TOOL_COSTS
    for tool, cost in TOOL_COSTS.items():
        print(f"  {tool:20s}  cost={cost:+.2f}")

    # ── 4. System prompt ─────────────────────────────────────────────
    section("4. SYSTEM PROMPT -- Instructions to the agent")

    system_prompt = build_system_prompt(k_max=5)
    for line in system_prompt.split("\n"):
        print(f"  {C.DIM}{line}{C.END}")

    # ── 5. Full chat messages ────────────────────────────────────────
    section("5. CHAT MESSAGES -- What gets sent to the model")

    env = RansomwareDetectionEnv(max_steps=5)
    ep_rng = np.random.RandomState(42)
    telemetry = env.reset(ScenarioType.BLITZ, 0.8, ep_rng, attack_progress=0.6)
    messages = build_chat_messages(
        system_prompt, telemetry,
        history_windows=env._episode.history_windows,
    )

    for msg in messages:
        role_color = C.GREEN if msg["role"] == "system" else C.BLUE
        content_preview = msg["content"][:200]
        if len(msg["content"]) > 200:
            content_preview += "..."
        print(f"  {role_color}[{msg['role']}]{C.END} {content_preview}\n")

    # ── 6. Simulated multi-step rollout ──────────────────────────────
    section("6. SIMULATED ROLLOUT -- What the loop looks like")
    print("  (Simulating what an LLM would output -- step by step)\n")

    env = RansomwareDetectionEnv(max_steps=5)
    ep_rng = np.random.RandomState(42)
    telemetry = env.reset(ScenarioType.BLITZ, 0.8, ep_rng, attack_progress=0.6)

    # Simulate a reasonable model output sequence
    simulated_steps = [
        (
            '<think>I see file events with high entropy deltas and extension changes '
            'to .locked. This is suspicious. Let me scan the Documents directory.</think>\n'
            '<tool_call>{"name": "scan_directory", "arguments": '
            '{"path": "C:/Users/A/Documents"}}</tool_call>'
        ),
        (
            '<think>Several files show high entropy. Let me check if there is a '
            'suspicious process running.</think>\n'
            '<tool_call>{"name": "check_process", "arguments": {"pid": 4}}</tool_call>'
        ),
        (
            '<think>The System process looks normal. Given the entropy spikes and '
            'extension changes to .locked, this is clearly ransomware encryption. '
            'I should quarantine immediately.</think>\n'
            '<tool_call>{"name": "DECIDE", "arguments": {"verdict": "quarantine", '
            '"explanation": "High entropy deltas + extension changes to .locked '
            'indicate active ransomware encryption"}}</tool_call>'
        ),
    ]

    for i, model_output in enumerate(simulated_steps):
        print(f"  {C.BOLD}Step {i+1}:{C.END}")

        # Show model output
        thinking = extract_thinking(model_output)
        if thinking:
            print(f"    {C.YELLOW}[thinking]{C.END} {thinking[:120]}...")

        # Parse and execute
        parsed = parse_tool_call(model_output)
        if parsed:
            print(f"    {C.CYAN}[tool_call]{C.END} {parsed.tool_name}({parsed.args})")
            step_result = env.step(parsed)
            result_preview = json.dumps(step_result.result)[:150]
            print(f"    {C.GREEN}[tool_result]{C.END} {result_preview}")
            print(f"    {C.DIM}cost={step_result.cost:+.2f}  "
                  f"terminal={step_result.is_terminal}{C.END}")

        if env.is_done:
            break
        print()

    # ── 7. Reward computation ────────────────────────────────────────
    section("7. REWARD COMPUTATION -- How the agent scored")

    rollout = env.finish()
    print(f"  Verdict:      {rollout.verdict}")
    print(f"  Ground truth: is_ransomware={rollout.ground_truth.is_ransomware}"
          f"  phase={rollout.ground_truth.attack_phase}")
    print(f"  Steps taken:  {len(rollout.steps)}")
    print(f"  Total cost:   {rollout.cumulative_cost:+.2f}")
    print(f"  Has thinking: {rollout.has_thinking}  Has tool call: {rollout.has_tool_call}")
    print()
    print(f"  {C.BOLD}Reward breakdown:{C.END}")
    r = rollout.reward
    print(f"    Verdict reward:    {r.verdict_reward:+.2f}")
    print(f"    Action cost:       {r.action_cost:+.2f}")
    print(f"    Efficiency bonus:  {r.efficiency_bonus:+.3f}")
    print(f"    Format reward:     {r.format_reward:+.2f}")
    print(f"    {C.BOLD}Total:             {r.total:+.3f}{C.END}")

    correct = (rollout.verdict in ("alert", "quarantine", "block")) == rollout.ground_truth.is_ransomware
    tag = f"{C.GREEN}CORRECT" if correct else f"{C.RED}WRONG"
    print(f"\n  Result: {tag}{C.END}")


# ── Live mode: run a real model ─────────────────────────────────────


def run_live(
    model_name: str = "Qwen/Qwen3-0.6B",
    n_episodes: int = 3,
    k_max: int = 5,
    seed: int = 42,
    max_new_tokens: int = 512,
) -> None:
    """Load a real model and run it through the environment."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Live mode requires: pip install torch transformers")
        sys.exit(1)

    section("Loading model")
    print(f"  Model: {model_name}")
    print(f"  This may download the model on first run...\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    device = next(model.parameters()).device
    print(f"  Loaded on: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build system prompt
    system_prompt = build_system_prompt(k_max=k_max)

    # Scenarios to test
    scenario_configs = [
        ("benign", ScenarioType.BENIGN, 0.8, 0.0),
        ("blitz ransomware", ScenarioType.BLITZ, 0.8, 0.6),
        ("sleeper ransomware", ScenarioType.SLEEPER, 0.7, 0.5),
    ]

    rng = np.random.RandomState(seed)

    for ep_idx in range(min(n_episodes, len(scenario_configs))):
        label, stype, obs, progress = scenario_configs[ep_idx]
        section(f"Episode {ep_idx+1}: {label}")

        # Reset environment
        env = RansomwareDetectionEnv(max_steps=k_max)
        ep_rng = np.random.RandomState(rng.randint(0, 2**31))
        telemetry = env.reset(stype, obs, ep_rng, attack_progress=progress)

        print(f"  {C.DIM}Ground truth: {env._episode.ground_truth.label}{C.END}\n")

        # Build initial messages
        messages = build_chat_messages(
            system_prompt, telemetry,
            history_windows=env._episode.history_windows,
        )

        # Multi-step inference loop
        for step_num in range(k_max):
            subsection(f"Step {step_num + 1}/{k_max}")

            # Tokenize
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            model_output = tokenizer.decode(new_tokens, skip_special_tokens=False)

            # Show thinking
            thinking = extract_thinking(model_output)
            if thinking:
                print(f"  {C.YELLOW}[thinking]{C.END} "
                      f"{textwrap.shorten(thinking, 200)}")

            # Parse tool call
            parsed = parse_tool_call(model_output)
            if parsed is None:
                print(f"  {C.RED}[no valid tool call found]{C.END}")
                print(f"  {C.DIM}Raw output: "
                      f"{textwrap.shorten(model_output, 300)}{C.END}")
                # No valid tool call — has_tool_call stays False for this turn
                # Add model output to conversation and continue
                messages.append({"role": "assistant", "content": model_output})
                messages.append({
                    "role": "user",
                    "content": "Please use a tool or call DECIDE with your verdict."
                })
                continue

            print(f"  {C.CYAN}[tool_call]{C.END} {parsed.tool_name}({parsed.args})")

            # Execute
            step_result = env.step(parsed)
            result_json = json.dumps(step_result.result, indent=2)
            print(f"  {C.GREEN}[tool_result]{C.END} {result_json[:200]}")
            print(f"  {C.DIM}cost={step_result.cost:+.2f}{C.END}")

            # Append to conversation
            messages.append({"role": "assistant", "content": model_output})
            messages.append({
                "role": "user",
                "content": step_result.result_text,
            })

            if env.is_done:
                break

        # Compute reward
        rollout = env.finish()
        subsection("Result")
        print(f"  Verdict:      {rollout.verdict}")
        print(f"  Ground truth: {rollout.ground_truth.label}")
        print(f"  Steps taken:  {len(rollout.steps)}")
        print(f"  Total cost:   {rollout.cumulative_cost:+.2f}")

        r = rollout.reward
        print(f"\n  Reward breakdown:")
        print(f"    Verdict reward:    {r.verdict_reward:+.2f}")
        print(f"    Action cost:       {r.action_cost:+.2f}")
        print(f"    Efficiency bonus:  {r.efficiency_bonus:+.3f}")
        print(f"    Format reward:     {r.format_reward:+.2f}")
        print(f"    {C.BOLD}Total:             {r.total:+.3f}{C.END}")

        correct = (rollout.verdict in ("alert", "quarantine", "block")) == \
                  rollout.ground_truth.is_ransomware
        tag = f"{C.GREEN}CORRECT" if correct else f"{C.RED}WRONG"
        print(f"\n  {tag}{C.END}")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run untrained model through ransomware detection environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m scripts.run_untrained --mode demo
              python -m scripts.run_untrained --mode live --model Qwen/Qwen3-0.6B
        """),
    )
    parser.add_argument("--mode", choices=["demo", "live"], default="demo",
                        help="demo = walkthrough with simulated output; "
                             "live = real model inference (default: demo)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model for live mode (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes for live mode")
    parser.add_argument("--k-max", type=int, default=5,
                        help="Tool call budget per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens per generation step")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(seed=args.seed)
    else:
        run_live(
            model_name=args.model,
            n_episodes=args.episodes,
            k_max=args.k_max,
            seed=args.seed,
            max_new_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
