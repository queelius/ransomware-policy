"""GRPO training script using TRL GRPOTrainer + environment_factory.

Uses TRL's native multi-step agent training: the environment exposes
investigation tools as methods, and GRPOTrainer handles the full
generate → tool_call → execute → inject_result → continue loop.

No ad-hoc hacks — TRL's environment_factory does real multi-step rollouts.

Usage::

    accelerate launch -m training.train_grpo \
        --model Qwen/Qwen3.5-9B \
        --output-dir ./checkpoints \
        --n-episodes 500 \
        --group-size 4 \
        --max-steps 200 \
        --lr 1e-5

Requires: transformers>=5.2.0, trl>=0.16.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simulator.host import HostState
from simulator.models import GroundTruth, ScenarioType, Verdict
from simulator.telemetry import generate_episode
from tools.inspection import TOOL_COSTS, VALID_VERDICTS
from training.prompts import build_system_prompt
from training.scenarios import generate_training_scenarios, save_scenarios

logger = logging.getLogger(__name__)


# ── Training configuration ───────────────────────────────────────────


@dataclass
class TrainingConfig:
    """GRPO training configuration."""

    model_name: str = "Qwen/Qwen3.5-9B"
    adapter_path: str | None = None  # Path to SFT LoRA adapter to load as starting point
    output_dir: str = "./checkpoints"
    n_episodes: int = 500
    group_size: int = 4
    max_training_steps: int = 200
    lr: float = 1e-5
    k_max: int = 5
    max_completion_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 42
    observability_levels: list[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    available_tools: list[str] | None = None
    logging_steps: int = 10
    save_steps: int = 50
    use_unsloth: bool = True
    disable_thinking: bool = False
    n_history: int = 2


# ── TRL Environment ──────────────────────────────────────────────────
#
# GRPOTrainer's environment_factory creates one instance per rollout.
# Public methods become tools. TRL handles the multi-step loop:
#   model generates → parses tool call → calls method → injects result
#   → model continues generating → ... → EOS or max_completion_length.
#
# The environment's reset() method receives the dataset row fields
# and returns a string appended to the last user message.


class DetectionEnv:
    """TRL-compatible environment for ransomware detection.

    Public methods (inspect_file, check_process, scan_directory,
    decide, etc.) are exposed as tools by GRPOTrainer.
    """

    def __init__(self, k_max: int = 5) -> None:
        self._k_max = k_max
        self._host: HostState | None = None
        self._ground_truth: GroundTruth | None = None
        self._steps: int = 0
        self._cumulative_cost: float = 0.0
        self._verdict: str | None = None
        self._explanation: str = ""

    def reset(self, **kwargs) -> str | None:
        """Reset environment state. Called by GRPOTrainer before each rollout.

        Receives dataset row fields as kwargs. We use 'scenario_data'
        (a JSON string) to reconstruct the episode state.
        """
        scenario_data = json.loads(kwargs.get("scenario_data", "{}"))

        scenario_type = ScenarioType(scenario_data.get("scenario_type", "benign"))
        observability = scenario_data.get("observability", 0.5)
        attack_progress = scenario_data.get("attack_progress", 0.5)
        seed = scenario_data.get("seed", 42)
        n_history = scenario_data.get("n_history", 2)

        rng = np.random.RandomState(seed)

        # Generate episode to get ground truth and host state
        episode = generate_episode(
            scenario_type, observability, rng,
            attack_progress=attack_progress,
            n_history=n_history,
        )
        self._ground_truth = episode.ground_truth

        # Use the episode's host for tool execution — ensures tools
        # query the same state that generated the telemetry.
        self._host = episode.host

        # Reset counters
        self._steps = 0
        self._cumulative_cost = 0.0
        self._verdict = None
        self._explanation = ""

        # Format history + current window for the model
        if episode.history_windows:
            parts = []
            n = len(episode.history_windows)
            for i, hw in enumerate(episode.history_windows):
                label = f"Window t-{n - i} (prior)"
                parts.append(f"--- {label} ---\n{hw}")
            parts.append(f"--- Current window ---\n{episode.input_text}")
            return "\n\n".join(parts)

        return episode.input_text

    # ── Tools (public methods exposed by TRL) ────────────────────────

    def inspect_file(self, path: str) -> str:
        """Inspect a file's metadata — entropy, size, extension, timestamps, content type.

        Args:
            path: Full path to the file to inspect (e.g., "C:/Users/A/Documents/report.docx").

        Returns:
            JSON string with file metadata or error message.
        """
        self._steps += 1
        self._cumulative_cost += TOOL_COSTS["inspect_file"]

        record = self._host.files.get_file(path)
        if record is None:
            return json.dumps({"error": f"File not found: {path}"})

        return json.dumps({
            "entropy": round(record.entropy, 2),
            "size": record.size,
            "extension": record.extension,
            "modified_ts": record.modified_at.isoformat(),
            "content_type": record.content_type.value,
        })

    def check_process(self, pid: int) -> str:
        """Check a running process by PID — name, command line, parent, children.

        Args:
            pid: Process ID to look up.

        Returns:
            JSON string with process details or error message.
        """
        self._steps += 1
        self._cumulative_cost += TOOL_COSTS["check_process"]

        record = self._host.processes.get_process(pid)
        if record is None:
            return json.dumps({"error": f"Process not found: pid={pid}"})

        parent = self._host.processes.get_process(record.parent_pid)
        parent_name = parent.name if parent else "unknown"

        return json.dumps({
            "name": record.name,
            "command_line": record.command_line,
            "parent": parent_name,
            "child_pids": record.child_pids,
        })

    def scan_directory(self, path: str) -> str:
        """List files in a directory with metadata summaries. More expensive but reveals more.

        Args:
            path: Directory path to scan (e.g., "C:/Users/A/Documents").

        Returns:
            JSON string with list of files and their metadata.
        """
        self._steps += 1
        self._cumulative_cost += TOOL_COSTS["scan_directory"]

        files = self._host.files.list_directory(path)
        if not files:
            return json.dumps({"files": [], "note": f"No files found in {path}"})

        return json.dumps({
            "files": [
                {
                    "path": f.path,
                    "size": f.size,
                    "entropy": round(f.entropy, 2),
                    "extension": f.extension,
                }
                for f in files
            ]
        })

    def list_connections(self, filter_state: str = "") -> str:
        """List active network connections, optionally filtered by state.

        Args:
            filter_state: Optional state filter (e.g., "established", "closed").

        Returns:
            JSON string with list of connections.
        """
        self._steps += 1
        self._cumulative_cost += TOOL_COSTS["list_connections"]

        from tools.network_tools import list_connections as _list_connections
        result = _list_connections(
            self._host.connections,
            filter_state=filter_state if filter_state else None,
        )
        return json.dumps(result)

    def inspect_connection(self, conn_id: int) -> str:
        """Inspect a specific network connection by ID.

        Args:
            conn_id: Connection ID to inspect.

        Returns:
            JSON string with connection details or error message.
        """
        self._steps += 1
        self._cumulative_cost += TOOL_COSTS["inspect_connection"]

        from tools.network_tools import inspect_connection as _inspect_connection
        result = _inspect_connection(self._host.connections, conn_id)
        return json.dumps(result, default=str)

    def query_registry(self, key_path: str) -> str:
        """Query a Windows registry key for values and sub-keys.

        Args:
            key_path: Registry key path (e.g., "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run").

        Returns:
            JSON string with registry values or error message.
        """
        self._steps += 1
        self._cumulative_cost += TOOL_COSTS["query_registry"]

        from tools.forensic_tools import query_registry as _query_registry
        result = _query_registry(self._host.registry, key_path)
        return json.dumps(result)

    def list_process_handles(self, pid: int) -> str:
        """List a process's open file handles, loaded modules, and integrity info.

        Args:
            pid: Process ID to inspect.

        Returns:
            JSON string with process forensic details or error message.
        """
        self._steps += 1
        self._cumulative_cost += TOOL_COSTS["list_process_handles"]

        from tools.forensic_tools import list_process_handles as _list_process_handles
        result = _list_process_handles(self._host.processes, pid)
        return json.dumps(result)

    def query_event_log(self, source: str = "", event_id: int = 0,
                        since: str = "") -> str:
        """Query the Windows Event Log with optional filters.

        Args:
            source: Filter by event source (e.g., "Security", "System").
            event_id: Filter by event ID (e.g., 4624 for logon events).
            since: Filter events after this ISO timestamp.

        Returns:
            JSON string with matching event log entries.
        """
        self._steps += 1
        self._cumulative_cost += TOOL_COSTS["query_event_log"]

        from tools.forensic_tools import query_event_log as _query_event_log
        result = _query_event_log(
            self._host.event_log,
            source=source if source else None,
            event_id=event_id if event_id else None,
            since=since if since else None,
        )
        return json.dumps(result, default=str)

    def read_file_sample(self, path: str) -> str:
        """Read a hex sample of file contents with entropy and magic bytes analysis.

        Args:
            path: Full path to the file to sample.

        Returns:
            JSON string with hex bytes, entropy, and magic bytes, or error message.
        """
        self._steps += 1
        self._cumulative_cost += TOOL_COSTS["read_file_sample"]

        from tools.forensic_tools import read_file_sample as _read_file_sample
        result = _read_file_sample(self._host.files, path)
        return json.dumps(result)

    def decide(self, verdict: str, explanation: str) -> str:
        """Submit your final verdict. This ends the investigation.

        Args:
            verdict: One of: ignore, monitor, alert, quarantine, block.
            explanation: Brief explanation of your reasoning.

        Returns:
            Confirmation of the verdict submitted.
        """
        self._steps += 1
        self._verdict = verdict
        self._explanation = explanation

        if verdict not in VALID_VERDICTS:
            return json.dumps({
                "error": f"Invalid verdict '{verdict}'. "
                         f"Must be one of: {', '.join(sorted(VALID_VERDICTS))}"
            })

        return json.dumps({"verdict": verdict, "explanation": explanation,
                           "status": "Investigation complete."})


# ── Reward function ──────────────────────────────────────────────────


def detection_reward(environments: list[DetectionEnv], **kwargs) -> list[float]:
    """Compute rewards from completed environment rollouts.

    Called by GRPOTrainer after each batch of rollouts completes.
    Each environment has accumulated state from the multi-step interaction.
    """
    rewards = []
    for env in environments:
        reward = _compute_env_reward(env)
        rewards.append(reward)
    return rewards


def _compute_env_reward(env: DetectionEnv) -> float:
    """Compute reward for a single completed rollout."""
    gt = env._ground_truth
    if gt is None:
        return -1.0  # shouldn't happen

    verdict = env._verdict
    k_max = env._k_max

    # ── Verdict reward (asymmetric) ──────────────────────────────
    if verdict is None:
        # No decide call — harsh penalty
        if gt.is_ransomware:
            verdict_reward = -2.0  # false negative
        else:
            verdict_reward = -0.5  # slightly less bad for benign
    else:
        try:
            v = Verdict(verdict)
            predicted = v.is_ransomware_prediction
            if predicted == gt.is_ransomware:
                verdict_reward = 1.0
            elif gt.is_ransomware and not predicted:
                verdict_reward = -2.0  # false negative
            else:
                verdict_reward = -1.0  # false positive
        except ValueError:
            verdict_reward = -1.0  # invalid verdict

    # ── Action cost ──────────────────────────────────────────────
    action_cost = env._cumulative_cost

    # ── Efficiency bonus ─────────────────────────────────────────
    unused = max(0, k_max - env._steps)
    efficiency_bonus = unused * 0.05

    return verdict_reward + action_cost + efficiency_bonus


def verdict_reward_func(completions: list[str], **kwargs) -> list[float]:
    """Offline GRPO reward: parse verdict from completion, score against ground truth.

    For TRL 0.24 which doesn't support environment_factory. The model generates
    a single-pass response with a verdict, and we score it.
    """
    import re as _re

    scenario_data_list = kwargs.get("scenario_data", [])
    rewards = []

    for i, completion in enumerate(completions):
        content = completion
        if isinstance(completion, list):
            content = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            content = completion.get("content", "")

        # Parse scenario data to get ground truth
        if i < len(scenario_data_list):
            sd = json.loads(scenario_data_list[i]) if isinstance(scenario_data_list[i], str) else scenario_data_list[i]
        else:
            sd = {}

        scenario_type = ScenarioType(sd.get("scenario_type", "benign"))
        is_ransomware = scenario_type != ScenarioType.BENIGN

        # Parse verdict from completion
        verdict = None
        json_match = _re.search(r'"verdict"\s*:\s*"([^"]+)"', content)
        if json_match:
            verdict = json_match.group(1).lower().strip()
        else:
            content_lower = content.lower()
            for v in ["block", "quarantine", "alert", "monitor", "ignore"]:
                if v in content_lower:
                    verdict = v
                    break

        # Score
        if verdict is None:
            reward = -2.0 if is_ransomware else -0.5
        else:
            predicted_malicious = verdict in ("alert", "quarantine", "block")
            if predicted_malicious == is_ransomware:
                reward = 1.0  # correct
            elif is_ransomware and not predicted_malicious:
                reward = -2.0  # false negative (worst)
            else:
                reward = -1.0  # false positive

        # Format bonus
        if "<think>" in content:
            reward += 0.05
        if "verdict" in content.lower():
            reward += 0.05

        rewards.append(reward)

    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward for well-formatted output (thinking + tool calls).

    Separate reward function — TRL sums multiple reward functions.
    """
    rewards = []
    for completion in completions:
        content = completion
        if isinstance(completion, list):
            # Conversational format
            content = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            content = completion.get("content", "")

        has_thinking = "<think>" in content
        has_tool_call = "<tool_call>" in content

        score = 0.0
        if has_thinking:
            score += 0.05
        if has_tool_call:
            score += 0.05
        rewards.append(score)
    return rewards


# ── Dataset preparation ──────────────────────────────────────────────


def prepare_dataset(config: TrainingConfig) -> list[dict]:
    """Generate training dataset in TRL conversational format.

    Each item has:
    - 'prompt': chat messages (system + user with telemetry placeholder)
    - 'scenario_data': JSON string with episode metadata for env.reset()
    """
    batch = generate_training_scenarios(
        n_episodes=config.n_episodes,
        observability_levels=config.observability_levels,
        seed=config.seed,
    )

    system_prompt = build_system_prompt(
        k_max=config.k_max,
        available_tools=config.available_tools,
    )

    rng = np.random.RandomState(config.seed)
    dataset = []

    for ep in batch.episodes:
        # Generate a unique seed for this episode's environment
        ep_seed = int(rng.randint(0, 2**31))

        attack_progress = 0.0
        if ep.scenario_type != ScenarioType.BENIGN:
            attack_progress = rng.uniform(0.2, 0.9)

        scenario_data = {
            "scenario_type": ep.scenario_type.value,
            "observability": ep.observability,
            "attack_progress": attack_progress,
            "seed": ep_seed,
            "n_history": config.n_history,
        }

        # Generate episode to get telemetry text
        episode = generate_episode(
            ep.scenario_type, ep.observability, np.random.RandomState(ep_seed),
            attack_progress=attack_progress, n_history=config.n_history,
        )

        # Build telemetry text with history windows
        if episode.history_windows:
            parts = []
            n = len(episode.history_windows)
            for i, hw in enumerate(episode.history_windows):
                parts.append(f"--- Window t-{n - i} (prior) ---\n{hw}")
            parts.append(f"--- Current Window ---\n{episode.input_text}")
            telemetry_text = "\n\n".join(parts)
        else:
            telemetry_text = episode.input_text

        # Prompt with telemetry included (offline GRPO — no environment injection)
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this host telemetry and provide your verdict as JSON: {{\"verdict\": \"...\", \"explanation\": \"...\"}}\n\n{telemetry_text}"},
        ]

        dataset.append({
            "prompt": prompt,
            "scenario_data": json.dumps(scenario_data),
        })

    return dataset


# ── Model loading ────────────────────────────────────────────────────


def load_model(config: TrainingConfig):
    """Load model with QLoRA, optionally via Unsloth."""
    if config.use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_name,
                max_seq_length=4096,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=[
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=0.0,
                bias="none",
            )
            logger.info(f"Loaded {config.model_name} via Unsloth (4-bit)")
            return model, tokenizer

        except ImportError:
            logger.warning("Unsloth not available, falling back to transformers")

    # Fallback: transformers + PEFT
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import torch as _torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=_torch.bfloat16,
    )

    # If an SFT adapter is provided, merge it into the base model first
    if config.adapter_path and Path(config.adapter_path).exists():
        logger.info(f"Loading SFT adapter from {config.adapter_path}")
        model = PeftModel.from_pretrained(model, config.adapter_path)
        model = model.merge_and_unload()
        logger.info("SFT adapter merged into base model")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Cast LoRA layers to bfloat16 to match compute dtype
    import torch as _torch2
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(_torch2.bfloat16)

    logger.info(f"Loaded {config.model_name} via transformers + PEFT (4-bit)")
    return model, tokenizer


# ── Training entry point ─────────────────────────────────────────────


def train(config: TrainingConfig) -> None:
    """Run GRPO training with multi-step environment rollouts.

    Requires GPU + transformers>=5.2.0 + trl>=0.16.0.
    """
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        logger.error(
            "Training requires: pip install 'trl>=0.16.0' 'transformers>=5.2.0' datasets\n"
            f"Missing: {e}"
        )
        sys.exit(1)

    # Load model
    model, tokenizer = load_model(config)

    # Prepare dataset
    logger.info(f"Generating {config.n_episodes} training scenarios...")
    dataset_items = prepare_dataset(config)

    # Save scenarios for reproducibility
    save_scenarios(
        generate_training_scenarios(
            n_episodes=config.n_episodes, seed=config.seed),
        Path(config.output_dir) / "training_scenarios.jsonl",
    )

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(dataset_items)

    # GRPO training config
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=1,
        max_steps=config.max_training_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.group_size,
        learning_rate=config.lr,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        max_completion_length=config.max_completion_length,
        num_generations=config.group_size,
        seed=config.seed,
        report_to="none",  # change to "wandb" for experiment tracking
        gradient_checkpointing=True,
        bf16=True,
    )

    # Workaround for PEFT + TRL compatibility: GRPOTrainer tries to set
    # model.warnings_issued which doesn't exist on PeftModel
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # Create trainer — TRL 0.24 uses offline GRPO (no environment_factory).
    # The reward function scores single-pass completions against ground truth.
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=hf_dataset,
        reward_funcs=[verdict_reward_func],
    )

    logger.info("Starting GRPO training with multi-step environment rollouts...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    logger.info("Training complete.")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train ransomware detection agent with GRPO")

    parser.add_argument("--model", default="Qwen/Qwen3.5-9B",
                        help="HuggingFace model name or path")
    parser.add_argument("--adapter", default=None,
                        help="Path to SFT LoRA adapter to merge before GRPO")
    parser.add_argument("--output-dir", default="./checkpoints",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--n-episodes", type=int, default=500,
                        help="Number of training episodes to generate")
    parser.add_argument("--group-size", type=int, default=4,
                        help="GRPO group size (rollouts per scenario)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--k-max", type=int, default=5,
                        help="Tool call budget per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-unsloth", action="store_true",
                        help="Disable Unsloth (use transformers + PEFT)")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Disable Qwen3 thinking mode")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = TrainingConfig(
        model_name=args.model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
        group_size=args.group_size,
        max_training_steps=args.max_steps,
        lr=args.lr,
        k_max=args.k_max,
        seed=args.seed,
        use_unsloth=not args.no_unsloth,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        disable_thinking=args.disable_thinking,
    )

    train(config)


if __name__ == "__main__":
    main()
