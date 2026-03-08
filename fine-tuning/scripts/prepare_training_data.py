#!/usr/bin/env python3
"""
Prepare training data from our telemetry generators for LLM fine-tuning.
Converts telemetry windows into prompt-completion pairs.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add telemetry directory to path
sys.path.append('./telemetry')

from comprehensive_telemetry_gen import ComprehensiveTelemetryGenerator
from finetune_format import FineTuningFormatter

class TrainingDataPreparer:
    """Prepare telemetry data for LLM fine-tuning."""

    def __init__(self):
        self.generator = ComprehensiveTelemetryGenerator(seed=42)
        self.formatter = FineTuningFormatter()

    def generate_training_data(
        self,
        output_file: str,
        num_sequences: int = 100,
        train_split: float = 0.8
    ):
        """Generate complete training dataset."""

        all_examples = []

        print(f"Generating {num_sequences} sequences...")

        for i in range(num_sequences):
            if i % 10 == 0:
                print(f"  Generated {i}/{num_sequences} sequences...")

            # Mix scenarios: 70% ransomware, 30% benign
            scenario = "benign" if i % 3 == 2 else "ransomware_attack"

            # Generate sequence
            windows = self.generator.generate_telemetry_sequence(
                num_windows=5,
                scenario=scenario
            )

            # Create training examples from windows
            for j in range(len(windows) - 3):
                example = self._create_training_example(
                    windows[j:j+4],  # 3 context + 1 target
                    scenario=scenario
                )
                all_examples.append(example)

        # Split into train/eval
        split_idx = int(len(all_examples) * train_split)
        train_examples = all_examples[:split_idx]
        eval_examples = all_examples[split_idx:]

        # Save training data
        train_file = output_file.replace('.jsonl', '_train.jsonl')
        with open(train_file, 'w') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\n')

        # Save evaluation data
        eval_file = output_file.replace('.jsonl', '_eval.jsonl')
        with open(eval_file, 'w') as f:
            for example in eval_examples:
                f.write(json.dumps(example) + '\n')

        print(f"\nGenerated {len(all_examples)} total examples")
        print(f"  Training: {len(train_examples)} examples -> {train_file}")
        print(f"  Evaluation: {len(eval_examples)} examples -> {eval_file}")

    def _create_training_example(self, windows: List, scenario: str) -> Dict[str, str]:
        """Create a single training example."""

        # Context windows (what model sees)
        context_windows = windows[:3]

        # Target window (what model predicts)
        target_window = windows[3]

        # Create prompt
        prompt = self._format_prompt(context_windows)

        # Create completion
        completion = self._format_completion(target_window, scenario)

        return {
            "prompt": prompt,
            "completion": completion
        }

    def _format_prompt(self, windows: List) -> str:
        """Format input prompt."""

        lines = []

        # System prompt
        lines.append("<|system|>")
        lines.append("You are a ransomware detection expert. Analyze telemetry and predict threats.")
        lines.append("</system>")
        lines.append("")

        # Telemetry context
        lines.append("<|telemetry|>")

        for i, window in enumerate(windows, 1):
            lines.append(f"\n[Window {i}]")

            # Summarize events
            event_summary = self._summarize_events(window.observable_events)
            lines.append(event_summary)

        lines.append("</telemetry>")
        lines.append("")

        # Task
        lines.append("<|task|>")
        lines.append("Based on the telemetry above:")
        lines.append("1. What is the current state?")
        lines.append("2. What will happen next?")
        lines.append("3. What actions should be taken?")
        lines.append("</task>")

        return '\n'.join(lines)

    def _format_completion(self, target_window, scenario: str) -> str:
        """Format expected completion."""

        lines = []

        lines.append("<|analysis|>")

        # State assessment
        state = target_window.latent_context.true_state.value
        lines.append(f"\n## Current State")
        lines.append(f"State: {state}")
        lines.append(f"Threat Level: {target_window.expert_annotation.risk_assessment}")
        lines.append(f"Attack Phase: {target_window.latent_context.attack_phase}")

        # Situation explanation
        lines.append(f"\n## Situation")
        lines.append(target_window.expert_annotation.plain_english_summary)

        # Key indicators
        lines.append(f"\n## Evidence")
        for indicator in target_window.expert_annotation.key_indicators[:3]:
            lines.append(f"- {indicator}")

        # Prediction
        lines.append(f"\n## Prediction")
        lines.append(f"Next Action: {target_window.latent_context.next_likely_action}")
        if target_window.latent_context.time_to_impact:
            lines.append(f"Time to Impact: {target_window.latent_context.time_to_impact} minutes")

        # Recommended actions
        lines.append(f"\n## Recommended Actions")
        for action in target_window.expert_annotation.recommended_actions[:3]:
            lines.append(f"- {action}")

        lines.append("\n</analysis>")

        return '\n'.join(lines)

    def _summarize_events(self, events: List) -> str:
        """Create concise summary of events."""

        if not events:
            return "No events detected"

        # Count event types
        event_types = {}
        suspicious_events = []

        for event in events[:20]:  # Limit to first 20
            event_type = event.event_type
            event_types[event_type] = event_types.get(event_type, 0) + 1

            # Identify suspicious events
            if event_type == "process":
                proc_name = event.details.get("process_name", "").lower()
                cmd_line = str(event.details.get("command_line", "")).lower()

                if any(s in proc_name for s in ["powershell", "cmd", "wmic", "vssadmin"]):
                    suspicious_events.append(f"Process: {proc_name}")
                elif "delete shadows" in cmd_line:
                    suspicious_events.append("CRITICAL: Shadow deletion")
                elif "-enc" in cmd_line:
                    suspicious_events.append("Encoded command execution")

            elif event_type == "file":
                if event.details.get("entropy_after", 0) > 7.5:
                    suspicious_events.append("File encryption detected")

            elif event_type == "network":
                if event.details.get("bytes_sent", 0) > 100000000:
                    suspicious_events.append("Large data exfiltration")

        # Format summary
        lines = []
        lines.append(f"Events: {event_types}")

        if suspicious_events:
            lines.append("Suspicious:")
            for evt in suspicious_events[:5]:
                lines.append(f"  ! {evt}")

        return '\n'.join(lines)


def create_simple_examples(output_file: str, num_examples: int = 50):
    """Create simple hand-crafted examples for quick testing."""

    examples = []

    for i in range(num_examples):
        is_ransomware = i % 3 != 2

        if is_ransomware:
            # Ransomware example
            prompt = """<|system|>
You are a ransomware detection expert. Analyze telemetry and predict threats.
</system>

<|telemetry|>
[Window 1]
Events: {'process': 2, 'file': 1}
Suspicious:
  ! Process: powershell
  ! Encoded command execution

[Window 2]
Events: {'process': 3, 'network': 1}
Suspicious:
  ! Process: cmd
  ! Process: wmic

[Window 3]
Events: {'process': 2, 'file': 10}
Suspicious:
  ! Process: vssadmin
  ! CRITICAL: Shadow deletion
</telemetry>

<|task|>
Based on the telemetry above:
1. What is the current state?
2. What will happen next?
3. What actions should be taken?
</task>"""

            completion = """<|analysis|>

## Current State
State: defense_evasion
Threat Level: CRITICAL
Attack Phase: Preparing for encryption

## Situation
Ransomware preparing for encryption phase. Shadow copies deleted to prevent recovery.

## Evidence
- Shadow copy deletion detected
- Encoded PowerShell execution
- System enumeration activity

## Prediction
Next Action: Mass file encryption
Time to Impact: 2-5 minutes

## Recommended Actions
- Isolate system immediately
- Kill suspicious processes
- Activate incident response

</analysis>"""

        else:
            # Benign example
            prompt = """<|system|>
You are a ransomware detection expert. Analyze telemetry and predict threats.
</system>

<|telemetry|>
[Window 1]
Events: {'process': 1, 'file': 3}
No suspicious activity

[Window 2]
Events: {'process': 2, 'network': 2}
No suspicious activity

[Window 3]
Events: {'file': 5}
No suspicious activity
</telemetry>

<|task|>
Based on the telemetry above:
1. What is the current state?
2. What will happen next?
3. What actions should be taken?
</task>"""

            completion = """<|analysis|>

## Current State
State: benign
Threat Level: LOW
Attack Phase: No attack

## Situation
Normal system activity observed with routine user operations.

## Evidence
- No suspicious indicators detected

## Prediction
Next Action: Continue normal work
Time to Impact: None

## Recommended Actions
- Continue routine monitoring
- No immediate action required

</analysis>"""

        examples.append({
            "prompt": prompt,
            "completion": completion
        })

    # Save examples
    with open(output_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Created {len(examples)} simple examples in {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for LLM fine-tuning")
    parser.add_argument("--output", default="ransomware_training.jsonl",
                       help="Output file for training data")
    parser.add_argument("--sequences", type=int, default=50,
                       help="Number of sequences to generate")
    parser.add_argument("--simple", action="store_true",
                       help="Generate simple hand-crafted examples")

    args = parser.parse_args()

    if args.simple:
        # Quick simple examples
        create_simple_examples(args.output, 50)
    else:
        # Full telemetry-based examples
        preparer = TrainingDataPreparer()
        preparer.generate_training_data(
            output_file=args.output,
            num_sequences=args.sequences
        )


if __name__ == "__main__":
    main()