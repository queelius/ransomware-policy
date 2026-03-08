#!/usr/bin/env python3
"""
Fine-tuning format generator for training LLMs on ransomware telemetry.
The LLM learns to predict both future telemetry and expert explanations.
"""

import json
from typing import List, Dict, Any, Optional
from comprehensive_telemetry_gen import (
    ComprehensiveTelemetryGenerator,
    TelemetryWindow
)

class FineTuningFormatter:
    """Format telemetry for LLM fine-tuning."""

    def __init__(self):
        self.generator = ComprehensiveTelemetryGenerator(seed=42)

    def create_training_example(
        self,
        telemetry_windows: List[TelemetryWindow],
        context_windows: int = 3
    ) -> Dict[str, str]:
        """
        Create a training example where the LLM:
        1. Sees N windows of telemetry (context)
        2. Predicts the next window's telemetry
        3. Explains what's happening
        4. Answers expert questions
        """

        if len(telemetry_windows) < context_windows + 1:
            raise ValueError(f"Need at least {context_windows + 1} windows")

        # Context: what the LLM sees
        context = telemetry_windows[:context_windows]

        # Target: what the LLM should predict
        target = telemetry_windows[context_windows]

        # Format the prompt (input)
        prompt = self._format_prompt(context)

        # Format the completion (what LLM should generate)
        completion = self._format_completion(target, context)

        return {
            "prompt": prompt,
            "completion": completion
        }

    def _format_prompt(self, context_windows: List[TelemetryWindow]) -> str:
        """Format the input prompt with telemetry context."""

        prompt_parts = []

        # System instruction
        prompt_parts.append(
            "<|system|>\n"
            "You are an advanced ransomware detection system. Analyze telemetry data to:\n"
            "1. Predict what will happen next\n"
            "2. Explain the current situation\n"
            "3. Assess the risk level\n"
            "4. Recommend defensive actions\n"
            "</|system|>\n"
        )

        # Telemetry context
        prompt_parts.append("<|telemetry_context|>")

        for i, window in enumerate(context_windows):
            prompt_parts.append(f"\n[Window {i+1}: {window.start_time} to {window.end_time}]")

            # Show observable events
            for event in window.observable_events[:20]:  # Limit for context length
                event_str = self._format_event(event)
                prompt_parts.append(event_str)

            # Add statistical summary
            stats = self._compute_window_stats(window)
            prompt_parts.append(f"Stats: {stats}")

        prompt_parts.append("</|telemetry_context|>\n")

        # Task instruction
        prompt_parts.append(
            "<|task|>\n"
            "Based on the telemetry above:\n"
            "1. What will likely happen in the next 2 minutes?\n"
            "2. What is currently happening?\n"
            "3. What is the risk level?\n"
            "4. What actions should be taken?\n"
            "</|task|>\n"
        )

        return "\n".join(prompt_parts)

    def _format_completion(
        self,
        target_window: TelemetryWindow,
        context_windows: List[TelemetryWindow]
    ) -> str:
        """Format what the LLM should generate."""

        completion_parts = []

        # Start with analysis marker
        completion_parts.append("<|analysis|>")

        # 1. Current State Assessment (from latent context)
        completion_parts.append("\n## Current State Assessment")
        completion_parts.append(f"True State: {target_window.latent_context.true_state.value}")
        completion_parts.append(f"Attack Phase: {target_window.latent_context.attack_phase}")
        completion_parts.append(f"Threat Actor: {target_window.latent_context.threat_actor}")
        completion_parts.append(f"Confidence: {target_window.latent_context.confidence:.1%}")

        # 2. Situation Explanation (from expert annotation)
        completion_parts.append("\n## Situation Explanation")
        completion_parts.append(target_window.expert_annotation.plain_english_summary)

        # 3. Key Indicators
        completion_parts.append("\n## Key Indicators Observed")
        for indicator in target_window.expert_annotation.key_indicators:
            completion_parts.append(f"- {indicator}")

        # 4. Risk Assessment
        completion_parts.append("\n## Risk Assessment")
        completion_parts.append(f"Risk Level: {target_window.expert_annotation.risk_assessment.upper()}")
        completion_parts.append(target_window.expert_annotation.confidence_explanation)

        # 5. Predicted Next Events
        completion_parts.append("\n## Predicted Next Events")
        completion_parts.append(f"Next Likely Action: {target_window.latent_context.next_likely_action}")
        if target_window.latent_context.time_to_impact:
            completion_parts.append(f"Time to Impact: {target_window.latent_context.time_to_impact} minutes")

        # Predict some actual telemetry events
        completion_parts.append("\nExpected telemetry in next window:")
        for event in target_window.observable_events[:5]:  # Show first 5 events
            completion_parts.append(f"- {self._format_event_prediction(event)}")

        # 6. Recommended Actions
        completion_parts.append("\n## Recommended Actions")
        for action in target_window.expert_annotation.recommended_actions:
            completion_parts.append(f"- {action}")

        # 7. Q&A Pairs
        completion_parts.append("\n## Expert Q&A")
        for qa in target_window.expert_annotation.qa_pairs:
            completion_parts.append(f"Q: {qa['question']}")
            completion_parts.append(f"A: {qa['answer']}\n")

        # 8. Missing Context (what we wish we knew)
        completion_parts.append("\n## Additional Context Needed")
        for missing in target_window.expert_annotation.missing_context:
            completion_parts.append(f"- {missing}")

        completion_parts.append("</|analysis|>")

        return "\n".join(completion_parts)

    def _format_event(self, event) -> str:
        """Format a single event for display."""
        if event.event_type == "process":
            return f"  [PROCESS] {event.details.get('process_name', 'unknown')} | {event.details.get('command_line', '')[:100]}"
        elif event.event_type == "file":
            path = event.details.get('path', 'unknown')
            op = event.details.get('operation', 'unknown')
            entropy = event.details.get('entropy_after', 0)
            return f"  [FILE] {op} {path} | entropy: {entropy:.2f}"
        elif event.event_type == "network":
            return f"  [NETWORK] {event.details.get('direction', '')} to {event.details.get('destination_ip', '')}:{event.details.get('destination_port', '')} | {event.details.get('bytes_sent', 0)} bytes"
        elif event.event_type == "registry":
            return f"  [REGISTRY] {event.details.get('operation', '')} {event.details.get('key', '')}"
        else:
            return f"  [{event.event_type.upper()}] {str(event.details)[:100]}"

    def _format_event_prediction(self, event) -> str:
        """Format predicted future event."""
        if event.event_type == "process":
            return f"Process execution: {event.details.get('process_name', 'unknown')}"
        elif event.event_type == "file":
            if event.details.get('entropy_after', 0) > 7.5:
                return f"File encryption: {event.details.get('path', 'unknown')}"
            else:
                return f"File {event.details.get('operation', 'access')}: {event.details.get('path', 'unknown')}"
        elif event.event_type == "network":
            return f"Network connection to {event.details.get('destination_ip', 'unknown')}"
        else:
            return f"{event.event_type.capitalize()} activity expected"

    def _compute_window_stats(self, window: TelemetryWindow) -> str:
        """Compute statistics for a window."""
        stats = {
            "total_events": len(window.observable_events),
            "process_events": sum(1 for e in window.observable_events if e.event_type == "process"),
            "file_events": sum(1 for e in window.observable_events if e.event_type == "file"),
            "network_events": sum(1 for e in window.observable_events if e.event_type == "network"),
            "high_entropy_files": sum(1 for e in window.observable_events
                                     if e.event_type == "file" and e.details.get("entropy_after", 0) > 7.5),
            "suspicious_processes": sum(1 for e in window.observable_events
                                       if e.event_type == "process" and
                                       any(p in str(e.details.get("process_name", "")).lower()
                                          for p in ["powershell", "cmd", "vssadmin", "wmic"]))
        }
        return json.dumps(stats)

    def create_conversation_format(
        self,
        telemetry_windows: List[TelemetryWindow]
    ) -> List[Dict[str, str]]:
        """
        Create conversational training format where the LLM learns through dialog.
        """

        conversation = []

        # Initial context
        conversation.append({
            "role": "system",
            "content": (
                "You are a ransomware detection expert analyzing real-time telemetry. "
                "Explain what you observe, predict what will happen next, and recommend actions. "
                "Be specific about indicators and confident in assessments."
            )
        })

        # Process windows in sequence
        for i, window in enumerate(telemetry_windows):
            # User shows telemetry
            telemetry_summary = self._summarize_window(window)
            conversation.append({
                "role": "user",
                "content": f"Telemetry Window {i+1}:\n{telemetry_summary}\n\nWhat's happening?"
            })

            # Assistant analyzes
            analysis = self._generate_analysis(window)
            conversation.append({
                "role": "assistant",
                "content": analysis
            })

            # Follow-up questions
            for qa in window.expert_annotation.qa_pairs[:2]:  # First 2 Q&As
                conversation.append({
                    "role": "user",
                    "content": qa["question"]
                })
                conversation.append({
                    "role": "assistant",
                    "content": qa["answer"]
                })

        return conversation

    def _summarize_window(self, window: TelemetryWindow) -> str:
        """Create concise summary of telemetry window."""
        lines = []

        # Time range
        lines.append(f"Time: {window.start_time} to {window.end_time}")

        # Event counts by type
        event_types = {}
        for event in window.observable_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        lines.append(f"Events: {event_types}")

        # Key events (suspicious ones)
        suspicious = []
        for event in window.observable_events[:10]:  # Check first 10
            if event.event_type == "process":
                if any(s in str(event.details.get("command_line", "")).lower()
                      for s in ["vssadmin", "bcdedit", "powershell -enc"]):
                    suspicious.append(f"SUSPICIOUS: {event.details.get('command_line', '')[:50]}")
            elif event.event_type == "file":
                if event.details.get("entropy_after", 0) > 7.5:
                    suspicious.append(f"ENCRYPTION: {event.details.get('path', '')}")

        if suspicious:
            lines.append("Notable events:")
            lines.extend(suspicious[:3])  # Show up to 3

        return "\n".join(lines)

    def _generate_analysis(self, window: TelemetryWindow) -> str:
        """Generate expert analysis response."""
        parts = []

        # State assessment
        state = window.latent_context.true_state.value
        if state == "benign":
            parts.append("✓ Normal activity detected. No immediate threats identified.")
        elif state == "encryption_active":
            parts.append("🚨 CRITICAL: Active ransomware encryption in progress!")
        else:
            parts.append(f"⚠️ WARNING: Suspicious activity detected - {window.latent_context.attack_phase} phase")

        # Key findings
        parts.append(f"\n{window.expert_annotation.plain_english_summary}")

        # Indicators
        if window.expert_annotation.key_indicators:
            parts.append("\nKey indicators:")
            for ind in window.expert_annotation.key_indicators[:3]:
                parts.append(f"• {ind}")

        # Risk and recommendation
        parts.append(f"\nRisk level: {window.expert_annotation.risk_assessment.upper()}")
        parts.append(f"Immediate action: {window.expert_annotation.recommended_actions[0]}")

        # Prediction
        if window.latent_context.next_likely_action:
            parts.append(f"\nPrediction: {window.latent_context.next_likely_action}")
            if window.latent_context.time_to_impact:
                parts.append(f"Time to impact: ~{window.latent_context.time_to_impact} minutes")

        return "\n".join(parts)


def generate_finetuning_dataset(
    output_file: str,
    num_sequences: int = 100,
    format_type: str = "completion"  # or "conversation"
):
    """Generate complete fine-tuning dataset."""

    formatter = FineTuningFormatter()
    generator = ComprehensiveTelemetryGenerator(seed=42)

    training_examples = []

    for i in range(num_sequences):
        # Generate telemetry sequence
        scenario = "benign" if i % 3 == 0 else "ransomware_attack"
        windows = generator.generate_telemetry_sequence(
            num_windows=5,
            scenario=scenario
        )

        if format_type == "completion":
            # Create training examples from sliding windows
            for j in range(len(windows) - 3):
                example = formatter.create_training_example(
                    windows[j:j+4],  # 3 context + 1 target
                    context_windows=3
                )
                training_examples.append(example)

        elif format_type == "conversation":
            # Create conversational format
            conversation = formatter.create_conversation_format(windows)
            training_examples.append({"messages": conversation})

    # Save dataset
    with open(output_file, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')

    print(f"Generated {len(training_examples)} training examples")
    print(f"Format: {format_type}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate fine-tuning data")
    parser.add_argument("--output", default="finetune_data.jsonl")
    parser.add_argument("--sequences", type=int, default=10)
    parser.add_argument("--format", choices=["completion", "conversation"], default="completion")

    args = parser.parse_args()

    generate_finetuning_dataset(
        output_file=args.output,
        num_sequences=args.sequences,
        format_type=args.format
    )