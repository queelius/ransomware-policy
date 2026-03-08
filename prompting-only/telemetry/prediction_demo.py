#!/usr/bin/env python3
"""
Demonstration of what the LLM actually learns to predict.
Shows the difference between simple next-token and our multi-layer approach.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

class PredictionDemo:
    """Demonstrate the multi-layer prediction approach."""

    def show_simple_next_token_prediction(self):
        """What a simple next-token model would learn."""
        print("=" * 60)
        print("SIMPLE NEXT-TOKEN PREDICTION (Shallow Pattern Matching)")
        print("=" * 60)

        context = [
            "process: powershell.exe",
            "process: cmd.exe",
            "process: whoami.exe"
        ]

        print("\nContext (last 3 events):")
        for event in context:
            print(f"  {event}")

        print("\nSimple model predicts next token:")
        print("  process: net.exe")  # Just statistical correlation
        print("\nThat's it! No understanding of WHY or WHAT IT MEANS")

    def show_our_multilayer_prediction(self):
        """What our model learns to predict."""
        print("\n" + "=" * 60)
        print("OUR MULTI-LAYER PREDICTION (Deep Understanding)")
        print("=" * 60)

        # Same context, but rich representation
        context_windows = [
            {
                "window_id": 1,
                "events": [
                    {"type": "process", "name": "winword.exe", "parent": "explorer.exe"}
                ]
            },
            {
                "window_id": 2,
                "events": [
                    {"type": "process", "name": "powershell.exe", "parent": "winword.exe",
                     "command": "powershell -enc SGVsbG8gV29ybGQ="}
                ]
            },
            {
                "window_id": 3,
                "events": [
                    {"type": "process", "name": "cmd.exe", "command": "whoami /all"},
                    {"type": "process", "name": "cmd.exe", "command": "net user"}
                ]
            }
        ]

        print("\nContext (3 telemetry windows):")
        for window in context_windows:
            print(f"\n  Window {window['window_id']}:")
            for event in window['events']:
                print(f"    {event['type']}: {event['name']} | {event.get('command', '')}")

        print("\n" + "-" * 40)
        print("Our model predicts 7 LAYERS of information:")
        print("-" * 40)

        predictions = self.generate_multilayer_prediction(context_windows)

        for layer_name, layer_content in predictions.items():
            print(f"\n{layer_name}:")
            if isinstance(layer_content, dict):
                for key, value in layer_content.items():
                    print(f"  {key}: {value}")
            elif isinstance(layer_content, list):
                for item in layer_content:
                    print(f"  - {item}")
            else:
                print(f"  {layer_content}")

    def generate_multilayer_prediction(self, context_windows: List[Dict]) -> Dict:
        """Generate all prediction layers."""

        return {
            "LAYER 1 - HIDDEN STATE INFERENCE": {
                "current_state": "reconnaissance",
                "previous_state": "initial_compromise",
                "confidence": "85%",
                "evidence": "PowerShell from Word + system enumeration"
            },

            "LAYER 2 - SEMANTIC UNDERSTANDING": (
                "A malicious document spawned PowerShell with an encoded command, "
                "indicating initial compromise via phishing. The attacker is now "
                "performing system reconnaissance to map the environment."
            ),

            "LAYER 3 - CAUSAL REASONING": [
                "Word→PowerShell: Malicious macro execution",
                "Encoded command: Attempting to evade detection",
                "whoami/net user: Attacker orienting in environment",
                "Sequence timing: Human-operated, not automated"
            ],

            "LAYER 4 - FUTURE TELEMETRY PREDICTION": {
                "next_5_min": [
                    "process: mimikatz.exe OR process: lsass.exe access",
                    "file: write C:\\Windows\\Temp\\[random].dat",
                    "network: SMB to internal IPs"
                ],
                "next_10_min": [
                    "process: vssadmin.exe delete shadows",
                    "registry: Disable Windows Defender",
                    "network: Large outbound transfer"
                ],
                "next_30_min": [
                    "file: Mass encryption (*.docx → *.locked)",
                    "file: Ransom note creation",
                    "process: Self-deletion"
                ]
            },

            "LAYER 5 - RISK ASSESSMENT": {
                "current_risk": "HIGH",
                "trending": "INCREASING",
                "time_to_impact": "20-30 minutes",
                "attack_stage": "Early (2 of 7)",
                "threat_actor_profile": "LockBit affiliate (75% confidence)"
            },

            "LAYER 6 - ACTIONABLE INTELLIGENCE": [
                "IMMEDIATE: Isolate host from network",
                "Kill PowerShell process (PID from telemetry)",
                "Check email for initial vector",
                "Monitor accounts that were enumerated",
                "Prepare for credential theft attempts"
            ],

            "LAYER 7 - UNCERTAINTY & MISSING CONTEXT": {
                "alternative_hypotheses": [
                    "30%: Penetration test (check with security team)",
                    "10%: Legitimate admin script (verify with user)",
                    "60%: Ransomware attack in progress"
                ],
                "what_we_need": [
                    "Email that delivered the document",
                    "Memory dump of PowerShell",
                    "Network packet captures",
                    "Other hosts with similar activity"
                ]
            }
        }

    def show_prediction_cascade(self):
        """Show how predictions evolve over time."""
        print("\n" + "=" * 60)
        print("PREDICTION CASCADE: How Confidence Evolves")
        print("=" * 60)

        timeline = [
            {
                "time": "T+0",
                "observed": "powershell -enc [base64]",
                "predicted_state": "possible_compromise",
                "confidence": 40,
                "predicted_next": "system commands OR normal activity"
            },
            {
                "time": "T+2",
                "observed": "whoami, net user",
                "predicted_state": "reconnaissance",
                "confidence": 75,
                "predicted_next": "credential theft tools"
            },
            {
                "time": "T+5",
                "observed": "mimikatz.exe",
                "predicted_state": "credential_theft",
                "confidence": 95,
                "predicted_next": "vssadmin delete (HIGH CONFIDENCE)"
            },
            {
                "time": "T+8",
                "observed": "vssadmin delete shadows",
                "predicted_state": "pre_encryption",
                "confidence": 99,
                "predicted_next": "ENCRYPTION IN 2-5 MINUTES!"
            }
        ]

        for step in timeline:
            print(f"\n{step['time']} - Observed: {step['observed']}")
            print(f"  → State: {step['predicted_state']} (confidence: {step['confidence']}%)")
            print(f"  → Predicting: {step['predicted_next']}")

    def compare_predictions_with_ground_truth(self):
        """Show how our predictions compare to actual outcomes."""
        print("\n" + "=" * 60)
        print("VALIDATION: Predictions vs Reality")
        print("=" * 60)

        test_cases = [
            {
                "context": "PowerShell encoded + whoami",
                "our_prediction": {
                    "next_event": "Credential theft",
                    "time": "5-10 min",
                    "confidence": 85
                },
                "simple_prediction": "net.exe",
                "actual_outcome": "mimikatz.exe at T+7min",
                "our_accuracy": "✓ Correct (event and timing)",
                "simple_accuracy": "✗ Wrong"
            },
            {
                "context": "vssadmin delete + bcdedit",
                "our_prediction": {
                    "next_event": "Mass file encryption",
                    "time": "1-3 min",
                    "confidence": 95
                },
                "simple_prediction": "wmic.exe",
                "actual_outcome": "Encryption started at T+2min",
                "our_accuracy": "✓ Correct (critical alert!)",
                "simple_accuracy": "✗ Missed critical moment"
            }
        ]

        for i, test in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test['context']}")
            print(f"  Simple Model: {test['simple_prediction']}")
            print(f"  Our Model: {test['our_prediction']['next_event']} "
                  f"in {test['our_prediction']['time']} "
                  f"(conf: {test['our_prediction']['confidence']}%)")
            print(f"  Reality: {test['actual_outcome']}")
            print(f"  Simple: {test['simple_accuracy']}")
            print(f"  Ours: {test['our_accuracy']}")

    def show_training_example(self):
        """Show actual training data format."""
        print("\n" + "=" * 60)
        print("ACTUAL TRAINING EXAMPLE")
        print("=" * 60)

        training_example = {
            "input": {
                "telemetry_windows": [
                    {"events": ["powershell.exe -enc", "connection to 185.220.101.45"]},
                    {"events": ["whoami", "net user", "net localgroup"]},
                    {"events": ["taskist", "netstat", "systeminfo"]}
                ]
            },
            "output_layers": {
                "1_state": "reconnaissance transitioning to credential_theft",
                "2_meaning": "Attacker completed initial recon, preparing credential access",
                "3_indicators": ["encoded PS", "C2 contact", "user enum", "systematic discovery"],
                "4_next_events": ["mimikatz.exe", "lsass dump", "SAM access"],
                "5_timeline": "5-10min to cred theft, 20-30min to encryption",
                "6_risk": "HIGH->CRITICAL",
                "7_actions": ["Isolate NOW", "Kill PS process", "Audit enumerated accounts"]
            }
        }

        print("\nTraining Input:")
        print(json.dumps(training_example["input"], indent=2))

        print("\nModel Learns to Output (7 layers):")
        print(json.dumps(training_example["output_layers"], indent=2))

        print("\n" + "-" * 40)
        print("Key Point: The model learns to predict ALL these layers")
        print("simultaneously, creating deep understanding from telemetry!")


def main():
    """Run the demonstration."""
    demo = PredictionDemo()

    # Show the difference
    demo.show_simple_next_token_prediction()
    demo.show_our_multilayer_prediction()

    # Show evolution over time
    demo.show_prediction_cascade()

    # Validate predictions
    demo.compare_predictions_with_ground_truth()

    # Show training format
    demo.show_training_example()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Our approach goes FAR beyond simple next-token prediction:

1. Simple next-token: "After A comes B" (pattern matching)

2. Our multi-layer:
   - WHY does B follow A? (causal understanding)
   - WHAT does this mean? (semantic interpretation)
   - WHEN will impact occur? (temporal reasoning)
   - HOW confident are we? (uncertainty quantification)
   - WHAT should we do? (actionable intelligence)

The LLM learns to be an expert analyst, not just a pattern matcher!
    """)


if __name__ == "__main__":
    main()