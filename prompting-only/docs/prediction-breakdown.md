# What the LLM Learns to Predict: Beyond Next-Token

## The Key Insight

Instead of just predicting "the next telemetry event" (which would be surface-level pattern matching), our LLM learns to predict **multiple interconnected layers of understanding** about what's happening.

## Training Example Breakdown

Let's walk through exactly what the LLM sees and predicts:

### What the LLM Sees (Input)

```
<|telemetry_context|>
[Window 1: 16:00:00 to 16:02:00]
  [PROCESS] winword.exe | parent: explorer.exe
  [FILE] read C:\Users\user\invoice.docx

[Window 2: 16:02:00 to 16:04:00]
  [PROCESS] powershell.exe | parent: winword.exe | cmd: -enc SGVsbG8g...
  [NETWORK] outbound to 185.220.101.45:443 | 1024 bytes

[Window 3: 16:04:00 to 16:06:00]
  [PROCESS] cmd.exe | whoami /all
  [PROCESS] cmd.exe | net user
  [PROCESS] cmd.exe | net localgroup administrators
</|telemetry_context|>

TASK: What happens next and why?
```

### What the LLM Learns to Predict (Output)

The LLM predicts **SEVEN different layers** of information:

```
<|analysis|>

## 1. HIDDEN STATE INFERENCE (Not directly observable!)
True State: reconnaissance → credential_theft
Attack Phase: Moving from Discovery to Credential Access
Threat Actor: LockBit affiliate (inferred from TTPs)
Confidence: 85%

## 2. SEMANTIC UNDERSTANDING (Not just syntax!)
"A malicious document has spawned PowerShell, which connected to a C2 server.
The attacker is now enumerating the system to understand privileges and
identify targets. This is consistent with human-operated ransomware."

## 3. CAUSAL REASONING (Why this sequence?)
Key Indicators:
- Document → PowerShell = likely malicious macro
- Encoded command = obfuscation attempt
- whoami after C2 contact = attacker orientation
- User enumeration = preparing for lateral movement

## 4. FUTURE EVENT PREDICTION (Specific telemetry)
Next window will likely contain:
- [PROCESS] mimikatz.exe or lsass.exe access
- [FILE] write to C:\Windows\Temp\[random].exe
- [REGISTRY] HKLM\SAM access attempts
- [NETWORK] SMB connections to internal IPs

## 5. TEMPORAL REASONING (When and how fast?)
Next Likely Action: Credential dumping
Time to Impact: ~20 minutes
Progression Speed: Moderate (human-operated)

## 6. RISK ASSESSMENT (Not just classification!)
Risk Level: HIGH → CRITICAL (escalating)
Evidence: Completed initial access, active reconnaissance
Certainty: High - multiple corroborating indicators

## 7. ACTIONABLE INTELLIGENCE (What to do?)
Immediate Actions:
1. Isolate the host from network (lateral movement imminent)
2. Check if winword.exe is still running (kill if yes)
3. Audit accounts that were enumerated
4. Monitor for new service creation

What We Need But Don't Have:
- Email that delivered the document
- Memory dump of PowerShell process
- Network packet captures of C2 traffic
</|analysis|>
```

## Why This is Different from Simple Next-Token Prediction

### Simple Next-Token (What GPT would naturally learn):
```
Input: "[PROCESS] cmd.exe | whoami"
Output: "[PROCESS] cmd.exe | net"  # Just pattern matching
```

### Our Multi-Layer Prediction:
```
Input: "[PROCESS] cmd.exe | whoami"
Output:
  State: "reconnaissance"
  Meaning: "Attacker orienting after compromise"
  Risk: "HIGH - post-exploitation activity"
  Next: "Credential theft likely"
  Action: "Isolate immediately"
```

## The Training Process

### Step 1: Observable → Latent State
The LLM learns to infer hidden states from partial observations:

```python
def infer_state(telemetry):
    if "vssadmin delete" in telemetry:
        return "pre_encryption"  # High confidence
    elif "whoami" in telemetry and recent("powershell -enc"):
        return "reconnaissance"  # Medium confidence
    elif only_normal_processes(telemetry):
        return "benign"  # But could be wrong!
```

### Step 2: Latent State → Future Observables
The LLM learns state transitions:

```python
state_transitions = {
    "reconnaissance": {
        "next_states": ["credential_theft", "lateral_movement"],
        "observable_signs": ["mimikatz.exe", "lsass access", "SAM dumps"],
        "time_estimate": "5-10 minutes"
    }
}
```

### Step 3: Context → Explanation
The LLM learns to explain in plain English:

```python
def explain(telemetry, inferred_state):
    return f"""
    Based on {telemetry}, this appears to be {inferred_state}.
    The attacker has {completed_actions} and will likely {next_actions}.
    Risk: {calculate_risk(inferred_state, telemetry)}
    """
```

## Concrete Example: Prediction Cascade

Let's trace through a real prediction:

### T=0: LLM sees:
```
powershell.exe -enc [base64]
```

### T=1: LLM predicts:
```
STATE: initial_compromise
NEXT: System discovery commands
WHY: Encoded PS typically followed by recon
```

### T=2: LLM sees (validating prediction):
```
whoami, net user, systeminfo
```

### T=3: LLM predicts (with higher confidence):
```
STATE: reconnaissance → credential_theft
NEXT: mimikatz.exe OR lsass.exe dumping
WHY: Pattern matches LockBit playbook
TIME: 5-10 minutes to encryption
```

### T=4: LLM sees:
```
lsass.exe accessed by suspicious process
```

### T=5: LLM predicts (urgent):
```
STATE: credential_theft → pre_encryption
NEXT: vssadmin delete shadows
WHY: Credentials obtained, preparing for encryption
ACTION: ISOLATE NOW - 2-3 minutes to impact!
```

## What Makes This Powerful

1. **Causal Understanding**: The LLM learns WHY events happen in sequence, not just that they do

2. **Uncertainty Quantification**: It predicts confidence levels based on partial observability

3. **Multi-Scale Temporal Reasoning**:
   - Immediate: Next process spawn
   - Short-term: Next attack phase (5 min)
   - Long-term: Time to impact (30 min)

4. **Actionable Predictions**: Not just "ransomware=yes" but "isolate now because X will happen in Y minutes"

5. **Counterfactual Reasoning**: "If we see X next, it's ransomware. If we see Y, it might be penetration testing."

## Training Data Example

Here's actual training data showing the multi-layer prediction:

```json
{
  "context_windows": [/* 3 windows of telemetry */],
  "predict": {
    // Layer 1: Raw telemetry prediction
    "next_telemetry_events": [
      {"type": "process", "name": "vssadmin.exe", "cmd": "delete shadows"}
    ],

    // Layer 2: State inference
    "latent_state": "defense_evasion",
    "confidence": 0.85,

    // Layer 3: Semantic explanation
    "explanation": "Ransomware preparing for encryption by removing recovery options",

    // Layer 4: Risk assessment
    "risk": "CRITICAL",
    "time_to_impact": 5,

    // Layer 5: Recommended actions
    "actions": ["Isolate", "Kill process", "Backup critical files NOW"]
  }
}
```

## Evaluation: How We Know It's Learning Deep Patterns

We can test if the LLM truly understands by:

1. **Counterfactual Testing**: Change one event - does prediction change appropriately?
2. **Novel Ransomware**: Test on unseen families - does it recognize patterns?
3. **Explanation Quality**: Are explanations accurate and useful?
4. **Time Predictions**: How accurate are time-to-impact estimates?
5. **Action Effectiveness**: Do recommended actions actually help?

## The Magic: Implicit Dynamic Bayes Net

The LLM essentially learns an implicit Dynamic Bayes Net:

```
P(State_t | Telemetry_1:t) → P(Telemetry_t+1 | State_t) → P(Action | State_t)
```

Without explicitly programming this, the LLM learns:
- State transitions
- Emission probabilities
- Optimal actions

This is far beyond simple next-token prediction - it's learning a complete model of ransomware behavior from partial observations!