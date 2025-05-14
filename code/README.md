## What I have working

- **Telemetry simulator**

  Generates raw log windows (process, file, network events, plus simple
  entropy / rename features).  Benign and blitz‑style ransomware
  behaviours are included.  Script: `code/telemetry_sim.py`. It's very
  simple right now -- making this more realistic is one of the most
  important next steps, as my approach will be data-driven.

- **Fine‑Tune a small LLM**

  Initially, I'm starting with a tiny 1.1 B‑parameter model (TinyLlama)
  fine‑tuned with 4‑bit LoRA on my 12 GB GPU. Script: `code/finetune_llm.py`.
  The model sees two things: (1) the raw telemetry window (as text) and
  (2) a supervised Q/A pair such as “Q: Is there suspicious activity?
  What is your evidence? A: Yes, I have seen evidence of exfiltration.
  Here are the processes that were involved: ...”  The model is fine-tuned
  to predict both the telemetry and the Q/A pair.

  I'm fine-tuning on a simulated data set right now, but I won't be able to
  look at the results for a few days.

## Why this implicit approach

Instead of building an explicit HMM or DBN I let the LLM learn its own
representation (from the telemetry simulator and Q/A pairs).

- Each (embedded) telemetry window + Q/A pairs are treated as the
  **emission** from the state (both latent and observable) of the system
  (e.g., whether there is suspicious activity, etc).

- We avoid hand‑crafting variables (Bitter Lesson) and
  have the model to learn its own representation. This is why
  the telemetry simulator is so important.

## Longer term (after I’m back)

- Also pretrain on text from a general knowledge base about ransomware
  (e.g., MITRE ATT&CK) to help the model predict known behaviors of
  ransomware. This can be part of an unsupervised pretraining step. Ablation
  studies may be conducted to see if this step helps.

- Experiment with a retrieval layer so the model can pull older
  telemetry when a single window is not enough. Then, we can add slower
  "sleeper" ransomware behaviour to test long‑horizon cases.

- Train a **small policy head** on top of the LLM embeddings so we have a
  lightweight, fixed action space. We can get very creative with this,
  e.g., treating it as a multi‑armed bandit problem that learns to
  act as a prompt engineer with ransomware expertise so that it can take
  actions to get more information (like more detailed logs about some
  aspect of the system), and so on.

