# End‑to‑End Ransomware‑Aware LLM Pipeline — Stage I Plan

---

## 1  Overview

Establish an **end‑to‑end learnable pipeline** that ingests synthetic host telemetry, produces rich natural‑language analyses, and outputs defender actions.  Core novelty: treat the system as an *implicit* Dynamic Bayes Net learned from data, with the LLM’s latent embedding playing the role of hidden state.

## 2  Motivation & Key Questions

1. **Can an LLM trained only on synthetic logs + domain text generalise to unseen ransomware tactics?**
2. **Does auxiliary telemetry reconstruction improve detection latency and explanation quality?**
3. **How well does a small policy head vs. a prompted LLM handle action selection?**

## 3  Theoretical Foundations (1‑paragraph summary)

* Factorisation inherits the DBN structure $P(X_t|X_{t-1})P(Z_t|X_t)$ but replaces conditionals with neural estimators.
* Training ≈ amortised variational filtering + supervised/ RL heads.
* Multi‑task loss mixes telemetry MLM, QA generation, and action cross‑entropy.

## 4  Synthetic Data Framework

| Component                 | Status                                                        | Next TODO                                     |
| ------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| **Telemetry schema**      | v0.1 in `telemetry_sim.py` (process, file, net, signal feats) | add registry + DLL injection events           |
| **Goodware generators**   | basic Office + backup                                         | diversify (browser, antivirus)                |
| **Ransomware generators** | blitz encryptor                                               | slow sleeper, exfil‑first, privilege‑escalate |
| **Rolling windows**       | 120 s window, 30 s stride                                     | parameter sweep for window sizes              |
| **QA templates**          | simple ‘Is ransomware?’ pair                                  | add action + evidence prompts                 |

## 4.1  Future memory/RAG layer (stage II)

* **Motivation** – sliding 120 s windows can miss stealthy, long‑horizon ransomware. A retrieval‑augmented memory (RAG) tier will later feed the LLM with compact summaries from the distant past.
* **Design sketch** – each JSON window will gain a `MEMORY` block holding *k* ≈ 3 high‑scoring historical summaries (selected heuristically at first, then by a learned key‑selector head).  The prompt becomes `WINDOW + MEMORY + QUESTION`.
* **Stage I scope** – **deferred**.  For this phase we simulate only adversaries whose tell‑tale evidence appears within a single 120 s window so that the baseline model’s capacity is sufficient.

## 5  Model Pipeline  Model Pipeline

1. **Domain pretraining**
   ‑ Masked‑LM on ATT\&CK articles, API docs, public IR reports.
2. **Telemetry MLM**
   ‑ Phase 0: train on raw synthetic windows (λ\_tel = 1).
3. **Supervised fine‑tune**
   ‑ Add QA + action heads (λ\_tel≈0.3, λ\_qa=λ\_act≈0.7).
4. **Optional RL phase**
   ‑ Scripted attacker ↔ frozen defender to start; later co‑evolution.

## 6  Evaluation

| Metric                           | Purpose                      |
| -------------------------------- | ---------------------------- |
| **Time‑to‑first‑correct‑alert**  | Detection latency            |
| **Precision / Recall / F1**      | Action correctness           |
| **Explanation BLEU / ROUGE**     | QA faithfulness              |
| **Perplexity on benign windows** | Generalisation & calibration |

Baselines: logistic‑regression on handcrafted features; small MLP on same synthetic features.

## 7  Milestones

| Week | Goal                                                    |
| ---- | ------------------------------------------------------- |
| 1    | Finalise telemetry schema & richer generators           |
| 2    | Generate 100 k pretrain windows; run domain pretraining |
| 3    | Produce 10 k labelled fine‑tune windows with QA/action  |
| 4    | Fine‑tune LLM + evaluate baseline metrics               |
| 5    | Introduce slow‑sleeper ransomware; test OOD detection   |
| 6    | Draft first internal report & demo for advisors         |

## 8  Risks & Mitigations

* **Simulator bias → poor OOD perf**
  → inject noise variants, small real traces.
* **Compute cost** (MLM on 100 k windows)
  → use LoRA adapters; prune model if needed.
* **Overfitting QA templates**
  → generate paraphrased queries via LLM augmentation.

## 10  Deliverables

1. **Telemetry simulator code** (ready: `telemetry_sim.py`).
2. **Dataset sample** (1 k windows, JSONL).
3. **Short slide‑deck** summarising theory & early metrics.
4. **Paper draft** maybe we'll already have enough for a short paper at end of this?
