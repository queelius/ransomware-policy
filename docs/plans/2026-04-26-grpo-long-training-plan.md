# GRPO Long-Training Research Plan

Date: 2026-04-26
Status: Pending. To be executed on remote GPU.

## The empirical question

**Does GRPO ever produce a meaningful improvement over the SFT-only baseline on this task, given enough training steps?**

Existing 700-step run shows +8 percentage points over base 1.7B on overlapping seeds (52.0% vs 44.0%), but the gain is entirely from FP reduction. False-negative rate is essentially unchanged (88.5% vs 84.6%). The asymmetric reward (-2.0 for missed ransomware) has not yet pressured the policy toward catching attacks. The question is whether longer training closes that gap, or whether the 1.7B base lacks the capacity.

Secondary questions answered along the way:

- Does the redesigned reward (post `(phase7)` format-hack hardening) produce a different learning trajectory than the gameable pre-fix reward?
- Where does the learning curve flatten? What is the GRPO ceiling on 1.7B before scaling to Qwen3.5-9B?
- Which scenario types are improving with training, which are stuck?

## State at the start of this run

- 408 tests passing (`python -m pytest tests/ -q` from `active-detective/`)
- Pre-training audit fully landed (commits `(phase1)` through `(phase9)` in `2026-04-13-comprehensive-pretraining-fixes.md`)
- Reward function redesigned post-audit (`67450b1` hardens `format_reward`, `1aade6f` aligns tests). All existing checkpoints predate this redesign.
- Best existing checkpoint: `checkpoints-grpo-sft-1.7b/` at 700 steps, trained on Qwen3-1.7B + SFT adapter from `../fine-tuning/checkpoints-sft-1.7b/final`
- Eval results landed in `active-detective/results/` (see commit `5940ba7` for provenance)

### What the 700-step checkpoint tells us

On 50 overlapping seeds vs base 1.7B (no tools):

| Metric | Base 1.7B | GRPO+SFT+tools (700 step) | Delta |
|--------|-----------|---------------------------|-------|
| Accuracy | 44.0% | 52.0% | +8.0pp |
| FN rate | 84.6% | 88.5% | +3.9pp (worse) |
| Tools/episode | 0 | 4.58 | +4.58 |
| Cost | 0 | -0.137 | spending |

Investigation behavior is real (4.58 tools/ep). Verdict-from-evidence is not. The agent looks but does not update.

## Why fresh run, not resume

The 700-step checkpoint was trained on the pre-`(phase7)` reward, where `format_reward` could be gamed (emit `<think>` and `<tool_call>` once for +0.1 forever). The policy may have absorbed that exploit. Resuming with the redesigned reward risks two artifacts:
1. Initial policy whiplash as the reward shape shifts under it.
2. Inability to attribute "did GRPO work?" cleanly. Outcomes get confounded by reward-change dynamics.

A fresh run gives a clean signal answering the actual research question.

If GPU budget is tight, **Option B** in this doc resumes the existing checkpoint as a cheaper, noisier alternative.

## Plan A: Fresh GRPO run on redesigned reward (recommended)

### Config

```bash
# active-detective/scripts/run_grpo_overnight.sh, with --max-steps 200 → 3000
# and --output-dir → ./checkpoints-grpo-sft-1.7b-fresh

python -m training.train_grpo \
    --model Qwen/Qwen3-1.7B \
    --adapter ../fine-tuning/checkpoints-sft-1.7b/final \
    --output-dir ./checkpoints-grpo-sft-1.7b-fresh \
    --n-episodes 200 \
    --group-size 2 \
    --max-steps 3000 \
    --lr 5e-6 \
    --k-max 5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --no-unsloth \
    --disable-thinking \
    --seed 42
```

Notes:
- `--max-steps 3000` is roughly 4x the existing run. At ~8 hours per 200 steps on RTX 3060 12GB, this is ~5 days continuous. On A5000 expect 2-3x faster.
- Save every 25 steps (existing default in `training/train_grpo.py:72`). Auto-resume from latest checkpoint is wired (`train_grpo.py:663-673`); interrupting and re-running the same command continues from the most recent save.
- Same hyperparams as existing run for direct comparability. Hyperparam search is out of scope; the question is whether GRPO improves at all.

### Eval cadence

Pin a 200-seed eval set held out from training. Eval at fixed milestones:

```
Step  250 → results/grpo_fresh_step0250.jsonl
Step  500 → results/grpo_fresh_step0500.jsonl
Step 1000 → results/grpo_fresh_step1000.jsonl
Step 1500 → results/grpo_fresh_step1500.jsonl
Step 2000 → results/grpo_fresh_step2000.jsonl
Step 2500 → results/grpo_fresh_step2500.jsonl
Step 3000 → results/grpo_fresh_step3000.jsonl
```

Plus a one-time baseline eval of the SFT-only adapter (no GRPO) on the same pinned set as the lower-bound comparator: `results/sft_baseline_held_out.jsonl`.

### Held-out eval seed set

Generate once before training, reuse forever:

```bash
cd active-detective
python -c "
from training.scenarios import generate_training_scenarios, save_scenarios
# Use a seed pool disjoint from training (training uses seed=42 -> 0..N)
save_scenarios(generate_training_scenarios(200, base_seed=900000), 'eval_held_out_200.jsonl')
"
```

`base_seed=900000` ensures no overlap with the training scenario seeds (which start at 0). Verify by checking unique seeds in the file before launching training.

## Plan B: Resume the 700-step checkpoint (fallback if GPU-budget tight)

```bash
# Same command as Plan A but pointing at existing checkpoint dir.
# Auto-resume picks up at step 700.

python -m training.train_grpo \
    --model Qwen/Qwen3-1.7B \
    --adapter ../fine-tuning/checkpoints-sft-1.7b/final \
    --output-dir ./checkpoints-grpo-sft-1.7b \
    --n-episodes 200 \
    --group-size 2 \
    --max-steps 2000 \
    --lr 5e-6 \
    --k-max 5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --no-unsloth \
    --disable-thinking \
    --seed 42
```

Caveats: outcome will be confounded with the reward-redesign dynamics. Useful only as a "does the trajectory keep improving" probe before committing to Plan A.

## Eval methodology fixes (must do, regardless of A/B)

These are the changes that make any future result interpretable.

1. **Pin the eval seed set** (above). Use it for every checkpoint, every model, every comparison. The +8 vs +0 discrepancy in headline numbers vs overlapping seeds was caused by different scenario draws. This eliminates that.

2. **Track FN rate as the leading indicator**, not accuracy. The asymmetric reward (-2.0 FN, -1.0 FP) means FN is the failure mode the reward function is *trying* to drive down. If FN doesn't drop, the model isn't learning what we asked.

3. **Eval mid-training, not just at the end.** Fixed milestones above. The interesting question is not "is the final checkpoint better than base"; it is "what does the learning curve look like, and where does it inflect or plateau."

4. **Compute per-scenario-type breakdown** for every eval. `semantic_shuffle` and `slow_sleeper` require temporal-context reasoning; if they stay at 0% while `blitz` improves, that's a different problem than uniform stagnation.

## Decision tree on outcomes

| Observation at 3000 steps | Diagnosis | Next action |
|---------------------------|-----------|-------------|
| FN drops from ~80% to <50% | GRPO works on this task | Scale to Qwen3.5-9B base, repeat with same protocol |
| FN flat (>75%), FP keeps dropping | Reward asymmetry insufficient | Increase FN penalty to -3.0 or -4.0; rerun |
| Both flat after 1500 steps, no further improvement | 1.7B capacity bottleneck | Switch to Qwen3.5-9B; QLoRA still fits in 24GB |
| Per-type: `blitz` improves, `semantic_shuffle` stuck at 0% | Temporal reasoning isn't being exercised | Increase `--n-history` from 2 to 4; add curriculum that emphasizes `semantic_shuffle` |
| Reward goes up but eval accuracy flat | `format_reward` gaming returns | Inspect rollouts; tighten format checks in `environment/reward.py` |
| Mode collapse to "always benign" | Negative cost dominates verdict reward | Reduce per-tool costs by 50%; rerun |
| Mode collapse to "always ransomware" (FP=100%) | Format/efficiency bonuses outweigh verdict cost | Lower `EFFICIENCY_BONUS_PER_STEP` |

## What to log and report back

After each milestone eval, append a row to `results/grpo_fresh_progress.md`:

```
| Step | Acc | FN | FP | Tools/ep | Reward | benign | blitz | exfil | shuffle | sleeper |
| 0 (SFT) | ... |
| 250 | ... |
| 500 | ... |
| ... |
```

Plus the raw JSONL per checkpoint in `results/grpo_fresh_step*.jsonl`. Both committed to the repo so the home machine can pull and the analysis can pick up there.

Optional: log key metrics to wandb (already in `requirements.txt`). Helpful for live monitoring; not load-bearing.

## First session on the GPU box: checklist

```bash
# 1. Pull latest
cd ~/github/repos/ransomware-policy   # or wherever cloned
git pull origin master

# 2. Re-read this plan
$EDITOR docs/plans/2026-04-26-grpo-long-training-plan.md

# 3. Verify environment (training-time deps, not the dev-test set)
cd active-detective
python -c "import transformers, trl, peft, unsloth; print(transformers.__version__, trl.__version__, peft.__version__)"
# Expect: transformers>=5.2.0, trl>=0.16.0, peft>=0.15.0
# If missing or wrong version: bash scripts/vastai_setup.sh

# 4. Run pure-Python tests as a sanity check (no GPU needed, ~30s)
python -m pytest tests/ -q

# 5. Run smoke test (~15-30 min on A5000)
accelerate launch -m training.train_grpo \
    --model Qwen/Qwen3.5-9B \
    --output-dir ./checkpoints-smoke \
    --n-episodes 50 --group-size 2 --max-steps 10
# Tests that the GRPO loop itself works on this hardware before committing to long run.

# 6. Generate held-out eval set
python -c "from training.scenarios import generate_training_scenarios, save_scenarios; \
    save_scenarios(generate_training_scenarios(200, base_seed=900000), 'eval_held_out_200.jsonl')"

# 7. Eval the SFT-only adapter on held-out set as the lower-bound baseline
PYTHONPATH=. python scripts/run_eval_local.py \
    --model Qwen/Qwen3-1.7B \
    --adapter ../fine-tuning/checkpoints-sft-1.7b/final \
    --n 200 --seed 900000 \
    --output results/sft_baseline_held_out.jsonl

# 8. (Optional) Re-eval 700-step checkpoint on held-out set as the resume-baseline reference point
PYTHONPATH=. python scripts/run_eval_local.py \
    --model Qwen/Qwen3-1.7B \
    --adapter ./checkpoints-grpo-sft-1.7b \
    --n 200 --seed 900000 \
    --output results/grpo_700_held_out.jsonl

# 9. Launch the fresh long run (Plan A)
nohup python -m training.train_grpo \
    --model Qwen/Qwen3-1.7B \
    --adapter ../fine-tuning/checkpoints-sft-1.7b/final \
    --output-dir ./checkpoints-grpo-sft-1.7b-fresh \
    --n-episodes 200 --group-size 2 --max-steps 3000 \
    --lr 5e-6 --k-max 5 --lora-r 16 --lora-alpha 32 \
    --no-unsloth --disable-thinking --seed 42 \
    > grpo_fresh_training.log 2>&1 &

# 10. Monitor
tail -f grpo_fresh_training.log
```

## Resumability across GPU sessions

- Training auto-resumes from latest checkpoint in `--output-dir` (`train_grpo.py:663-673`). If the GPU disconnects, just re-run step 9 verbatim. It picks up from the most recent `checkpoint-N`.
- Eval JSONLs are append-only; safe to interrupt and rerun.
- After each milestone eval, `git add results/ && git commit -m "eval: step N held-out results" && git push` so the home machine has the trajectory.
- If `environment/reward.py` is modified mid-run, **discard the checkpoint dir and start over**. Mixing reward functions in one training run produces uninterpretable results.

## What "done" looks like

- Eight checkpoints evaluated (SFT-only baseline + 7 milestones from 250 to 3000 steps), all on the same pinned 200-seed held-out set.
- A learning curve table in `results/grpo_fresh_progress.md` showing accuracy, FN, FP, tools/ep, and per-scenario-type breakdown at each step.
- One of the decision-tree branches has fired, and the implied next action is queued (scale to 9B, adjust reward, switch curriculum, etc.).
- Everything committed and pushed so analysis can resume on the home machine.

## References

Code:
- `active-detective/training/train_grpo.py`: trainer plus auto-resume
- `active-detective/environment/env.py`: `RansomwareDetectionEnv` and `DetectionEnv`
- `active-detective/environment/reward.py`: redesigned reward
- `active-detective/scripts/run_grpo_overnight.sh`: long-run launcher (template for Plan A)
- `active-detective/scripts/vastai_setup.sh`: pinned training-time dep versions
- `active-detective/scripts/run_eval_local.py`: eval entry point
- `active-detective/scripts/compute_metrics.py`: post-hoc metrics on a results JSONL

Plans and design:
- `docs/plans/2026-03-05-active-detective-system-design.md`: system design
- `docs/plans/2026-03-07-pomdp-environment-design.md`: POMDP environment
- `docs/plans/2026-03-09-pre-training-audit-design.md`: audit motivation
- `docs/plans/2026-04-13-comprehensive-pretraining-fixes.md`: `(phase1)` through `(phase9)` plan
- `active-detective/CLAUDE.md`: architectural reference

Existing results (pre-redesigned-reward):
- `active-detective/results/grpo_sft_tools_700_eval.jsonl`: best existing checkpoint
- `active-detective/results/heuristic_baseline.jsonl`: strongest non-Opus baseline (76%)
- `active-detective/results/opus_eval_20.jsonl`: frontier ceiling (~90%)
