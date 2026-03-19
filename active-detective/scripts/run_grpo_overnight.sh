#!/bin/bash
# Overnight GRPO training run on RTX 3060 12GB
# Expected: ~8 hours for 200 steps with group_size=2
#
# Usage: cd active-detective && bash scripts/run_grpo_overnight.sh

set -e

echo "=== GRPO Overnight Training ==="
echo "Start: $(date)"
echo "Model: Qwen/Qwen3-1.7B + SFT adapter"
echo "Steps: 200, Episodes: 200, Group size: 2"
echo ""

python -m training.train_grpo \
    --model Qwen/Qwen3-1.7B \
    --adapter ../fine-tuning/checkpoints-sft-1.7b/final \
    --output-dir ./checkpoints-grpo-sft-1.7b \
    --n-episodes 200 \
    --group-size 2 \
    --max-steps 200 \
    --lr 5e-6 \
    --k-max 5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --disable-thinking \
    --seed 42 \
    2>&1 | tee grpo_training.log

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
echo "Checkpoints: ./checkpoints-grpo-sft-1.7b/"
echo ""
echo "Run evaluation:"
echo "  PYTHONPATH=. python scripts/run_eval_local.py \\"
echo "    --model Qwen/Qwen3-1.7B \\"
echo "    --adapter ./checkpoints-grpo-sft-1.7b \\"
echo "    --n 200 --seed 42"
