#!/usr/bin/env bash
# Vast.ai setup script for Active Detective GRPO training.
#
# Usage:
#   scp scripts/vastai_setup.sh root@<vast-ip>:/root/
#   ssh root@<vast-ip> 'bash /root/vastai_setup.sh'
#
# Target: RTX A5000 (24GB) or any GPU with >=16GB VRAM.
# Model: Qwen/Qwen3.5-9B with QLoRA 4-bit (~6GB VRAM).

set -euo pipefail

echo "=== Active Detective: Vast.ai Training Setup ==="

# ── System deps ──────────────────────────────────────────────────
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1
echo "[1/5] System packages ready"

# ── Clone repo ───────────────────────────────────────────────────
REPO_DIR="/root/ransomware-policy"
if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 https://github.com/queelius/ransomware-policy.git "$REPO_DIR"
fi
cd "$REPO_DIR/active-detective"
echo "[2/5] Repository cloned"

# ── Python deps ──────────────────────────────────────────────────
pip install -q --upgrade pip
pip install -q \
    "transformers>=4.51.0" \
    "trl>=0.16.0" \
    "peft>=0.15.0" \
    "bitsandbytes>=0.45.0" \
    "accelerate>=1.5.0" \
    "datasets>=3.0.0" \
    "unsloth[cu124-ampere]>=2025.3" \
    torch \
    numpy \
    pytest
echo "[3/5] Python dependencies installed"

# ── Verify setup ─────────────────────────────────────────────────
echo "[4/5] Running test suite..."
python -m pytest tests/ -q --tb=short 2>&1 | tail -3

echo "[5/5] Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU: {gpu} ({vram:.1f} GB)')
else:
    print('  WARNING: No GPU detected!')
"

# ── Print training command ───────────────────────────────────────
cat << 'READY'

=== Setup complete! ===

To start training:

  cd /root/ransomware-policy/active-detective

  # Standard run (500 episodes, ~2-6 hours on A5000)
  accelerate launch -m training.train_grpo \
      --model Qwen/Qwen3.5-9B \
      --output-dir ./checkpoints \
      --n-episodes 500 \
      --group-size 4 \
      --max-steps 200

  # Quick smoke test (50 episodes, ~15-30 min)
  accelerate launch -m training.train_grpo \
      --model Qwen/Qwen3.5-9B \
      --output-dir ./checkpoints-smoke \
      --n-episodes 50 \
      --group-size 4 \
      --max-steps 20

  # Larger group (if VRAM allows, better gradient signal)
  accelerate launch -m training.train_grpo \
      --model Qwen/Qwen3.5-9B \
      --output-dir ./checkpoints \
      --n-episodes 500 \
      --group-size 8 \
      --max-steps 200

Tips:
  - Monitor with: watch -n5 nvidia-smi
  - Checkpoints saved every 50 steps to --output-dir
  - Copy results back: scp -r root@<ip>:~/ransomware-policy/active-detective/checkpoints ./

READY
