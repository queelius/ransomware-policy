#!/bin/bash
# Quick start script for fine-tuning ransomware detection LLM

echo "==================================="
echo "Ransomware Detection LLM Fine-tuning"
echo "==================================="

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
pip install -r requirements_finetune.txt

# Step 2: Generate training data
echo -e "\nStep 2: Generating training data..."
echo "Option A: Simple examples (for quick testing)"
python prepare_training_data.py --simple --output simple_train.jsonl

echo -e "\nOption B: Full telemetry data (better quality)"
# Uncomment for full data generation (takes longer)
# python prepare_training_data.py --sequences 100 --output full_train.jsonl

# Step 3: Fine-tune model
echo -e "\nStep 3: Fine-tuning model..."
echo "Choose your model based on GPU memory:"
echo "  - 4GB VRAM: --model tiny (TinyLlama 1.1B)"
echo "  - 8GB VRAM: --model small (Phi-2)"
echo "  - 12GB+ VRAM: --model medium (Mistral 7B)"

# Example with TinyLlama (works on most GPUs)
python finetune_ransomware_llm.py \
    --model tiny \
    --train-data simple_train.jsonl \
    --epochs 3 \
    --output-dir ./ransomware_model

echo -e "\n==================================="
echo "Training complete!"
echo "Model saved to: ./ransomware_model"
echo "==================================="

# Step 4: Test the model
echo -e "\nStep 4: Testing the model..."
python -c "
from finetune_ransomware_llm import RansomwareDetectionTrainer

trainer = RansomwareDetectionTrainer(output_dir='./ransomware_model')
trainer.setup_model_and_tokenizer()

test_prompt = '''<|telemetry|>
Process: powershell.exe -enc [encoded]
Process: vssadmin delete shadows
File: multiple files encrypted
Network: outbound to suspicious IP

Analyze this telemetry.'''

response = trainer.generate_detection(test_prompt)
print('Test Prompt:', test_prompt)
print('\\nModel Response:', response)
"