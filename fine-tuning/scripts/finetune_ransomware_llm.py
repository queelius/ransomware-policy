#!/usr/bin/env python3
"""
Fine-tune an LLM for ransomware detection using our telemetry data.
Uses QLoRA for efficient training on consumer GPUs (8GB+ VRAM).
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset, load_dataset
import evaluate
import numpy as np

# Models that work well for this task (pick based on your GPU)
MODEL_OPTIONS = {
    "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",        # 4GB VRAM
    "small": "microsoft/phi-2",                           # 6GB VRAM
    "medium": "mistralai/Mistral-7B-Instruct-v0.2",      # 8-12GB VRAM
    "large": "meta-llama/Llama-2-13b-chat-hf"            # 16GB+ VRAM
}

class RansomwareDataset:
    """Dataset handler for ransomware telemetry training data."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_and_prepare(self) -> Dataset:
        """Load and tokenize the dataset."""

        # Load JSONL data
        raw_data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                raw_data.append(json.loads(line))

        # Format for training
        formatted_data = []
        for item in raw_data:
            # Combine prompt and completion
            text = f"{item['prompt']}\n{item['completion']}"
            formatted_data.append({"text": text})

        # Create dataset
        dataset = Dataset.from_list(formatted_data)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Set labels for language modeling
        def set_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples

        tokenized_dataset = tokenized_dataset.map(set_labels, batched=True)

        return tokenized_dataset

class RansomwareDetectionTrainer:
    """Fine-tune LLM for ransomware detection."""

    def __init__(
        self,
        model_name: str = "tiny",
        output_dir: str = "./ransomware_detector",
        use_4bit: bool = True
    ):
        self.model_name = MODEL_OPTIONS.get(model_name, model_name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_4bit = use_4bit

        print(f"Initializing with model: {self.model_name}")

    def setup_model_and_tokenizer(self):
        """Initialize model with QLoRA configuration."""

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Quantization config for QLoRA
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.use_4bit else None
        )

        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=16,                      # Rank
            lora_alpha=32,            # Alpha scaling
            target_modules=[          # Modules to apply LoRA to
                "q_proj", "v_proj",   # Common attention modules
                "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"  # MLP modules
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_training_args(self, num_epochs: int = 3):
        """Configure training arguments."""

        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # Effective batch size = 16
            warmup_ratio=0.03,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="none",  # or "tensorboard", "wandb"
            gradient_checkpointing=True,
            optim="paged_adamw_8bit"  # Memory efficient optimizer
        )

    def train(
        self,
        train_data_path: str,
        eval_data_path: Optional[str] = None,
        num_epochs: int = 3
    ):
        """Fine-tune the model."""

        # Load datasets
        train_dataset = RansomwareDataset(
            train_data_path,
            self.tokenizer
        ).load_and_prepare()

        eval_dataset = None
        if eval_data_path:
            eval_dataset = RansomwareDataset(
                eval_data_path,
                self.tokenizer
            ).load_and_prepare()

        # Training arguments
        training_args = self.prepare_training_args(num_epochs)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Save final model
        trainer.save_model(self.output_dir / "final_model")
        self.tokenizer.save_pretrained(self.output_dir / "final_model")

        print(f"Training complete! Model saved to {self.output_dir}")

    def generate_detection(self, telemetry_prompt: str) -> str:
        """Generate ransomware detection analysis."""

        # Tokenize input
        inputs = self.tokenizer(
            telemetry_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )

        return response

class RansomwareEvaluator:
    """Evaluate the fine-tuned model."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_detection_accuracy(self, test_data_path: str) -> Dict:
        """Evaluate detection accuracy on test set."""

        correct = 0
        total = 0
        results = []

        with open(test_data_path, 'r') as f:
            for line in f:
                data = json.loads(line)

                # Generate prediction
                response = self.generate_detection(data['prompt'])

                # Extract predicted state (simple parsing)
                predicted_state = "unknown"
                if "ransomware" in response.lower():
                    predicted_state = "ransomware"
                elif "benign" in response.lower():
                    predicted_state = "benign"

                # Check ground truth
                true_state = "ransomware" if "ransomware" in data['completion'].lower() else "benign"

                if predicted_state == true_state:
                    correct += 1
                total += 1

                results.append({
                    "predicted": predicted_state,
                    "true": true_state,
                    "correct": predicted_state == true_state
                })

        accuracy = correct / total if total > 0 else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results[:10]  # Sample results
        }

    def generate_detection(self, prompt: str) -> str:
        """Helper to generate detection."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_sample_data(output_path: str, num_samples: int = 100):
    """Generate sample training data."""

    print(f"Generating {num_samples} training samples...")

    samples = []
    for i in range(num_samples):
        is_ransomware = i % 3 != 0  # 2/3 ransomware, 1/3 benign

        if is_ransomware:
            prompt = f"""<|telemetry|>
Process: powershell.exe -enc SGVsbG8=
Process: cmd.exe whoami /all
Process: vssadmin.exe delete shadows
File: Multiple files changing to .encrypted
Network: Outbound to 185.220.101.45

Analyze this telemetry for ransomware."""

            completion = """<|analysis|>
State: RANSOMWARE DETECTED - encryption_active
Risk: CRITICAL
Evidence: Shadow copy deletion, file encryption, suspicious PowerShell
Action: Isolate system immediately
Confidence: 95%"""
        else:
            prompt = f"""<|telemetry|>
Process: chrome.exe
Process: notepad.exe
File: C:\\Users\\user\\document.docx accessed
Network: Connection to microsoft.com

Analyze this telemetry for ransomware."""

            completion = """<|analysis|>
State: BENIGN - normal activity
Risk: LOW
Evidence: Standard user applications, no suspicious indicators
Action: Continue monitoring
Confidence: 90%"""

        samples.append({
            "prompt": prompt,
            "completion": completion
        })

    # Save to file
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Sample data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for ransomware detection")
    parser.add_argument("--model", choices=["tiny", "small", "medium"], default="tiny",
                       help="Model size to use")
    parser.add_argument("--train-data", default="train_data.jsonl",
                       help="Training data path")
    parser.add_argument("--eval-data", default=None,
                       help="Evaluation data path")
    parser.add_argument("--output-dir", default="./ransomware_detector",
                       help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--generate-sample", action="store_true",
                       help="Generate sample training data")
    parser.add_argument("--test", action="store_true",
                       help="Test the model after training")

    args = parser.parse_args()

    if args.generate_sample:
        # Generate sample data
        generate_sample_data("train_data.jsonl", 100)
        generate_sample_data("test_data.jsonl", 20)
        print("\nSample data generated. Now run without --generate-sample to train.")
        return

    # Initialize trainer
    trainer = RansomwareDetectionTrainer(
        model_name=args.model,
        output_dir=args.output_dir
    )

    # Setup model
    print("Setting up model and tokenizer...")
    trainer.setup_model_and_tokenizer()

    # Train
    print("Starting training...")
    trainer.train(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        num_epochs=args.epochs
    )

    # Test if requested
    if args.test:
        print("\nTesting model...")
        test_prompt = """<|telemetry|>
Process: powershell.exe -enc [encoded]
Process: vssadmin delete shadows
File: encryption detected

Analyze this telemetry."""

        response = trainer.generate_detection(test_prompt)
        print(f"\nTest Input: {test_prompt}")
        print(f"\nModel Response: {response}")


if __name__ == "__main__":
    main()