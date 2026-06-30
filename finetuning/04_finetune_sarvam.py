"""
04_finetune_sarvam.py
=====================
Fine-tunes the Sarvam-2b model on the Malayalam-to-SQL dataset using
QLoRA (4-bit quantized LoRA).

Input:  data/train_dataset.jsonl, data/val_dataset.jsonl
Output: sarvam-malayalam-sql-lora/ (LoRA adapter weights)

Requirements:
  - GPU with >= 16GB VRAM (T4, A100, RTX 3090+)
  - Or run on Google Colab (free T4)

Usage:
  python 04_finetune_sarvam.py
  python 04_finetune_sarvam.py --epochs 5 --batch_size 2
  python 04_finetune_sarvam.py --resume  # Resume from checkpoint
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ─── Defaults ───────────────────────────────────────────────
BASE_MODEL = "sarvamai/sarvam-2b-v0.5"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sarvam-malayalam-sql-lora")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train_dataset.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val_dataset.jsonl")

# LoRA config
LORA_R = 16           # Rank of low-rank matrices
LORA_ALPHA = 32       # Scaling factor
LORA_DROPOUT = 0.05   # Dropout for LoRA layers
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# Training defaults
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 2e-4
DEFAULT_MAX_SEQ_LEN = 512
GRADIENT_ACCUMULATION = 4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 200
EVAL_STEPS = 200


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Sarvam-2b for Malayalam-to-SQL")
    parser.add_argument("--model", default=BASE_MODEL,
                        help=f"Base model name (default: {BASE_MODEL})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Per-device batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN,
                        help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LEN})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a quick 10-step smoke test")
    return parser.parse_args()


def print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  WARNING: No GPU detected! Training will be very slow.")
        print("  Consider using Google Colab with a T4 GPU.")


def main():
    args = parse_args()

    print("=" * 60)
    print("Sarvam-2b Fine-Tuning (QLoRA)")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Output:      {args.output_dir}")
    print_gpu_info()

    # ─── Check data files ─────────────────────────────────────
    if not os.path.exists(TRAIN_FILE):
        print(f"\nERROR: Training data not found: {TRAIN_FILE}")
        print("Run scripts 01-03 first!")
        sys.exit(1)

    # ─── Load dataset ─────────────────────────────────────────
    print("\nStep 1: Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE if os.path.exists(VAL_FILE) else TRAIN_FILE,
    })
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Val:   {len(dataset['validation'])} examples")

    # Smoke test: use tiny subset
    if args.smoke_test:
        print("  SMOKE TEST MODE: Using 50 examples, 10 steps")
        dataset["train"] = dataset["train"].select(range(min(50, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(10, len(dataset["validation"]))))
        args.epochs = 1

    # ─── Load tokenizer ───────────────────────────────────────
    print("\nStep 2: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ─── Quantization config (4-bit for QLoRA) ────────────────
    print("\nStep 3: Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ─── Load base model ──────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Silence warning

    # ─── LoRA config ──────────────────────────────────────────
    print("\nStep 4: Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # ─── Training arguments ───────────────────────────────────
    print("\nStep 5: Configuring training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        max_steps=10 if args.smoke_test else -1,
        seed=42,
    )

    # ─── Trainer ──────────────────────────────────────────────
    print("\nStep 6: Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
    )

    # Resume from checkpoint if requested
    resume_from = None
    if args.resume:
        checkpoints = [d for d in os.listdir(args.output_dir)
                       if d.startswith("checkpoint-")] if os.path.exists(args.output_dir) else []
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            resume_from = os.path.join(args.output_dir, latest)
            print(f"  Resuming from: {resume_from}")
        else:
            print("  No checkpoint found, starting fresh")

    # Train!
    print("\n" + "─" * 60)
    print("Training started...")
    print("─" * 60)

    trainer.train(resume_from_checkpoint=resume_from)

    # ─── Save final model ─────────────────────────────────────
    print("\nStep 7: Saving LoRA adapter...")
    final_path = os.path.join(args.output_dir, "final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  LoRA adapter saved to: {final_path}")
    print(f"  Base model: {args.model}")
    print(f"  To run inference: python 05_inference.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
