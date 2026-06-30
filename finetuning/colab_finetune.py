# ============================================================
# MLSQL — Fine-Tune Sarvam-2b for Malayalam → SQL
# ============================================================
# Run this in Google Colab with T4 GPU enabled.
# Go to: Runtime > Change runtime type > T4 GPU
#
# Instructions: Copy each section into a separate Colab cell
# and run them one by one.
# ============================================================

# %%
# ===================== CELL 1: Setup =====================
# Check GPU and install dependencies

!nvidia-smi

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %%
# ===================== CELL 2: Install Packages =====================

!pip install -q datasets transformers peft trl bitsandbytes accelerate
!pip install -q deep-translator sentencepiece protobuf tqdm

# %%
# ===================== CELL 3: Download Spider Dataset =====================

import json
import os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

print("Downloading Spider dataset...")
dataset = load_dataset("xlangai/spider", trust_remote_code=True)
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

# Save training records
records = []
for row in dataset["train"]:
    records.append({
        "question": row["question"],
        "query": row["query"],
        "db_id": row["db_id"],
        "schema_context": ""
    })

with open("data/spider_train.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved {len(records)} records")
for i in range(3):
    print(f"  [{i+1}] {records[i]['question']}")
    print(f"       → {records[i]['query']}")

# %%
# ===================== CELL 4: Translate to Malayalam =====================
# This takes ~1-2 hours for 7000 questions. Checkpoints every 100.

import time
from tqdm.notebook import tqdm
from deep_translator import GoogleTranslator

with open("data/spider_train.json", "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Translating {len(records)} questions to Malayalam...")

# Checkpoint support
checkpoint_file = "data/translation_checkpoint.json"
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    print(f"Resuming from: {checkpoint['completed']} done")
else:
    checkpoint = {"completed": 0, "translations": {}}

start_idx = checkpoint["completed"]

for i in tqdm(range(start_idx, len(records)), initial=start_idx, total=len(records)):
    if str(i) in checkpoint["translations"]:
        continue

    question = records[i]["question"]
    try:
        translated = GoogleTranslator(source='en', target='ml').translate(question)
    except Exception as e:
        print(f"  Error at {i}: {e}")
        translated = question  # Keep English as fallback

    checkpoint["translations"][str(i)] = translated if translated else question
    checkpoint["completed"] = i + 1
    time.sleep(0.5)

    if (i + 1) % 100 == 0:
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False)
        print(f"  Checkpoint: {i+1}/{len(records)}")

# Final save
with open(checkpoint_file, "w", encoding="utf-8") as f:
    json.dump(checkpoint, f, ensure_ascii=False)

# Build output
output_records = []
for i, rec in enumerate(records):
    output_records.append({
        "question_en": rec["question"],
        "question_ml": checkpoint["translations"].get(str(i), rec["question"]),
        "query": rec["query"],
        "db_id": rec["db_id"],
        "schema_context": rec.get("schema_context", "")
    })

with open("data/spider_malayalam.json", "w", encoding="utf-8") as f:
    json.dump(output_records, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved {len(output_records)} translated records")
for i in range(5):
    print(f"  EN: {output_records[i]['question_en']}")
    print(f"  ML: {output_records[i]['question_ml']}")
    print()

if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

# %%
# ===================== CELL 5: Format Training Data =====================

import random

with open("data/spider_malayalam.json", "r", encoding="utf-8") as f:
    records = json.load(f)

TEMPLATE = """### Instruction:
Convert the following Malayalam question to an SQL query.

### Question:
{question}

### SQL:
{sql}"""

formatted = []
for rec in records:
    if not rec.get("question_ml") or not rec.get("query"):
        continue
    text = TEMPLATE.format(question=rec["question_ml"], sql=rec["query"])
    formatted.append({"text": text})

random.seed(42)
random.shuffle(formatted)
val_size = int(len(formatted) * 0.05)
train_data = formatted[val_size:]
val_data = formatted[:val_size]

with open("data/train_dataset.jsonl", "w", encoding="utf-8") as f:
    for rec in train_data:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

with open("data/val_dataset.jsonl", "w", encoding="utf-8") as f:
    for rec in val_data:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Train: {len(train_data)} | Val: {len(val_data)}")
print(f"\nSample:\n{'='*50}")
print(train_data[0]["text"])

# %%
# ===================== CELL 6: Load Model with QLoRA =====================

import torch
from datasets import load_dataset as load_ds
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

BASE_MODEL = "sarvamai/sarvam-2b-v0.5"
OUTPUT_DIR = "sarvam-malayalam-sql-lora"

print("Loading dataset...")
ds = load_ds("json", data_files={
    "train": "data/train_dataset.jsonl",
    "validation": "data/val_dataset.jsonl"
})
print(f"Train: {len(ds['train'])} | Val: {len(ds['validation'])}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("\nLoading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

print("\nApplying LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\n✅ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# %%
# ===================== CELL 7: Train! 🚀 =====================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    save_total_limit=3,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
)

print("🚀 Training started...")
trainer.train()
print("\n✅ Training complete!")

# Save
FINAL_PATH = f"{OUTPUT_DIR}/final"
trainer.model.save_pretrained(FINAL_PATH)
tokenizer.save_pretrained(FINAL_PATH)
print(f"✅ Saved to: {FINAL_PATH}")

# %%
# ===================== CELL 8: Test Inference 🧠 =====================

from peft import PeftModel

print("Loading model for inference...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
inference_model = PeftModel.from_pretrained(base_model, FINAL_PATH)
inference_model.eval()

def generate_sql(question, max_new_tokens=256):
    prompt = f"""### Instruction:
Convert the following Malayalam question to an SQL query.

### Question:
{question}

### SQL:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(inference_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = inference_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    sql = tokenizer.decode(generated, skip_special_tokens=True).strip()
    for stop in ["\n###", "\n\n", "### "]:
        if stop in sql:
            sql = sql[:sql.index(stop)]
    return sql.strip()

# Test
test_questions = [
    "വിദ്യാർത്ഥികളുടെ പേരുകൾ കാണിക്കുക",
    "56-ന് മുകളിൽ പ്രായമുള്ള വിഭാഗ തലവന്മാർ എത്ര?",
    "ആകെ വിദ്യാർത്ഥികൾ എത്ര?",
]

print("=" * 60)
print("INFERENCE TEST RESULTS")
print("=" * 60)
for q in test_questions:
    sql = generate_sql(q)
    print(f"\n🗣️  {q}")
    print(f"📝  {sql}")
    print("-" * 60)

# %%
# ===================== CELL 9: Download Model 📦 =====================

import shutil
zip_path = shutil.make_archive("sarvam-malayalam-sql-lora", "zip", FINAL_PATH)
print(f"✅ Model zipped: {zip_path}")
print(f"\n📌 Download from Files panel (📁 icon on left)")
print(f"   Then extract to: finetuning/sarvam-malayalam-sql-lora/final/")
print(f"   And run: python 05_inference.py --serve")
