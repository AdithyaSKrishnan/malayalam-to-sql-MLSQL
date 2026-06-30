"""
03_prepare_training_data.py
===========================
Converts the translated Spider dataset into instruction-tuning format
for supervised fine-tuning of the Sarvam-2b model.

Input:  data/spider_malayalam.json
Output: data/train_dataset.jsonl

Each training example follows the format:
    ### Instruction:
    Convert the following Malayalam question to an SQL query.

    ### Schema:
    {CREATE TABLE statements}

    ### Question:
    {Malayalam question}

    ### SQL:
    {Expected SQL query}
"""

import json
import os
import sys
import random
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_FILE = os.path.join(DATA_DIR, "spider_malayalam.json")
OUTPUT_TRAIN = os.path.join(DATA_DIR, "train_dataset.jsonl")
OUTPUT_VAL = os.path.join(DATA_DIR, "val_dataset.jsonl")

# Validation split ratio
VAL_RATIO = 0.05
RANDOM_SEED = 42

INSTRUCTION_TEMPLATE = """### Instruction:
Convert the following Malayalam question to an SQL query based on the given database schema.

### Schema:
{schema}

### Question:
{question}

### SQL:
{sql}"""

INSTRUCTION_TEMPLATE_NO_SCHEMA = """### Instruction:
Convert the following Malayalam question to an SQL query.

### Question:
{question}

### SQL:
{sql}"""


def format_record(record):
    """Format a single record into instruction-tuning format."""
    question_ml = record["question_ml"]
    sql = record["query"]
    schema = record.get("schema_context", "").strip()

    if schema:
        text = INSTRUCTION_TEMPLATE.format(
            schema=schema,
            question=question_ml,
            sql=sql
        )
    else:
        text = INSTRUCTION_TEMPLATE_NO_SCHEMA.format(
            question=question_ml,
            sql=sql
        )

    return {"text": text}


def main():
    print("=" * 60)
    print("Training Data Preparation")
    print("=" * 60)

    # Load translated data
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run 02_translate_to_malayalam.py first!")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"  Loaded {len(records)} records from {INPUT_FILE}")

    # Format all records
    print("\n  Formatting records for instruction tuning...")
    formatted = []
    skipped = 0

    for record in tqdm(records, desc="Formatting"):
        # Skip records with empty questions or SQL
        if not record.get("question_ml") or not record.get("query"):
            skipped += 1
            continue

        formatted_record = format_record(record)
        formatted.append(formatted_record)

    print(f"  Formatted: {len(formatted)} | Skipped: {skipped}")

    # Split into train and validation
    random.seed(RANDOM_SEED)
    random.shuffle(formatted)

    val_size = int(len(formatted) * VAL_RATIO)
    train_data = formatted[val_size:]
    val_data = formatted[:val_size]

    print(f"\n  Train set: {len(train_data)} examples")
    print(f"  Val set:   {len(val_data)} examples")

    # Save train set
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for record in train_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save validation set
    with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
        for record in val_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"  Train: {OUTPUT_TRAIN}")
    print(f"  Val:   {OUTPUT_VAL}")
    print(f"{'=' * 60}")

    # Print samples
    print("\n--- Sample Training Prompts ---")
    for i in range(min(2, len(train_data))):
        print(f"\n{'─' * 40}")
        print(train_data[i]["text"])

    # Print statistics
    total_tokens_approx = sum(len(r["text"].split()) for r in formatted)
    avg_tokens = total_tokens_approx / len(formatted) if formatted else 0
    print(f"\n--- Statistics ---")
    print(f"  Total approx words: {total_tokens_approx:,}")
    print(f"  Avg words per example: {avg_tokens:.1f}")


if __name__ == "__main__":
    main()
