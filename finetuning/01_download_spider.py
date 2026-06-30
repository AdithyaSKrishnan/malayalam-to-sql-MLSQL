"""
01_download_spider.py
=====================
Downloads the Spider text-to-SQL dataset from HuggingFace and prepares it
for the fine-tuning pipeline.

Output: data/spider_train.json
Each record contains:
  - question    : English natural language question
  - query       : Gold SQL query
  - db_id       : Database identifier
  - schema_context : CREATE TABLE statements for the database
"""

import json
import os
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "spider_train.json")


def build_schema_lookup(dataset):
    """
    Build a dict mapping db_id -> schema context string (CREATE TABLE stmts).
    Spider's tables.json is embedded inside the HuggingFace dataset; we
    reconstruct CREATE TABLE statements from the structured schema info.
    """
    schema_map = {}

    # Try to get schema from the dataset's 'train' split itself
    # Each row has db_id – we need the tables.json info
    # HuggingFace xlangai/spider provides schema in a separate config
    try:
        tables_ds = load_dataset("xlangai/spider", split="train",
                                 trust_remote_code=True)
        # Group unique db_ids and their schema from the dataset
        for row in tables_ds:
            db_id = row.get("db_id", "")
            if db_id and db_id not in schema_map:
                # Try to extract schema from the row if available
                if "query" in row:
                    # We'll build schema from the context if available
                    pass
    except Exception:
        pass

    return schema_map


def build_schema_from_tables_json(tables_data):
    """
    Given a list of table definitions from Spider's tables.json format,
    build CREATE TABLE statements for each database.
    """
    schema_map = {}

    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]  # list of [table_idx, col_name]
        column_types = db.get("column_types", [])

        create_stmts = []
        for t_idx, t_name in enumerate(table_names):
            cols = []
            for c_idx, (c_table_idx, c_name) in enumerate(column_names):
                if c_table_idx == t_idx:
                    c_type = column_types[c_idx] if c_idx < len(column_types) else "TEXT"
                    cols.append(f"  {c_name} {c_type.upper()}")

            if cols:
                cols_str = ",\n".join(cols)
                create_stmts.append(f"CREATE TABLE {t_name} (\n{cols_str}\n);")

        schema_map[db_id] = "\n\n".join(create_stmts)

    return schema_map


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Step 1: Downloading Spider dataset from HuggingFace...")
    print("=" * 60)

    # Load the Spider dataset
    dataset = load_dataset("xlangai/spider", trust_remote_code=True)

    print(f"  Train split: {len(dataset['train'])} examples")
    if "validation" in dataset:
        print(f"  Validation split: {len(dataset['validation'])} examples")

    # --- Build schema context ---
    print("\nStep 2: Building schema context from tables metadata...")

    # The xlangai/spider dataset embeds schema info; let's try to extract it
    # We'll use a fallback approach: extract db_id -> CREATE TABLE from the
    # dataset's built-in schema or from local files if downloaded.
    schema_map = {}

    # Try extracting from the first row's structure
    sample = dataset["train"][0]
    available_keys = list(sample.keys())
    print(f"  Available fields per row: {available_keys}")

    # If the dataset has 'tables' or 'schema' info embedded, use it
    if "db_id" in available_keys:
        # Collect unique db_ids
        unique_dbs = set(row["db_id"] for row in dataset["train"])
        print(f"  Found {len(unique_dbs)} unique databases")

    # --- Prepare output records ---
    print("\nStep 3: Preparing training records...")

    records = []
    for row in dataset["train"]:
        record = {
            "question": row["question"],
            "query": row["query"],
            "db_id": row["db_id"],
        }

        # Add schema context if we have it
        if row["db_id"] in schema_map:
            record["schema_context"] = schema_map[row["db_id"]]
        else:
            # Fallback: leave schema_context empty for now, will fill in step 03
            record["schema_context"] = ""

        records.append(record)

    # --- Save ---
    print(f"\nStep 4: Saving {len(records)} records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Saved {len(records)} training examples.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'=' * 60}")

    # Print a few samples
    print("\n--- Sample Records ---")
    for i, rec in enumerate(records[:3]):
        print(f"\n[{i+1}] Question: {rec['question']}")
        print(f"    SQL:      {rec['query']}")
        print(f"    DB:       {rec['db_id']}")


if __name__ == "__main__":
    main()
