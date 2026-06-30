"""
02_translate_to_malayalam.py
============================
Translates English questions from the Spider dataset to Malayalam.

Supports two backends:
  1. Sarvam AI Translate API (preferred, set SARVAM_API_KEY env variable)
  2. googletrans library (fallback, no API key needed)

Input:  data/spider_train.json
Output: data/spider_malayalam.json

Usage:
  # With Sarvam AI API key:
  set SARVAM_API_KEY=your_key_here
  python 02_translate_to_malayalam.py

  # Without API key (uses googletrans fallback):
  python 02_translate_to_malayalam.py
"""

import json
import os
import time
import sys
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_FILE = os.path.join(DATA_DIR, "spider_train.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "spider_malayalam.json")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "translation_checkpoint.json")

# Sarvam AI API config
SARVAM_API_URL = "https://api.sarvam.ai/translate"
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

# Rate limiting
REQUESTS_PER_SECOND = 5
BATCH_SIZE = 50  # Save checkpoint every N translations


def translate_with_sarvam(text, api_key):
    """Translate English text to Malayalam using Sarvam AI Translate API."""
    import requests

    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key,
    }

    payload = {
        "input": text,
        "source_language_code": "en-IN",
        "target_language_code": "ml-IN",
        "model": "mayura:v1",
        "enable_preprocessing": True,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(SARVAM_API_URL, json=payload,
                                     headers=headers, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get("translated_text", text)
            elif response.status_code == 429:
                # Rate limited – wait and retry
                wait_time = (attempt + 1) * 2
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  API error {response.status_code}: {response.text}")
                if attempt == max_retries - 1:
                    return None
        except Exception as e:
            print(f"  Request error: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)

    return None


def translate_with_googletrans(text):
    """Translate English text to Malayalam using deep-translator (fallback)."""
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source='en', target='ml').translate(text)
        return result
    except Exception as e:
        print(f"  deep-translator error: {e}")
        return None


def load_checkpoint():
    """Load translation checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint: {data['completed']} translations done")
        return data
    return {"completed": 0, "translations": {}}


def save_checkpoint(checkpoint):
    """Save translation checkpoint."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 60)
    print("Malayalam Translation Pipeline")
    print("=" * 60)

    # Load input data
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run 01_download_spider.py first!")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"  Loaded {len(records)} records from {INPUT_FILE}")

    # Determine translation backend
    use_sarvam = bool(SARVAM_API_KEY)
    if use_sarvam:
        print("  Using: Sarvam AI Translate API")
    else:
        print("  Using: googletrans (fallback)")
        print("  Tip: Set SARVAM_API_KEY env variable for better translations")

    # Load checkpoint
    checkpoint = load_checkpoint()
    start_idx = checkpoint["completed"]

    if start_idx >= len(records):
        print("  All translations already completed!")
    else:
        print(f"\n  Translating {len(records) - start_idx} remaining questions...\n")

        for i in tqdm(range(start_idx, len(records)),
                      initial=start_idx, total=len(records),
                      desc="Translating"):
            question_en = records[i]["question"]

            # Check if already translated
            if str(i) in checkpoint["translations"]:
                continue

            # Translate
            if use_sarvam:
                translated = translate_with_sarvam(question_en, SARVAM_API_KEY)
                time.sleep(1.0 / REQUESTS_PER_SECOND)  # Rate limit
            else:
                translated = translate_with_googletrans(question_en)
                time.sleep(0.5)  # Be gentle with free API

            if translated:
                checkpoint["translations"][str(i)] = translated
            else:
                # Keep original English if translation fails
                checkpoint["translations"][str(i)] = question_en
                print(f"  WARNING: Failed to translate index {i}, keeping English")

            checkpoint["completed"] = i + 1

            # Save checkpoint periodically
            if (i + 1) % BATCH_SIZE == 0:
                save_checkpoint(checkpoint)
                print(f"  Checkpoint saved at {i + 1}/{len(records)}")

        # Final checkpoint save
        save_checkpoint(checkpoint)

    # --- Build output records ---
    print(f"\nBuilding output file...")

    output_records = []
    for i, record in enumerate(records):
        output_record = {
            "question_en": record["question"],
            "question_ml": checkpoint["translations"].get(str(i), record["question"]),
            "query": record["query"],
            "db_id": record["db_id"],
            "schema_context": record.get("schema_context", ""),
        }
        output_records.append(output_record)

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Saved {len(output_records)} translated records.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'=' * 60}")

    # Print samples
    print("\n--- Sample Translations ---")
    for i in range(min(5, len(output_records))):
        rec = output_records[i]
        print(f"\n[{i+1}] EN: {rec['question_en']}")
        print(f"     ML: {rec['question_ml']}")
        print(f"     SQL: {rec['query']}")

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("\n  Checkpoint file cleaned up.")


if __name__ == "__main__":
    main()
