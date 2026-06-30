# Fine-Tuning Pipeline: Malayalam → SQL

This directory contains the complete pipeline to fine-tune **Sarvam-2b** on the
**Spider dataset** for converting **Malayalam natural language** to **SQL queries**.

## Prerequisites

- **Python 3.10+**
- **GPU** with ≥16 GB VRAM (or use Google Colab with T4)
- **Sarvam AI API Key** (optional, for high-quality translation)

## Setup

```bash
cd finetuning
pip install -r requirements.txt
```

## Pipeline Steps

Run the scripts **in order**:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `python 01_download_spider.py` | Download Spider dataset from HuggingFace |
| 2 | `python 02_translate_to_malayalam.py` | Translate English questions → Malayalam |
| 3 | `python 03_prepare_training_data.py` | Format data for instruction tuning |
| 4 | `python 04_finetune_sarvam.py` | Fine-tune Sarvam-2b with QLoRA |
| 5 | `python 05_inference.py --serve` | Start inference API server (port 5000) |

### Step 2 — Translation Options

```bash
# Option A: Sarvam Translate API (recommended)
set SARVAM_API_KEY=your_key_here
python 02_translate_to_malayalam.py

# Option B: Free fallback (googletrans)
python 02_translate_to_malayalam.py
```

### Step 4 — Fine-Tuning Options

```bash
# Quick smoke test (10 steps)
python 04_finetune_sarvam.py --smoke_test

# Full training with custom params
python 04_finetune_sarvam.py --epochs 5 --batch_size 2 --lr 1e-4

# Resume from checkpoint
python 04_finetune_sarvam.py --resume
```

### Step 5 — Inference Modes

```bash
# Interactive CLI
python 05_inference.py

# Single question
python 05_inference.py --question "വിദ്യാർത്ഥികളുടെ പേരുകൾ കാണിക്കുക"

# Flask API server (for Spring Boot integration)
python 05_inference.py --serve --port 5000
```

## Spring Boot Integration

Once the inference server is running (`05_inference.py --serve`), the Spring Boot
app exposes a full-pipeline endpoint:

```bash
POST /api/malayalam-to-nosql
Content-Type: application/json

{
  "question": "വിദ്യാർത്ഥികളുടെ പേരുകൾ കാണിക്കുക",
  "schema": "{\"students\": {\"type\": \"collection\"}}"
}
```

Response:
```json
{
  "generatedSql": "SELECT name FROM students",
  "mongoQuery": "db.students.find({}, {name: 1})",
  "originalQuestion": "വിദ്യാർത്ഥികളുടെ പേരുകൾ കാണിക്കുക",
  "error": null
}
```

## Directory Structure

```
finetuning/
├── 01_download_spider.py       # Download Spider dataset
├── 02_translate_to_malayalam.py# Translate to Malayalam
├── 03_prepare_training_data.py # Format for training
├── 04_finetune_sarvam.py       # QLoRA fine-tuning
├── 05_inference.py             # Inference (CLI + API)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # [Generated] Dataset files
│   ├── spider_train.json
│   ├── spider_malayalam.json
│   ├── train_dataset.jsonl
│   └── val_dataset.jsonl
└── sarvam-malayalam-sql-lora/  # [Generated] LoRA weights
    └── final/
```
