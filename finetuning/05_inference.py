"""
05_inference.py
===============
Inference script for the fine-tuned Sarvam-2b Malayalam-to-SQL model.

Can run in two modes:
  1. Interactive CLI mode: python 05_inference.py
  2. Flask API server:     python 05_inference.py --serve

The Flask API server provides a REST endpoint that the Spring Boot
application can call to get SQL from Malayalam prompts.

API Endpoint:
  POST /api/malayalam-to-sql
  Body: {"question": "...", "schema": "..."}
  Response: {"sql": "...", "error": null}
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ─── Defaults ───────────────────────────────────────────────
BASE_MODEL = "sarvamai/sarvam-2b-v0.5"
LORA_PATH = os.path.join(os.path.dirname(__file__),
                         "sarvam-malayalam-sql-lora", "final")
DEFAULT_PORT = 5000

# Inference prompt template (must match training format)
PROMPT_WITH_SCHEMA = """### Instruction:
Convert the following Malayalam question to an SQL query based on the given database schema.

### Schema:
{schema}

### Question:
{question}

### SQL:
"""

PROMPT_NO_SCHEMA = """### Instruction:
Convert the following Malayalam question to an SQL query.

### Question:
{question}

### SQL:
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Malayalam-to-SQL Inference")
    parser.add_argument("--base_model", default=BASE_MODEL,
                        help=f"Base model (default: {BASE_MODEL})")
    parser.add_argument("--lora_path", default=LORA_PATH,
                        help=f"LoRA adapter path (default: {LORA_PATH})")
    parser.add_argument("--serve", action="store_true",
                        help="Run as Flask API server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--question", type=str, default=None,
                        help="Single question to translate (non-interactive)")
    parser.add_argument("--schema", type=str, default="",
                        help="Schema context for the question")
    parser.add_argument("--mock", action="store_true",
                        help="Use a mock model to avoid loading 2B parameters")
    return parser.parse_args()


class MalayalamToSQLModel:
    """Wrapper for the fine-tuned Malayalam-to-SQL model."""

    def __init__(self, base_model_name, lora_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model...")
        print(f"  Base model: {base_model_name}")
        print(f"  LoRA adapter: {lora_path}")
        print(f"  Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            # A known Windows issue with 5GB+ safetensors shards causes silent crashes.
            # However `use_safetensors` may invoke mmap by default. 

        # Load LoRA adapter
        if os.path.exists(lora_path):
            print(f"  Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print("  LoRA adapter loaded successfully!")
        else:
            print(f"  WARNING: LoRA adapter not found at {lora_path}")
            print("  Using base model without fine-tuning")

        self.model.eval()
        print("Model ready!\n")

    def generate_sql(self, question, schema="", max_new_tokens=256,
                     temperature=0.1, top_p=0.95):
        """Generate SQL from a Malayalam question."""
        # Build prompt
        if schema.strip():
            prompt = PROMPT_WITH_SCHEMA.format(schema=schema, question=question)
        else:
            prompt = PROMPT_NO_SCHEMA.format(question=question)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode – only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        sql = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Clean up: stop at first newline or ### marker
        for stop_token in ["\n###", "\n\n", "### "]:
            if stop_token in sql:
                sql = sql[:sql.index(stop_token)]

        return sql.strip()


def run_interactive(model):
    """Run interactive CLI mode."""
    print("=" * 60)
    print("Malayalam-to-SQL Interactive Mode")
    print("Type your Malayalam question (or 'quit' to exit)")
    print("Prefix with 'schema:' to set schema context")
    print("=" * 60)

    schema = ""
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input.lower().startswith("schema:"):
            schema = user_input[7:].strip()
            print(f"Schema set: {schema[:100]}...")
            continue

        sql = model.generate_sql(user_input, schema=schema)
        print(f"\nSQL: {sql}")


def run_server(model, port):
    """Run Flask API server."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("ERROR: Flask not installed. Run: pip install flask")
        sys.exit(1)

    app = Flask(__name__)

    @app.route("/api/malayalam-to-sql", methods=["POST"])
    def translate():
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"sql": None, "error": "Missing 'question' field"}), 400

        question = data["question"]
        schema = data.get("schema", "")

        try:
            sql = model.generate_sql(question, schema=schema)
            return jsonify({"sql": sql, "error": None})
        except Exception as e:
            return jsonify({"sql": None, "error": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": BASE_MODEL})

    print(f"\n{'=' * 60}")
    print(f"Flask API server starting on port {port}")
    print(f"  POST /api/malayalam-to-sql")
    print(f"  GET  /health")
    print(f"{'=' * 60}\n")

    app.run(host="0.0.0.0", port=port, debug=False)


def main():
    args = parse_args()

    # Load model
    if args.mock or os.environ.get("USE_MOCK_MODEL") == "1":
        print("Starting in mock mode (bypassing PyTorch HF loading)...")
        class MockModel:
            def generate_sql(self, question, schema=""):
                return "SELECT * FROM mock_table WHERE condition = 'true';"
                
        model = MockModel()
    else:
        try:
            model = MalayalamToSQLModel(args.base_model, args.lora_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Starting in mock mode...")
            
            class MockModel:
                def generate_sql(self, question, schema=""):
                    return "SELECT * FROM mock_table WHERE condition = 'true';"
                    
            model = MockModel()

    if args.question:
        # Single question mode
        sql = model.generate_sql(args.question, schema=args.schema)
        print(f"SQL: {sql}")
    elif args.serve:
        # Flask server mode
        run_server(model, args.port)
    else:
        # Interactive mode
        run_interactive(model)


if __name__ == "__main__":
    main()
