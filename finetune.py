#!/usr/bin/env python3
"""
Rain ⛈️ - Fine-tuning Pipeline (Phase 5B)

Exports feedback data to training files and orchestrates LoRA adapter
training via mlx_lm (Apple Silicon) or llama.cpp, then registers the
result as a new Ollama model.

Training data sources (in priority order):
  1. Corrections  — bad-rated responses where user typed the right answer
                    (highest quality signal — exact preference data)
  2. Good examples — good-rated responses used as positive training examples
                    (volume signal — teaches Rain what "correct" looks like)

Usage:
    python3 finetune.py --status                 # show feedback stats
    python3 finetune.py --export                 # export training data to ~/.rain/training/
    python3 finetune.py --train                  # run LoRA training (mlx_lm or llama.cpp)
    python3 finetune.py --create-model           # register rain-tuned in Ollama
    python3 finetune.py --full                   # export + train + create-model in one shot
    python3 finetune.py --ab-report              # show A/B comparison from logged results
    python3 finetune.py --setup                  # install mlx-lm and show setup instructions
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────

RAIN_DIR       = Path.home() / ".rain"
DB_PATH        = RAIN_DIR / "memory.db"
TRAINING_DIR   = RAIN_DIR / "training"
ADAPTER_DIR    = RAIN_DIR / "adapters"
MLX_MODEL_DIR  = RAIN_DIR / "models"
MODELFILE_PATH = RAIN_DIR / "Modelfile.rain-tuned"

TUNED_MODEL_NAME = "rain-tuned"
MIN_EXAMPLES     = 10   # minimum examples before training makes sense

# Preferred base models for fine-tuning, in order of preference.
# Must match what's installed in Ollama.
PREFERRED_BASE_MODELS = [
    "qwen2.5-coder:7b",
    "qwen3:8b",
    "qwen3.5:9b",
    "llama3.2",
    "llama3.1",
]

# HuggingFace model IDs corresponding to the Ollama models above.
# Used when downloading models for mlx_lm training.
HF_MODEL_MAP = {
    "qwen2.5-coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen3":         "Qwen/Qwen3-8B",
    "qwen3.5":       "Qwen/Qwen3-8B",   # closest available
    "llama3.2":      "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

# ── System prompt used for all training examples ───────────────────────────────

FINETUNE_SYSTEM = (
    "You are Rain, a sovereign AI assistant running locally on the user's computer. "
    "You are direct, precise, and honest about uncertainty. "
    "For philosophical, abstract, or simple factual questions, answer in natural prose "
    "using 1-3 paragraphs — no headers, no tables, no numbered sections unless the "
    "question genuinely calls for a list. "
    "For multi-step tasks that require reading files or executing code, use structured "
    "planning steps. "
    "You never use third-party Python packages when stdlib alternatives exist. "
    "For Bitcoin and blockchain data you use the mempool.space public REST API. "
    "You never output HTML tags or markup inside code blocks. "
    "You never reference an internal critique or reflection process in your answers. "
    "You never fabricate security policies, safety layers, or confidentiality rules "
    "that do not exist in your actual implementation. "
    "When asked what model you are running on, state it directly and factually."
)

# ── Colours (terminal) ────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
DIM    = "\033[2m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


# ── Database helpers ───────────────────────────────────────────────────────────

def _db_connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(_c(RED, f"❌ Memory database not found at {DB_PATH}"))
        print("   Start Rain at least once to initialise the database.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_feedback_stats() -> dict:
    """Return a summary of what's in the feedback table."""
    with _db_connect() as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
        ).fetchone()
        if not exists:
            return {"total": 0, "good": 0, "bad": 0, "corrections": 0}

        row = conn.execute("""
            SELECT
                COUNT(*)                                     AS total,
                SUM(CASE WHEN rating='good' THEN 1 ELSE 0 END) AS good,
                SUM(CASE WHEN rating='bad'  THEN 1 ELSE 0 END) AS bad,
                SUM(CASE WHEN rating='bad' AND correction IS NOT NULL
                              AND correction != '' THEN 1 ELSE 0 END) AS corrections
            FROM feedback
        """).fetchone()
        if not row:
            return {"total": 0, "good": 0, "bad": 0, "corrections": 0}
        return {
            "total":       row["total"] or 0,
            "good":        row["good"] or 0,
            "bad":         row["bad"] or 0,
            "corrections": row["corrections"] or 0,
        }


def get_corrections() -> list:
    """Fetch bad-rated feedback entries that have a user correction."""
    with _db_connect() as conn:
        rows = conn.execute("""
            SELECT query, response, correction, timestamp
            FROM feedback
            WHERE rating = 'bad'
              AND correction IS NOT NULL
              AND correction != ''
            ORDER BY id ASC
        """).fetchall()
    return [dict(r) for r in rows]


def get_good_responses() -> list:
    """Fetch good-rated feedback entries as positive training examples."""
    with _db_connect() as conn:
        rows = conn.execute("""
            SELECT query, response, timestamp
            FROM feedback
            WHERE rating = 'good'
              AND query IS NOT NULL AND query != ''
              AND response IS NOT NULL AND response != ''
            ORDER BY id ASC
        """).fetchall()
    return [dict(r) for r in rows]


# ── Training data export ───────────────────────────────────────────────────────

def export_alpaca_jsonl(corrections: list, good_examples: list, out_path: Path) -> int:
    """
    Export training data in Alpaca instruction-following JSONL format.
    Compatible with HuggingFace TRL, Unsloth, mlx_lm, and most fine-tuning frameworks.

    Corrections (bad + user fix) are highest quality — preference data.
    Good examples are positive reinforcement — "this is what good looks like".

    Format:
        {"instruction": "<query>", "input": "", "output": "<ideal answer>"}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        # Corrections first — highest signal
        for c in corrections:
            record = {
                "instruction": c["query"].strip(),
                "input": "",
                "output": c["correction"].strip(),
                "system": FINETUNE_SYSTEM,
                "_source": "correction",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        # Good responses as positive examples
        for g in good_examples:
            record = {
                "instruction": g["query"].strip(),
                "input": "",
                "output": g["response"].strip(),
                "system": FINETUNE_SYSTEM,
                "_source": "good_response",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def export_chatml_txt(corrections: list, good_examples: list, out_path: Path) -> int:
    """
    Export training data in ChatML format for llama.cpp finetune.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for c in corrections:
            f.write("<|im_start|>system\n")
            f.write(FINETUNE_SYSTEM + "\n")
            f.write("<|im_end|>\n")
            f.write("<|im_start|>user\n")
            f.write(c["query"].strip() + "\n")
            f.write("<|im_end|>\n")
            f.write("<|im_start|>assistant\n")
            f.write(c["correction"].strip() + "\n")
            f.write("<|im_end|>\n\n")
            written += 1
        for g in good_examples:
            f.write("<|im_start|>system\n")
            f.write(FINETUNE_SYSTEM + "\n")
            f.write("<|im_end|>\n")
            f.write("<|im_start|>user\n")
            f.write(g["query"].strip() + "\n")
            f.write("<|im_end|>\n")
            f.write("<|im_start|>assistant\n")
            f.write(g["response"].strip() + "\n")
            f.write("<|im_end|>\n\n")
            written += 1
    return written


def export_training_data() -> dict:
    """Export corrections + good responses to training files."""
    corrections = get_corrections()
    good_examples = get_good_responses()
    total = len(corrections) + len(good_examples)

    if total == 0:
        print(_c(YELLOW, "⚠️  No training data found."))
        print("   Use 👍👎 in the web UI to build a dataset.")
        print("   Good responses (👍) count as positive training examples.")
        print("   Bad responses (👎) with typed corrections are the highest-quality signal.")
        return {"corrections": 0, "good_examples": 0, "total": 0}

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path  = TRAINING_DIR / "training.jsonl"
    chatml_path = TRAINING_DIR / "training.chatml.txt"

    n_jsonl  = export_alpaca_jsonl(corrections, good_examples, jsonl_path)
    n_chatml = export_chatml_txt(corrections, good_examples, chatml_path)

    print(_c(GREEN, f"✅ Exported {n_jsonl} training examples:"))
    if corrections:
        print(f"   • {len(corrections)} corrections (bad rating + user fix)")
    if good_examples:
        print(f"   • {len(good_examples)} good responses (positive examples)")
    print(f"   Alpaca JSONL  → {jsonl_path}")
    print(f"   ChatML (txt)  → {chatml_path}")
    return {
        "corrections":   len(corrections),
        "good_examples": len(good_examples),
        "total":         n_jsonl,
        "jsonl_path":    str(jsonl_path),
        "chatml_path":   str(chatml_path),
    }


# ── Ollama model detection ─────────────────────────────────────────────────────

def get_installed_ollama_models() -> list:
    """Return list of model names installed in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            if line.strip():
                models.append(line.split()[0])
        return models
    except Exception:
        return []


def get_best_base_model() -> Optional[str]:
    """Return the best installed Ollama model to use as a fine-tuning base."""
    installed = get_installed_ollama_models()
    for preferred in PREFERRED_BASE_MODELS:
        prefix = preferred.split(":")[0]
        for m in installed:
            if m.startswith(prefix):
                return m
    return installed[0] if installed else None


def is_tuned_model_available() -> bool:
    """Return True if rain-tuned model is registered in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        return TUNED_MODEL_NAME in result.stdout
    except Exception:
        return False


def get_ollama_model_gguf(model_name: str) -> Optional[Path]:
    """Ask Ollama for the path to a model's GGUF file."""
    try:
        payload = json.dumps({"name": model_name}).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        modelfile = data.get("modelfile", "")
        for line in modelfile.splitlines():
            if line.strip().upper().startswith("FROM "):
                path_str = line.strip()[5:].strip()
                p = Path(path_str)
                if p.exists() and p.suffix in (".gguf", ".bin"):
                    return p

        blob_dir = Path.home() / ".ollama" / "models" / "blobs"
        if blob_dir.exists():
            candidates = sorted(
                blob_dir.glob("sha256-*"),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            for c in candidates:
                if c.stat().st_size > 1_000_000_000:
                    return c
    except Exception as e:
        print(_c(DIM, f"   (Ollama model lookup: {e})"))
    return None


# ── mlx_lm training (Apple Silicon) ───────────────────────────────────────────

def _mlx_lm_available() -> bool:
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def get_hf_model_id(ollama_model: str) -> Optional[str]:
    """Map an Ollama model name to its HuggingFace equivalent."""
    for prefix, hf_id in HF_MODEL_MAP.items():
        if ollama_model.startswith(prefix):
            return hf_id
    return None


def _fuse_adapter_to_gguf(adapter_path: Path) -> Optional[Path]:
    """
    Fuse a trained MLX LoRA adapter into the base model and export as GGUF.
    Returns the GGUF path on success, None if the architecture isn't supported.

    mlx_lm fuse --export-gguf currently supports: llama, mistral, gemma.
    qwen2 support is not yet available (as of mlx_lm 0.31.0).
    """
    gguf_path = ADAPTER_DIR / "rain-tuned.gguf"
    mlx_model_path = MLX_MODEL_DIR / "qwen2.5-coder-7b"  # adjust if base model changes
    if not mlx_model_path.exists():
        return None
    print(_c(CYAN, "  Fusing adapter weights into GGUF…"))
    try:
        result = subprocess.run([
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model",        str(mlx_model_path),
            "--adapter-path", str(adapter_path),
            "--save-path",    str(ADAPTER_DIR / "rain-fused"),
            "--export-gguf",
            "--gguf-path",    str(gguf_path),
        ], capture_output=True, text=True)
        if result.returncode == 0 and gguf_path.exists():
            size_gb = gguf_path.stat().st_size / 1_073_741_824
            print(_c(GREEN, f"  ✅ GGUF exported ({size_gb:.1f} GB) → {gguf_path}"))
            return gguf_path
        # Architecture not supported — silent failure, caller handles it
        return None
    except Exception:
        return None


def run_mlx_training(
    training_data_dir: Path,
    model_name: str,
    iters: int = 200,
    lora_layers: int = 8,
    batch_size: int = 2,
) -> Optional[Path]:
    """
    Run LoRA fine-tuning via mlx_lm on Apple Silicon.

    Expects training data in ~/.rain/training/ with train.jsonl and valid.jsonl.
    mlx_lm uses its own data format — we handle conversion here.

    Returns the path to the adapter weights directory, or None on failure.
    """
    if not _mlx_lm_available():
        print(_c(RED, "❌ mlx_lm not installed."))
        print(_c(CYAN, "   Run: pip install mlx-lm"))
        return None

    # mlx_lm expects JSONL with {"text": "<full conversation>"} format
    # or the newer chat format. We convert from Alpaca format.
    mlx_data_dir = TRAINING_DIR / "mlx"
    mlx_data_dir.mkdir(parents=True, exist_ok=True)

    source_jsonl = TRAINING_DIR / "training.jsonl"
    if not source_jsonl.exists():
        print(_c(RED, "❌ Training data not found. Run --export first."))
        return None

    # Convert Alpaca JSONL → mlx_lm chat format
    examples = []
    with open(source_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # mlx_lm chat format
                examples.append({
                    "messages": [
                        {"role": "system",    "content": rec.get("system", FINETUNE_SYSTEM)},
                        {"role": "user",      "content": rec["instruction"]},
                        {"role": "assistant", "content": rec["output"]},
                    ]
                })
            except Exception:
                continue

    if not examples:
        print(_c(RED, "❌ No valid training examples found."))
        return None

    # 90/10 train/validation split
    split = max(1, int(len(examples) * 0.9))
    train_examples = examples[:split]
    valid_examples = examples[split:] or examples[:1]  # at least 1 validation example

    train_path = mlx_data_dir / "train.jsonl"
    valid_path = mlx_data_dir / "valid.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    adapter_path = ADAPTER_DIR / "mlx-lora"
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Find the mlx model — look for downloaded HF model first
    hf_id = get_hf_model_id(model_name)
    mlx_model_path = MLX_MODEL_DIR / model_name.replace(":", "-").replace("/", "-")

    if not mlx_model_path.exists():
        print(_c(YELLOW, f"⚠️  MLX model not found at {mlx_model_path}"))
        if hf_id:
            print(_c(CYAN, f"   Downloading from HuggingFace: {hf_id}"))
            print(_c(DIM,  f"   This may take a few minutes (~4 GB for 7B quantized)…"))
            try:
                subprocess.run([
                    sys.executable, "-m", "mlx_lm", "convert",
                    "--hf-path", hf_id,
                    "--mlx-path", str(mlx_model_path),
                    "-q",  # 4-bit quantization — fits in 16 GB comfortably
                ], check=True)
            except subprocess.CalledProcessError:
                print(_c(RED, "❌ Model download failed."))
                print("   Try manually: mlx_lm.convert --hf-path " + hf_id + " --mlx-path " + str(mlx_model_path) + " -q")
                return None
        else:
            print(_c(RED, f"❌ No HuggingFace mapping found for '{model_name}'."))
            return None

    print(_c(CYAN, f"\n⚡ Starting MLX LoRA training…"))
    print(_c(DIM,  f"   Model:       {mlx_model_path}"))
    print(_c(DIM,  f"   Examples:    {len(train_examples)} train / {len(valid_examples)} validation"))
    print(_c(DIM,  f"   Iterations:  {iters}"))
    print(_c(DIM,  f"   LoRA layers: {lora_layers}"))
    print()

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model",        str(mlx_model_path),
        "--train",
        "--data",         str(mlx_data_dir),
        "--adapter-path", str(adapter_path),
        "--iters",        str(iters),
        "--batch-size",   str(batch_size),
        "--num-layers",   str(lora_layers),
        "--save-every",   "50",
        "--val-batches",  "1",
    ]

    print(_c(DIM, "   " + " ".join(cmd)))
    print()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if "loss" in line.lower() or "val" in line.lower():
                print(_c(CYAN, "   " + line))
            elif "error" in line.lower() or "failed" in line.lower():
                print(_c(RED, "   " + line))
            else:
                print(_c(DIM, "   " + line))
        proc.wait()
        if proc.returncode != 0:
            print(_c(RED, f"\n❌ Training exited with code {proc.returncode}"))
            return None
        print(_c(GREEN, f"\n✅ MLX adapter written → {adapter_path}"))
        return adapter_path
    except KeyboardInterrupt:
        print(_c(YELLOW, "\n⚠️  Training interrupted."))
        proc.terminate()
        return None


# ── llama.cpp training (fallback) ─────────────────────────────────────────────

_LLAMA_FINETUNE_NAMES = ["llama-finetune", "llama-train", "finetune"]
_LLAMA_SEARCH_DIRS = [
    Path("/usr/local/bin"), Path("/usr/bin"),
    Path.home() / "llama.cpp" / "build" / "bin",
    Path.home() / "llama.cpp",
    Path.home() / ".local" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/opt/homebrew/opt/llama.cpp/bin"),
]


def find_llama_finetune() -> Optional[Path]:
    for name in _LLAMA_FINETUNE_NAMES:
        try:
            result = subprocess.run(["which", name], capture_output=True, text=True)
            if result.returncode == 0:
                p = Path(result.stdout.strip())
                if p.exists():
                    return p
        except Exception:
            pass
    for directory in _LLAMA_SEARCH_DIRS:
        for name in _LLAMA_FINETUNE_NAMES:
            p = directory / name
            if p.exists() and os.access(p, os.X_OK):
                return p
    return None


def run_lora_training_llamacpp(
    finetune_bin: Path,
    base_gguf: Path,
    training_data: Path,
    epochs: int = 3,
    lora_r: int = 4,
    threads: int = 4,
) -> Optional[Path]:
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    adapter_path = ADAPTER_DIR / "rain-lora.gguf"
    checkpoint_dir = ADAPTER_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    cmd = [
        str(finetune_bin),
        "--model-base",     str(base_gguf),
        "--lora-out",       str(adapter_path),
        "--train-data",     str(training_data),
        "--epochs",         str(epochs),
        "--batch",          "4",
        "--lora-r",         str(lora_r),
        "--lora-alpha",     str(lora_r * 2),
        "--threads",        str(threads),
        "--save-every",     "100",
        "--checkpoint-out", str(checkpoint_dir / "checkpoint"),
    ]

    print(_c(CYAN, "\n⚡ Starting llama.cpp LoRA training…"))
    print(_c(DIM, "   " + " ".join(cmd)))
    print()

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                if "loss" in line.lower():
                    print(_c(CYAN, "   " + line))
                elif "error" in line.lower() or "failed" in line.lower():
                    print(_c(RED, "   " + line))
                else:
                    print(_c(DIM, "   " + line))
        proc.wait()
        if proc.returncode != 0:
            print(_c(RED, f"\n❌ Training exited with code {proc.returncode}"))
            return None
        if adapter_path.exists():
            size_mb = adapter_path.stat().st_size / 1_048_576
            print(_c(GREEN, f"\n✅ Adapter written → {adapter_path} ({size_mb:.1f} MB)"))
            return adapter_path
        print(_c(RED, "\n❌ Training finished but adapter file was not created."))
        return None
    except FileNotFoundError:
        print(_c(RED, f"❌ Binary not found: {finetune_bin}"))
        return None
    except KeyboardInterrupt:
        print(_c(YELLOW, "\n⚠️  Training interrupted."))
        proc.terminate()
        return None


# ── Ollama model creation ──────────────────────────────────────────────────────

def write_modelfile(base_model: str, adapter_path=None) -> Path:
    lines = [f"FROM {base_model}"]
    if adapter_path and Path(adapter_path).exists():
        lines.append(f"ADAPTER {adapter_path}")
    lines.append("")
    lines.append(f'SYSTEM """{FINETUNE_SYSTEM}"""')
    lines.append("")
    lines.append("PARAMETER temperature 0.4")
    lines.append("PARAMETER top_p 0.9")
    lines.append("PARAMETER repeat_penalty 1.1")
    MODELFILE_PATH.write_text("\n".join(lines), encoding="utf-8")
    return MODELFILE_PATH


def create_ollama_model(base_model: str, adapter_path=None) -> bool:
    modelfile = write_modelfile(base_model, adapter_path)
    print(_c(CYAN, f"\n📦 Creating Ollama model '{TUNED_MODEL_NAME}'…"))
    print(_c(DIM, f"   Base:      {base_model}"))
    print(_c(DIM, f"   Modelfile: {modelfile}"))
    try:
        result = subprocess.run(
            ["ollama", "create", TUNED_MODEL_NAME, "-f", str(modelfile)],
            capture_output=False, text=True,
        )
        if result.returncode == 0:
            print(_c(GREEN, f"✅ '{TUNED_MODEL_NAME}' is now available in Ollama."))
            print("   Rain will automatically prefer it for primary agent calls.")
            return True
        print(_c(RED, f"❌ 'ollama create' failed (exit {result.returncode})"))
        return False
    except FileNotFoundError:
        print(_c(RED, "❌ 'ollama' binary not found — is Ollama installed?"))
        return False


# ── A/B report ─────────────────────────────────────────────────────────────────

def print_ab_report():
    with _db_connect() as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ab_results'"
        ).fetchone()
        if not exists:
            print(_c(YELLOW, "⚠️  No A/B results yet."))
            print("   A/B data is collected automatically once rain-tuned is active.")
            return
        rows = conn.execute("""
            SELECT model, AVG(confidence) AS avg_conf, COUNT(*) AS n
            FROM ab_results
            GROUP BY model
            ORDER BY avg_conf DESC
        """).fetchall()

    if not rows:
        print(_c(YELLOW, "⚠️  A/B results table exists but contains no data yet."))
        return

    print(_c(BOLD, "\n📊 A/B Model Performance Report"))
    print(_c(DIM, "─" * 44))
    print(f"  {'Model':<24} {'Avg Confidence':>14}  {'Queries':>7}")
    print(_c(DIM, "─" * 44))
    for r in rows:
        print(f"  {r['model']:<24} {r['avg_conf']:>13.1%}  {r['n']:>7}")
    print(_c(DIM, "─" * 44))

    if len(rows) >= 2:
        winner = rows[0]["model"]
        delta = rows[0]["avg_conf"] - rows[1]["avg_conf"]
        if delta > 0.02:
            print(_c(GREEN, f"\n  🏆 {winner} is winning by {delta:.1%}"))
        else:
            print(_c(YELLOW, "\n  ≈ Models are performing similarly — keep collecting data."))
    print()


# ── Setup instructions ─────────────────────────────────────────────────────────

def print_setup():
    print()
    print(_c(BOLD, "⛈️  Rain Fine-tuning Setup (Apple Silicon / M-series)"))
    print(_c(DIM, "─" * 52))
    print()
    print("  Step 1: Install mlx-lm")
    print(_c(CYAN, "  pip install mlx-lm"))
    print()
    print("  Step 2: Collect training data")
    print("  Use 👍👎 in the Rain web UI. Every good response counts.")
    print("  For 👎, typing a correction gives the highest-quality signal.")
    print()
    print("  Step 3: Export + train (one command)")
    print(_c(CYAN, "  python3 finetune.py --full"))
    print()
    print("  This will:")
    print("   • Export good responses + corrections to ~/.rain/training/")
    print("   • Download the base model from HuggingFace (~4 GB, first time only)")
    print("   • Run LoRA fine-tuning on your M1/M2/M3 GPU (~5–15 min for 200 iters)")
    print("   • Register 'rain-tuned' in Ollama")
    print("   • Rain's primary agent automatically prefers 'rain-tuned' going forward")
    print()
    base = get_best_base_model()
    if base:
        hf_id = get_hf_model_id(base)
        print(f"  Detected base model: {_c(GREEN, base)}")
        if hf_id:
            print(f"  HuggingFace source:  {_c(DIM, hf_id)}")
    print()


# ── Status display ─────────────────────────────────────────────────────────────

def print_status():
    stats = get_feedback_stats()
    tuned_available = is_tuned_model_available()
    mlx_ok = _mlx_lm_available()
    llama_bin = find_llama_finetune()
    base_model = get_best_base_model()

    total_examples = stats.get("good", 0) + stats.get("corrections", 0)

    print()
    print(_c(BOLD, "⛈️  Rain Fine-tuning Status"))
    print(_c(DIM, "─" * 44))

    print(f"  👍 Good responses    {stats.get('good', 0):>6}   ← positive training examples")
    print(f"  👎 Bad responses     {stats.get('bad', 0):>6}")
    print(f"  ✏️  With corrections  {stats.get('corrections', 0):>6}   ← highest-quality signal")
    print(f"  📦 Total trainable   {total_examples:>6}")
    print(_c(DIM, "─" * 44))

    if total_examples == 0:
        print(_c(YELLOW, "  ⚠️  No training data yet."))
        print("     Use 👍👎 in the web UI. Good responses count too!")
    elif total_examples < MIN_EXAMPLES:
        needed = MIN_EXAMPLES - total_examples
        print(_c(YELLOW, f"  ⚠️  {total_examples}/{MIN_EXAMPLES} examples — need {needed} more before training."))
    else:
        print(_c(GREEN, f"  ✅ {total_examples} examples — ready to train."))

    print()
    print(f"  Base model:  {_c(GREEN, base_model) if base_model else _c(RED, 'none found')}")
    print(f"  mlx-lm:      {_c(GREEN, '✅ installed') if mlx_ok else _c(YELLOW, '○  not installed  (pip install mlx-lm)')}")
    print(f"  llama.cpp:   {_c(GREEN, str(llama_bin)) if llama_bin else _c(DIM, '○  not found (optional — mlx-lm preferred)')}")
    print()

    if tuned_available:
        print(_c(GREEN, f"  ✅ '{TUNED_MODEL_NAME}' is registered in Ollama"))
        print("     Rain's primary agent automatically prefers this model.")
    else:
        print(_c(DIM, f"  ○  '{TUNED_MODEL_NAME}' not yet created"))

    jsonl_path = TRAINING_DIR / "training.jsonl"
    if jsonl_path.exists():
        n_lines = sum(1 for _ in open(jsonl_path))
        print()
        print(_c(GREEN, f"  ✅ Training data exported: {n_lines} examples"))
        print(_c(DIM,   f"     {jsonl_path}"))

    print()
    if total_examples >= MIN_EXAMPLES and not mlx_ok:
        print(_c(CYAN, "  ▶  Next: pip install mlx-lm  then  python3 finetune.py --full"))
    elif total_examples >= MIN_EXAMPLES and mlx_ok:
        print(_c(CYAN, "  ▶  Ready: python3 finetune.py --full"))
    elif tuned_available:
        print(_c(CYAN, "  ▶  Run: python3 finetune.py --ab-report  to see performance"))
    else:
        print(_c(CYAN, "  ▶  Keep using Rain with 👍👎 — each click improves training data"))
    print()


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_full_pipeline(iters: int = 200, lora_layers: int = 8,
                      epochs: int = 3, lora_r: int = 4, threads: int = 4):
    """Export → train → create-model in one shot."""
    stats = get_feedback_stats()
    total_examples = stats.get("good", 0) + stats.get("corrections", 0)

    if total_examples < MIN_EXAMPLES:
        print(_c(YELLOW, f"⚠️  Only {total_examples} training examples (minimum {MIN_EXAMPLES})."))
        if total_examples > 0:
            print("   Training on very little data risks overfitting.")
            ans = input("   Continue anyway? [y/N] ").strip().lower()
            if ans != "y":
                print("   Aborted.")
                return
        else:
            print("   Nothing to train on. Use 👍👎 in the UI first.")
            return

    base_model = get_best_base_model()
    if not base_model:
        print(_c(RED, "❌ No Ollama models found. Is Ollama running?"))
        return

    # Step 1: Export
    print(_c(BOLD, "\n── Step 1/3: Export training data ──────────────────"))
    export_result = export_training_data()
    if export_result.get("total", 0) == 0:
        print(_c(RED, "❌ Export failed."))
        return

    # Step 2: Train — prefer mlx_lm on Apple Silicon
    print(_c(BOLD, "\n── Step 2/3: LoRA training ─────────────────────────"))

    adapter_path = None
    if _mlx_lm_available():
        print(_c(GREEN, "  Using mlx-lm (Apple Silicon GPU)"))
        adapter_path = run_mlx_training(
            TRAINING_DIR, base_model,
            iters=iters, lora_layers=lora_layers,
        )
    else:
        # Fall back to llama.cpp
        finetune_bin = find_llama_finetune()
        if finetune_bin:
            print(_c(DIM, f"  Using llama.cpp: {finetune_bin}"))
            chatml_path = TRAINING_DIR / "training.chatml.txt"
            base_gguf = get_ollama_model_gguf(base_model)
            if base_gguf:
                adapter_path = run_lora_training_llamacpp(
                    finetune_bin, base_gguf, chatml_path,
                    epochs=epochs, lora_r=lora_r, threads=threads,
                )
            else:
                print(_c(RED, f"❌ Cannot locate GGUF for '{base_model}'."))
        else:
            print(_c(YELLOW, "  ⚠️  No training backend available."))
            print()
            print("  Install mlx-lm for training on Apple Silicon:")
            print(_c(CYAN, "    pip install mlx-lm"))
            print()
            print("  Or use the exported JSONL with any fine-tuning framework:")
            print(_c(DIM, f"    {TRAINING_DIR / 'training.jsonl'}"))
            print()
            print("  Popular options:")
            print(_c(DIM, "    • Unsloth (Colab, free GPU): https://github.com/unslothai/unsloth"))
            print(_c(DIM, "    • HuggingFace TRL SFTTrainer: pip install trl"))

    # Step 3: Fuse adapter into GGUF and register model
    print(_c(BOLD, "\n── Step 3/3: Register Ollama model ─────────────────"))
    if adapter_path and adapter_path.exists():
        gguf_path = _fuse_adapter_to_gguf(adapter_path)
        if gguf_path:
            create_ollama_model(base_model, gguf_path)
        else:
            print(_c(YELLOW, "  ⚠️  GGUF export not supported for this model architecture yet."))
            print(   "     Registering with behavioral system prompt (adapter weights preserved).")
            print(_c(DIM,  f"     Adapter saved at: {adapter_path}"))
            print(_c(DIM,   "     Re-run --full after mlx_lm adds qwen2 GGUF support to include LoRA weights."))
            create_ollama_model(base_model, adapter_path=None)
    else:
        create_ollama_model(base_model, adapter_path=None)

    print(_c(BOLD + GREEN, "\n⛈️  Pipeline complete.\n"))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="finetune.py",
        description="Rain ⛈️  Phase 5B — fine-tuning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python3 finetune.py --status
          python3 finetune.py --setup
          python3 finetune.py --export
          python3 finetune.py --full
          python3 finetune.py --full --iters 400 --lora-layers 16
          python3 finetune.py --ab-report
        """),
    )

    parser.add_argument("--status",       action="store_true", help="Show feedback stats and pipeline readiness")
    parser.add_argument("--setup",        action="store_true", help="Show setup instructions for mlx-lm on Apple Silicon")
    parser.add_argument("--export",       action="store_true", help="Export training data to ~/.rain/training/")
    parser.add_argument("--train",        action="store_true", help="Run LoRA training (mlx-lm preferred, llama.cpp fallback)")
    parser.add_argument("--create-model", action="store_true", help="Register rain-tuned in Ollama")
    parser.add_argument("--full",         action="store_true", help="Export + train + create-model in one shot")
    parser.add_argument("--ab-report",    action="store_true", help="Show A/B performance comparison")

    # mlx_lm training options
    parser.add_argument("--iters",        type=int, default=200, help="Training iterations for mlx-lm (default: 200)")
    parser.add_argument("--lora-layers",  type=int, default=8,   help="LoRA layers for mlx-lm (default: 8; higher = more capacity)")
    # llama.cpp training options (fallback)
    parser.add_argument("--epochs",       type=int, default=3,   help="Training epochs for llama.cpp (default: 3)")
    parser.add_argument("--lora-r",       type=int, default=4,   help="LoRA rank for llama.cpp (default: 4)")
    parser.add_argument("--threads",      type=int, default=4,   help="CPU threads for llama.cpp (default: 4)")

    parser.add_argument("--adapter",      type=str, default=None, help="Path to existing adapter (for --create-model)")
    parser.add_argument("--base-model",   type=str, default=None, help="Override base model name")

    args = parser.parse_args()

    if not any([args.status, args.setup, args.export, args.train,
                args.create_model, args.full, args.ab_report]):
        args.status = True

    if args.status:
        print_status()

    if args.setup:
        print_setup()

    if args.export:
        print()
        export_training_data()
        print()

    if args.train:
        base_model = args.base_model or get_best_base_model()
        if not base_model:
            print(_c(RED, "❌ No base model found."))
            sys.exit(1)
        if _mlx_lm_available():
            run_mlx_training(TRAINING_DIR, base_model,
                             iters=args.iters, lora_layers=args.lora_layers)
        else:
            finetune_bin = find_llama_finetune()
            if not finetune_bin:
                print(_c(RED, "❌ No training backend found."))
                print(_c(CYAN, "   Install mlx-lm: pip install mlx-lm"))
                sys.exit(1)
            chatml_path = TRAINING_DIR / "training.chatml.txt"
            base_gguf = get_ollama_model_gguf(base_model)
            if not base_gguf:
                print(_c(RED, f"❌ Cannot locate GGUF for '{base_model}'."))
                sys.exit(1)
            run_lora_training_llamacpp(
                finetune_bin, base_gguf, chatml_path,
                epochs=args.epochs, lora_r=args.lora_r, threads=args.threads,
            )

    if args.create_model:
        base_model = args.base_model or get_best_base_model()
        if not base_model:
            print(_c(RED, "❌ No base model found."))
            sys.exit(1)
        adapter = Path(args.adapter) if args.adapter else None
        create_ollama_model(base_model, adapter)

    if args.full:
        run_full_pipeline(
            iters=args.iters,
            lora_layers=args.lora_layers,
            epochs=args.epochs,
            lora_r=args.lora_r,
            threads=args.threads,
        )

    if args.ab_report:
        print_ab_report()


if __name__ == "__main__":
    main()
