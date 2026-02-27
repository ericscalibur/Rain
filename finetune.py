#!/usr/bin/env python3
"""
Rain ⛈️ - Fine-tuning Pipeline (Phase 5B)

Exports correction feedback to training data and orchestrates LoRA adapter
training via llama.cpp, then registers the result as a new Ollama model.

Usage:
    python3 finetune.py --status                 # show feedback stats
    python3 finetune.py --export                 # export training data to ~/.rain/training/
    python3 finetune.py --train                  # run llama.cpp LoRA training
    python3 finetune.py --create-model           # register rain-tuned in Ollama
    python3 finetune.py --full                   # export + train + create-model in one shot
    python3 finetune.py --ab-report              # show A/B comparison from logged results
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

RAIN_DIR      = Path.home() / ".rain"
DB_PATH       = RAIN_DIR / "memory.db"
TRAINING_DIR  = RAIN_DIR / "training"
ADAPTER_DIR   = RAIN_DIR / "adapters"
MODELFILE_PATH = RAIN_DIR / "Modelfile.rain-tuned"

TUNED_MODEL_NAME = "rain-tuned"
BASE_MODEL       = "llama3.1"
MIN_EXAMPLES     = 10   # minimum corrections before training makes sense

# ── System prompt used for all training examples ───────────────────────────────

FINETUNE_SYSTEM = (
    "You are Rain, a sovereign AI assistant running locally on the user's computer. "
    "You are direct, precise, and honest about uncertainty. "
    "You never use third-party Python packages when stdlib alternatives exist. "
    "For Bitcoin and blockchain data you use the mempool.space public REST API. "
    "You never output HTML tags or markup inside code blocks. "
    "You never reference an internal critique or reflection process in your answers."
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
        # Check if feedback table exists yet
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
    """Fetch all bad-rated feedback entries that have a user correction."""
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


# ── Training data export ───────────────────────────────────────────────────────

def export_alpaca_jsonl(corrections: list, out_path: Path) -> int:
    """
    Export corrections in Alpaca instruction-following JSONL format.
    Compatible with HuggingFace TRL, Unsloth, and most fine-tuning frameworks.

    Format:
        {"instruction": "<query>", "input": "", "output": "<correction>"}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for c in corrections:
            record = {
                "instruction": c["query"].strip(),
                "input": "",
                "output": c["correction"].strip(),
                "system": FINETUNE_SYSTEM,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def export_chatml_txt(corrections: list, out_path: Path) -> int:
    """
    Export corrections in ChatML format for llama.cpp finetune.
    Each example is a fully-formatted conversation separated by a blank line.

    Format:
        <|im_start|>system
        {system}
        <|im_end|>
        <|im_start|>user
        {query}
        <|im_end|>
        <|im_start|>assistant
        {correction}
        <|im_end|>
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
    return written


def export_training_data() -> dict:
    """Export all corrections to both JSONL and ChatML formats."""
    corrections = get_corrections()
    if not corrections:
        print(_c(YELLOW, "⚠️  No corrections found in feedback database."))
        print(   "   Use the 👎 button in the web UI to save corrections first.")
        return {"corrections": 0}

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = TRAINING_DIR / "corrections.jsonl"
    chatml_path = TRAINING_DIR / "corrections.chatml.txt"

    n_jsonl  = export_alpaca_jsonl(corrections, jsonl_path)
    n_chatml = export_chatml_txt(corrections, chatml_path)

    print(_c(GREEN, f"✅ Exported {n_jsonl} corrections:"))
    print(f"   Alpaca JSONL  → {jsonl_path}")
    print(f"   ChatML (txt)  → {chatml_path}")
    return {
        "corrections": n_jsonl,
        "jsonl_path":  str(jsonl_path),
        "chatml_path": str(chatml_path),
    }


# ── llama.cpp discovery ────────────────────────────────────────────────────────

_LLAMA_FINETUNE_NAMES = [
    "llama-finetune",
    "finetune",
    "llama-finetune-gguf",
]

_LLAMA_SEARCH_DIRS = [
    Path("/usr/local/bin"),
    Path("/usr/bin"),
    Path.home() / "llama.cpp" / "build" / "bin",
    Path.home() / "llama.cpp",
    Path.home() / ".local" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/opt/homebrew/opt/llama.cpp/bin"),
]


def find_llama_finetune() -> Optional[Path]:
    """Search common locations for the llama.cpp finetune binary."""
    # First, check PATH
    for name in _LLAMA_FINETUNE_NAMES:
        try:
            result = subprocess.run(
                ["which", name], capture_output=True, text=True
            )
            if result.returncode == 0:
                p = Path(result.stdout.strip())
                if p.exists():
                    return p
        except Exception:
            pass

    # Then scan known directories
    for directory in _LLAMA_SEARCH_DIRS:
        for name in _LLAMA_FINETUNE_NAMES:
            p = directory / name
            if p.exists() and os.access(p, os.X_OK):
                return p

    return None


# ── Ollama model discovery ─────────────────────────────────────────────────────

def get_ollama_model_gguf(model_name: str = BASE_MODEL) -> Optional[Path]:
    """
    Ask Ollama for the path to a model's GGUF file.
    Ollama stores GGUF blobs in ~/.ollama/models/blobs/ keyed by sha256.
    We get the hash via the /api/show endpoint.
    """
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

        # The modelfile may have a FROM line pointing to the GGUF
        modelfile = data.get("modelfile", "")
        for line in modelfile.splitlines():
            if line.strip().upper().startswith("FROM "):
                path_str = line.strip()[5:].strip()
                p = Path(path_str)
                if p.exists() and p.suffix in (".gguf", ".bin"):
                    return p

        # Try the blobs directory using model details
        details = data.get("details", {})
        blob_dir = Path.home() / ".ollama" / "models" / "blobs"
        if blob_dir.exists():
            # Find the largest GGUF blob (the model weights)
            candidates = sorted(
                blob_dir.glob("sha256-*"),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            for c in candidates:
                if c.stat().st_size > 1_000_000_000:  # >1 GB — likely a model
                    return c

    except Exception as e:
        print(_c(DIM, f"   (Ollama model lookup: {e})"))

    return None


def is_tuned_model_available() -> bool:
    """Return True if rain-tuned model is registered in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        return TUNED_MODEL_NAME in result.stdout
    except Exception:
        return False


# ── Training ───────────────────────────────────────────────────────────────────

def run_lora_training(
    finetune_bin: Path,
    base_gguf: Path,
    training_data: Path,
    epochs: int = 3,
    lora_r: int = 4,
    threads: int = 4,
) -> Optional[Path]:
    """
    Run llama.cpp finetune to produce a LoRA adapter GGUF.

    Returns the path to the generated adapter, or None on failure.
    """
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    adapter_path = ADAPTER_DIR / "rain-lora.gguf"
    checkpoint_dir = ADAPTER_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    cmd = [
        str(finetune_bin),
        "--model-base",    str(base_gguf),
        "--lora-out",      str(adapter_path),
        "--train-data",    str(training_data),
        "--epochs",        str(epochs),
        "--batch",         "4",
        "--lora-r",        str(lora_r),
        "--lora-alpha",    str(lora_r * 2),
        "--threads",       str(threads),
        "--save-every",    "100",
        "--checkpoint-out", str(checkpoint_dir / "checkpoint"),
    ]

    print(_c(CYAN, "\n⚡ Starting LoRA training…"))
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
            if line:
                # Highlight loss lines
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
        else:
            print(_c(RED, "\n❌ Training finished but adapter file was not created."))
            return None
    except FileNotFoundError:
        print(_c(RED, f"❌ Binary not found: {finetune_bin}"))
        return None
    except KeyboardInterrupt:
        print(_c(YELLOW, "\n⚠️  Training interrupted by user."))
        proc.terminate()
        return None


# ── Ollama model creation ──────────────────────────────────────────────────────

def write_modelfile(adapter_path: Optional[Path] = None) -> Path:
    """
    Write an Ollama Modelfile for rain-tuned.
    If an adapter path is given, it includes the ADAPTER directive.
    """
    lines = [f"FROM {BASE_MODEL}"]
    if adapter_path and adapter_path.exists():
        lines.append(f"ADAPTER {adapter_path}")
    lines.append("")
    lines.append(f'SYSTEM """{FINETUNE_SYSTEM}"""')
    lines.append("")
    lines.append("PARAMETER temperature 0.7")
    lines.append("PARAMETER top_p 0.9")
    lines.append("PARAMETER repeat_penalty 1.1")

    MODELFILE_PATH.write_text("\n".join(lines), encoding="utf-8")
    return MODELFILE_PATH


def create_ollama_model(adapter_path: Optional[Path] = None) -> bool:
    """
    Register rain-tuned in Ollama using the generated Modelfile.
    Returns True on success.
    """
    modelfile = write_modelfile(adapter_path)
    print(_c(CYAN, f"\n📦 Creating Ollama model '{TUNED_MODEL_NAME}'…"))
    print(_c(DIM, f"   Modelfile → {modelfile}"))

    try:
        result = subprocess.run(
            ["ollama", "create", TUNED_MODEL_NAME, "-f", str(modelfile)],
            capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            print(_c(GREEN, f"✅ '{TUNED_MODEL_NAME}' is now available in Ollama."))
            print(   "   Rain will automatically prefer it for primary agent calls.")
            return True
        else:
            print(_c(RED, f"❌ 'ollama create' failed (exit {result.returncode})"))
            return False
    except FileNotFoundError:
        print(_c(RED, "❌ 'ollama' binary not found — is Ollama installed?"))
        return False


# ── A/B report ─────────────────────────────────────────────────────────────────

def print_ab_report():
    """
    Print a comparison of base model vs rain-tuned model performance
    from the ab_results table (written by MultiAgentOrchestrator when
    the tuned model is active).
    """
    with _db_connect() as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ab_results'"
        ).fetchone()
        if not exists:
            print(_c(YELLOW, "⚠️  No A/B results yet."))
            print(   "   A/B data is collected automatically once rain-tuned is active.")
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
        bar_len = int(r["avg_conf"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {r['model']:<24} {r['avg_conf']:>13.1%}  {r['n']:>7}")
    print(_c(DIM, "─" * 44))

    if len(rows) >= 2:
        winner = rows[0]["model"]
        delta  = rows[0]["avg_conf"] - rows[1]["avg_conf"]
        if delta > 0.02:
            print(_c(GREEN, f"\n  🏆 {winner} is winning by {delta:.1%}"))
            if winner == TUNED_MODEL_NAME:
                print("     Rain is routing primary queries to the tuned model.")
        else:
            print(_c(YELLOW, "\n  ≈ Models are performing similarly — keep collecting data."))
    print()


# ── Status display ─────────────────────────────────────────────────────────────

def print_status():
    stats = get_feedback_stats()
    tuned_available = is_tuned_model_available()

    print()
    print(_c(BOLD, "⛈️  Rain Fine-tuning Status"))
    print(_c(DIM, "─" * 44))

    # Feedback stats
    total       = stats.get("total", 0)
    good        = stats.get("good", 0)
    bad         = stats.get("bad", 0)
    corrections = stats.get("corrections", 0)

    print(f"  👍 Good responses    {good:>6}")
    print(f"  👎 Bad responses     {bad:>6}")
    print(f"  ✏️  With corrections  {corrections:>6}   ← training data")
    print(_c(DIM, "─" * 44))

    # Readiness
    if corrections == 0:
        print(_c(YELLOW, "  ⚠️  No corrections yet."))
        print(   "     Use 👎 in the web UI and write corrections to build a dataset.")
    elif corrections < MIN_EXAMPLES:
        needed = MIN_EXAMPLES - corrections
        print(_c(YELLOW, f"  ⚠️  {corrections}/{MIN_EXAMPLES} corrections — need {needed} more before training."))
    else:
        print(_c(GREEN, f"  ✅ {corrections} corrections — ready to train."))

    # Tuned model
    print()
    if tuned_available:
        print(_c(GREEN, f"  ✅ '{TUNED_MODEL_NAME}' is registered in Ollama"))
        print(   "     Rain's primary agent automatically prefers this model.")
    else:
        print(_c(DIM, f"  ○  '{TUNED_MODEL_NAME}' not yet created"))

    # llama.cpp
    finetune_bin = find_llama_finetune()
    print()
    if finetune_bin:
        print(_c(GREEN, f"  ✅ llama.cpp finetune found: {finetune_bin}"))
    else:
        print(_c(DIM, "  ○  llama.cpp finetune binary not found"))
        print(_c(DIM, "     Install: https://github.com/ggerganov/llama.cpp"))

    # Training data
    jsonl_path = TRAINING_DIR / "corrections.jsonl"
    if jsonl_path.exists():
        n_lines = sum(1 for _ in open(jsonl_path))
        print()
        print(_c(GREEN, f"  ✅ Training data exported: {jsonl_path}"))
        print(_c(DIM,   f"     {n_lines} examples (Alpaca JSONL)"))
        print(_c(DIM,   f"     {TRAINING_DIR / 'corrections.chatml.txt'} (ChatML)"))

    print()

    # Next step hint
    if corrections >= MIN_EXAMPLES and not tuned_available:
        print(_c(CYAN, "  ▶  Next: python3 finetune.py --full"))
    elif tuned_available:
        print(_c(CYAN, "  ▶  Run: python3 finetune.py --ab-report  to see performance"))
    elif corrections < MIN_EXAMPLES and corrections > 0:
        print(_c(CYAN, "  ▶  Keep giving Rain feedback — each correction improves training"))
    print()


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_full_pipeline(epochs: int = 3, lora_r: int = 4, threads: int = 4):
    """Export → train → create-model in one shot."""
    stats = get_feedback_stats()
    corrections = stats.get("corrections", 0)

    if corrections < MIN_EXAMPLES:
        print(_c(YELLOW, f"⚠️  Only {corrections} corrections (minimum {MIN_EXAMPLES})."))
        if corrections > 0:
            print("   Training on very little data risks overfitting.")
            ans = input("   Continue anyway? [y/N] ").strip().lower()
            if ans != "y":
                print("   Aborted.")
                return
        else:
            print("   Nothing to train on. Use 👎 in the UI to save corrections first.")
            return

    # Step 1: Export
    print(_c(BOLD, "\n── Step 1/3: Export training data ──────────────────"))
    export_result = export_training_data()
    chatml_path = Path(export_result.get("chatml_path", ""))
    if not chatml_path.exists():
        print(_c(RED, "❌ Export failed."))
        return

    # Step 2: Train
    print(_c(BOLD, "\n── Step 2/3: LoRA training ─────────────────────────"))
    finetune_bin = find_llama_finetune()
    if not finetune_bin:
        print(_c(YELLOW, "⚠️  llama.cpp finetune binary not found."))
        print()
        print("   To train locally, install llama.cpp and build the finetune binary:")
        print(_c(DIM, "   git clone https://github.com/ggerganov/llama.cpp"))
        print(_c(DIM, "   cd llama.cpp && cmake -B build && cmake --build build --target llama-finetune -j"))
        print()
        print("   Alternatively, use the exported JSONL with any fine-tuning framework:")
        print(_c(DIM, f"   {TRAINING_DIR / 'corrections.jsonl'}"))
        print()
        print("   Popular options:")
        print(_c(DIM, "   • Unsloth (fast, free, Colab-compatible): https://github.com/unslothai/unsloth"))
        print(_c(DIM, "   • HuggingFace TRL SFTTrainer:             pip install trl"))
        print(_c(DIM, "   • LM Studio fine-tune:                    https://lmstudio.ai"))
        print()
        print(_c(CYAN, "   Once you have an adapter GGUF, run:"))
        print(_c(DIM,  "   python3 finetune.py --create-model --adapter /path/to/adapter.gguf"))
        return

    base_gguf = get_ollama_model_gguf(BASE_MODEL)
    if not base_gguf:
        print(_c(RED, f"❌ Could not locate base model GGUF for '{BASE_MODEL}'."))
        print(   "   Ensure Ollama is running and the model is pulled:")
        print(_c(DIM, f"   ollama pull {BASE_MODEL}"))
        return

    print(f"   Base model GGUF: {_c(DIM, str(base_gguf))}")
    adapter_path = run_lora_training(
        finetune_bin=finetune_bin,
        base_gguf=base_gguf,
        training_data=chatml_path,
        epochs=epochs,
        lora_r=lora_r,
        threads=threads,
    )

    # Step 3: Create model
    print(_c(BOLD, "\n── Step 3/3: Register Ollama model ─────────────────"))
    if adapter_path:
        create_ollama_model(adapter_path)
    else:
        print(_c(YELLOW, "⚠️  Training did not produce an adapter."))
        print("   Creating rain-tuned with system prompt only (no LoRA weights).")
        create_ollama_model(adapter_path=None)

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
          python3 finetune.py --export
          python3 finetune.py --full
          python3 finetune.py --full --epochs 5 --lora-r 8
          python3 finetune.py --create-model --adapter ~/.rain/adapters/rain-lora.gguf
          python3 finetune.py --ab-report
        """),
    )

    parser.add_argument("--status",       action="store_true", help="Show feedback stats and pipeline readiness")
    parser.add_argument("--export",       action="store_true", help="Export corrections to ~/.rain/training/")
    parser.add_argument("--train",        action="store_true", help="Run llama.cpp LoRA training")
    parser.add_argument("--create-model", action="store_true", help="Register rain-tuned in Ollama")
    parser.add_argument("--full",         action="store_true", help="Export + train + create-model in one shot")
    parser.add_argument("--ab-report",    action="store_true", help="Show A/B performance comparison")

    parser.add_argument("--epochs",  type=int, default=3,  help="Training epochs (default: 3)")
    parser.add_argument("--lora-r",  type=int, default=4,  help="LoRA rank (default: 4; higher = more capacity)")
    parser.add_argument("--threads", type=int, default=4,  help="CPU threads for training (default: 4)")
    parser.add_argument("--adapter", type=str, default=None, help="Path to existing adapter GGUF (for --create-model)")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL, help=f"Base Ollama model (default: {BASE_MODEL})")

    args = parser.parse_args()

    # Default to --status if no action given
    if not any([args.status, args.export, args.train, args.create_model, args.full, args.ab_report]):
        args.status = True

    if args.status:
        print_status()

    if args.export:
        print()
        export_training_data()
        print()

    if args.train:
        finetune_bin = find_llama_finetune()
        if not finetune_bin:
            print(_c(RED, "❌ llama.cpp finetune binary not found."))
            print("   Build it from https://github.com/ggerganov/llama.cpp")
            sys.exit(1)
        # Ensure training data is exported first
        chatml_path = TRAINING_DIR / "corrections.chatml.txt"
        if not chatml_path.exists():
            print("   Training data not yet exported — exporting now…")
            export_training_data()
        if not chatml_path.exists():
            print(_c(RED, "❌ No training data available."))
            sys.exit(1)
        base_gguf = get_ollama_model_gguf(args.base_model)
        if not base_gguf:
            print(_c(RED, f"❌ Cannot locate GGUF for '{args.base_model}'."))
            sys.exit(1)
        run_lora_training(
            finetune_bin=finetune_bin,
            base_gguf=base_gguf,
            training_data=chatml_path,
            epochs=args.epochs,
            lora_r=args.lora_r,
            threads=args.threads,
        )

    if args.create_model:
        adapter = Path(args.adapter) if args.adapter else (ADAPTER_DIR / "rain-lora.gguf")
        if args.adapter and not adapter.exists():
            print(_c(RED, f"❌ Adapter not found: {adapter}"))
            sys.exit(1)
        if not args.adapter and not adapter.exists():
            adapter = None  # create model with system prompt only
        create_ollama_model(adapter)

    if args.full:
        run_full_pipeline(
            epochs=args.epochs,
            lora_r=args.lora_r,
            threads=args.threads,
        )

    if args.ab_report:
        print_ab_report()


if __name__ == "__main__":
    main()
