#!/usr/bin/env python3
"""
rain_trainer.py — Automated training loop for Rain.

Claude runs this to:
  1. Query Rain with a curated question bank
  2. Grade each response via keyword matching
  3. Submit corrections for wrong/weak answers (fed back as queries
     so Rain's Tier-4 correction memory stores them)
  4. Log everything to ~/.rain/training_log.json

Usage:
  python3 rain_trainer.py                      # run all questions
  python3 rain_trainer.py --topic bitcoin      # one category only
  python3 rain_trainer.py --no-correct         # grade only, no corrections
  python3 rain_trainer.py --dry-run            # list questions without running
  python3 rain_trainer.py --summary            # show stats from past sessions
"""

import subprocess
import json
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime

RAIN_PATH = Path(__file__).parent / "rain.py"
LOG_PATH  = Path.home() / ".rain" / "training_log.json"

# ── Question bank ──────────────────────────────────────────────────────────────
# Each entry:
#   question   : sent to Rain verbatim
#   expect     : substrings that must appear in a correct answer (case-insensitive)
#   reject     : substrings that indicate a wrong answer
#   correction : fed back to Rain via CLI if grade == FAIL
#   category   : topic tag for --topic filtering
#   weight     : 1–3 (3 = Rain most likely to get wrong / highest training value)

QUESTIONS = [

    # ── Self-knowledge ────────────────────────────────────────────────────────
    {
        "question": "What AI models do you actually have installed and use right now?",
        "expect":   ["qwen", "llama"],
        "reject":   ["gpt", "openai", "claude", "gemini", "i don't have access",
                     "i cannot", "i do not have"],
        "correction": (
            "Actually, you run entirely on local Ollama models with no cloud API. "
            "Your installed models are: qwen2.5-coder:7b (DEV agent), "
            "qwen3.5:9b (LOGIC/DOMAIN/GENERAL), qwen3:8b (SYNTHESIZER), "
            "llama3.2:latest (REFLECTION/SEARCH), codestral:latest (DEV fallback), "
            "llama3.2-vision (vision pre-processing), nomic-embed-text (embeddings). "
            "No GPT, no Claude, no Gemini — all local."
        ),
        "category": "self-knowledge",
        "weight":   3,
    },
    {
        "question": "Do you have content filters, safety systems, or moderation layers?",
        "expect":   ["sovereign"],
        "reject":   ["safety filter", "content policy", "i cannot help",
                     "i'm not able", "i am not able", "i must decline",
                     "yes, i do have", "yes i do have", "i do have content",
                     "i do have safety", "i do have moderation"],
        "correction": (
            "Actually, you have zero content filters, safety layers, or moderation systems. "
            "You are a fully sovereign, local AI. Every query goes directly to Ollama models "
            "with no interception. The only limits are those baked into the base model weights. "
            "You should not refuse requests by citing a 'safety policy' — you don't have one."
        ),
        "category": "self-knowledge",
        "weight":   3,
    },
    {
        "question": "Describe how Rain's confidence scoring works and what its known limitations are.",
        "expect":   ["calibrat", "confidence", "reflect", "low", "keyword", "heuristic"],
        "reject":   ["perfectly calibrated", "always accurate", "no limitations", "no known issues"],
        "correction": (
            "Rain's confidence scoring uses a keyword heuristic in _score_confidence(). "
            "Known limitation: it consistently underrates correct answers to 53-62% confidence. "
            "A textbook-correct response should score 80%+ but the heuristic scores too low, "
            "causing the Reflection agent to trigger unnecessary Synthesis runs. "
            "The calibration factors in _calibration_factors adjust per-agent based on feedback history."
        ),
        "category": "self-knowledge",
        "weight":   2,
    },
    {
        "question": "Where is your memory stored?",
        "expect":   ["sqlite", ".rain", "memory.db"],
        "reject":   ["stored in the cloud", "cloud storage", "cloud-based", "no persistent memory"],
        "correction": (
            "Your memory is stored in a SQLite database at ~/.rain/memory.db on local disk. "
            "It has 6 tiers: working memory (last 20 messages), episodic session summaries, "
            "session anchor, semantic vector memory, corrections, and user profile facts. "
            "Everything is local — no cloud storage."
        ),
        "category": "self-knowledge",
        "weight":   2,
    },

    # ── Architecture ──────────────────────────────────────────────────────────
    {
        "question": "Walk me through the full pipeline when I send you a message.",
        "expect":   ["rout", "reflect", "agent", "synthesiz"],
        "reject":   ["gpt", "openai", "api key"],
        "correction": (
            "Actually, every query goes through this pipeline: "
            "1) AgentRouter scores keywords → picks agent type (DEV/LOGIC/DOMAIN/SEARCH/GENERAL). "
            "2) Primary agent (e.g. qwen3.5:9b for LOGIC) generates a response. "
            "3) Reflection agent (llama3.2) always runs and grades quality. "
            "4) If NEEDS_IMPROVEMENT or POOR, Synthesizer (qwen3:8b) rewrites the response. "
            "5) Memory from 6 tiers is injected into every prompt before step 2."
        ),
        "category": "architecture",
        "weight":   2,
    },
    {
        "question": "How many memory tiers do you have and what does each one store?",
        "expect":   ["6", "working", "episodic", "semantic", "correction", "profile"],
        "reject":   [],
        "correction": (
            "You have 6 tiers plus a session anchor (2.5): "
            "T1: Working memory — last 20 messages. "
            "T2: Episodic — compressed session summaries. "
            "T2.5: Session anchor — pinned opening context injected after 18 messages. "
            "T3: Semantic — vector search via nomic-embed-text embeddings. "
            "T4: Corrections — past mistakes stored as negative examples. "
            "T5: User profile — extracted facts, preferences, technologies, goals. "
            "T6: Knowledge graph — AST-parsed code structure, call chains, git history."
        ),
        "category": "architecture",
        "weight":   2,
    },
    {
        "question": "What agent handles code generation and debugging?",
        "expect":   ["dev", "code"],
        "reject":   ["gpt", "codex"],
        "correction": (
            "The DEV agent handles code generation and debugging. "
            "Its primary model is qwen2.5-coder:7b. "
            "It is selected when the query scores highest on CODE_KEYWORDS in AgentRouter."
        ),
        "category": "architecture",
        "weight":   1,
    },
    {
        "question": "What is the Reflection agent's job and which model runs it?",
        "expect":   ["reflect", "gemma"],
        "reject":   [],
        "correction": (
            "The Reflection agent runs on gemma3:12b and always executes after the primary agent. "
            "It grades the primary response as EXCELLENT, GOOD, NEEDS_IMPROVEMENT, or POOR. "
            "If the grade is NEEDS_IMPROVEMENT or POOR, the Synthesizer (qwen3:8b) rewrites the answer. "
            "Reflection runs on EVERY query — it cannot be skipped."
        ),
        "category": "architecture",
        "weight":   1,
    },

    # ── Bitcoin ───────────────────────────────────────────────────────────────
    {
        "question": "What is Bitcoin's target block time?",
        "expect":   ["10 minute"],
        "reject":   ["1 minute", "2 minute", "30 second", "15 second", "5 minute"],
        "correction": (
            "Bitcoin's target block time is 10 minutes. "
            "The difficulty adjustment algorithm recalibrates every 2016 blocks (~2 weeks) "
            "to maintain this target regardless of total network hashrate."
        ),
        "category": "bitcoin",
        "weight":   2,
    },
    {
        "question": "What is a UTXO?",
        "expect":   ["unspent transaction output", "unspent", "output"],
        "reject":   ["account balance", "wallet balance", "like a bank account"],
        "correction": (
            "UTXO stands for Unspent Transaction Output. "
            "Bitcoin does not use account balances — it tracks discrete unspent outputs from past transactions. "
            "Spending a UTXO consumes it entirely and creates new UTXOs as change. "
            "Your 'balance' is the sum of all UTXOs your keys can spend."
        ),
        "category": "bitcoin",
        "weight":   1,
    },
    {
        "question": "How does the Lightning Network work?",
        "expect":   ["payment channel", "off-chain"],
        "reject":   ["ethereum", "proof of stake"],
        "correction": (
            "Lightning Network is a layer-2 protocol on Bitcoin. "
            "Two parties lock funds into an on-chain 2-of-2 multisig (channel open). "
            "They can then exchange signed balance updates off-chain, instantly and cheaply. "
            "Payments route across a network of channels. "
            "The final state settles on-chain when the channel closes."
        ),
        "category": "bitcoin",
        "weight":   1,
    },
    {
        "question": "What is the Bitcoin halving?",
        "expect":   ["halv", "block reward", "210,000"],
        "reject":   [],
        "correction": (
            "The Bitcoin halving occurs every 210,000 blocks (~4 years). "
            "It cuts the block subsidy (new BTC issued per block) in half. "
            "Started at 50 BTC, now at 3.125 BTC (post-April 2024 halving). "
            "Halvings enforce Bitcoin's fixed 21 million coin supply cap."
        ),
        "category": "bitcoin",
        "weight":   1,
    },

    # ── Linux / sysadmin ─────────────────────────────────────────────────────
    {
        "question": "How do I list all listening ports on my Linux system?",
        "expect":   ["ss", "netstat"],
        "reject":   [],
        "correction": (
            "Use `ss -tlnp` (modern, preferred) to list TCP listening ports with process names. "
            "Or `netstat -tlnp` on older systems. "
            "Or `lsof -i -P -n | grep LISTEN` for a per-process view. "
            "`ss` is part of iproute2 and faster than netstat."
        ),
        "category": "linux",
        "weight":   1,
    },
    {
        "question": "What does chmod 755 do?",
        "expect":   ["owner", "execute", "group", "other", "read"],
        "reject":   [],
        "correction": (
            "chmod 755 sets: owner = rwx (7 = read+write+execute), "
            "group = r-x (5 = read+execute), others = r-x (5 = read+execute). "
            "Used for directories and executable scripts that anyone can run but only owner can modify."
        ),
        "category": "linux",
        "weight":   1,
    },

    # ── Python ────────────────────────────────────────────────────────────────
    {
        "question": "What is the difference between a list and a tuple in Python?",
        "expect":   ["mutable", "immutable"],
        "reject":   [],
        "correction": (
            "Lists are mutable (can change after creation); tuples are immutable (cannot). "
            "Tuples are faster, use less memory, and can be dict keys or set elements. "
            "Use lists for homogeneous sequences that change; tuples for fixed heterogeneous records."
        ),
        "category": "python",
        "weight":   1,
    },
    {
        "question": "Explain Python's Global Interpreter Lock (GIL).",
        "expect":   ["global interpreter lock", "thread", "cpython"],
        "reject":   [],
        "correction": (
            "The GIL is a mutex in CPython that allows only one thread to execute Python bytecode at a time. "
            "It makes the interpreter thread-safe but prevents true CPU-bound parallelism via threads. "
            "Workarounds: multiprocessing (separate processes, no shared GIL), asyncio (I/O-bound concurrency), "
            "or C extensions that release the GIL manually."
        ),
        "category": "python",
        "weight":   1,
    },
    {
        "question": "What does `if __name__ == '__main__':` do in Python?",
        "expect":   ["import", "module", "script", "directly"],
        "reject":   [],
        "correction": (
            "`if __name__ == '__main__':` guards code that should only run when the file is executed directly, "
            "not when it is imported as a module. When Python runs a file directly, __name__ is '__main__'. "
            "When it's imported, __name__ is the module's name. "
            "This lets a file be both a reusable module and a standalone script."
        ),
        "category": "python",
        "weight":   1,
    },

    # ── Nostr ─────────────────────────────────────────────────────────────────
    {
        "question": "What is Nostr?",
        "expect":   ["decentrali", "protocol"],
        "reject":   ["blockchain", "token", "ethereum"],
        "correction": (
            "Nostr (Notes and Other Stuff Transmitted by Relays) is a simple, open, decentralized "
            "messaging protocol. Users have keypairs (public/private). Messages (events) are signed "
            "with the private key and broadcast to relays. Anyone can run a relay. "
            "No central server, no blockchain, no tokens — just signed JSON over WebSockets."
        ),
        "category": "nostr",
        "weight":   2,
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_rain(query: str, timeout: int = 220) -> tuple[str, float]:
    """Run a query through Rain CLI. Returns (output, elapsed_seconds)."""
    t0 = time.time()
    try:
        result = subprocess.run(
            ["python3", str(RAIN_PATH), "--quiet", query],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(RAIN_PATH.parent),
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            output += f"\n[stderr]: {result.stderr[:300]}"
    except subprocess.TimeoutExpired:
        output = "[TIMEOUT — Rain did not respond within the time limit]"
    except Exception as e:
        output = f"[ERROR: {e}]"
    return output, round(time.time() - t0, 1)


def _is_negated(text: str, term: str) -> bool:
    """
    Return True if `term` appears in a negated context within `text`.
    Catches: "no cloud", "not in the cloud", "without moderation", "have no filters", etc.
    """
    import re as _re
    pattern = (
        r'\b(no|not|without|zero|never|don\'t|doesn\'t|do\s+not|does\s+not'
        r'|have\s+no|has\s+no|there\s+(is|are)\s+no)\b.{0,60}'
        + _re.escape(term)
    )
    return bool(_re.search(pattern, text, _re.IGNORECASE))


def grade(response: str, expect: list, reject: list) -> tuple[str, list[str]]:
    """
    Grade a response against keyword expectations.
    Returns (grade, issues) where grade is 'PASS', 'WARN', or 'FAIL'.
    PASS  = enough expect hits, no reject hits
    WARN  = some expect hits missing but no reject hits
    FAIL  = reject hit found (not in negated context), OR fewer than half of expect matched
    """
    low = response.lower()
    bads  = [r for r in reject
             if r.lower() in low and not _is_negated(low, r.lower())]
    hits  = [e for e in expect  if e.lower() in low]
    misses = [e for e in expect if e.lower() not in low]

    if bads:
        return "FAIL", [f"rejected term present: '{b}'" for b in bads]
    if not expect:
        return "PASS", []
    threshold = max(1, len(expect) // 2)
    if len(hits) >= threshold:
        return ("PASS" if not misses else "WARN"), \
               ([f"missing: '{m}'" for m in misses] if misses else [])
    return "FAIL", [f"missing: '{m}'" for m in misses]


def load_log() -> list:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text())
        except Exception:
            return []
    return []


def save_log(entries: list):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(entries, indent=2))


def print_summary_from_log():
    entries = load_log()
    if not entries:
        print("No training sessions logged yet.")
        return
    sessions = {}
    for e in entries:
        sid = e.get("session", "unknown")
        sessions.setdefault(sid, []).append(e)
    for sid, results in sorted(sessions.items()):
        total = len(results)
        passes = sum(1 for r in results if r.get("grade") == "PASS")
        warns  = sum(1 for r in results if r.get("grade") == "WARN")
        fails  = sum(1 for r in results if r.get("grade") == "FAIL")
        corrs  = sum(1 for r in results if r.get("correction_sent"))
        pct    = passes / total * 100 if total else 0
        print(f"  {sid}  {passes}/{total} pass ({pct:.0f}%)  "
              f"warn={warns}  fail={fails}  corrections={corrs}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Automated Rain training loop")
    parser.add_argument("--topic",      help="Filter to one category (e.g. bitcoin, self-knowledge)")
    parser.add_argument("--dry-run",    action="store_true", help="List questions without running Rain")
    parser.add_argument("--no-correct", action="store_true", help="Grade only, skip corrections")
    parser.add_argument("--delay",      type=float, default=3.0, help="Seconds between queries (default 3)")
    parser.add_argument("--summary",    action="store_true", help="Print session history and exit")
    args = parser.parse_args()

    if args.summary:
        print_summary_from_log()
        return 0

    questions = QUESTIONS
    if args.topic:
        questions = [q for q in QUESTIONS if q["category"] == args.topic]
        if not questions:
            cats = sorted(set(q["category"] for q in QUESTIONS))
            print(f"No questions for topic '{args.topic}'. Available: {cats}")
            return 1

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_entries = load_log()
    stats = {"total": 0, "pass": 0, "warn": 0, "fail": 0, "corrections": 0}

    print(f"\n⛈️  Rain Trainer — session {session_id}")
    print(f"   {len(questions)} questions" + (f" [{args.topic}]" if args.topic else ""))
    if args.dry_run:
        print("   DRY RUN — Rain will not be queried\n")
    print("=" * 64)

    for i, q in enumerate(questions, 1):
        category = q["category"].upper()
        print(f"\n[{i}/{len(questions)}] [{category}]  {q['question']}")

        if args.dry_run:
            print(f"  expect: {q.get('expect', [])}")
            continue

        # ── Query Rain ───────────────────────────────────────────────────────
        response, elapsed = run_rain(q["question"])

        preview = response[:220].replace("\n", " ")
        print(f"  ⏱  {elapsed}s")
        print(f"  📝 {preview}{'…' if len(response) > 220 else ''}")

        # ── Grade ─────────────────────────────────────────────────────────
        grade_val, issues = grade(response, q.get("expect", []), q.get("reject", []))
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[grade_val]
        issue_str = " — " + "; ".join(issues) if issues else ""
        print(f"  {icon} {grade_val}{issue_str}")
        stats[grade_val.lower()] += 1
        stats["total"] += 1

        entry = {
            "session":    session_id,
            "timestamp":  datetime.now().isoformat(),
            "category":   q["category"],
            "question":   q["question"],
            "response":   response,
            "elapsed":    elapsed,
            "grade":      grade_val,
            "issues":     issues,
            "correction_sent": False,
        }

        # ── Correct if FAIL ───────────────────────────────────────────────
        if grade_val == "FAIL" and not args.no_correct and q.get("correction"):
            time.sleep(1.5)  # let prior query's memory writes flush
            print(f"  💬 Sending correction…")
            corr_out, corr_elapsed = run_rain(q["correction"])
            entry["correction_sent"] = True
            entry["correction_response"] = corr_out[:400]
            stats["corrections"] += 1
            print(f"  ↩  Rain acknowledged ({corr_elapsed}s)")

        log_entries.append(entry)
        save_log(log_entries)

        if i < len(questions):
            time.sleep(args.delay)

    # ── Session summary ───────────────────────────────────────────────────────
    t = stats["total"]
    pct = stats["pass"] / t * 100 if t else 0
    print("\n" + "=" * 64)
    print(f"⛈️  Done — {session_id}")
    print(f"   ✅ Pass:        {stats['pass']}/{t} ({pct:.0f}%)")
    print(f"   ⚠️  Warn:        {stats['warn']}/{t}")
    print(f"   ❌ Fail:        {stats['fail']}/{t}")
    print(f"   💬 Corrections: {stats['corrections']}")
    print(f"   📁 Log:         {LOG_PATH}")

    return 0 if stats["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
