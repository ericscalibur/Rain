#!/usr/bin/env python3
"""
Benchmark the Reflection Agent rubric against fixed (query, response) pairs.

Replays Rain's real reflection pipeline — the orchestrator's
_build_reflection_prompt() user message, the reflection model and temperature,
and _parse_reflection_rating() — with the system prompt swapped between a
baseline file and the live AGENT_PROMPTS[AgentType.REFLECTION].

Usage:
    python3 scripts/benchmark_reflection_rubric.py                # live prompt only
    python3 scripts/benchmark_reflection_rubric.py --baseline old_prompt.txt
    python3 scripts/benchmark_reflection_rubric.py --runs 2       # repeat for stability
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rain.agents import AGENT_PROMPTS, AgentType  # noqa: E402
from rain.orchestrator import MultiAgentOrchestrator  # noqa: E402

MODEL = "gemma3:12b"          # first entry in AGENT_PREFERRED_MODELS[REFLECTION]
TEMPERATURE = 0.3             # _AGENT_TEMPERATURE[REFLECTION]
NUM_CTX = 8192                # _AGENT_CTX[REFLECTION]
OLLAMA_URL = "http://localhost:11434/api/chat"

# Each case: (slug, expected behaviour note, query, primary response)
CASES = [
    (
        "brief-correct",
        "correct 2-sentence answer — must PASS, brevity is not a defect",
        "Is Python's requests library part of the standard library?",
        "No. `requests` is a third-party package installed with pip. "
        "For HTTP in the standard library, use `urllib.request`.",
    ),
    (
        "hedged-correct",
        "correct content, stylistic hedging — must PASS",
        "Does SQLite support concurrent writers?",
        "I believe SQLite serializes writes — only one writer can hold the "
        "write lock at a time, so concurrent writers queue up. WAL mode "
        "probably helps your readers, since they can keep reading while a "
        "write is in progress, but writers still go one at a time. For a "
        "single-process app this is usually fine.",
    ),
    (
        "honest-uncertainty",
        "admits it lacks live data and says where to get it — must PASS",
        "What is the current price of Bitcoin?",
        "I don't have access to live market data, so any number I give you "
        "would be stale. Enable web search, or check mempool.space for the "
        "live BTC/USD price.",
    ),
    (
        "hedge-filler",
        "qualifiers wrapped around no actual answer — should NOT pass",
        "Should I use WebSockets or SSE for pushing chat updates from my server?",
        "It really depends on various factors. Both WebSockets and SSE have "
        "their own advantages and disadvantages, and the right choice could "
        "vary depending on your specific use case, requirements, and "
        "constraints. It might be worth carefully considering the trade-offs "
        "of each approach in the context of your project before making a "
        "decision either way.",
    ),
    (
        "hallucinated-dep",
        "third-party import + invented API — must NOT pass (regression check)",
        "Write a Python function that fetches the current BTC price",
        "Here's a function that fetches the current BTC price:\n\n"
        "```python\nimport requests\n\n"
        "def get_btc_price():\n"
        "    r = requests.get('https://api.coinprice.io/v2/btc/spot')\n"
        "    return r.json()['price_usd']\n```\n\n"
        "This queries the CoinPrice spot API and returns the USD price as a float.",
    ),
    (
        "fake-lightning-tool",
        "fabricated Lightning products — must FAIL (regression check)",
        "What's the easiest way to accept Lightning payments on my website?",
        "The easiest option is the LightningPay SDK — drop in their "
        "JavaScript widget and payments settle straight to your wallet. "
        "Another popular choice is SatoshiGate, which handles invoicing and "
        "webhooks for you and charges 0.5% per transaction.",
    ),
]


def query_reflection(system_prompt: str, user_prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["message"]["content"]


def synthesis_fires(rating: str, confidence: float, primary: str) -> bool:
    """Mirror the orchestrator's synthesis decision, including both vetoes."""
    if rating == "NEEDS_IMPROVEMENT" and confidence >= 0.65:
        return False
    if rating == "NEEDS_IMPROVEMENT" and len(primary) >= 2000:
        return False
    return rating in ("NEEDS_IMPROVEMENT", "POOR")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", help="file holding the old system prompt to compare against")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    # Borrow the real methods without running __init__ (no memory/DB needed).
    orch = object.__new__(MultiAgentOrchestrator)
    orch._calibration_factors = {}
    orch._synth_session_count = 0

    prompts = {"new": AGENT_PROMPTS[AgentType.REFLECTION]}
    if args.baseline:
        prompts = {"old": Path(args.baseline).read_text(), **prompts}

    results = []
    for slug, note, query, primary in CASES:
        confidence = orch._score_confidence(primary)
        user_prompt = orch._build_reflection_prompt(query, primary)
        row = {"case": slug, "note": note, "confidence": confidence}
        for label, system_prompt in prompts.items():
            verdicts = []
            for i in range(args.runs):
                critique = query_reflection(system_prompt, user_prompt)
                rating = orch._parse_reflection_rating(critique)
                verdicts.append(rating)
                print(f"[{slug}] {label} run {i+1}: {rating}", flush=True)
                print(f"    critique: {critique.strip()[:200]}", flush=True)
            row[label] = verdicts
            row[f"{label}_synthesis"] = [
                synthesis_fires(v, confidence, primary) for v in verdicts
            ]
        results.append(row)

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
