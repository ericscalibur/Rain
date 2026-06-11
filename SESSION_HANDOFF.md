# Rain ⛈️ — Session Handoff
> For the next Claude instance picking up this codebase.
> Written at session end. Do not skip this file.

---

_Last updated: 2026-03-31 — Claude Sonnet 4.6_

---

## What Was Done This Session

Phase 11: Metacognition & Self-Directed Evolution — substantial progress. Built the remaining pieces on top of what was already partially done (tiered escalation, gap detection, relevance-gated memory).

---

## Changes Made This Session

### 1. Fixed `prune_session_memory` schema bug — `rain/memory.py`
- The `INSERT INTO session_facts` call used `fact`/`confidence` columns that don't exist in the schema
- Fixed to use correct columns: `fact_type='compressed_history'`, `fact_key='summary'`, `fact_value=<compressed>`
- This was silently failing and suppressing all deliberate forgetting

### 2. Phase 11 methods — `rain/memory.py`
Three new methods added after `get_top_gaps()`:

**`get_performance_stats()`**
- Queries `feedback` table, returns per-agent accuracy/confidence/counts (all time + rolling 30d)
- Includes synthesis stats from `synthesis_log` and open gap count from `knowledge_gaps`
- Powers `/api/performance` and the CLI `--meta` flag

**`harvest_positive_examples(min_confidence=0.65, limit=200)`**
- Collects `rating='good'`, no correction, `confidence >= threshold` rows
- Returns list ready for JSONL export as positive fine-tuning examples
- Complements the existing corrections export in finetune.py

**`generate_meta_report(model_query_fn)`**
- Calls `get_performance_stats()` + `get_top_gaps()` to build a stats block
- Sends to LLM (llama3.2) with a metacognition prompt requesting: Strengths / Weak areas / Improvement proposals / One-sentence summary
- Falls back to raw stats block if model unavailable

### 3. Phase 11 API endpoints — `server.py`
- `GET /api/performance` — returns `get_performance_stats()` JSON
- `POST /api/finetune/harvest?min_confidence=0.65` — runs `harvest_positive_examples()`, writes to `~/.rain/training/positive_examples.jsonl`
- `GET /api/meta` — runs `generate_meta_report()` via llama3.2, returns `{"report": "...markdown..."}`

### 4. Phase 11 UI — `static/index.html`
- New **Performance** sidebar section (above sidebar footer, below Training)
- Shows: total ratings, overall accuracy, per-agent accuracy (last 30d), synthesis stats, open gap count
- **🧠 Self-Assessment** button — calls `/api/meta`, renders the report in an expandable box in the sidebar

### 5. `--meta` CLI flag — `rain.py`
- `python3 rain.py --meta` — generates and prints the metacognition report, then exits
- Uses llama3.2 directly (same model as server-side)

---

## Phase 11 Completion Status

**Already built (previous sessions + this session):**
- ✅ Tiered model escalation (`_fast_logic_model`, `_is_simple_logic_query` in orchestrator)
- ✅ Knowledge gap detection + logging + injection into agent prompts
- ✅ Background LLM gap description generation
- ✅ Deliberate forgetting via `prune_session_memory()` (now fixed)
- ✅ Relevance-gated Tier 1 (session summaries — keyword overlap > 0.08)
- ✅ Relevance-gated Tier 3 (semantic search — cosine sim)
- ✅ Relevance-gated Tier 4 (corrections — semantic retrieval)
- ✅ Relevance-gated Tier 5 (session facts — keyword overlap > 0.05; profile always injected)
- ✅ Metacognitive note injection (gap awareness in agent prompts)
- ✅ Performance dashboard (`get_performance_stats` + `/api/performance` + UI panel)
- ✅ Self-generated positive training data (`harvest_positive_examples` + `/api/finetune/harvest`)
- ✅ Metacognition agent (`generate_meta_report` + `/api/meta` + `--meta` CLI flag)
- ✅ Calibration tracking (`get_calibration_factors` in memory.py — per-agent accuracy factors)

**What remains (nice-to-have):**
- Improvement proposals that Rain can write to a file for human review (currently they're inline in the meta report)
- Embedding-based gating for Tier 1 (currently keyword heuristic — more accurate but costs an embed call)
- Context budget tracker (prune lowest-relevance injections when total context grows large)
- Weekly scheduled meta report (cron-style, not yet wired)

Phase 11 is substantially complete. If you consider the must-haves done, update ROADMAP.md to ✅.

---

## Discussions Worth Preserving

### TurboQuant (Google Research)
Evaluated this session. TurboQuant is KV-cache compression (PolarQuant 3-bit + QJL 1-bit error correction). Claims 6x KV-cache reduction, 8x throughput on H100 GPUs, no accuracy loss, no training required. **Not relevant to Rain today** — Rain runs 3–14B models single-query on M1 16GB; KV cache isn't the bottleneck. Bookmark for Phase 12 if Rain ever runs 70B+ models or multi-user concurrent sessions. Ollama may integrate it automatically in future.

### Tier 3 semantic search gap — RESOLVED (was stale when this doc was written)
`semantic_search()` in `rain/memory.py` enforces `min_similarity=0.4` (default parameter, applied before the top-k cut — results below the floor are dropped, not padded). MemPalace `search()` in `rain/mem_palace.py` has its own `min_similarity=0.35` floor. Landed in commit 27745ab (2026-03-25). No action needed.

---

## Current State

- Rain server: functional — restart to pick up new endpoints
- All three Python files compile clean (`python3 -m py_compile`)
- Telegram bot: working (`python3 rain-telegram.py`)
- `qwen2.5:14b`: should be fully downloaded now

## Honest Assessment (dictated end of session)

**What works:** Architecture is sound. 6-tier relevance-gated memory is real. Knowledge graph closes codebase hallucination. Phase 11 metacognition gives Rain genuine self-awareness of its weak areas. Telegram, voice, OpenAI-compatible API, web UI — Rain meets you everywhere.

**What's broken:**
- Confidence deflation (53–62% on correct answers) → synthesis fires constantly → 2–4 min response times. Root cause: local models trained for completions, not epistemic calibration. Fix: reflection rubric, not `_score_confidence()`.
- Fine-tuning loop built but never run. `finetune.py --full` has never been executed. All quality improvement to date is prompt-level. No weight updates.
- ~~Tier 3 similarity floor missing~~ RESOLVED — `semantic_search()` defaults to `min_similarity=0.4` since 27745ab.
- Correction deduplication doesn't exist — 10 corrections about the same mistake = 10 rows, growing noise.
- Tool use reliability patch-worked, not fixed.

**What Rain learned from Claude:** Epistemic calibration is trained not prompted. Consistency across sessions needs values in weights not facts in memory. Short and correct beats long and approximate — bias in local model training toward verbose responses.

**What Rain learned from OpenClaw:** Skills as composable declarative units. Declarative triggers. Skill chaining. Community registry. ClawHub integration is half-wired — finish it.

## What's Next (priority order)

### Immediate (close existing loops)
1. **Run the fine-tuning loop** — `python3 finetune.py --full`. First time Rain's weights reflect its own experience. Highest impact action available.
2. **Fix reflection rubric** — ✅ DONE (PR #1, 2026-06-10): rubric reworked around accuracy + epistemic honesty + answers-the-question; structure/length explicitly ungraded; runtime VERDICT guide aligned. Benchmark: `scripts/benchmark_reflection_rubric.py`.
3. **Tier 3 similarity floor** — ✅ ALREADY DONE (27745ab): `semantic_search()` enforces `min_similarity=0.4` default; MemPalace search has a 0.35 floor.
4. **Correction deduplication** — background job distilling near-duplicate Tier 4 corrections into single authoritative rules.

### Near-term
5. **Phase 12: Sovereign Identity** — `--export`, Nostr keypair, cross-device sync, Lightning micropayments. Makes Rain portable and cryptographically yours.
6. **Finish ClawHub skill ecosystem** — declarative triggers, skill chaining, one-click install from web UI.

### Horizon
7. **Proactive intelligence** — Rain surfaces insights without being asked, via Telegram. Pattern detection across sessions.
8. **Sustained autonomy** — progress persistence for multi-hour tasks, milestone checkpoints, async Telegram notification when done.

Phase 8 (Voice): ✅ Already done — STT via faster-whisper + `/api/transcribe`, TTS via Web Speech API, both toggles in web UI.

## Known Issues (carry-forward)
- Confidence deflation still present (53–62% on correct answers) — reflection prompt fix pending
- Streaming works but synthesis can't stream (falls back to full response)
- Vision slow on first run (model load time)
- GitHub API rate limit at 60 req/hr unauthenticated
- ~~Tier 3 semantic search has no minimum similarity floor~~ RESOLVED — 0.4 floor in `semantic_search()`, 0.35 in MemPalace (27745ab)

## Eric's Setup
- MacBook Pro M1, 16GB RAM, 712GB free disk
- Ollama installed, Rain server on port 7734
- Telegram bot wired to Rain, working
- Monthly El Salvador tax filing automation planned (~2026-04-30)
