# Rain ⛈️ — Session Handoff
> For the next Claude instance picking up this codebase.
> Written at session end. Do not skip this file.

---

_Last updated: 2026-03-25 — Claude Sonnet 4.6_

---

## What Was Done This Session

### Starting State
Picked up from previous session (streaming, DOMAIN fast tier, gemma3:4b reflection, rubric overhaul). Rain was functional. This session focused on getting Rain closer to Claude-level quality: synthesis misfires, Phase 8 voice, Phase 11 metacognition, model correctness, and memory hygiene.

---

## Fixes Committed This Session

### Voice TTS + STT (Phase 8 — complete)

| Fix | Detail |
|-----|--------|
| **Web Speech API TTS** | Voice Response toggle below Voice Dictate. 🔊 button on every response bubble. Auto-speaks on `done` event when toggle is on. Cancel on new send. |
| **Female voice preference** | `onvoiceschanged` cache prevents empty array on first call. Ordered preference list: Zoe → Samantha → Serena → Karen → Moira → Tessa → other female voices. Uses `reduce()` not `find()` to avoid Daniel (male) matching first. |
| **faster-whisper installed** | `pip install faster-whisper` in system python3 (miniconda). Previously only in `.venv` — server uses system python3. Mic button now works end-to-end. `/api/voice-status` returns `available: true`. |

### Active Project Context

| Fix | Detail |
|-----|--------|
| **`state.activeProject` sent with every chat request** | `project_path` injected into every `/api/chat` POST body. Auto-pinned when only one project indexed. 📌 pin button for multiple projects. |
| **Server-side auto-detect** | When `req.project_path` not sent, uses single indexed project automatically. `effective_project_path` replaces `req.project_path` throughout chat handler. |

### GitHub Auto-Fetch + Audit Rules

| Fix | Detail |
|-----|--------|
| **Unconditional GitHub prefetch** | When query contains a GitHub URL, `_fetch_github_data()` runs before model sees the query — regardless of web search toggle. Prevents hallucinated repo analysis. |
| **README + source files fetched** | Always fetches README. For audit/compare queries, also fetches up to 6 key source files × 1500 chars each via file tree walk. |
| **Audit prompt rules** | Distinguish committed credentials vs README setup instructions. Ignore ANSI codes. Only report what's in fetched files. Prevents `.env` false positives. |

### Synthesis Misfires

| Fix | Detail |
|-----|--------|
| **Synthesis veto at 0.76** | `if rating == 'NEEDS_IMPROVEMENT' and primary_confidence >= 0.76: rating = 'GOOD'` — skips synthesis when confidence is high enough. |
| **Reflection rating tail-scan** | Parses final 5 lines of critique before rfind fallback — prevents preamble text from swallowing the rating word. |
| **TOPIC DRIFT softened** | Reflection prompt: only flag drift that actively misleads, not helpful adjacent context. |

### Model Roster Fixes

| Fix | Detail |
|-----|--------|
| **Exact match before base-name fallback** | `_best_model_for()` now tries exact match first (`qwen3:8b` → `qwen3:8b`), base-name only for tagless entries (`llama3.2` → `llama3.2:latest`). Fixed Synthesizer using `qwen3:1.7b` instead of `qwen3:8b`. |
| **qwen3.5:9b + qwen2.5-coder:7b re-pulled** | Both were deleted during Docker space reclaim. Re-pulled and restored as primary models. |
| **codestral removed** | 12GB dead weight on M1. Removed from DEV fallback list. |
| **gemma3:12b promoted to primary REFLECTION** | Stronger on structured rubric tasks than gemma3:4b. |
| **Model preferences realigned** | `AGENT_PREFERRED_MODELS` updated to match installed models. |

### Phase 11 Metacognition

| Fix | Detail |
|-----|--------|
| **Tier 1 relevance gating** | Session summaries filtered by keyword overlap with current query (> 0.08 threshold). Most recent summary always kept as recency anchor. |
| **Tier 5 relevance gating** | `get_fact_context(query=query)` — session facts filtered by relevance when query provided, cap 12 vs 20. |
| **Correction decay** | Corrections >90 days pruned unless `access_count >= 3`. Schema migration adds `access_count` column. |
| **Knowledge gap logging** | When `final_confidence < 0.55 AND rating == 'POOR'`, logs gap. Background thread asks reflection agent for LLM-described gap text. |
| **Knowledge gap schema fix** | Orchestrator was creating `knowledge_gaps` without `gap_description` column — all LLM descriptions failed silently. Fixed: `_init_db` owns the schema, schema migration adds missing columns, orchestrator delegates to `memory.log_knowledge_gap()`. `get_top_gaps()` now returns results. |
| **Knowledge gap context injection** | `_build_memory_context()` checks recent gaps against current query (2+ content word overlap). Injects metacognitive warning: "Past uncertainty: [desc]. Be especially careful here." |
| **Gap display on startup** | Shows up to 3 recent gaps with confidence % at startup. |
| **Session pruning** | `prune_session_memory()` fires in background thread after each response; compresses old messages when session > 40 messages. |

### Tier 5 User Profile Cleanup

| Fix | Detail |
|-----|--------|
| **Model name fixed** | `extract_session_facts()` was using `llama3.1` (not installed) → failing silently for months. Fixed to `llama3.2`. |
| **Canonical key list** | Extraction prompt now provides canonical keys: `user_name`, `preferred_language`, `active_project`, `os_platform`, `tech_stack`, `goal`, etc. |
| **Rain-as-subject filter** | Extraction now asks for USER facts only. `_normalize()` drops facts where key is in `_AI_SYSTEM_KEYS` or value is `Rain`/`Sovereign AI`. |
| **Garbage value filter** | Drops values like `my_project`, `time`, `array`, `unknown`. |
| **Key alias normalization** | `_KEY_ALIASES` maps noisy LLM keys to canonical: `language` → `preferred_language`, `project_name` → `active_project`, etc. |
| **DB wipe + reseed** | Wiped 264 polluted user_profile rows and 420 session_facts rows (topic extractions masquerading as user facts). Seeded 4 known-correct facts: `preferred_language=Python`, `active_project=Rain`, `project_type=sovereign local AI ecosystem on Ollama`, `goal=build Rain to replace Claude entirely`. |

### Reflection Agent Improvements

| Fix | Detail |
|-----|--------|
| **Epistemic boundary** | LOGIC and DOMAIN agents: if URL in query and no web search, Rain states it cannot access URLs rather than fabricating content. |
| **URL/REPO FABRICATION CHECK** | REFLECTION prompt auto-rates POOR if response contains fabricated repo analysis not based on fetched data. |

---

## Current State (post-session)

### Installed Models
```
qwen3.5:9b       → LOGIC, DOMAIN, GENERAL (primary)
qwen2.5-coder:7b → DEV (primary)
gemma3:12b       → REFLECTION (primary)
gemma3:4b        → REFLECTION (fallback), SYNTHESIZER (fallback)
qwen3:8b         → SYNTHESIZER (primary)
qwen3:4b         → fallback
qwen3:1.7b       → fast tier fallback
llama3.2         → REFLECTION fast tier, SEARCH
rain-tuned       → DEV (preferred when available)
nomic-embed-text → embeddings
```

### User Profile DB
Clean — 4 seeded facts, garbage wiped. Future sessions will accumulate cleanly.

### Knowledge Gaps DB
Working — schema fixed, gap injection active. 3 gaps from prior sessions visible at startup.

---

## What's Still Outstanding

### LoRA Weights Not Fused
- `rain-tuned` = qwen2.5-coder:7b + behavioral system prompt only (no LoRA weights)
- Adapter weights at `~/.rain/adapters/mlx-lora/adapters.safetensors` (22MB)
- Blocked: mlx_lm doesn't support qwen2 → GGUF export
- Monitor: `pip install --upgrade mlx-lm` then re-run `python3 finetune.py --full`

### Phase 11 Remaining
- Gap-driven learning only injects warnings — doesn't proactively seek to fill gaps
- No self-directed "I should learn more about X" behavior yet
- Knowledge gap resolution (marking gaps as resolved when Rain answers correctly) not wired

### Synthesis Still Occasionally Misfires
- Veto at 0.76 catches most cases; remaining fires are genuine POOR ratings
- Watch `↳` log lines to identify which reflection check is triggering

### Phase 12: Distributed Rain
- Not started

---

## Priority Order for Next Session

1. **Gap resolution** — when Rain gives a confident correct answer on a topic it previously had a gap for, mark the gap `resolved = 1`. Close the loop.
2. **Phase 11 self-directed learning** — Rain identifies its weakest topics from gap log and proactively asks clarifying questions or suggests it needs examples.
3. **LoRA weight fusion** — check mlx_lm for qwen2 GGUF support; re-run `python3 finetune.py --full` when available.
4. **Phase 12: Distributed Rain** — multi-node Rain instances sharing memory.

---

## Key Facts for Next Claude Instance

- **Streaming**: done — tokens buffer silently, final response renders in one shot
- **TTS**: Web Speech API, female voice, 🔊 per bubble, auto-speak toggle
- **STT**: faster-whisper installed in system python3. Mic button fully functional.
- **Active project**: sent with every chat request. Auto-detected server-side.
- **GitHub prefetch**: unconditional when URL in query. Prevents hallucinated analysis.
- **Synthesis veto**: NEEDS_IMPROVEMENT + conf ≥ 0.76 → vetoed to GOOD
- **Model matching**: exact match first, base-name fallback only for tagless preferences
- **Tier 5 profile**: clean 4-fact baseline, extraction now uses llama3.2 + canonical keys
- **Knowledge gaps**: schema fixed, injection working, startup display shows recent gaps
- **RAIN.md**: injected into every agent EXCEPT REFLECTION and SYNTHESIZER (context overflow protection)
- **All changes committed** to main branch
