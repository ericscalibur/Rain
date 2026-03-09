# Rain ⛈️ — Session Handoff
> For the next Claude instance picking up this codebase.
> Written at session end. Do not skip this file.

---

_Last updated: 2026-03-08 — Claude Sonnet 4.6_

---

## What Was Done This Session

### Diagnostic Prompt Battery
Ran 4 structured prompts to baseline Rain's quality before fixing anything:
1. Merge sort (DEV) — correct, 61% conf, 155s, over-structured
2. "Why is it easier to destroy than build?" (LOGIC) — applied task planning template to a philosophy question (numbered sections, tables, Python code)
3. "What models are you running?" — hallucinated corporate security policy
4. GPT-4 false premise — correctly rejected, then fabricated a memory citation

Root causes identified:
- LOGIC agent TASK PLANNING PATTERN applied to all queries — no bypass for conversational/conceptual
- `_score_confidence()` flat keyword heuristic → always 0.75 × calibration_factor = 0.61
- No grounded self-knowledge in agent context → hallucination of security policies
- `_needs_synthesis()` returned `rating != 'EXCELLENT'` (GOOD also triggered synthesis)
- `get_recent_messages()` had `WHERE session_id != ?` — excluded current session entirely (Rain had zero in-session memory)

---

## Fixes Committed This Session

### rain.py
| Fix | Detail |
|-----|--------|
| LOGIC agent conversational bypass | Guard clause before TASK PLANNING PATTERN: prose for philosophical/factual questions, no headers/bullets/numbered sections |
| One-shot prose example | Explicit WRONG/CORRECT demonstration in LOGIC agent prompt showing prose vs bullets |
| Reasoning rule | Changed "Prefer structured responses" → context-dependent: task→steps, concept→prose |
| `_score_confidence()` rewrite | Hedging-aware: base 0.80 for substantive responses, -0.05 per hedge phrase, floor 0.45 |
| `_build_runtime_context()` | New method injecting factual model roster at front of system_content — kills hallucinated security policy |
| Runtime context position | Moved from after system_prompt to BEFORE it — model reads identity facts first |
| GENERAL agent SELF-KNOWLEDGE | Updated: remove stale model list + deflection to `--agents`; point to runtime context block |
| `_needs_synthesis()` | Changed from `rating != 'EXCELLENT'` to `rating in ('NEEDS_IMPROVEMENT', 'POOR')` — GOOD bypasses synthesis |
| Self-identity short-circuit | Programmatic bypass in `recursive_reflect` for "what models are you running" queries — returns factual answer at 99% conf in <1s, never touches the LLM |
| `rain-tuned` scoped to DEV only | Was selecting rain-tuned for LOGIC/DOMAIN/GENERAL (wrong — it's a fine-tuned code model). Now DEV only until a reasoning base model's LoRA can be fused |
| Duration bug | `time.monotonic() - _time.time()` → `_time.time() - start_time` (was producing huge negative durations) |
| `get_recent_messages()` | Removed `WHERE session_id != ?` — current session messages now included in working memory |
| `get_current_session_messages()` | New method: current session only, chronological. Used for "what did we talk about" queries to avoid cross-session contamination |
| Conversation history injection | When user asks "what was my first question / what did we discuss", inject current session chat log directly into user message turn (system prompt injection alone is reliably ignored by models for this query type) |

### server.py
| Fix | Detail |
|-----|--------|
| `resource.setrlimit(RLIMIT_NOFILE, 8192)` | Raises FD limit at startup — macOS default 256 was exhausted by Ollama + SQLite + SSE sockets → `[Errno 24] Too many open files` crash |
| Monkey-patch restores in `finally` | `_query_agent`, `router.route`, `_parse_reflection_rating` were restored inside `try` — exceptions left patches applied permanently across all subsequent requests |

### Calibration Reset
- LOGIC agent had 9 bad ratings, 2 good from test sessions (debugging sessions that generated false negatives)
- Calibration factor ~0.81 → all LOGIC responses scored 65% → synthesis fired on correct answers
- **Fix**: `DELETE FROM feedback WHERE agent_type = 'logic'` — reset to no data, factor defaults to 1.0, baseline confidence now 0.80
- Do not let future test/debug sessions accumulate bad ratings without counterbalancing good ones

---

## Commits This Session
```
5ed2fd2  fix: memory, self-knowledge, calibration, and stability  ← HEAD on main
b718d90  fix: finetune.py — graceful GGUF export fallback + fuse step for supported arches
b935d4d  feat: prompt quality, self-knowledge, calibration, and fine-tuning pipeline
```

---

## Current Performance Baseline (post-fixes)
| Query type | Before | After |
|-----------|--------|-------|
| Conceptual (no synthesis) | 135–155s | 38–80s |
| Conceptual (synthesis fires) | 270s+ | ~140s (decreasing as calibration stabilizes) |
| Self-knowledge ("what models?") | 50s + hallucination | <1s, 99% conf, accurate |
| In-session memory recall | Broken (session excluded) | Working (current session injected into user turn) |
| Server stability | Crashed after ~30 min (FD exhaustion) | FD limit raised to 8192 |

---

## What's Still Outstanding

### Bullet Point Formatting
- qwen3.5:9b has strong RLHF preference for bullet lists on multi-component answers
- Tried: system prompt prohibition, removing escape hatch, one-shot WRONG/CORRECT example
- All had partial effect but model still reaches for bullets on enumerable answers (4+ distinct points)
- **Decision**: accepted. User noted "in 2-3 sentences" in the user message is more reliable than system prompt rules. Not worth more engineering time.

### LoRA Weights Not in Ollama
- `rain-tuned` in Ollama = qwen2.5-coder:7b + behavioral system prompt only (no LoRA weights)
- Adapter weights exist at `~/.rain/adapters/mlx-lora/adapters.safetensors` (22MB)
- Blocked by: mlx_lm 0.31.0 doesn't support qwen2 → GGUF export
- Monitor: `pip install --upgrade mlx-lm` then re-run `python3 finetune.py --full`

### Rain.md (Phase 11 — Recommended Next Work)
- Claude has CLAUDE.md giving it accurate Rain context at session start
- Rain has NO equivalent — no self-knowledge document it reads at startup
- Currently relies on runtime context injection (model names only) and RLHF training (unreliable for everything else)
- **Proposal**: Create `RAIN.md` in project root, read at server startup, injected into every agent's system prompt alongside the runtime context block
- Should contain: phase completion status, capability list, known limitations, architecture overview, what "Rain" is
- This is Phase 11 (metacognition) territory — Rain knowing about itself accurately

### Calibration Monitoring
- LOGIC calibration reset this session — will rebuild from scratch
- Check periodically: `sqlite3 ~/.rain/memory.db "SELECT agent_type, rating, COUNT(*) FROM feedback GROUP BY agent_type, rating;"`
- Don't let test/debug sessions contaminate calibration — clear if needed

### Training Data
- 12 trainable examples (all from 👍 ratings — no written corrections yet)
- Each 👎 with a written correction = highest-quality training signal
- Re-run `python3 finetune.py --full` after accumulating more corrections

---

## Priority Order for Next Session
1. **RAIN.md** — Rain's self-knowledge document injected at startup. High value, straightforward to build.
2. **Calibration health check** — verify LOGIC isn't drifting back into low accuracy from test sessions
3. **LoRA weight fusion** — check if mlx_lm added qwen2 GGUF support; if so, re-run `python3 finetune.py --full`
4. **Phase 8: Voice** — whisper.cpp for STT, piper-tts for TTS, `--voice` CLI flag

---

## Key Facts for Next Claude Instance
- Streaming responses: **done** (completed before this session, not in old handoff)
- Implicit feedback plausibility gate: **done** (same)
- Calibration death spiral: **fixed this session** — LOGIC reset to clean slate at `5ed2fd2`
- `rain-tuned` in Ollama: behavioral system prompt only, no LoRA weights — don't confuse for fully fine-tuned
- Worktree `claude/gallant-jang` branch exists and is pushed; main is canonical at `5ed2fd2`
- LOGIC agent calibration was deliberately cleared — this is intentional, not a bug
