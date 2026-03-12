# Rain ⛈️ — Session Handoff
> For the next Claude instance picking up this codebase.
> Written at session end. Do not skip this file.

---

_Last updated: 2026-03-11 — Claude Sonnet 4.6_

---

## What Was Done This Session

### Starting State
Picked up from the previous session's fixes (memory, self-knowledge, calibration, streaming). Rain was functional but had several remaining issues:
- Synthesis still firing on correct responses (confidence deflation)
- No fast-path for DOMAIN agent — 9-word factual queries hitting qwen3.5:9b (180s+)
- Response streaming was raw/unformatted — user preferred status updates + polished final output
- gemma3:4b not wired into model preferences
- Reflection agent crashing (HTTP 500) due to RAIN.md context overflow at 4096 ctx
- Sandbox self-awareness missing — Rain didn't know whether sandbox was active

---

## Fixes Committed This Session

### rain.py

| Fix | Detail |
|-----|--------|
| **Sandbox self-awareness** | `Agent.__init__` now sets `self.sandbox_enabled` / `self.sandbox` BEFORE calling `_get_default_system_prompt()` so sandbox state is available. `_get_default_system_prompt()` now injects a `sandbox_fact` line into the system prompt: "Code sandbox: ACTIVE — Rain will automatically run any Python code you generate..." |
| **gemma3:4b as reflection model** | Added gemma3:4b to `AGENT_PREFERRED_MODELS`: position #1 for REFLECTION, #2 for SYNTHESIZER, #5 for LOGIC/DOMAIN/GENERAL. Added to `_LOGIC_FAST_PREFERRED`. Confirmed running in production. |
| **Reflection rubric overhaul** | Added explicit RATING GUIDE to `AGENT_PROMPTS[AgentType.REFLECTION]`: EXCELLENT/GOOD/NEEDS_IMPROVEMENT/POOR with clear definitions. GOOD = correct + no hallucinations; style and length preferences do NOT drop a response from GOOD. NEEDS_IMPROVEMENT reserved for genuine problems. |
| **UNVERIFIABLE CLAIMS CHECK fix** | Reflection prompt's unverifiable claims check now explicitly exempts standard domain knowledge and well-reasoned elaboration from the NEEDS_IMPROVEMENT trigger. |
| **RAIN.md excluded from REFLECTION/SYNTHESIZER** | `_build_runtime_context()` now skips RAIN.md injection for REFLECTION and SYNTHESIZER agent types. Was causing HTTP 500 on gemma3:4b (RAIN.md ~3-4K tokens + system prompt + primary response > 4096 ctx). |
| **REFLECTION ctx bump** | `num_ctx` for REFLECTION raised from 4096 → 8192. |
| **Two-tier DOMAIN routing** | Added fast-path for DOMAIN agent — mirrors LOGIC's existing two-tier pattern. Short queries (≤20 words, no complexity markers) → `_fast_logic_model()` (llama3.2, 5–15s). Complex domain analysis → qwen3.5:9b (120–180s). Reduced "is bitcoin blockchain a merkle tree?" from 236s → 32s. |
| **Critique summary fix** | `↳` log line after synthesis fires now skips bare rating words (`EXCELLENT`, `GOOD`, `NEEDS_IMPROVEMENT`, `POOR`) and `Rating:` prefixes to show actual critique text instead of just the rating word. |
| **Reflection model name in log** | Log line now shows `🔍 Reflection Agent reviewing... (gemma3:4b)` with actual model name. |

### server.py

| Fix | Detail |
|-----|--------|
| **Streaming suppressed in UI** | Token events now buffer silently — no streaming bubble. Status updates (draft ready, reflecting, synthesizing) show pipeline progress. Final response renders in one shot with full markdown + badges. |
| **Progress messages with timing** | `patched_query_agent` now emits: `✓ Draft ready (Xs) — reflecting...`, `🔍 Reflection Agent reviewing... (model)`, `🔍 Reflection: good ✓` or `🔍 Reflection: needs improvement — synthesizing...`, `⚡ Synthesizer rewriting...` |
| **Monkey-patch timing** | Added `_stage_times` dict tracking primary start and synth start; elapsed times included in progress messages. |

### static/index.html

| Fix | Detail |
|-----|--------|
| **Token buffering** | `case "token"` now sets `_streamingContent += event.content` and breaks — no bubble created. `done` event renders complete final response via `appendRainMessage()`. |

### RAIN.md

- Updated Weakness #2 to ✅ Fixed (reflection rubric)
- Updated REFLECTION row to note gemma3:4b
- Updated "What Would Actually Make Rain Better" priority #1

---

## Performance Baseline (post-session)

| Query type | Before this session | After |
|-----------|---------------------|-------|
| Short factual domain query | 181s (qwen3.5:9b) | 15–20s (llama3.2 fast tier) |
| Short factual logic query | 10–20s (already had fast tier) | unchanged |
| Complex domain/logic | 120–180s | unchanged (qwen3.5:9b still correct for these) |
| Total w/ no synthesis | 200–240s | 30–50s |
| Total w/ synthesis | 300–370s | 60–90s |

---

## What's Still Outstanding

### Synthesis still fires occasionally
- With the rubric fix and RATING GUIDE, synthesis over-firing has reduced significantly
- Root cause of remaining misfires: unknown — need actual critique text to diagnose (the `↳` fix helps here; watch logs on next synthesis fire)
- TOPIC DRIFT rule in the reflection prompt may still be too strict — if synthesis fires on a correct answer that mentions adjacent concepts, soften this rule

### LoRA Weights Not Fused
- `rain-tuned` in Ollama = qwen2.5-coder:7b + behavioral system prompt only (no LoRA weights baked in)
- Adapter weights exist at `~/.rain/adapters/mlx-lora/adapters.safetensors` (22MB)
- Blocked by: mlx_lm 0.31.0 doesn't support qwen2 → GGUF export
- Monitor: `pip install --upgrade mlx-lm` then re-run `python3 finetune.py --full`

### Calibration
- Check periodically: `sqlite3 ~/.rain/memory.db "SELECT agent_type, rating, COUNT(*) FROM feedback GROUP BY agent_type, rating;"`
- Don't let test/debug sessions contaminate calibration — clear if needed

### Confidence scoring still keyword-based
- `_score_confidence()` was rewritten last session to hedge-aware (base 0.80, -0.05 per hedge phrase)
- Still not ideal — longer correct answers score lower than short ones; question type not factored in
- Deprioritized since synthesis over-firing is now mostly addressed by the rubric fix

---

## Priority Order for Next Session

1. **Phase 8: Voice** — `whisper.cpp` for STT, `piper-tts` for TTS, wake word detection, `--voice` CLI flag. Fully local. Marked as ⭐ Next in ROADMAP.md.
2. **Synthesis misfires** — if synthesis is still firing on correct responses, read the actual critique text (now visible via `↳` fix) and tune the TOPIC DRIFT rule or whichever check is triggering.
3. **Phase 11: Metacognition** — deliberate forgetting, relevance-gated memory injection, self-directed knowledge gap resolution.
4. **LoRA weight fusion** — check mlx_lm for qwen2 GGUF support; re-run `python3 finetune.py --full` when available.

---

## Key Facts for Next Claude Instance

- **Streaming**: done — tokens buffer silently, final response renders in one shot
- **gemma3:4b**: pulled and confirmed running as REFLECTION model (3.3GB, Google, strong on structured rubric tasks)
- **DOMAIN fast tier**: done — same pattern as LOGIC fast tier, using `_is_simple_logic_query` + `_fast_logic_model`
- **RAIN.md**: exists at project root, loaded at server startup via `_load_rain_md()` in server.py, injected into every agent via `_build_runtime_context()` — EXCEPT REFLECTION and SYNTHESIZER (context overflow protection)
- **Reflection ctx**: 8192 (was 4096) — required for gemma3:4b + primary response to fit
- **Worktree**: `focused-dubinsky` branch exists but main repo at `/Users/ericscalibur/Documents/Rain/` is canonical — all changes applied to main
- **No commits this session** — all changes are uncommitted edits in the working tree; commit before starting next session
