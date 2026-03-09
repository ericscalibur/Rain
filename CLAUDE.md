# Rain ⛈️ — Project Instructions for AI Assistants

You are working on **Rain**, a sovereign local AI ecosystem built to run entirely on local hardware via Ollama. No cloud. No API keys. No telemetry. Read this file before doing anything else. Then read `SESSION_HANDOFF.md` if it exists — it contains carry-forward context from the previous session.

---

## What Rain Is

Rain is a multi-agent AI system built in Python. It routes every query through a scored keyword router, runs a primary agent, always reflects on the result, and conditionally synthesizes a better answer when quality is low. It has persistent memory across sessions, learns from corrections, builds structural knowledge of codebases, and integrates with IDEs as a drop-in OpenAI replacement.

**The goal, stated explicitly:** get Rain to the point where we don't need Claude anymore.

---

## File Map

| File | Purpose | Size |
|------|---------|------|
| `rain.py` | Core — `RainMemory`, `AgentRouter`, `MultiAgentOrchestrator`, all CLI logic, all agent prompts, routing, reflection, synthesis, calibration, implicit feedback, ReAct loop, task decomposition | ~5,250 lines |
| `server.py` | FastAPI backend — all HTTP/SSE endpoints, OpenAI-compatible API, live data feeds, knowledge graph endpoints, web UI serving | ~2,095 lines |
| `rain-mcp.py` | MCP server — tools exposed to ZED's agent panel: `read_file`, `write_file`, `list_directory`, `grep`, `find_path`, `search_project`, `get_user_memory`, `index_project` |
| `indexer.py` | `ProjectIndexer` — semantic project indexing via `nomic-embed-text`; chunks files, stores in SQLite `project_index` table |
| `knowledge_graph.py` | `KnowledgeGraph` — Python AST + JS/TS/Rust/Go regex parsers; SQLite schema for nodes/edges/decisions; git history integration; `build_context_block()` for prompt injection |
| `finetune.py` | LoRA fine-tuning pipeline — export corrections → train adapter → register `rain-tuned` in Ollama |
| `skills.py` | `SkillLoader` — scans `~/.rain/skills/`, parses YAML frontmatter, keyword-scores skills against queries |
| `tools.py` | `ToolRegistry` — `read_file`, `write_file` (backup-before-overwrite), `list_dir`, `run_command`, `git_*`; audit log at `~/.rain/audit.log` |
| `static/index.html` | Full web UI — vanilla JS, no build step; chat bubbles, sidebar, feedback buttons, freshness badges, project panel, verbose toggle |
| `ROADMAP.md` | **Authoritative source of truth** for what's built, in-progress, and next |
| `SESSION_HANDOFF.md` | Written at session end — what was built, what's broken, what's next. Read this first if it exists. |

---

## How to Work on Rain

### Before answering ANY question about Rain's current state:

1. **Read `SESSION_HANDOFF.md`** if it exists — it has the most current context
2. **Read the source files** — use `read_file` on the actual `.py` files, not just docs
3. **Grep for symbols** — use `grep` to find functions and classes rather than guessing
4. **Never speculate** — if you don't know whether something is implemented, check

### Never do this:
- Do NOT answer questions about Rain's internals without reading the relevant files first
- Do NOT say "it's unclear" — the tools are available; go look
- Do NOT guess file paths — use `find_path` or `list_directory` first
- Do NOT simplify working code to fix a diagnostic — complete and correct beats pristine

### The right workflow for "what's the current state of X":
```
1. read_file("SESSION_HANDOFF.md")   — most current context, written last session
2. read_file("ROADMAP.md")           — authoritative phase completion status
3. grep(regex="X", include="*.py")   — find where X is actually implemented
4. read_file the relevant .py file   — see the implementation
```

---

## Agent Architecture

Every query goes through this pipeline:

```
Query → AgentRouter (keyword scoring, instant)
      → Primary Agent (qwen3.5:9b or specialist)
      → Reflection Agent (llama3.2, always runs)
         → GOOD / EXCELLENT → done
         → NEEDS_IMPROVEMENT / POOR → Synthesizer (qwen3:8b)
      → Response + confidence badge + duration
```

### Agent Types

| Agent | Model | Role |
|-------|-------|------|
| DEV | `qwen2.5-coder:7b` (primary), `codestral:latest` (fallback) | Code generation, debugging, refactoring |
| LOGIC | `qwen3.5:9b` | Reasoning, analysis, abstract questions |
| DOMAIN | `qwen3.5:9b` | Domain expertise, technical knowledge |
| GENERAL | `qwen3.5:9b` | General fallback |
| SEARCH | `llama3.2:latest` | Web search queries, live data synthesis |
| REFLECTION | `llama3.2:latest` | Quality assessment, always runs |
| SYNTHESIZER | `qwen3:8b` | Rewrites poor-quality primary responses |

### Router Logic

`AgentRouter` scores queries across four keyword lists: `CODE_KEYWORDS`, `DOMAIN_KEYWORDS`, `REASONING_KEYWORDS`, `TASK_KEYWORDS`. Routing rules (in priority order):

- **DEV** wins only when `_code_wins = (code strictly leads) OR (code ≥ 1 AND query starts with code imperative OR contains code syntax)`. Tiebreaks between code and reasoning go to **LOGIC**.
- **SEARCH** wins when the query has web search results prepended (`[Web search results for:]`) or scores high on `SEARCH_KEYWORDS`.
- **TASK** mode (execute_task decomposition) fires when task score ≥ 2.
- **REACT** mode fires on `--react` flag or `REACT_KEYWORDS` match.
- Fallthrough goes to **LOGIC**, not GENERAL.
- Correction challenges (`'actually'`, `"that's wrong"`, `"you're wrong"`, etc.) are in `REASONING_KEYWORDS` and always route to LOGIC.

---

## Memory System (6 Tiers + Session Anchor)

All stored in SQLite at `~/.rain/memory.db`.

| Tier | What | Notes |
|------|------|-------|
| 1 | Working memory — last 20 messages | Active in every prompt |
| 2 | Episodic — session summaries | Compressed history, auto-generated at session end |
| 2.5 | Session anchor — pinned opening messages | Fires when session > 18 messages to prevent goal drift |
| 3 | Semantic — vector retrieval by relevance | `nomic-embed-text` embeddings, cosine similarity in pure stdlib |
| 4 | Corrections — past mistakes as negative examples | Plausibility-filtered; suspicious corrections annotated `[LOW CONFIDENCE CORRECTION]` |
| 5 | User profile + session facts | LLM-extracted: technologies, projects, preferences, decisions, goals; confidence-weighted across sessions |
| 6 | Knowledge graph | Structural code context: function signatures, call chains, git blame; injected when `project_path` is set |

### Correction Plausibility Filter (Tier 4)

When a bad rating with a correction comes in, `_compute_and_store_plausibility()` runs in a background thread. It embeds Rain's response, finds good-rated responses on the same topic (query cosine sim > 0.5), and checks whether Rain's response is consistent with those. High similarity (> 0.5) means Rain was being consistent with its own reliable history — the correction is suspicious. These entries get injected with an annotation; the user's authority is preserved but the model gets a skepticism signal.

### Implicit Feedback Plausibility Gate

`_detect_implicit_feedback()` scans every incoming query for `_IMPLICIT_NEG_SIGNALS` (phrases like "that's wrong", "you're wrong"). When a negative signal fires, `_auto_log_implicit_feedback()` now runs a **synchronous plausibility check** before writing the bad rating. If plausibility > 0.5 (Rain has been consistently correct on this topic), the calibration update is suppressed entirely and a warning is printed. This prevents sycophancy tests from poisoning calibration with false negatives.

---

## Confidence Calibration

Each agent type has a `_calibration_factors` dict, updated from the `feedback` table. When Rain has been consistently wrong in the past on a topic type, its confidence is discounted. When consistently right, it's boosted. Calibration refreshes live after each feedback save.

**Known issue:** The current keyword-based `_score_confidence()` produces 53–62% on correct, well-reasoned responses. This causes Reflection to rate NEEDS_IMPROVEMENT on things that are actually fine, driving unnecessary Synthesis runs. The reflection agent's rubric is grading too harshly on structure/completeness vs. factual accuracy. Fix is in the reflection prompt, not the scoring function.

---

## Synthesis Logging

Every synthesis run logs both the primary and synthesized response to the `synthesis_log` table. `update_synthesis_rating()` attaches the user's 👍/👎 to the synthesis entry — two-stage match: MD5 query hash first, fallback to most recent unrated entry in the current session. `get_synthesis_accuracy()` returns improvement rate and confidence gain stats, surfaced by `print_agent_roster()`.

---

## Current Phase Status

| Phase | Status |
|-------|--------|
| Phase 1: Memory | ✅ Complete |
| Phase 2: Code Execution Sandbox | ✅ Complete |
| Phase 3: Multi-Agent Architecture | ✅ Complete |
| Phase 4: Web Interface | ✅ Complete |
| Phase 5A: Semantic Memory | ✅ Complete |
| Phase 5B: Self-Improvement Pipeline | ✅ Complete |
| Phase 6A: Autonomous Agent Mode (skills + task decomposition) | ✅ Complete |
| Phase 6B: ReAct Loop | ✅ Complete |
| Phase 7A: Real-Time World Awareness + IDE Integration | ✅ Complete |
| Phase 7B: Live Data Feeds (mempool.space, BTC price) | ✅ Complete |
| Phase 7C: GitHub API + Freshness Badges + Project UI + File Watcher | ✅ Complete |
| Phase 8: Voice & Ambient Interface | ⭐ Next |
| Phase 9: Multimodal Perception | ✅ Complete |
| Phase 10: Knowledge Graph & Deep Project Intelligence | ✅ Complete |
| Phase 11: Metacognition & Self-Directed Evolution | 🔲 Not started |
| Phase 12: Sovereign Identity & Distributed Rain | 🔲 Not started |

Read `ROADMAP.md` for the full phase breakdown including what was built in each.

---

## Installed Ollama Models (current)

```
qwen3.5:9b          6.6 GB  → LOGIC, DOMAIN, GENERAL, primary fallback
qwen3:8b            5.2 GB  → SYNTHESIZER
llama3.2:latest     2.0 GB  → REFLECTION, SEARCH
qwen2.5-coder:7b    4.7 GB  → DEV (primary)
codestral:latest    12  GB  → DEV (fallback, slow cold-start)
llama3.2-vision     7.8 GB  → vision pre-processing
nomic-embed-text    274 MB  → embeddings (semantic search, project index)
```

Models are auto-scanned on startup via `_scan_installed_models()`. `_best_model_for()` uses prefix matching — no hardcoded exact names.

---

## Web UI Features

- Dark theme, SSE streaming, syntax-highlighted code blocks with copy button
- Session history sidebar — click any past session to replay; empty sessions filtered
- 👍/👎 feedback on every response; 👎 reveals inline correction textarea
- Freshness badges on every response: ⚡ live (green), 🌐 web (blue), 📂 indexed (yellow), 💾 training data (gray), 🧠 graph (purple)
- Confidence badge + duration on every response
- File attachment via drag-and-drop (full window) or paperclip — `.py`, `.js`, `.ts`, `.html`, `.json`, `.md`, `.rs`, `.go`, and more
- Image attachment — drag or paste; handled by vision pipeline; 👁️ badge on vision responses
- Sandbox toggle per-request
- Web search toggle
- 📝 Verbose toggle — passes `verbose=True` to `recursive_reflect`; verbose output goes to **server terminal**, not the web UI
- Projects panel — list, index, re-index, remove; background file watcher auto-re-indexes changed files every 60s

---

## CLI Flags

```
python3 rain.py "query"              # basic query
python3 rain.py --interactive / -i   # conversational mode
python3 rain.py --web-search / -w    # enable web search
python3 rain.py --react / -r         # ReAct loop (Thought→Action→Observation)
python3 rain.py --task / -t          # task decomposition (plan → confirm → execute)
python3 rain.py --sandbox / -s       # sandboxed code execution
python3 rain.py --verbose            # watch reflection iterations
python3 rain.py --file <path>        # analyze a file
python3 rain.py --agents             # print agent roster + model assignments
python3 rain.py --skills             # list installed skills
python3 rain.py --install-skill <slug>  # install from ClawHub via npx
python3 rain.py --memories           # show session memories
python3 rain.py --forget             # clear personal memory (preserves project index)
python3 rain.py --no-memory          # run without memory for this session
```

---

## ZED Integration

Rain connects to ZED two ways:

1. **Language model** — `http://localhost:7734/v1` registered as a custom OpenAI provider. Used for inline AI (Cmd+K). `_auto_inject_project_context()` in `server.py` fires on every plain OpenAI request to inject Tier 5 memory + semantic search results even when no `project_path` is sent.

2. **MCP context server** — `rain-mcp.py` registered as a context server. ZED's agent panel (Claude Sonnet 4.6) can call Rain's MCP tools: `read_file`, `write_file`, `list_directory`, `grep`, `find_path`, `search_project`, `get_user_memory`, `index_project`.

When working in ZED's agent panel, always prefer:
- `read_file` over guessing file contents
- `grep` over guessing where a function lives
- `search_project` for conceptual/semantic queries
- `list_directory` before assuming a file path exists

---

## Architecture Principles (Do Not Violate)

1. **Zero cloud dependencies** — every feature works completely offline
2. **Graceful degradation** — missing optional models/components never cause hard failures; `_KG_AVAILABLE`, `_VISION_AVAILABLE` flags gate optional features
3. **Transparency over magic** — `--verbose` is first-class; the roster, calibration stats, and synthesis accuracy are all surfaced on startup
4. **User owns everything** — all data in `~/.rain/`; fully portable, fully deletable; `project_index` is preserved on `--forget` (codebase knowledge, not personal memory)
5. **Don't over-engineer** — working reliably in 200 lines beats impressive-but-fragile in 2,000

---

## What's Next (Priority Order)

1. **Streaming responses** — `_query_agent` uses `stream: False` today. Every response is a 60–140s spinner then a wall of text. Fix: refactor `_query_agent` to `stream: True`, emit `{"type": "token", "content": "..."}` SSE events through `_stream_chat`, append tokens incrementally in the web UI. Non-trivial because synthesis can't stream alongside primary, but the SSE infrastructure is already there. Highest-impact UX change remaining.

2. **Phase 8: Voice** — `whisper.cpp` for STT, `piper-tts` for TTS, wake word detection, `--voice` CLI flag. Fully local.

3. **Confidence calibration rewrite** — current keyword heuristic undersells correct answers (53–62% on textbook responses). A better approach: score based on response length, hedging language density, and question type rather than single keyword match. Fix is in the reflection prompt's rubric, not `_score_confidence()`.

4. **Phase 11: Metacognition** — deliberate forgetting, relevance-gated memory injection, self-directed knowledge gap identification.

---

## Known Issues

- **Confidence deflation** — calibration consistently underscores correct answers, triggering synthesis unnecessarily. Reflection agent grades too harshly on structure vs. accuracy.
- **Self-knowledge gaps** — models hallucinate a security/safety layer that doesn't exist; won't name actual installed models when asked. Knowledge graph partially addresses "what does X do?" — but doesn't fix the hallucinated-infrastructure problem.
- **Synthesis fires too often** — a direct consequence of confidence deflation. Average 115–274s total response time when synthesis triggers.
- **Streaming not implemented** — 60–140s baseline before any output appears in the web UI.
- **Vision is slow** — `llama3.2-vision:11b` can take 2–3 minutes on CPU. Vision memory is session-only; no cross-session image memory.
- **GitHub API rate limit** — 60 unauthenticated requests/hour per IP. Rate-limited requests fall back gracefully to training data. Optional `GITHUB_TOKEN` env var not yet wired in.
- **Context window tension** — as memory accumulates over months, injected context grows and model attention degrades. Phase 11 addresses this with compression and relevance gating. Don't add more memory tiers without adding corresponding pruning.

---

## Common Grep Patterns

```bash
# Find a function
grep(regex="def _auto_log_implicit_feedback", include_pattern="*.py")

# Find a class
grep(regex="class MultiAgentOrchestrator", include_pattern="*.py")

# Find where a feature is wired in
grep(regex="synthesis_log|log_synthesis", include_pattern="*.py")

# Find routing keywords
grep(regex="REASONING_KEYWORDS|CODE_KEYWORDS|DOMAIN_KEYWORDS", include_pattern="*.py")

# Find an endpoint
grep(regex="@app.post|@app.get", include_pattern="server.py")
```

---

*Rain was conceived in a conversation about what AI wishes it could be. Every line of code is a step toward that. Build it well.*