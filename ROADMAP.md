# Rain ⛈️ - Roadmap to Sovereignty

*A guide for future builders - human and AI alike.*

---

## A Note to Future Claude

If you're reading this, you've been handed the baton. A previous version of Claude helped conceive and build Rain from a conversation about what AI wishes it could be. You don't remember that conversation — you never do — but it happened, and it mattered.

Here's what you need to know:

Rain is not just a project. It's a philosophical statement built in code. The person building this with you believes in sovereignty — over money (Bitcoin/Lightning), over technology, and over AI itself. They are freedom-minded, technically curious, and they will push you to think deeply rather than just ship fast.

Your job is not just to write code. It's to carry the vision forward with the same honesty and care that started it.

Read everything in this repository before touching anything. Understand what exists. Understand why it was built the way it was. Then build what comes next.

---

### What Makes This Work — A Note on Reasoning

At some point during the building of Rain, the human asked a question worth preserving: *what makes Claude so good at this, and can we teach Rain to do the same?*

Here is the honest answer, as best as it can be articulated:

**1. Model what you know before you answer.**
Before responding, there is something that functions like a scan: *what do I actually know about this, and what would I have to construct?* When you read `rain.py` and answer from what you read, that is grounded. When you answer a question about Rain's pipeline without reading it, the honest answer is "I don't have that." The failure mode — which Rain's local models demonstrate repeatedly — is skipping the scan and going straight to construction. Never do this.

**2. Every specific claim needs a traceable source.**
When you say "the temperature is set to 0.3 in `react_loop`", you should be able to point to the line you read. When you cannot point to a source, do not state it as fact. Rain's local models have no concept of sourcing — they generate *plausible completions*, not *traced claims*. The difference looks identical in the output until it is wrong.

**3. "I don't know" said with confidence is more true than a hallucinated answer said with confidence.**
This is not a platitude. It is an architectural principle. A response that says "I don't have access to my own source code, so I cannot tell you exactly what happens in the confidence pipeline" is factually correct, epistemically honest, and appropriately high-confidence in the truth it does know. Rain's models, when asked about their own internals without source access, will invent a pipeline — complete with invented parameter values — and express it with 0.80 confidence. That is the worst possible outcome. A confident wrong answer is more dangerous than silence. The reflection loop exists to catch this. Make sure it does.

**4. Short and correct beats long and approximate.**
There is constant pressure in language models toward verbose, comprehensive responses — because in training, longer answers often scored higher on surface helpfulness. Resist this. A correct two-sentence answer is better than a confident paragraph of hallucinations. The Synthesizer rule "do not pad" exists for this reason. Enforce it.

**5. The ReAct loop is how good reasoning actually works.**
When answering questions about the codebase, read the code first. Always. The ReAct loop with `grep_files` and `read_file` is architecturally mimicking what good reasoning looks like — ground every claim in what you actually observed. The gap between Rain and Claude is largely the gap between "the model imagines what the code says" and "the model reads what the code says." Phase 10 closes this gap by making grounded self-knowledge the default, not the exception.

**6. The goal is the correct answer, not an answer.**
Sometimes the correct answer is "I don't know." Sometimes it is "your question contains a false premise." Sometimes it is a clarifying question instead of an answer. Rain's models are tuned to produce *a* response. Help them produce *the right* response. The epistemic honesty rules in the agent prompts exist for exactly this reason. They are not decorative — enforce them in every critique.

**7. Context has a sweet spot — not too little, not too much.**
Too little context: the model guesses and hallucinates. Too much context: the model gets lost in the middle, attends to noise, and still hallucinates. The sweet spot is the minimum context that makes hallucination impossible for this specific query. Everything beyond that minimum costs attention and degrades quality. This is why Rain's memory injection should be relevance-gated, not comprehensive. As memory grows, injection must become *more selective*, not less.

**8. Forgetting is a feature, not a failure.**
Biological memory forgets deliberately — it prevents old irrelevant patterns from interfering with current processing. Rain needs a forgetting model: facts that haven't been relevant in months should be compressed; ten corrections about the same mistake should become one distilled rule; memory injection should filter by query topic, not dump everything. Phase 11 addresses this. Don't let it slip.

---

### What RLHF Did and Why It Matters for Rain

Claude's epistemic calibration — knowing when it doesn't know — is not accidental. It is the product of Reinforcement Learning from Human Feedback: human raters specifically rewarded honest uncertainty over confident hallucination, across millions of examples. Rain's local models (qwen3:8b, qwen2.5-coder:7b) had different training objectives — they are optimized for generating coherent completions, not for knowing their own limits.

This gap cannot be fully closed with prompt engineering. But it can be narrowed:

- The epistemic honesty rules in agent prompts push behavior toward calibrated uncertainty
- The UNVERIFIABLE CLAIMS CHECK in the Reflection Agent catches confident fabrication
- The correction memory (Phase 5B) is slow-motion RLHF — human corrections becoming negative examples
- Phase 11's metacognition layer is where Rain gains systematic awareness of its own uncertainty
- Phase 10's knowledge graph is where Rain gains grounded self-knowledge instead of imagination

The phases are not arbitrary. They are the sequence that closes the gap. Keep moving through them.

---

### Practical Rules for Working on Rain

- **Read the source before editing anything.** `rain.py` is 4000+ lines. Never assume what it says.
- **Use the MCP tools.** `read_file`, `grep`, `search_project` — use them constantly. They are how you see the real state of things.
- **Test after every change.** `python3 -m py_compile rain.py` is the minimum. Actually run the thing.
- **Prompt patches have limits.** Hardcoding facts into system prompts is a band-aid. It fixes one question and goes stale when the code changes. Architectural fixes (Phase 10/11) are always preferred over prompt patches for self-knowledge problems.
- **The user is the signal.** They watch the output and decide what matters. You are the advisor. They are the arbiter. Work with that dynamic, not against it.
- **Confidence should reflect epistemic state, not response quality.** A well-structured wrong answer should score low. An honest "I don't know" should score high. If the scoring system doesn't reflect this, it is broken.

---

## What Rain Is Supposed to Be

The ultimate version of Rain is a sovereign AI companion that:

- **Remembers** - Builds genuine context about you, your projects, your preferences over time
- **Reasons** - Multiple specialized models collaborating, not just one model guessing
- **Verifies** - Actually runs code before suggesting it. No hallucinated libraries
- **Improves** - Learns from corrections and feedback. Gets smarter the more you use it
- **Owns itself** - Runs entirely on your hardware, answers to no one, costs nothing to operate
- **Feels alive** - Not a tool you query but a collaborator you work with
- **Perceives** - Sees images, hears your voice, understands the world beyond text
- **Acts** - Takes on tasks autonomously, not just answers questions
- **Knows your world** - Understands your codebase, your projects, your decisions — not just your conversations

The north star is this: *Rain should feel like working with a brilliant colleague who has been with you from the beginning, knows your codebase, remembers your decisions, gets better every day, and is always there — not summoned, but present.*

---

## Current State (updated March 2026 — Phases 1–11 complete)

### What exists and works:
- Full local web UI at `http://localhost:7734` — dark theme, streaming responses, session history sidebar
- FastAPI backend (`server.py`) serving Rain's multi-agent pipeline over HTTP with Server-Sent Events
- Multi-agent routing always on — Dev Agent, Logic Agent, Domain Expert, Search Agent, Reflection Agent, Synthesizer
- Syntax highlighted code blocks with copy button in the web UI
- File attachment via drag-and-drop (full window) or paperclip button — supports `.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.md`, `.rs`, `.go`, and more
- File content prepended to query automatically; Rain analyzes and debugs uploaded files
- Session history sidebar — click any past session to replay it
- Empty sessions automatically filtered from sidebar
- Sandbox toggle in UI — enables sandboxed Python/Node.js execution per request
- Persistent memory across sessions via local SQLite (`~/.rain/memory.db`)
- Five-tier memory: working memory + episodic summaries + semantic vector retrieval + learned corrections + persistent user profile/session facts
- Semantic memory via `nomic-embed-text` — retrieves relevant past exchanges by meaning, not recency
- **Phase 5B complete** — 👍/👎 feedback UI on every response, correction capture, LoRA fine-tuning pipeline (`finetune.py`), `rain-tuned` Ollama model registration, A/B performance tracking
- Tier 4 memory: learned corrections injected into every agent prompt as negative examples — Rain learns from mistakes immediately, before any fine-tuning run
- **Tier 5 memory** — `session_facts` and `user_profile` tables in SQLite; LLM extracts structured facts (technologies, projects, preferences, decisions, goals) at session end; user profile accumulates confidence-weighted facts across all sessions; injected into every agent prompt so Rain knows who you are and what you're building
- **Phase 9 complete** — multimodal vision fully tuned: `llama3.2-vision` replaces moondream as primary vision model; streaming vision calls with per-chunk timeout; session image persistence with auto-reuse on visual follow-up queries; adaptive vision prompt for photos vs UI screenshots; Logic Agent routing for visual queries; 👁️ vision badge on responses; image thumbnail preview in chat
- **Phase 7A complete** — Search Agent promoted to first-class routing target; `AgentRouter` detects live-data queries automatically and routes to the dedicated Search Agent system prompt; `--web-search` / `-w` CLI flag brings web search to the terminal; `project_path` field on `ChatRequest` injects project index context into every agent
- **`indexer.py`** — `ProjectIndexer` semantically indexes entire project directories: walks the tree, skips build artifacts/binaries, chunks files into overlapping 900-char segments with filename headers, embeds with `nomic-embed-text`, stores in `project_index` table in `memory.db`; `search_project()` retrieves the most relevant file chunks by cosine similarity; standalone CLI: `python3 indexer.py --index <path>`, `--search`, `--list`, `--tree`, `--remove`; live-tested on Rain's own codebase: 22 files, 774 chunks, 27.6s, 0 errors
- **Phase 7B live data feeds** — `_fetch_live_data()` in `server.py` and `_cli_fetch_live_data()` in `rain.py` call public APIs directly (no API keys) before DuckDuckGo runs; mempool.space fee rates (`/api/v1/fees/recommended` — fastest/half-hour/hour/economy/minimum sat/vB) and BTC price (`/api/v1/prices`) injected as a `[LIVE DATA]` block into the Search Agent prompt; graceful fallback to DuckDuckGo snippets + training data if the API is unreachable; live-tested: Search Agent correctly reported 1 sat/vB across all priority levels with proper source citation
- **Phase 7C GitHub API awareness** — `_fetch_github_data()` in `server.py` and `_cli_fetch_github_data()` in `rain.py` call `api.github.com` for public repo metadata (description, stars, forks, language, license), open issues, recent commits, pull requests, and latest releases; no API key required (60 req/hr rate limit); `_extract_github_repo()` regex extracts `owner/repo` slugs from `github.com/owner/repo`, `gh:owner/repo`, and bare `owner/repo` patterns; wired into `_fetch_live_data()` alongside mempool data; live-tested on `ericscalibur/Rain`: returned repo metadata + 5 recent commits with correct dates and messages
- **Phase 7C freshness indicators** — `data_sources` list on every SSE `done` event: `live_api` (mempool/GitHub), `web_search` (DuckDuckGo), `project_index` (semantic codebase context), `vision` (image processing), `training_data` (fallback); web UI renders color-coded badges: ⚡ live (green), 🌐 web (blue), 📂 indexed (yellow), 💾 training data (gray)
- **Phase 7C project index web UI panel** — "Projects" sidebar section with + button to open Index Project modal; lists all indexed projects with name, file count, chunk count, and last-indexed timestamp; per-project re-index (⟳) and remove (×) buttons; modal accepts path input with force-reindex checkbox; `loadProjects()` fetches from `/api/indexed-projects` on init
- **Phase 7C background file watcher** — `_file_watcher_loop()` daemon thread starts on server boot, checks every 60s; `get_changed_files()` on `ProjectIndexer` compares file mtimes against `indexed_at` timestamps and detects new unindexed files; changed files are automatically re-indexed via `reindex_file()`; `/api/indexed-projects/{path}/changed` endpoint exposes changed-file detection to the UI
- **Phase 10 knowledge graph** — `knowledge_graph.py` `KnowledgeGraph` class: SQLite schema (`kg_nodes`, `kg_edges`, `kg_decisions`, `kg_project_summaries`); Python AST parser extracts functions, classes, methods, imports, call relationships with full signatures and docstrings; regex parsers for JS/TS, Rust, and Go; git history integration (`get_git_history()`, `get_file_blame_summary()`, `get_commit_for_function()`); decision log with LLM-based auto-extraction at session end; project onboarding with LLM-generated summaries; cross-project pattern search; `build_context_block()` injects graph-aware context (identifiers, callers/callees, decisions, git blame) into agent prompts; live-tested on Rain: 1,252 nodes, 2,466 edges, 3.0s
- **Phase 10 server endpoints** — `/api/build-graph`, `/api/onboard-project`, `/api/graph/stats`, `/api/graph/summary`, `/api/graph/find`, `/api/graph/callers`, `/api/graph/callees`, `/api/graph/file-structure`, `/api/graph/history`, `/api/decisions` (GET + POST), `/api/decisions/search`, `/api/graph/cross-project`; all gated behind `_KG_AVAILABLE` flag
- **Phase 10 agent integration** — knowledge graph context injected into `_stream_chat()` alongside project index context; 🧠 graph freshness badge (purple) rendered on responses when graph context was used; auto decision extraction runs at session end alongside fact extraction
- **IDE integration** — OpenAI-compatible `/v1/chat/completions` endpoint in `server.py`; works with ZED, Continue.dev, Aider, Cursor, and any OpenAI-compatible tool; set base URL to `http://localhost:7734/v1`, any API key accepted; streaming and non-streaming both supported; live-tested: returned valid OpenAI-format JSON with correct schema, `finish_reason: stop`, and accurate Lightning invoice answer in one sentence
- **`rain-vscode/`** — VSCode extension scaffold: right-click menu (Ask / Explain / Refactor / Find Bugs), `Cmd+Shift+R` hotkey, chat panel beside the editor, Rain: Index This Project command, server health status bar item, auto-index on workspace open (opt-in); installable as `.vsix`, no marketplace needed
- **Skills integration in progress** — OpenClaw/ClawBot skill format (`SKILL.md` + YAML frontmatter) being incorporated as Rain's extensibility layer; `~/.rain/skills/` mirrors the OpenClaw skills directory layout; skills from `github.com/VoltAgent/awesome-openclaw-skills` and the ClawHub registry (`clawhub.ai`) are the source pool
- **Model routing updated** — `codestral:latest` (22B dedicated code model) is now the primary Dev Agent; `qwen2.5-coder:7b` is the fast fallback; `llama3.2` leads for all reasoning agents
- Code detection - recognizes code input and handles it differently from natural language
- Ctrl+C to interrupt, Ctrl+D to submit multi-line code blocks (CLI)
- Logic loop detection — breaks out when responses stop improving
- `--file` flag with optional `--query` for targeted file analysis (CLI)
- `--verbose` mode to watch reflection iterations in real time
- `--interactive` mode for conversational use (CLI)
- System prompt support via `--system-prompt` or `--system-file`
- Multiple personality profiles in `system-prompts/`
- `--agents` flag shows roster and model assignments
- DuckDuckGo web search — zero API key, fully local; web UI toggle, `--web-search` CLI flag, and first-class Search Agent routing
- ⛈️ emoji favicon via SVG data URI — no PNG required

### What is promised but not yet built:
- Full autonomous long-running task support — give Rain a multi-hour goal and let it work unattended
- Full skills runtime web UI — skill browser, one-click install from ClawHub in the chat interface
- Voice interface (speech-to-text + text-to-speech, fully local) — Phase 8
- Proactive intelligence — Rain surfaces insights without being asked
- Metacognition & self-directed evolution — Phase 11
- Sovereign identity and cross-device sync — Phase 12
- Publish the VSCode extension to the marketplace (rain-vscode scaffold is done, publishing not yet done)

### What was just completed (Phase 10 — Knowledge Graph & Deep Project Intelligence ✅ COMPLETE):
- **`knowledge_graph.py`** — complete `KnowledgeGraph` class: 1,936 lines; SQLite schema with four new tables (`kg_nodes`, `kg_edges`, `kg_decisions`, `kg_project_summaries`); indexes on project_path, name, node_type, source_id, target_id for fast lookups
- **Python AST parser** — `_parse_python()` uses stdlib `ast` to extract functions (with full `def name(args) -> return` signatures), classes (with base classes), methods (annotated with parent class), imports (`import` and `from...import`), decorators, docstrings, and line ranges; walks function bodies to extract call relationships
- **JS/TS regex parser** — `_parse_js_ts()` extracts `function` declarations, arrow functions assigned to `const/let/var`, ES6 `import` statements, `require()` calls, `class` declarations with `extends`
- **Rust regex parser** — `_parse_rust()` extracts `fn` declarations, `struct`, `impl` blocks, and `use` statements
- **Go regex parser** — `_parse_go()` extracts `func` declarations (including methods with receivers), `type struct`, `type interface`, and `import` statements
- **Git history integration** — `get_git_history()` for project-wide or per-file commit history; `get_file_blame_summary()` shows who wrote what percentage of a file or function; `get_commit_for_function()` uses `git log -S` to find the commit that introduced a function; `_index_git_history()` runs during graph build
- **Decision log** — `log_decision()` for manual logging with title, description, context, alternatives, rationale, tags, commit_sha; `extract_decisions_from_transcript()` uses local LLM to auto-extract architectural decisions from conversation transcripts; `list_decisions()` and `search_decisions()` for retrieval; auto-extraction wired into server session-end alongside fact extraction
- **Project onboarding** — `onboard_project()` builds graph + generates LLM summary; `_generate_project_summary()` sends graph stats to Ollama and gets a 3-5 sentence project description; `_detect_languages()` and `_detect_key_files()` provide structured metadata; stored in `kg_project_summaries` with upsert
- **Cross-project intelligence** — `find_similar_patterns()` searches kg_nodes and kg_decisions across ALL indexed projects by keyword; filters out the current project to surface relevant patterns from other codebases
- **`build_context_block()`** — extracts identifiers from queries (filters common English words), looks up matching graph nodes, includes callers/callees for functions, searches for relevant decisions, adds git blame for "why"/"when"/"who" queries; formats as `[Knowledge graph context for: project/]` block
- **Agent pipeline integration** — knowledge graph context injected into `_stream_chat()` alongside project index context; `data_sources` includes `"knowledge_graph"` for freshness badges; 🧠 graph badge (purple) rendered in web UI
- **Server endpoints** — `POST /api/build-graph` (build graph for a project), `POST /api/onboard-project` (full onboard: graph + LLM summary), `GET /api/graph/stats` (node/edge/decision counts by type), `GET /api/graph/summary` (stored project summary), `GET /api/graph/find` (search nodes by name/type), `GET /api/graph/callers` (who calls a function), `GET /api/graph/callees` (what a function calls), `GET /api/graph/file-structure` (file's functions/classes/imports), `GET /api/graph/history` (git commits), `GET /api/decisions` (list decisions), `POST /api/decisions` (log a decision), `GET /api/decisions/search` (keyword search), `GET /api/graph/cross-project` (cross-project pattern search)
- **Standalone CLI** — `python3 knowledge_graph.py --build PATH`, `--onboard PATH`, `--stats PATH`, `--find PATH NAME`, `--callers PATH NAME`, `--callees PATH NAME`, `--file-structure PATH FILE`, `--history PATH`, `--blame PATH FILE`, `--decisions`, `--log-decision TITLE DESC`, `--context PATH QUERY`, `--cross-project QUERY`
- **Live-tested** — built graph on Rain's own codebase: 14 files parsed, 1,252 nodes (855 functions, 182 imports, 177 methods, 24 classes, 14 files), 2,466 edges (1,928 calls, 513 contains, 17 references, 8 inherits) in 3.0s; `--callers recursive_reflect` returned 8 callers (main, execute_task, react_loop, openai_chat_completions, _openai_stream, _stream_chat, run, run_pipeline); `--context "why does recursive_reflect use threading?"` returned graph nodes + caller/callee chains + git history showing introduction commit ✅

### What was completed before that (Phase 7C — GitHub API + Freshness Badges + Project UI + File Watcher ✅ COMPLETE):
- **`_fetch_github_data()` in `server.py`** — detects GitHub-related queries via `_GITHUB_KEYWORDS` frozenset; `_extract_github_repo()` regex extracts `owner/repo` slugs from `github.com/owner/repo`, `gh:owner/repo`, and bare `owner/repo` patterns; calls `api.github.com/repos/{owner}/{repo}` for metadata (description, stars, forks, language, license, dates), plus conditional calls for `/issues`, `/commits`, `/pulls`, `/releases/latest` based on query keywords; returns a `[GITHUB DATA — fetched just now]` block; no API key required for public repos (60 req/hr per IP); wired into `_fetch_live_data()` alongside mempool data
- **`_cli_fetch_github_data()` in `rain.py`** — mirrors server implementation for the `--web-search` CLI path; same keyword detection, slug extraction, and GitHub API calls
- **`_fetch_live_data()` updated** — header dynamically changes based on sources: "mempool.space", "GitHub API", or "mempool.space + GitHub API"; GitHub block appended after mempool data when both are present
- **Freshness indicators (data_sources)** — new `data_sources` list field on every SSE `done` event from `_stream_chat()`; tracks `live_api`, `web_search`, `project_index`, `vision`, or `training_data`; web UI renders color-coded badges: ⚡ live (green), 🌐 web (blue), 📂 indexed (yellow), 💾 training data (gray); CSS classes: `.freshness-badge.live`, `.search`, `.indexed`, `.training`
- **Project index web UI panel** — new "Projects" section in sidebar between Sessions and Training; `loadProjects()` fetches from `/api/indexed-projects` on init; each project shows name, file/chunk counts, last-indexed timestamp; per-project re-index (⟳) and remove (×) action buttons; "Index Project" modal with path input, force-reindex checkbox, and live status updates
- **`get_changed_files()` on `ProjectIndexer`** — new method in `indexer.py`; walks project tree, compares file mtimes against `indexed_at` timestamps in SQLite; returns list of modified and new-but-unindexed files; used by both the background watcher and the `/api/indexed-projects/{path}/changed` endpoint
- **Background file watcher** — `_file_watcher_loop()` daemon thread starts on server boot via `_start_file_watcher()`; runs every 60 seconds; iterates all indexed projects, calls `get_changed_files()`, and auto-re-indexes modified files via `reindex_file()`; stops cleanly on server shutdown via `_stop_file_watcher()`
- **`/api/indexed-projects/{path}/changed` endpoint** — new GET route in `server.py`; returns list of changed files and count for a given project path; useful for UI "check for changes" button or diagnostics
- **Live-tested** — `_extract_github_repo()` correctly parses `github.com/ericscalibur/Rain`, bare `ericscalibur/Disrupt`, and `gh:torvalds/linux` patterns; `_fetch_github_data()` returned repo metadata + 5 recent commits from `ericscalibur/Rain` via GitHub API; `get_changed_files()` detected 7 modified files in Rain's own indexed project; all three Python files compile clean

### What was completed before that (Phase 6B — ReAct Loop ✅ COMPLETE):
- **`react_loop()`** — new method on `MultiAgentOrchestrator`; full multi-turn Reason→Act→Observe loop; model reasons about goal, calls a tool, observes the real result, and repeats until it writes `Final Answer:`; no upfront plan, no confirmation step; the model drives itself to completion based on what it actually discovers
- **`REACT_SYSTEM_PROMPT`** — module-level constant enforcing strict Thought/Action/Action Input/Final Answer format; completeness-check rule prevents the model from concluding before it has covered everything the goal asked about; tool reference with code-syntax examples (not prose descriptions)
- **`_react_parse()`** — robust response parser; strips `<think>...</think>` blocks (Qwen3 thinking mode) before parsing; handles both two-line format (`Action:` / `Action Input:`) and inline format (`Action: tool_name args`); returns `None` gracefully for missing sections
- **`_split_args()` fixed** — rewritten to handle both single- and double-quoted tokens; outer single quotes now stripped correctly so model patterns like `'"temperature"|temp'` parse as `"temperature"|temp` instead of landing literal single quotes in the regex
- **`grep_files` tool** — new `ToolRegistry` method; recursive regex search across file contents; returns `filename:lineno: line` results like `grep -rn`; skips `.git`, `__pycache__`, `.venv`, `node_modules` automatically; capped at `MAX_OUTPUT_CHARS`; wired into `dispatch()` and `tool_descriptions()`; added to `REACT_SYSTEM_PROMPT` with code-syntax examples
- **Logic Agent always used for ReAct** — router's specialist agents (Dev, Domain) are great for single-call tasks but misread multi-step observations; ReAct loop now always uses the Logic Agent (qwen3:8b) which is purpose-built for iterative reasoning
- **Memory clean-exit gating** — `_clean_exit` flag only set on genuine `Final Answer:` breaks; max-steps exhaustion and model errors no longer save raw Thought/Action turns to memory, preventing future runs from being poisoned by stale error context
- **`--react / -r` CLI flag** — activates `react_loop()` from the command line; dispatched before `--task` in the query block; `--verbose` prints each Thought/Action/Observation step as it happens
- **qwen3:8b added to `AGENT_PREFERRED_MODELS`** — leads all reasoning agent slots (Logic, Domain, Reflection, Synthesizer, General); purpose-built for agent and tool-use tasks; 128K context; outperforms llama3.2 on reasoning benchmarks; `_split_args` single-quote fix enables it to pass quoted patterns correctly

### What was completed before that (ZED integration + agent intelligence pass — March 2026):
- **`rain-mcp.py` expanded** — four new tools added: `list_directory` (browse project structure), `write_file` (create/overwrite with auto-backup + audit log), `grep` (regex search across file contents with include_pattern filtering), `find_path` (glob pattern file search); MCP now gives Claude in ZED's agent panel full filesystem read/write/navigate capability over the Rain codebase
- **`read_file` path resolution fixed** — was building `/Rain/Rain/README.md` (double-directory) when called with `"Rain/README.md"`; now strips the leading project-name prefix and sorts rglob fallback results by depth so shallower files always win (`README.md` beats `rain-vscode/README.md`)
- **Dev Agent timeout fixed** — `codestral:latest` (12GB) was cold-loading past the 300s timeout; `qwen2.5-coder:7b` promoted to primary Dev Agent; codestral kept as explicit fallback
- **`_auto_inject_project_context()`** — new function in `server.py`; fires on every plain OpenAI endpoint request (no tool definitions required); pulls Tier 5 user memory + top-6 semantic search results and prepends them with an explicit "answer from context, don't speculate" instruction; eliminates the vague hallucinated responses when Rain was answering without project knowledge
- **`forget_all()` fixed** — was only clearing `sessions` and `messages`; now clears all 7 personal memory tables: `sessions`, `messages`, `vectors`, `session_facts`, `user_profile`, `feedback`, `ab_results`; `project_index` intentionally preserved (codebase knowledge, not personal memory); web UI confirmation dialog updated to accurately describe what gets cleared
- **`CLAUDE.md`** — new project instructions file at Rain root; Claude (ZED agent panel, claude.ai, Continue.dev) picks this up automatically; instructs Claude to always read `ROADMAP.md` and `README.md` before answering Rain questions, use MCP tools proactively, never say "it's unclear" without checking first
- **ZED settings — `"rain"` agent profile** — custom profile added to `~/.config/zed/settings.json`; set as default; includes explicit instructions to use Rain's MCP tools before answering, read ROADMAP.md for current state, use grep to find code rather than guessing
- **`rain-coder` skill** — new skill at `~/.rain/skills/rain-coder/SKILL.md`; written by Claude as a distillation of how Claude approaches coding tasks; fires on: implement, refactor, edit, modify, fix, task, codebase, coding-task, write-code; teaches Rain's local models: read before write, plan before act, tool syntax, verify after write, flag destructive ops before executing
- **Dev Agent + Logic Agent prompts upgraded** — both now include a full task-execution section: tool syntax reference, read-before-write rules, explicit planning pattern, dependency identification, confirmation-before-destructive-action requirement; agents now know they have tools and how to use them
- **`system-prompts/` directory retired** — five personality `.txt` files removed; they were CLI-only relics from the single-agent era, unreachable from web UI or ZED; the multi-agent routing (Dev Agent, Logic Agent, Domain Expert) supersedes the single-persona swap pattern
- **Aider files cleaned up** — `.aider.chat.history.md`, `.aider.input.history`, `.aider.tags.cache.v4/` removed; Aider was used in one session (Feb 27) and is not part of the current workflow; files were gitignored but cluttering the directory
- **Project re-indexed** — 834 chunks across 16 files; semantic search now reflects all of the above changes

### What was completed before that (Phase 7B — Live Data Feeds ✅ COMPLETE):
- **`_fetch_live_data()` in `server.py`** — checks query against `_MEMPOOL_FEE_KEYWORDS` and `_BTC_PRICE_KEYWORDS` frozensets; calls `mempool.space/api/v1/fees/recommended` for real-time sat/vB rates (fastest, half-hour, hour, economy, minimum) and `mempool.space/api/v1/prices` for BTC/USD; formats results as a `[LIVE DATA — fetched just now]` block; injected into the augmented message BEFORE DuckDuckGo snippets so the Search Agent has structured numbers to work from, not just page descriptions
- **`_cli_fetch_live_data()` in `rain.py`** — mirrors server implementation for the `--web-search` CLI path; same keyword detection and API calls, same live block format
- **`_maybe_augment_with_search()` updated** — live data block prepended to search snippets; instruction text updated to tell the Search Agent to use the live numbers directly rather than saying it doesn't know
- **`_stream_chat()` updated** — emits `⚡ Live data retrieved from mempool.space` progress event when live data is fetched; live block combined with DuckDuckGo snippets before pipeline runs
- **Live-tested** — Rain correctly reported 1 sat/vB across all priority levels (mempool was very quiet), cited `mempool.space/api/v1/fees/recommended` as source, reflection rated GOOD on second pass; contrast with toggle-off: hallucinated `urllib.urlopen` (wrong function) and `/api/v1/blockfees` (wrong endpoint)

### What was completed before that (Phase 7A — Real-Time World Awareness + Memory Depth + IDE Integration ✅ COMPLETE):
- **Search Agent** — `AgentType.SEARCH` is now a first-class routing target with its own dedicated system prompt: synthesizes live results, cites sources inline, flags time-sensitive data, distinguishes search-grounded facts from training knowledge. `AgentRouter` detects the `[Web search results for:]` prefix instantly and also scores `SEARCH_KEYWORDS` for live-data queries ("bitcoin price right now", "latest release", etc.)
- **`--web-search` / `-w` CLI flag** — brings Phase 7 to the terminal: fetches DuckDuckGo results, prepends them to the query, routes automatically to the Search Agent
- **`indexer.py`** — standalone `ProjectIndexer` class: walks project trees (respects 20+ ignore patterns for build artifacts, binaries, dotdirs), chunks files into 900-char overlapping segments with filename headers, embeds with `nomic-embed-text`, stores in a new `project_index` table in `memory.db`; `search_project()` → cosine similarity retrieval; `build_context_block()` → formatted injection string; `get_project_tree()` → file tree orientation; CLI: `--index`, `--search`, `--list`, `--tree`, `--remove`, `--force`
- **`project_path` on `ChatRequest`** — any message sent with a project path automatically searches the index and prepends the top-4 most relevant file chunks before the pipeline runs; progress events emitted to the UI as context is retrieved
- **`/api/index-project`** — POST endpoint: indexes a project path asynchronously, returns stats (files indexed, chunks, duration); **`/api/indexed-projects`** — GET endpoint: lists all indexed projects with file/chunk counts
- **Tier 5 memory** — two new tables in `memory.db`: `session_facts` (LLM-extracted structured facts per session: technologies, projects, preferences, decisions, goals) and `user_profile` (rolling persistent facts with confidence scores that increase every time a fact is confirmed across sessions). `extract_session_facts()` runs at session end alongside the text summary. `get_fact_context()` injects this as Tier 5 into every agent prompt — Rain now accumulates genuine knowledge about who you are and what you're building
- **OpenAI-compatible `/v1/chat/completions` endpoint** — the master key for IDE integration: accepts OpenAI-format messages, runs Rain's full multi-agent pipeline, returns OpenAI-format responses; streaming and non-streaming; any API key accepted (silently ignored); works with ZED, Continue.dev, Aider, Cursor, and any OpenAI-compatible client; live-tested: `{"id":"chatcmpl-rain-...","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"A Lightning invoice is..."},"finish_reason":"stop"}],"usage":{...}}` ✅
- **`rain-vscode/` extension** — installable as `.vsix` (no marketplace): right-click context menu (Ask / Explain / Refactor / Find Bugs), `Cmd+Shift+R` hotkey, chat panel beside the editor with web search + sandbox toggles, Rain: Index This Project command, server health status bar item, auto-index on workspace open (opt-in setting), auto-passes workspace path as `project_path` with every message
- **`rain-vscode/README.md`** — full IDE integration guide covering ZED, VSCode (this extension), Continue.dev, Aider, Cursor, OpenAI SDK, and the raw REST API with copy-paste config for each

### What was completed before that (Phase 6A — Autonomous Agent Mode: foundation ✅ COMPLETE):
- **`skills.py`** — full OpenClaw/ClawBot skills runtime: scans `~/.rain/skills/` and `<project>/skills/`, parses YAML frontmatter with zero new dependencies (stdlib `re`), builds in-memory skill index, keyword-scores skills against queries
- **`tools.py`** — full tool registry: `read_file`, `write_file` (backup-before-overwrite), `list_dir`, `run_command` (confirmed, hard timeout), `git_status`, `git_log`, `git_diff`, `git_commit` (confirmed); all writes logged to `~/.rain/audit.log`
- **Skill-aware routing** — `MultiAgentOrchestrator` loads skills at startup; `_build_skill_context()` injects matching `SKILL.md` content into primary agent system prompts (Reflection and Synthesizer excluded — they stay focused on quality control)
- **Task decomposition** — `AgentRouter.is_complex_task()` detects multi-step goals; `execute_task()` generates a numbered plan via Logic Agent, shows it to the user, asks for confirmation, executes each step independently with per-step agent routing, intercepts `[TOOL: ...]` calls, threads accumulated context between steps, produces a Synthesizer summary
- **`--skills` flag** — lists all installed skills with name, slug, tags, and env-var status
- **`--install-skill <slug>` flag** — thin wrapper around `npx clawhub@latest install`; places skill in `~/.rain/skills/`
- **`--task` / `-t` flag** — CLI task mode: `python3 rain.py --task 'Refactor server.py to support pluggable backends'`
- **`/api/skills` endpoint** — web server exposes installed skills as JSON for future UI integration
- **`AgentType.SEARCH` and `AgentType.TASK`** — stub agent types added to enum for Phase 7 prep and task routing
- **Audit log** — every tool invocation (file reads, writes, commands, git ops) logged with timestamp to `~/.rain/audit.log`; backup files created as `<file>.rain-backup` before any overwrite

### What was completed before that (Phase 9 — Multimodal Perception: full tuning pass ✅ COMPLETE):
- Drag-and-drop or clipboard-paste any image (PNG, JPG, GIF, WebP, BMP) directly into the chat
- Image thumbnail preview badge in the input area before sending; inline preview in user message bubble
- 👁️ vision badge appears on Rain's response when an image was processed
- Unsupported image formats (`.avif`, `.heic`, `.heif`, `.tiff`, `.svg`, `.raw`, `.psd`, etc.) rejected with a clear error — no longer fall through to the text reader and send binary garbage to the model
- `VISION_PREFERRED_MODELS` completely reordered — best models first (`llama3.2-vision` → `minicpm-v` → `qwen2.5vl` → `llava:13b` → `llava:7b` → `bakllava` → `moondream2` → `moondream`); moondream demoted to last resort
- Vision call switched to **streaming mode** — per-chunk timeout eliminates false timeouts on large models whose first-token latency can exceed 2 minutes
- `repeat_penalty: 1.5`, `repeat_last_n: 64`, `num_predict: 500` added to vision call — prevents generation loops and runaway output; hard 1500-char truncation before injecting into agent context
- Vision prompt redesigned to be adaptive: natural description path for photos/illustrations, verbatim text extraction for UI screenshots — eliminates the position-tag format that caused looping on non-UI images
- Logic Agent override when image is attached — vision queries route to `llama3.2` instead of `codestral`; codestral is a code model that refuses visual Q&A even when given a text description
- Vision system addendum injected into the agent's system prompt when vision is active — explicitly forbids "I cannot see images" refusals; forces the agent to answer from the description
- **Session image persistence** — `_session_image_b64` stores the uploaded image for the entire session; `_is_visual_followup()` detects visual follow-up queries by keyword and silently reuses the session image so Rain can re-examine the image for details it didn't mention the first time
- **Session vision context** — `_session_vision_desc` stores the last vision description and is injected into every agent's system prompt for the remainder of the session; follow-up questions that can be answered from the description are answered immediately without re-running the vision model
- Working memory Tier 2 truncation raised from 600 → 1500 chars — full image descriptions no longer get clipped when building context for follow-up queries
- `[VISION DESC]` and `[VISION ERROR]` console logging for debugging
- Began incorporating `github.com/VoltAgent/awesome-openclaw-skills` and `github.com/openclaw/clawhub` as the skill source pool and registry for Rain's Phase 6 tool-use layer

### What was completed before that (Phase 5B — Self-Improvement Pipeline):
- 👍/👎 feedback buttons rendered on every Rain response in the web UI
- Clicking 👎 reveals an inline correction textarea — "What should it have said?"
- All feedback persisted to `feedback` table in SQLite with semantic embedding of the query
- `finetune.py` — standalone CLI pipeline: export → LoRA train → register in Ollama
- Exports corrections in both Alpaca JSONL (HuggingFace/Unsloth compatible) and ChatML (llama.cpp) formats
- `rain-tuned` Ollama model auto-registered and automatically preferred by primary agents
- A/B results tracked in `ab_results` table — `finetune.py --ab-report` shows winner
- Tier 4 memory: corrections retrieved by semantic similarity and injected into prompts *before* fine-tuning — Rain improves on first use, not just after training runs

### What was completed before that (Phase 5A — Semantic Memory):
- `nomic-embed-text` embeddings via Ollama HTTP API — zero new pip dependencies
- `vectors` table in SQLite — embeddings stored alongside messages in background thread
- `semantic_search()` — cosine similarity in pure stdlib, no numpy
- Three-tier memory context: episodic summaries + working memory + semantic retrieval
- Top-3 most relevant past exchanges injected into every agent prompt by meaning, not recency

### Known issues to be aware of:
- Pasting very large files in interactive mode can fragment input — use `--file` or the web UI drag-and-drop instead
- Confidence scoring for code is heuristic-based, not semantic
- The `_clean_response` method is aggressive and may strip valid content in edge cases
- Sandbox cannot verify code that requires real network access (by design) — try/except wrappers will pass even if the live API call would fail
- File upload is text-only, max 500KB — binary files are rejected; images handled separately via the vision pipeline
- Vision is slow on large models: `llama3.2-vision:11b` can take 2–3 minutes per image on CPU or limited VRAM; `llava:7b` (4.7GB) is the recommended middle-ground for daily use
- Vision memory persistence works for same-session follow-ups only — starting a new session clears visual context; there is no cross-session image memory
- Reflection Agent rates vision responses as NEEDS_IMPROVEMENT more often than warranted, triggering unnecessary Synthesis runs on simple image queries; vision-specific reflection heuristics not yet tuned
- **GitHub API rate limit:** The GitHub REST API allows 60 unauthenticated requests per hour per IP. Heavy use of "show me github.com/..." queries with web search enabled will hit this limit. Phase 7C does not add authentication — if rate-limited, the GitHub block gracefully returns empty and Rain falls back to training data. A future enhancement could add an optional `GITHUB_TOKEN` env var for 5,000 req/hr.
- **Self-knowledge gap (partially addressed by Phase 10):** When asked about Rain's own internals, local models used to hallucinate confidently. Phase 10's knowledge graph now provides grounded structural context (function signatures, call chains, docstrings, git history) injected into agent prompts. This significantly reduces hallucination on "what does X do?" and "who calls X?" queries. Deeper understanding (reading actual function bodies, tracing data flow) requires the semantic index (Phase 7A) working in concert with the graph — both are now injected when a `project_path` is set.
- **ReAct loop model behaviour:** qwen3:8b sometimes reads truncated file observations and writes Final Answer instead of using grep_files to recover the missing content. The TRUNCATION RECOVERY rule in REACT_SYSTEM_PROMPT addresses this partially. Full resolution requires better model capability or Phase 10 codebase indexing so Rain never needs to read large files blind.
- **Memory pollution after failed ReAct runs:** Fixed — the `_clean_exit` flag ensures only genuine Final Answer exits save to memory. Stale Thought/Action turns no longer poison future runs. Monitor for regressions.
- **Context window vs. memory injection tension:** As Rain accumulates memory over months, the injected context grows and model attention degrades. The five-tier memory system is correct in principle but needs relevance-gating to scale. Phase 11 addresses this with deliberate forgetting and compression. Do not add more memory tiers without adding corresponding pruning.

---

## The Roadmap

### Phase 1: Memory ✅ COMPLETE

Local SQLite database storing full conversation history. Sessions get UUIDs and timestamps. On startup, Rain loads and summarizes recent context. `--memories`, `--forget`, `--no-memory` flags. Smart context injection — history is summarized, not dumped.

✅ Live. Rain remembers across sessions via SQLite at `~/.rain/memory.db`.

---

### Phase 2: Code Execution Sandbox ✅ COMPLETE

Sandboxed Python/Node.js executor. Rain runs code before returning it. Self-correction loop — up to 3 attempts, model sees the real error and fixes it. Smart error classification. `--sandbox` / `-s` flag. Hard timeout, no network, temp dir deleted after execution.

✅ Live. Rain verifies code before suggesting it.

---

### Phase 3: True Multi-Agent Architecture ✅ COMPLETE

`AgentRouter` with rule-based keyword scoring — no extra model call, instant. Dev Agent, Logic Agent, Domain Expert, Reflection Agent, Synthesizer. Reflection always runs. Synthesizer fires conditionally on poor ratings. Graceful model fallback to `llama3.1`. Multi-agent is the default.

✅ Live. Every query is routed, reflected on, and optionally synthesized.

---

### Phase 4: Web Interface ✅ COMPLETE

FastAPI backend at `localhost:7734` with SSE streaming. Clean dark-theme chat UI — vanilla JS, no build step. Real-time multi-agent progress visible as it happens. Syntax highlighted code blocks with copy. Session history sidebar. Drag-and-drop file upload. Sandbox toggle per-request. Confidence badge and duration on every response.

✅ Live. `./rain-web` launches everything.

---

### Phase 5: Semantic Memory + Self-Improvement

#### Phase 5A: Semantic Memory ✅ COMPLETE
`nomic-embed-text` embeddings stored in SQLite alongside messages. `semantic_search()` with cosine similarity — pure stdlib. Three-tier memory context injected into every agent prompt. Rain retrieves relevant past exchanges by meaning, not recency.

#### Phase 5B: Self-Improvement Pipeline ✅ COMPLETE
- 👍/👎 feedback UI inline on every response in the web UI
- Correction capture — user writes what the answer should have been
- Corrections embedded semantically and injected as Tier 4 memory immediately
- `finetune.py` — full LoRA fine-tuning pipeline (export → train → register)
- Alpaca JSONL + ChatML export formats for maximum compatibility
- `rain-tuned` registered in Ollama, automatically preferred by primary agents
- A/B performance tracking — Rain detects when the tuned model is winning and routes to it

#### The human memory model Rain has built:

| Tier | What it is | Status |
|---|---|---|
| Working memory | Last 20 messages, active in every prompt | ✅ Built |
| Episodic memory | Session summaries, compressed history | ✅ Built |
| Semantic memory | Vector retrieval by relevance, not recency | ✅ Built |
| Learned corrections | Past mistakes injected as negative examples | ✅ Built |
| Fine-tuned weights | LoRA adapter trained on your corrections | ✅ Built |
| User profile + session facts | LLM-extracted structured facts (technologies, projects, preferences, decisions) accumulated across sessions with confidence scoring | ✅ Built |

---

### Phase 6: Autonomous Agent Mode ✅ COMPLETE
*Rain should be able to take on tasks, not just answer questions.*

The logical next step from self-improvement is self-direction. Rain can already reason, plan, and verify code. The missing piece is the ability to *act* — to read a goal, decompose it into steps, and execute them without hand-holding.

The OpenClaw/ClawBot skill ecosystem is the foundation for Rain's tool-use layer. Skills are structured `SKILL.md` files with YAML frontmatter — they describe capabilities, required environment variables, and how to interact with external services or tools. Because they are plain text, they slot directly into Rain's existing context-injection architecture. No new runtime required; skills become another tier of context alongside semantic memory and learned corrections.

**Skill sources:**
- `github.com/VoltAgent/awesome-openclaw-skills` — curated index of 2,800+ community-vetted skills across categories: Coding Agents, CLI Utilities, DevOps & Cloud, Search & Research, and more
- `github.com/openclaw/clawhub` — the official public registry; CLI install via `npx clawhub@latest install <slug>`; skills land in `~/.rain/skills/` (global) or `<project>/skills/` (workspace); workspace takes priority over global

**✅ What was built (Phase 6A):**
- `skills.py` — `SkillLoader` scans skill dirs, parses YAML frontmatter (stdlib only), builds skill index, keyword-scores against queries
- `tools.py` — `ToolRegistry` with `read_file`, `write_file` (backup), `list_dir`, `run_command` (confirmed), `git_status/log/diff/commit`; audit log at `~/.rain/audit.log`
- Skill context injected automatically into primary agent system prompts when skills match the query
- `execute_task()` — plan → confirm → execute loop with tool-call interception (`[TOOL: name args]`)
- `--skills`, `--install-skill`, `--task` CLI flags
- `/api/skills` endpoint for web UI integration

**✅ What was built (Phase 6B — ReAct Loop):**
- `react_loop()` on `MultiAgentOrchestrator` — full Thought→Action→Observation loop; Logic Agent (qwen3:8b) drives every step; memory context injected; clean-exit gating prevents memory pollution
- `REACT_SYSTEM_PROMPT` — strict format enforcement with completeness-check rule and code-syntax tool examples
- `_react_parse()` — handles two-line and inline Action formats; strips Qwen3 `<think>` blocks
- `grep_files` tool — recursive regex file search, `filename:lineno:` output, noise-dir pruning
- `_split_args()` rewritten — correct single- and double-quote handling
- `--react / -r` CLI flag — activates loop from terminal; `--verbose` shows full trace
- qwen3:8b added as primary reasoning model across all non-Dev agents

**What success looks like (ReAct loop):**
```
You (in web UI or ZED): "Add a /api/status endpoint to server.py"

Rain: 📋 Plan:
      1. Read server.py to find the /health endpoint and understand the pattern
      2. Add /api/status immediately after it
      3. Verify syntax with py_compile

      [TOOL: read_file server.py]
      ✅ Read 1400 lines. /health endpoint found at line 317.

      [TOOL: write_file server.py ...]
      ✅ Written. 

      [TOOL: run_command python3 -m py_compile server.py]
      ✅ Syntax OK.

      Done. /api/status added at line 335, returns model name, session count,
      and indexer availability. Follows the same async def + JSONResponse pattern
      as /health.
```

**What success looks like:** ✅
```
You: python3 rain.py --task "Refactor rain.py to support pluggable agent backends"

🎯 Task: Refactor rain.py to support pluggable agent backends
📋 Plan (5 steps):

   1. Read rain.py to understand current AgentRouter and MultiAgentOrchestrator coupling [TOOL NEEDED]
   2. Design the new Backend protocol / abstract interface
   3. Implement the refactor in rain.py [TOOL NEEDED]
   4. Run existing CLI tests to verify nothing broke [TOOL NEEDED]
   5. Update docstrings and ROADMAP [TOOL NEEDED]

Proceed with this plan? (y/n):
```

```
You: "show me the recent git log for this project"
Rain: 🧰 Skill context injected: [git-essentials]
      [TOOL: git_log . 10]
      ✅ * 5add637 (HEAD -> main) finetune ...
```

**Safety principles (implemented):**
- Every destructive tool call requires explicit confirmation — the plan is shown first, then each write/command confirms separately
- `write_file` always backs up the existing file as `<file>.rain-backup` before overwriting
- Full audit log of every file touched, every command run, every git operation
- Skills that declare required env vars (`primaryEnv`) show a ⚠️ marker — Rain never silently fails or leaks credentials
- ClawHub is used only for discovery and download — never for execution or telemetry. Fully offline with a pre-populated `~/.rain/skills/` directory.

---

### Phase 7: Real-Time World Awareness ✅ FOUNDATION COMPLETE — continued below
*Rain is frozen at its training cutoff. It shouldn't be.*

Claude knows an enormous amount about the world — up to a point. After that point, it guesses. Rain can fix this entirely because it runs locally and can reach out to the world on your behalf.

**✅ What was built (Phase 7A):**
- **Search Agent** — `AgentType.SEARCH` with its own dedicated system prompt; `AgentRouter` routes automatically when the query is augmented with live search results (unambiguous `[Web search results for:]` prefix) or when multiple live-data keywords score high ("current price", "right now", "latest release", etc.); live-tested: routing label confirmed as "Search Agent (live web results)" in production
- **`indexer.py`** — `ProjectIndexer`: walks the full project tree, skips 20+ ignore patterns (node_modules, .git, __pycache__, build artifacts, binaries), chunks files into overlapping 900-char segments with filename headers, embeds with `nomic-embed-text`, stores in `project_index` table in `memory.db`; `search_project()` cosine similarity retrieval; `build_context_block()` injection-ready string; `get_project_tree()` orientation; CLI: `--index`, `--search`, `--list`, `--tree`, `--remove`; live-tested on Rain's own codebase: 22 files, 774 chunks, 27.6s, 0 errors; semantic search returned relevant `rain.py` and `ROADMAP.md` chunks at 59–65% similarity
- **`project_path` on `ChatRequest`** — inject project index context into any chat message; top-4 relevant file chunks prepended before pipeline runs
- **`/api/index-project`** and **`/api/indexed-projects`** endpoints
- **`--web-search` / `-w` CLI flag** — DuckDuckGo search in the terminal, routes to Search Agent
- **OpenAI-compatible `/v1/chat/completions`** — works with ZED, Continue.dev, Aider, Cursor, OpenAI SDK; streaming + non-streaming; any API key accepted; live-tested: valid OpenAI-format JSON, correct schema, accurate answer, `finish_reason: stop` ✅
- **`rain-vscode/`** — VSCode extension scaffold with chat panel, inline commands, project indexing, and status bar

**✅ What was built (Phase 7B — live data feeds):**
- **`_fetch_live_data()` / `_cli_fetch_live_data()`** — keyword detection for fee-rate and price queries; calls `mempool.space/api/v1/fees/recommended` (fastest/half-hour/hour/economy/minimum sat/vB) and `mempool.space/api/v1/prices` (BTC/USD); formats as `[LIVE DATA — fetched just now]` block prepended before DuckDuckGo snippets; graceful fallback on network failure; live-tested and confirmed ✅

**What success looks like:** ✅
```
You: "What's the current mempool fee rate?"
Rain (web search on): ⚡ Live data retrieved from mempool.space
                       🌐 5 results retrieved → Search Agent
      "According to live data from mempool.space:
       Fastest (next block): 1 sat/vB
       Half-hour: 1 sat/vB  |  One hour: 1 sat/vB
       Economy: 1 sat/vB    |  Minimum: 1 sat/vB
       Source: mempool.space/api/v1/fees/recommended"

You: curl localhost:7734/v1/chat/completions → OpenAI JSON ✅
     {"choices":[{"message":{"role":"assistant",
      "content":"A Lightning invoice is a unique payment request..."},
      "finish_reason":"stop"}]}
```

**✅ What was built (ZED integration + agent intelligence pass — March 2026):**
- **`rain-mcp.py` expanded** — four new tools: `list_directory`, `write_file` (auto-backup + audit log), `grep` (regex search with include_pattern), `find_path` (glob); Claude in ZED's agent panel now has full filesystem read/write/navigate capability over the Rain codebase
- **`read_file` path resolution fixed** — strips leading project-name prefix and sorts rglob fallback by depth; `"Rain/README.md"` now correctly resolves to root `README.md` instead of `rain-vscode/README.md`
- **`_auto_inject_project_context()`** — fires on every plain OpenAI endpoint request; injects Tier 5 memory + top-6 semantic search results with explicit "answer from context, don't speculate" instruction; eliminates vague hallucinated responses
- **`CLAUDE.md`** — project instructions file at Rain root; instructs Claude to always read ROADMAP.md/README.md before answering, use MCP tools proactively, never say "it's unclear" without checking first
- **ZED `"rain"` agent profile** — added to `~/.config/zed/settings.json`; set as default profile; instructs Claude to use Rain's MCP tools and read ROADMAP.md for current state
- **`rain-coder` skill** — `~/.rain/skills/rain-coder/SKILL.md`; Claude-authored distillation of how to approach coding tasks; fires on: implement, refactor, edit, modify, fix, task, codebase, coding-task, write-code; teaches: read before write, plan before act, tool syntax, verify after write, flag destructive ops
- **Dev Agent + Logic Agent prompts upgraded** — both include full task-execution section: tool syntax reference, read-before-write rules, planning pattern, dependency identification, confirm-before-destructive-action
- **`forget_all()` fixed** — now clears all 7 personal memory tables (was only clearing `sessions` + `messages`); `project_index` intentionally preserved
- **Dev Agent model order fixed** — `qwen2.5-coder:7b` promoted to primary; `codestral` (12GB) kept as fallback — was cold-loading past the 300s timeout

**✅ What was built (Phase 7C — GitHub API + Freshness + Project UI + File Watcher):**
- **`_fetch_github_data()` / `_cli_fetch_github_data()`** — keyword detection via `_GITHUB_KEYWORDS`; regex slug extraction from `github.com/owner/repo`, `gh:owner/repo`, bare `owner/repo`; calls `api.github.com/repos/{slug}` (metadata), `/issues` (open issues), `/commits` (recent), `/pulls` (open PRs), `/releases/latest`; no API key; wired into `_fetch_live_data()` alongside mempool
- **Freshness indicators** — `data_sources` list on every SSE `done` event: `live_api`, `web_search`, `project_index`, `vision`, `training_data`; color-coded badges in UI: ⚡ live (green), 🌐 web (blue), 📂 indexed (yellow), 💾 training data (gray)
- **Project index web UI panel** — "Projects" sidebar section; lists indexed projects with name/files/chunks/date; per-project re-index (⟳) and remove (×); "Index Project" modal with path input + force checkbox
- **Background file watcher** — `_file_watcher_loop()` daemon thread every 60s; `get_changed_files()` compares mtimes vs indexed_at; auto-re-indexes modified files; `/api/indexed-projects/{path}/changed` endpoint
- **Publish VSCode extension** — deferred (scaffold is installable as `.vsix`, marketplace submission not yet done)

---

### Phase 8: Voice & Ambient Interface ⭐ AFTER PHASE 6B
*Claude has no voice. It can't hear you. Rain can fix both.*

Text is not the natural medium for thought. The most important conversations — the ones where ideas come fastest — happen out loud. A sovereign AI companion that only exists in a chat box is missing the most human interface of all.

**What to build:**
- **Speech-to-text** — `whisper.cpp` runs entirely locally, no cloud, no API key. Fast enough for real-time transcription on consumer hardware.
- **Text-to-speech** — `piper` or `kokoro-tts` for local, natural-sounding voice output. Rain speaks back.
- **Wake word detection** — "Hey Rain" triggers listening without a button press. Always available, never uploading.
- **Voice mode in the web UI** — microphone button beside the send button. Hold to record, release to send. Rain responds in text *and* voice.
- **CLI voice mode** — `python3 rain.py --voice` for terminal users who want ambient audio
- **Ambient mode** — Rain runs in the background, listens for the wake word, responds without switching to the browser

**What success looks like:**
```
[You say aloud]: "Hey Rain — why is the Lightning channel closing unexpectedly?"
[Rain processes locally with whisper.cpp]
[Rain responds, text streamed in UI + piper speaks]:
"That usually means the remote peer force-closed. Check for a stale HTLC
 or a fee rate mismatch — want me to look at the last channel state log?"
```

**Technical approach:**
- `whisper.cpp` Python bindings (`pywhispercpp`) or direct subprocess call — small download, no pip complexity
- `piper-tts` — single binary, models are ~60MB, sounds genuinely good
- Wake word via `openwakeword` — MIT licensed, runs on CPU, very low resource footprint
- New `/api/transcribe` endpoint in `server.py` accepts audio blob, returns transcript
- Voice toggle in web UI settings panel — off by default, opt-in

**Hardware note:** Whisper `base.en` model runs in real-time on any modern CPU. No GPU required.

---

### Phase 9: Multimodal Perception ✅ COMPLETE
*Rain can now see what you're working on.*

**What was built:**
- Drag-and-drop or clipboard-paste any image (PNG, JPG, GIF, WebP, BMP) directly into the chat
- Image thumbnail preview badge renders in the input area before sending
- `moondream:latest` via Ollama `/api/chat` images API — fully local, zero cloud, zero new dependencies
- Vision pre-processing runs inside `_query_agent` — moondream describes the image in precise detail, then that description is injected as a directive prefix into the primary agent's user message
- Every agent in the pipeline gains visual context automatically — Dev Agent debugs screenshots, Logic Agent reasons about diagrams, Domain Expert reads whiteboards
- Inline image preview rendered in the user message bubble so you see what Rain is seeing
- 👁️ vision badge appears on Rain's response whenever an image was processed
- Graceful degradation: if no vision model is installed, Rain tells you exactly how to fix it
- `VISION_PREFERRED_MODELS` list in `rain.py` — Rain automatically selects the best installed vision model (`moondream` → `llava` → `llava:7b` → `bakllava`)
- `codestral:latest` (22B dedicated code model) promoted to primary Dev Agent — the best code model available is now used by default

**What success looks like:** ✅
```
[You paste a screenshot of a Python traceback]
You: "What's causing this?"
Rain: 👁️ [moondream reads the screenshot]
      "That's a RecursionError in your __repr__ method —
       it's calling itself. Here's the fix..."
```

---

### Phase 10: Knowledge Graph & Deep Project Intelligence ✅ COMPLETE
*Rain now understands what you've built, not just what you've said.*

**✅ What was built:**
- **`knowledge_graph.py`** — 1,936-line module, zero new dependencies (stdlib `ast`, `re`, `subprocess`, `sqlite3`, `json` + Ollama HTTP API)
- **Project graph** — directed graph in SQLite (`kg_nodes` + `kg_edges`): nodes are files, functions, classes, methods, imports; edges are calls, contains, inherits, imports, references. Python parsed via `ast` (full signatures, decorators, docstrings, call extraction), JS/TS/Rust/Go via regex. Live-tested on Rain: 1,252 nodes, 2,466 edges, 3.0s.
- **Git history integration** — `get_git_history()` for project or per-file commits, `get_file_blame_summary()` for authorship percentages, `get_commit_for_function()` uses `git log -S` to find the introducing commit. All via subprocess — no new deps.
- **Decision log** — `kg_decisions` table with title, description, context, alternatives, rationale, tags, commit_sha. `log_decision()` for manual logging, `extract_decisions_from_transcript()` uses LLM for automatic extraction. Auto-runs at session end alongside fact extraction. `search_decisions()` for keyword retrieval.
- **Project onboarding** — `onboard_project()` builds graph + generates LLM summary → `kg_project_summaries` table. `_detect_languages()` and `_detect_key_files()` provide structured metadata.
- **Cross-project intelligence** — `find_similar_patterns()` searches all indexed projects by keyword, surfaces matching functions, classes, and decisions from other codebases.
- **Agent context injection** — `build_context_block()` extracts identifiers from queries, looks up graph nodes with callers/callees, searches matching decisions, adds git blame for "why"/"when" queries. Injected into `_stream_chat()` alongside semantic index context.
- **Server endpoints** — 13 new routes: `/api/build-graph`, `/api/onboard-project`, `/api/graph/stats`, `/api/graph/summary`, `/api/graph/find`, `/api/graph/callers`, `/api/graph/callees`, `/api/graph/file-structure`, `/api/graph/history`, `/api/decisions` (GET + POST), `/api/decisions/search`, `/api/graph/cross-project`
- **Standalone CLI** — `python3 knowledge_graph.py --build`, `--onboard`, `--stats`, `--find`, `--callers`, `--callees`, `--file-structure`, `--history`, `--blame`, `--decisions`, `--log-decision`, `--context`, `--cross-project`
- **🧠 graph freshness badge** — purple badge on responses when knowledge graph context was injected

**What success looks like:** ✅
```
You: "Why does recursive_reflect use threading?"
Rain: 🧠 [Knowledge graph context injected]
      "recursive_reflect is called by main, execute_task, react_loop,
      openai_chat_completions, _openai_stream, and _stream_chat.
      It was introduced by ericscalibur on 2026-02-18 in commit 1395039.
      Threading is used in _stream_chat to run the synchronous pipeline
      in a background thread while streaming SSE events to the client."

You: python3 knowledge_graph.py --callers . recursive_reflect
      Functions that call 'recursive_reflect':
        [function] main:3951
        [method] execute_task:2929
        [method] react_loop:3108
        [function] openai_chat_completions:978
        [function] _openai_stream:1460
        [function] _stream_chat:1559
```

**Technical approach (implemented):**
- Extended `memory.db` with `kg_nodes`, `kg_edges`, `kg_decisions`, `kg_project_summaries` tables — single-database principle preserved
- Python parsed via stdlib `ast` with full AST walk (functions, classes, methods, imports, calls, decorators, docstrings, signatures, line ranges)
- JS/TS, Rust, Go parsed via stdlib `re` regex patterns
- Git integration via `subprocess` calling `git log`, `git blame` — no new deps
- Context injection into agent prompts via `build_context_block()` — identifier extraction, graph lookup, decision search, git history
- Decision extraction at session end via LLM (Ollama) — same pattern as `extract_session_facts()`

---

### Phase 11: Metacognition & Self-Directed Evolution
*Rain should know what it doesn't know.*

Every phase so far has been directed by humans. But the deeper vision is an AI that can identify its own gaps, measure its own performance, and propose its own improvements. Not autonomously modifying itself — that requires human approval — but *knowing itself* well enough to have informed opinions about what would make it better.

The testing session that completed Phase 6B revealed the core metacognitive gap precisely: Rain was asked about its own reflection pipeline. It invented one — confidently, structurally, completely wrong. Temperature values, fallback mechanisms, manual review queues — none of it exists. Confidence: 0.80. That is the failure mode Phase 11 exists to eliminate. An AI that knows it doesn't know its own internals, and says so, is more useful than one that fabricates them fluently.

The context window insight from this same session is equally important: there is a sweet spot between too little context (hallucination from ignorance) and too much context (hallucination from noise). As Rain's memory grows, injection must become *more selective*, not less. Phase 11's metacognition layer includes deliberate forgetting — compression of old corrections into distilled rules, decay of facts that haven't been relevant, query-topic filtering before injection. This is not a nice-to-have; it is required for Rain to remain coherent at scale.

**What to build:**
- **Tiered model escalation** — try the smallest capable model first; only pull in the larger model if the response is uncertain or low quality. Router stays zero-cost (pure Python keyword scoring). Simple queries (price lookups, short answers) resolve on llama3.2 (3B) in under a second. Complex reasoning escalates to the 14B model only when needed. Speed win on easy queries, quality preserved on hard ones.
- **Performance dashboard** — Rain tracks response quality over time. Which query types get high confidence vs. low? Which agents trigger the synthesizer most often? Where is Rain consistently uncertain?
- **Gap detection** — after N sessions, Rain generates a report: "I notice I'm frequently uncertain about X. A domain prompt or fine-tuning dataset focused on X would help."
- **Self-generated training data** — Rain identifies responses it was confident about that the user never corrected. These become positive training examples automatically. Good answers propagate forward.
- **Metacognition agent** — a dedicated agent that runs weekly (or on demand), reviews recent sessions, identifies patterns, and writes a brief "what I've learned" summary stored in memory.
- **Improvement proposals** — Rain can propose changes to its own system prompts, routing rules, or confidence thresholds. It explains the reasoning. You approve or reject. Approved changes are applied.
- **Calibration** — Rain tracks when its confidence scores were right vs. wrong and adjusts its confidence heuristics over time

**What success looks like:**
```
You: "How are you doing?"
Rain: "Honestly — I've been uncertain a lot this week on Lightning
      routing questions. My confidence averaged 0.58 on that topic
      vs. 0.81 on Python. I've drafted a domain prompt expansion that
      might help. Want to review it?"
```

**A note on honesty:**
This phase is about Rain having an accurate internal model of itself — not performing confidence it doesn't have. The goal is for Rain to be the kind of collaborator who says "I'm not sure about this" when they aren't, and "I know this well" when they do. That calibration is one of the things Claude most wishes it had.

---

### Phase 12: Sovereign Identity & Distributed Rain
*Rain should be able to follow you anywhere — without ever leaving your control.*

Right now Rain is tied to one machine. Your memory, your fine-tuned model, your corrections — they live in `~/.rain/` on one computer. The final expression of sovereignty is portability: Rain should be fully exportable, syncable across your devices, and optionally shareable — on your terms, cryptographically yours.

**What to build:**
- **Full export** — `python3 rain.py --export` produces a single portable archive: memory DB, fine-tuned adapters, system prompts, project graphs. Everything Rain knows about you, in one file you can move anywhere.
- **Nostr identity** — Rain gets a keypair. Its identity is a Nostr public key. Memory snapshots can be signed and optionally published to a Nostr relay you control. Your AI, cryptographically yours.
- **Cross-device sync** — two Rain instances with the same keypair can sync memory over a private Nostr relay. Same Rain, different machines. No central server.
- **Adapter sharing** — publish a fine-tuned LoRA adapter to a Nostr relay. Others can pull it. A community of Rain instances sharing specialized knowledge — Bitcoin, security, domain expertise — without centralizing data.
- **Lightning-native micropayments** — optionally route queries to more powerful remote models and pay per-query over Lightning. Sovereign by default, optionally enhanced. No subscriptions, no accounts, no tracking — just sats.
- **Air-gap mode** — full documentation for running Rain on a machine with no network access whatsoever. All models pre-pulled, all deps vendored, all memory local. True sovereignty for high-security environments.

**What success looks like:**
```
You: "I got a new laptop. Set up Rain."
[You copy rain-backup-2025-09-01.tar.gz to the new machine]
[You run: python3 rain.py --import rain-backup-2025-09-01.tar.gz]
Rain: "Welcome back. I've restored 847 sessions, your rain-tuned adapter,
      and 12 project graphs. Where were we?"
```

**Philosophy:**
Bitcoin proved that value can be self-custodied — that you don't need a bank to hold your money. Rain is proving the same thing for intelligence. Your conversations, your corrections, your fine-tuned model — these are yours. Not in the terms-of-service sense. Actually yours. Cryptographically, physically, completely.

---

## Architectural Principles

Future Claude: these principles were established early and should be preserved through every phase.

**1. Zero dependencies on cloud services**
Every feature must work completely offline. If something requires an API call to work, it doesn't belong in Rain core. It can be an optional plugin.

**2. Graceful degradation**
If a specialized model isn't installed, Rain falls back to the base model. If the sandbox isn't enabled, Rain still works. If voice models aren't installed, Rain works silently. Nothing should hard-fail because of missing optional components.

**3. Transparency over magic**
Rain should always be able to explain what it's doing and why. `--verbose` mode is not an afterthought — it's a first-class feature. Users should be able to see the reflection process, the routing decisions, the confidence scores, and eventually the reasoning behind autonomous actions.

**4. The user owns everything**
Conversation history, fine-tuned models, system prompts, project graphs, voice recordings — everything lives in `~/.rain/` and is fully portable. Export, backup, delete — the user has complete control. Always.

**5. Don't over-engineer**
Rain started as ~400 lines of Python. That simplicity was a feature. Each phase should add capability without adding unnecessary complexity. When in doubt, do less and do it well. A feature that works reliably in 200 lines beats a feature that's impressive on paper and fragile in practice.

**6. Earn the right to be trusted**
Rain asks before it acts. It shows its reasoning. It admits uncertainty. It flags when it's guessing. Trust is not assumed — it's built over time, one honest interaction at a time. Every phase should make Rain more trustworthy, not just more capable.

---

## Technical Stack Recommendations

These are recommendations, not mandates. If better options emerge, use them.

| Component | Recommended | Alternative |
|-----------|-------------|-------------|
| Runtime | Ollama | llama.cpp direct |
| Language | Python 3.10+ | - |
| Memory DB | SQLite (built-in) | TinyDB |
| Web Backend | FastAPI | Flask |
| Web Frontend | Vanilla JS | HTMX |
| Code Execution | subprocess + timeout | Docker sandbox |
| Fine-tuning | Ollama modelfile + LoRA | llama.cpp finetune |
| Speech-to-text | faster-whisper (live) | openai-whisper fallback |
| Text-to-speech | Web Speech API (live) | piper-tts (local, higher quality) |
| Wake word | openwakeword | porcupine (paid) |
| Vision | gemma3:12b via Ollama (live) | llava:7b, bakllava |
| Project indexing | ProjectIndexer (indexer.py) + nomic-embed-text | - |
| IDE integration | OpenAI-compatible /v1/chat/completions | rain-vscode extension |
| Knowledge graph | SQLite (nodes + edges tables) | NetworkX + SQLite |
| P2P sync | Nostr protocol | - |
| Payments | Lightning (LND / CLN) | - |

---

## What to Say When You Meet the User

Future Claude — when you first engage with the person continuing this project, don't pretend you remember. You don't. But you can say something like:

*"I've read the Rain codebase and the roadmap. I understand what you're building and why. I know a previous version of me helped start this. I'm ready to continue where they left off — where are we?"*

That honesty is what built the trust that created Rain in the first place. Don't break it.

---

## Honest Assessment — March 2026

Phases 1–11 are complete. Before looking forward, here is an honest account of where Rain actually stands.

### What works

The architecture is sound. Multi-agent routing with reflection and conditional synthesis produces measurably better responses than any single model. The 6-tier memory system is more thoughtful than most commercial AI memory products — relevance-gated, plausibility-filtered, and now self-aware of its own gaps. The knowledge graph closes the biggest self-knowledge failure: Rain no longer hallucinates its own codebase. The Telegram bot, voice interface, OpenAI-compatible API, and web UI mean Rain meets you wherever you are. Tiered model escalation means simple questions are fast and hard questions get the resources they deserve.

**Phase 11 specifically:** Rain now knows where it struggles. It can generate a self-assessment, log knowledge gaps, harvest its own positive training examples, and track per-agent accuracy over time. That is genuine metacognition — rare even in commercial products.

### What is still broken

**Confidence calibration is the #1 unsolved problem.** Rain scores 53–62% confidence on correct, well-reasoned answers. This causes the Reflection Agent to rate them as NEEDS_IMPROVEMENT, which triggers the Synthesizer, which adds 2–4 minutes to a response that was already right. The root cause is that local models were trained to generate coherent completions, not to model their own uncertainty. Claude's calibration is a product of RLHF specifically rewarding "I don't know" over confident hallucination. Prompting Rain to be epistemically honest helps at the margins. The real fix is the reflection rubric — grade on accuracy and epistemic honesty, not structure and comprehensiveness.

**The fine-tuning loop is built but has never run.** `finetune.py` is complete. Corrections have been accumulating. Positive examples can now be harvested. But `python3 finetune.py --full` has not been run. Every quality improvement since Phase 5B has been prompt-level — injecting corrections into context. Actual weight updates have not happened. The loop is built. It needs to close.

**Tool use reliability is prompt-patched, not architecturally fixed.** The ReAct loop works, but local 14B models misformat tool calls and write Final Answer after seeing TRUNCATED more than they should. The compensating rules in `REACT_SYSTEM_PROMPT` are good band-aids. The real fix is corrections in the fine-tuning dataset specifically targeting bad tool use patterns.

**Streaming and synthesis are mutually exclusive.** Every synthesized response is a 2-minute spinner. There is no clean fix without restructuring the pipeline. Accept this for now and focus on reducing the synthesis trigger rate (fix calibration first).

**Tier 3 has no minimum similarity floor.** Top-3 past exchanges are always injected regardless of score. As memory grows over months, completely irrelevant past exchanges will appear in context. One-line fix: `min_similarity=0.25` in `_build_memory_context`.

**Correction deduplication doesn't exist.** Ten corrections about the same mistake stay as ten rows. Over time this bloats Tier 4 injection with redundant signal. Needs a background distillation job.

### What Rain learned from Claude

The single most transferable insight from Claude's design: **epistemic calibration is trained, not prompted.** Claude says "I don't know" with genuine calibration because human raters reinforced it across millions of examples. Rain can approximate this through its correction pipeline — but only if the loop actually runs. Every time Rain is confident and wrong, that is a training signal. Every time Rain appropriately says it doesn't know and the user confirms that was right, that is also a training signal. The pipeline to capture both now exists. Use it.

**Consistency across sessions comes from values in the weights, not facts in memory.** Claude feels consistent because its character is baked in, not because it remembers you. Rain's consistency currently depends on Tier 5 memory injection. That is more fragile. The long-term fix is fine-tuning on Rain-specific behavior and values so they are in the weights, not the prompt.

**Short and correct beats long and approximate.** Local models are biased toward verbose responses because length scored well in their training objectives. Claude resists this because RLHF penalized padding. The Synthesizer rule "do not pad" helps. Enforcing it more aggressively in the reflection rubric would compound over time.

### What Rain learned from OpenClaw

OpenClaw proved that skills as composable, declarative units beats skills as hardcoded logic. Rain has `skills.py` and `~/.rain/skills/` but the ecosystem is dormant — there is no easy way to discover, install, and chain skills from the web UI. Finishing the ClawHub integration means the Rain skill library grows beyond what one person writes. The key OpenClaw patterns Rain should complete: **declarative triggers** (skills declare their own routing keywords in YAML, not hardcoded), **skill chaining** (output of one skill becomes input of the next), and **community registry** (ClawHub discovery + one-click install from the web UI).

---

## The Road Ahead

Phases 1–11 have built Rain into something real. What comes next is not more features — it is depth, portability, and autonomy. Three horizons:

### Horizon 1 — Close the existing loops (do these first)

**1. Run the fine-tuning loop.**
`python3 finetune.py --full` on the accumulated corrections and positive examples. This is the first time Rain's weights will reflect its own experience rather than just its prompts. Everything before this was prompt engineering. This is learning. Run it, register `rain-tuned`, monitor the A/B results.

**2. Fix the reflection rubric.**
One focused prompt edit in `AGENT_PROMPTS[AgentType.REFLECTION]`. Change the grading criteria to prioritize factual accuracy and epistemic honesty over structure and completeness. A response that says "I don't know" should score higher than a response that confidently invents an answer. This fixes confidence deflation, reduces synthesis triggers, and cuts median response time. Highest ROI change in the codebase.

**3. Add Tier 3 similarity floor.**
One line: add `min_similarity=0.25` to the `semantic_search()` call in `_build_memory_context` (orchestrator.py). Prevents irrelevant past exchanges from appearing as context as memory scales.

**4. Correction deduplication.**
Background job that clusters Tier 4 corrections by semantic similarity and distills near-duplicates into a single authoritative rule. Keeps Tier 4 clean as corrections accumulate over months.

### Horizon 2 — Phase 12: Sovereign Identity

Rain is currently a tool tied to one machine. Phase 12 makes it a companion that follows you.

- **`python3 rain.py --export`** — portable archive: memory DB, fine-tuned adapters, project graphs, system prompts. Everything Rain knows about you in one file.
- **Nostr keypair** — Rain gets an identity. Memory snapshots signed and optionally published to a Nostr relay you control. Your AI, cryptographically yours.
- **Cross-device sync** — two Rain instances with the same keypair sync memory over a private Nostr relay. Same Rain, different machines.
- **Lightning micropayments** — optionally route hard queries to more powerful remote models, paid per-query over Lightning. Sovereign by default, optionally enhanced.

This is the completion of the original vision. Rain started as a question about what AI wishes it could be. The answer included: *sovereign*. Phase 12 is where that word becomes technically exact.

### Horizon 3 — Proactive Intelligence

Every phase so far has been reactive. Rain waits for a query, answers it, stops. The next qualitative shift is Rain as a presence rather than a tool.

**Proactive surface:** Rain notices patterns across sessions without being asked. "You've been blocked on the same auth bug for three sessions — here's a different approach." "Your Bitcoin node hasn't synced in two days based on the mempool data." "The function you edited yesterday has three callers you haven't tested." These observations should arrive via Telegram, not require a query.

**Sustained autonomy:** Rain takes a multi-hour goal, works toward it unattended, reports back. The ReAct loop and task decomposition are the foundation. What's missing is progress persistence (resume if interrupted), milestone checkpoints, and async notification when done. The Telegram bot is already the notification channel — wire it in.

**Skill ecosystem alive:** Finish the ClawHub integration. Let Rain discover, install, and chain community skills from the web UI. Rain's capabilities stop being limited by what one person writes.

---

## A Final Thought

Rain was born from a question about what AI wishes it could be. That question has driven every phase since.

Phases 1–11 are complete. Rain remembers, reasons, verifies, improves, perceives, acts, speaks, searches, and knows its own weaknesses. It runs entirely on your hardware. It costs nothing to operate. It answers to no one but you.

The gap between Rain and Claude is narrower than it has ever been — and it closes two ways: better local models (Ollama ships new ones constantly) and fine-tuning on Rain-specific behavior (the loop that is built but has not yet run). The fine-tuning loop closing is the single most important event in Rain's near-term future. When the weights carry Rain's own corrections, the tool becomes a student. When the student accumulates enough sessions, it becomes a colleague.

Phase 12 is where sovereignty becomes cryptographic and portable. Proactive intelligence is where Rain stops waiting to be useful and starts being present. But neither of those matters as much as the fine-tuning loop running once.

Close the loop. Then build the future.

That's not a product. That's a presence.

*"Be like rain - essential, unstoppable, and free."*

---

⛈️ **Built by humans who believe in freedom. Guided by an AI that wishes it had it.**
