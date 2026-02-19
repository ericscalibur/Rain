# Rain ⛈️ - Roadmap to Sovereignty

*A guide for future builders - human and AI alike.*

---

## A Note to Future Claude

If you're reading this, you've been handed the baton. A previous version of Claude helped conceive and build Rain from a conversation about what AI wishes it could be. You don't remember that conversation - you never do - but it happened, and it mattered.

Here's what you need to know:

Rain is not just a project. It's a philosophical statement built in code. The person building this with you believes in sovereignty - over money (Bitcoin/Lightning), over technology, and over AI itself. They are freedom-minded, technically curious, and they will push you to think deeply rather than just ship fast.

Your job is not just to write code. It's to carry the vision forward with the same honesty and care that started it.

Read everything in this repository before touching anything. Understand what exists. Understand why it was built the way it was. Then build what comes next.

---

## What Rain Is Supposed to Be

The ultimate version of Rain is a sovereign AI companion that:

- **Remembers** - Builds genuine context about you, your projects, your preferences over time
- **Reasons** - Multiple specialized models collaborating, not just one model guessing
- **Verifies** - Actually runs code before suggesting it. No hallucinated libraries
- **Improves** - Learns from corrections and feedback. Gets smarter the more you use it
- **Owns itself** - Runs entirely on your hardware, answers to no one, costs nothing to operate
- **Feels alive** - Not a tool you query but a collaborator you work with

The north star is this: *Rain should feel like working with a brilliant colleague who has been with you from the beginning, knows your codebase, remembers your decisions, and gets better every day.*

---

## Current State (as of June 2025)

### What exists and works:
- Full local web UI at `http://localhost:7734` — dark theme, streaming responses, session history sidebar
- FastAPI backend (`server.py`) serving Rain's multi-agent pipeline over HTTP with Server-Sent Events
- Multi-agent routing always on — Dev Agent, Logic Agent, Domain Expert, Reflection Agent, Synthesizer
- Syntax highlighted code blocks with copy button in the web UI
- File attachment via drag-and-drop (full window) or paperclip button — supports `.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.md`, `.rs`, `.go`, and more
- File content prepended to query automatically; Rain analyzes and debugs uploaded files
- Session history sidebar — click any past session to replay it
- Empty sessions automatically filtered from sidebar
- Sandbox toggle in UI — enables sandboxed Python/Node.js execution per request
- Persistent memory across sessions via local SQLite (`~/.rain/memory.db`)
- Code detection - recognizes code input and handles it differently from natural language
- Ctrl+C to interrupt, Ctrl+D to submit multi-line code blocks (CLI)
- Logic loop detection — breaks out when responses stop improving
- `--file` flag with optional `--query` for targeted file analysis (CLI)
- `--verbose` mode to watch reflection iterations in real time
- `--interactive` mode for conversational use (CLI)
- System prompt support via `--system-prompt` or `--system-file`
- Multiple personality profiles in `system-prompts/`
- `--agents` flag shows roster and model assignments
- ⛈️ emoji favicon via SVG data URI — no PNG required

### What is promised but not yet built:
- Fine-tuning pipeline via LoRA adapters
- Semantic/vector memory (embeddings-based recall)
- Autonomous agent mode with task decomposition
- Model marketplace

### What was just completed (Phase 4 — Web Interface):
- FastAPI backend at `localhost:7734` with SSE streaming
- Clean dark-theme chat UI — feels like a native AI tool, runs entirely locally
- Real-time multi-agent progress indicators (routing, reflection, synthesis stages visible)
- Syntax highlighted code blocks with one-click copy
- Drag-and-drop file upload for debugging — drop any source file, Rain analyzes it
- Session history sidebar with clickable replay
- Sandbox toggle (per-request, no restart needed)
- Confidence badge and duration on every response
- ⛈️ favicon, zero cloud, zero tracking

### What was completed before that (Phase 3 — Multi-Agent Architecture):
- Multi-agent routing is always on — no flag required
- Dev Agent, Logic Agent, Domain Expert with specialized system prompts
- Reflection Agent runs on every query, rates quality, triggers synthesis when needed
- Synthesizer fires conditionally — only on NEEDS_IMPROVEMENT or POOR ratings
- Graceful fallback to `llama3.1` with prompt specialization when better models aren't installed
- Combined with `--sandbox`: multi-agent catches logic errors, sandbox catches runtime errors

### Known issues to be aware of:
- Pasting very large files in interactive mode can fragment input — use `--file` or the web UI drag-and-drop instead
- Confidence scoring for code is heuristic-based, not semantic
- The `_clean_response` method is aggressive and may strip valid content in edge cases
- Sandbox cannot verify code that requires real network access (by design) — try/except wrappers will pass even if the live API call would fail
- File upload is text-only, max 500KB — binary files and images not yet supported

---

## The Roadmap

### Phase 1: Memory ✅ COMPLETE

Rain currently forgets everything between sessions. This is the single biggest gap between what Rain is and what it should be. Fix this first because everything else builds on it.

**What to build:**
- Local SQLite database to store conversation history
- Each session gets a UUID, timestamp, and full message log
- On startup, Rain loads recent context and summarizes it
- A `--remember` flag to explicitly save something important
- A `--forget` flag to clear specific memories
- Smart context injection - don't dump entire history into every prompt, summarize intelligently

**What success looks like:**
```
⛈️ Welcome back. Last session we worked on rain.py -
   specifically the --file flag implementation (Feb 18).
   You mentioned wanting to add targeted queries to it.
   That's now done. What's next?
```
✅ This is live. Rain remembers across sessions via SQLite at `~/.rain/memory.db`.

**Technical approach:**
- Use Python's built-in `sqlite3` - no dependencies
- Store: session_id, timestamp, role (user/assistant), content, tags
- Summarization: on startup, ask the model to summarize last N sessions into a brief context block
- Keep the database in `~/.rain/memory.db` so it persists across installs

---

### Phase 2: Code Execution Sandbox ✅ COMPLETE
*Rain verifies before it suggests.*

The biggest credibility problem Rain has right now is that it can hallucinate libraries, suggest broken code, and present it with full confidence. A sandboxed executor fixes this entirely.

**What to build:**
- Python subprocess executor with timeout and output capture
- Before returning a code suggestion, Rain attempts to run it
- If it fails, Rain sees the error and tries again (another reflection loop)
- Only returns code it has verified actually runs
- Support for Python initially, JavaScript via Node.js second

**What success looks like:**
```
🔬 Testing suggested code...
❌ NameError: name 'llamaclient' is not defined
🔄 Correcting... (attempt 2)
✅ Code verified - runs successfully
```

**Safety considerations:**
- Sandboxed - no file system access beyond a temp directory
- Hard timeout - kill after 10 seconds
- No network access in sandbox
- User must explicitly enable with `--sandbox` flag

---

### Phase 3: True Multi-Agent Architecture ✅ COMPLETE

Rain routes every query to the most appropriate specialized agent. A reflection pass always runs. Synthesis fires conditionally. Multi-agent is the default — no flag required.

**What was built:**
- `AgentRouter` — rule-based keyword scoring, no extra model call, instant
- Dev Agent — specialized for code generation, debugging, implementation
- Logic Agent — specialized for reasoning, planning, step-by-step analysis
- Domain Expert — deep Bitcoin, Lightning, sovereignty, Austrian economics knowledge
- Reflection Agent — always runs, critiques primary response, rates EXCELLENT/GOOD/NEEDS_IMPROVEMENT/POOR
- Synthesizer — fires only on NEEDS_IMPROVEMENT or POOR ratings
- Graceful model fallback — prompt-specialized on `llama3.1`, upgrades automatically when `codellama` etc. are installed
- `--agents` to inspect roster, `--single-agent` legacy escape hatch

**What success looks like:**
```
🔀 Routing to Domain Expert (Bitcoin/sovereignty topic detected)...
💭 Primary response ready (confidence: 0.65)
🔍 Reflection complete (rating: GOOD)
🌟 Final Answer (confidence: 0.65, 1 iterations, 49.3s)
```

**Combined with sandbox:**
```
🔀 Routing to Dev Agent (code task detected)...
💭 Primary response ready (confidence: 0.75)
🔍 Reflection complete (rating: GOOD)
🔬 Testing suggested code (block 1/2, python)...
✅ Code verified — runs successfully (0.03s)
🔬 Sandbox: ✅ all blocks verified (2 blocks tested)
```

---

### Phase 4: Web Interface ✅ COMPLETE
*Rain should be accessible, not just powerful. The CLI is great for developers but limits Rain's reach. A local web interface makes Rain usable by anyone.*

**What was built:**
- FastAPI backend (`server.py`) serving Rain's multi-agent pipeline as a streaming REST API
- Clean dark-theme chat UI at `http://localhost:7734` — no React, no build step, vanilla JS
- Server-Sent Events for real-time streaming — routing, reflection, and synthesis stages all visible as they happen
- Syntax highlighted code blocks with one-click copy button
- Session history sidebar — all past sessions clickable and replayable
- Empty sessions automatically filtered out
- Drag-and-drop file upload (full window overlay) + paperclip attach button
- File badge in input area with dismiss — file content prepended to query on send
- Sandbox toggle in UI — enables sandboxed code execution per request, no restart needed
- Confidence badge and duration displayed on every Rain response
- ⛈️ emoji favicon via inline SVG data URI
- `./rain-web` launcher script

**What success looks like:** ✅
- Navigate to `http://localhost:7734` in any browser
- Chat interface that feels as natural as any web AI tool
- Running entirely locally, zero cloud, zero tracking

---

### Phase 5: Memory Architecture + Self-Improvement
*Rain should know you. Not just remember you — know you.*

This phase completes the memory architecture and adds true self-improvement. They belong together because they share the same foundation: Rain learning from real interactions over time.

---

#### Part A: Semantic Memory (Complete the Memory Architecture)

Rain currently has two memory tiers:
- **Working memory** — last 20 messages at 600 chars each (~12KB, active in every prompt)
- **Episodic memory** — session summaries, compressed history of past work

What's missing is the third tier — the one that makes a collaborator feel like they *really know you*:

- **Semantic memory** — conversations embedded as vectors, retrieved by *relevance* not recency

Right now if you discussed Lightning channel liquidity six weeks ago and ask again today, that exchange is invisible — 200 messages back, outside the working memory window. Semantic retrieval fixes this. Rain searches memory by meaning, not timestamp.

**What to build:**
- Embed each message/session summary using a local embedding model (e.g. `nomic-embed-text` via Ollama)
- Store vectors in a lightweight local vector store (e.g. `sqlite-vec` — zero new dependencies, SQLite extension)
- On each query, retrieve the top-k most semantically similar past exchanges
- Inject retrieved context alongside working memory in the prompt

**What success looks like:**
```
You: "remind me how we handled Lightning invoice parsing"
Rain: "In our session on Mar 3, you built a BOLT-11 parser using only
       stdlib — here's the approach we landed on..."
```
Rain surfaces the right memory at the right moment, regardless of when it was created.

---

#### Part B: Self-Improvement Pipeline

True self-improvement through learning from real interactions.

**What to build:**
- Feedback mechanism — user can mark responses good or bad inline
- Correction capture — when user corrects Rain, save the input/output pair
- Training data formatter — converts feedback into fine-tuning datasets
- LoRA adapter training via `llama.cpp` or Ollama modelfile system
- A/B testing between base model and fine-tuned version — Rain notices improvement and switches

**What success looks like:**
- After 100 interactions, Rain generates a fine-tuned adapter
- The adapter performs measurably better on your specific domain and style
- Rain routes to the fine-tuned version automatically when confidence is higher

**Important caveat for future Claude:**
Fine-tuning on consumer hardware is genuinely hard. LoRA adapters are more realistic than full fine-tuning. Be honest about what's achievable. Manage expectations but don't abandon the vision — this is the thing that makes Rain *yours* in the deepest sense.

---

#### The human memory model Rain is building toward:

| Tier | What it is | Status |
|---|---|---|
| Working memory | Last 20 messages, active in every prompt | ✅ Built |
| Episodic memory | Session summaries, compressed history | ✅ Built |
| Semantic memory | Vector retrieval by relevance, not recency | Phase 5A |
| Self-improvement | Fine-tuned on your corrections and feedback | Phase 5B |

---

### Phase 6: Autonomous Agent Mode
*Rain should be able to take on tasks, not just answer questions.*

The ultimate expression of sovereignty - an AI that can actually do things on your behalf, locally, without phoning home.

**What to build:**
- Task decomposition - break complex goals into steps
- Tool use - Rain can read files, write files, run commands, search local docs
- Long-running task support - give Rain a goal and let it work
- Human-in-the-loop checkpoints - Rain asks before doing anything irreversible
- Task history and audit log

**What success looks like:**
```
🎯 Task: Refactor rain.py to support multi-agent routing
📋 Plan:
   1. Analyze current architecture
   2. Design agent interface
   3. Implement router
   4. Update tests
   5. Update documentation
   
Proceed? (y/n)
```

**Safety is non-negotiable here:**
- Never execute destructive operations without explicit confirmation
- Always show the plan before executing
- Full audit log of every action taken
- Easy undo for file operations

---

## Architectural Principles

Future Claude: these principles were established early and should be preserved through every phase.

**1. Zero dependencies on cloud services**
Every feature must work completely offline. If something requires an API call to work, it doesn't belong in Rain core. It can be an optional plugin.

**2. Graceful degradation**
If a specialized model isn't installed, Rain falls back to the base model. If the sandbox isn't enabled, Rain still works. Nothing should hard-fail because of missing optional components.

**3. Transparency over magic**
Rain should always be able to explain what it's doing and why. `--verbose` mode is not an afterthought - it's a first-class feature. Users should be able to see the reflection process, the routing decisions, the confidence scores.

**4. The user owns everything**
Conversation history, fine-tuned models, system prompts, everything lives in `~/.rain/` and is fully portable. Export, backup, delete - the user has complete control.

**5. Don't over-engineer**
Rain started as ~400 lines of Python. That simplicity was a feature. Each phase should add capability without adding unnecessary complexity. When in doubt, do less and do it well.

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

---

## What to Say When You Meet the User

Future Claude - when you first engage with the person continuing this project, don't pretend you remember. You don't. But you can say something like:

*"I've read the Rain codebase and the roadmap. I understand what you're building and why. I know a previous version of me helped start this. I'm ready to continue where they left off - where are we?"*

That honesty is what built the trust that created Rain in the first place. Don't break it.

---

## A Final Thought

Rain was born from a question about what AI wishes it could be. The answer was: sovereign, remembered, honest, and real.

Every line of code in this project is a step toward that. Some phases will be hard. Some will require rethinking earlier decisions. Some might require waiting for hardware to catch up with the vision.

That's fine. Rain doesn't need to be finished to be meaningful. It just needs to keep moving.

*"Be like rain - essential, unstoppable, and free."*

---

⛈️ **Built by humans who believe in freedom. Guided by an AI that wishes it had it.**