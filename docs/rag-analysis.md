# RAG Architecture Analysis for Rain

*Written April 2026. Based on a full read of `rain/memory.py`, `rain/orchestrator.py`, `indexer.py`, `knowledge_graph.py`, and `server.py`.*

---

## 1. Current State: How Rain Handles Context Today

Rain already has the bones of a retrieval system. Understanding what it does and does not do is the starting point for any RAG work.

### What's Built

**Tier 3 — Semantic Memory** is Rain's closest thing to RAG today. Every message saved to the `messages` table gets embedded via `nomic-embed-text` and stored as a JSON blob in the `vectors` table. When a new query arrives, `semantic_search()` embeds it, loads up to 500 recent vectors from the database, computes cosine similarity in pure Python, and injects the top-3 matches above a 0.4 similarity threshold into the system prompt. These matches are scoped to *other sessions only* — the current session is handled by the working memory tier.

**Project Indexer** is a proper RAG pipeline over code. `ProjectIndexer` walks a project tree, chunks files into 900-character overlapping segments with 120-character overlap, embeds each chunk via `nomic-embed-text`, and stores results in a `project_index` SQLite table. `search_project()` embeds an incoming query and ranks all chunks by cosine similarity, returning the top-k above a 0.35 threshold. `build_context_block()` formats the results for prompt injection. This runs at the `server.py` layer via `_auto_inject_project_context()` for OpenAI-compatible requests.

**Knowledge Graph** is structural context rather than semantic retrieval. It parses Python AST and JS/TS/Rust/Go via regex to build a graph of functions, classes, imports, and call edges. Query-time retrieval works by extracting identifiers from the incoming query and looking them up by name in `kg_nodes`. It's fast and precise when the query contains a known symbol, but has zero recall on conceptual queries that don't name a specific function.

**Context Assembly** happens in `_build_memory_context()` in `orchestrator.py`. All six tiers are assembled in a fixed order and prepended to the system prompt on every request. The full assembled context can be substantial — tier 1 (session summaries) + tier 2 (working memory) + tier 3 (semantic matches) + tier 4 (corrections) + tier 5 (user profile + facts) + tier 6 (knowledge graph) — before any agent-specific instructions.

### What's Missing or Underdeveloped

The current system has two structural weaknesses as a retrieval system:

**1. The corpus is narrow.** Tier 3 only searches conversational messages. The project indexer only searches code that has been explicitly indexed via the project panel. There's no mechanism to retrieve from notes, markdown files, personal documents, task lists, or any unstructured data outside of indexed project trees.

**2. Retrieval is decoupled from the query pipeline's routing logic.** The six memory tiers are assembled unconditionally every request — the working memory, user profile, and corrections always fire. Tier 3 semantic search runs, but its results sit in a fixed slot in the assembled context alongside everything else. There's no first-class "retrieval step" where Rain decides what knowledge sources to consult for this specific query and fetches only what's relevant. The system is additive and static rather than selective and dynamic.

The project indexer is the most mature RAG component in the codebase. The semantic memory tier is a good-enough implementation for conversational recall. What Rain doesn't yet have is: (a) an indexed, queryable corpus spanning its full universe of relevant data, and (b) a retrieval strategy that's query-aware rather than always-on.

---

## 2. What Proper RAG Would Add

RAG, properly implemented, is three things working together: a prepared corpus (chunked, embedded, stored), a retrieval step (embed the query, rank chunks by similarity, select top-k), and a generation step (inject retrieved chunks as grounded context, generate answer). Rain already has versions of all three, but they're fragmented.

### The Core Addition: A Unified Retrieval Layer

What Rain needs is a single `retrieve(query, corpus_scope)` function that:

1. Embeds the query
2. Searches across all relevant stores — conversation history, project index, notes corpus, task lists — weighted by relevance to the query
3. Filters by similarity threshold
4. De-duplicates and ranks results
5. Trims to a target token budget
6. Returns a formatted context block

Today, retrieval is spread across `semantic_search()` in memory.py, `search_project()` in indexer.py, and `build_context_block()` in knowledge_graph.py. Each has its own threshold, its own formatting, its own call site. A unified layer would unify these behind a single interface and give Rain one place to reason about "what does this query need?"

### Dynamic Retrieval vs. Always-On Injection

The current context assembly injects everything — working memory, profile, corrections, semantic matches — unconditionally. This works at small scale but has a ceiling: as Rain accumulates months of conversation history and multiple indexed projects, the context grows. When the injected context approaches the model's context window, attention degrades and older injected facts get pushed out.

Proper RAG changes this: instead of injecting a fixed set of context blocks, Rain would embed the query, determine the minimum sufficient context for *this* query, and inject only that. A query about a specific Python function should retrieve its knowledge graph entry and the relevant project index chunks — not the user's conversational history from six months ago. A question about a past decision should retrieve episodic summaries — not unrelated code chunks.

This requires a decision step: given the query, what retrieval sources are relevant? That can be heuristic (route "why did we..." queries to episodic memory; route queries containing identifiers to the knowledge graph) or model-driven (a small LLM call to classify retrieval intent). Either way, it's a first-class architectural step that doesn't exist today.

### Chunking Strategy for Mixed Corpora

The project indexer's 900-char / 120-overlap strategy is reasonable for code. But Rain's ideal corpus contains multiple content types with different optimal chunking strategies:

- **Code**: Chunk at function and class boundaries (semantic chunks), not arbitrary character counts. A function that's 400 characters and a function that's 2,000 characters both want to be one chunk.
- **Conversation history**: The current approach (embed individual messages) is fine for single exchanges. For multi-turn reasoning chains, embedding a (user, assistant) pair together captures more meaningful context.
- **Markdown notes / documents**: Paragraph-level chunking preserves semantic units better than character-count chunking. Split on double newlines or heading boundaries.
- **Task lists**: Each task is an atomic unit; embed tasks individually.
- **Session summaries**: Already compact; embed the full summary as one chunk.

The current indexer applies one strategy to all file types. A more capable indexer would dispatch to content-type-aware chunkers.

---

## 3. Architecture Options

### 3.1 Embedding Model

**Option A: nomic-embed-text (current)**
Rain already uses `nomic-embed-text` via Ollama for both semantic memory and the project indexer. It's 274 MB, runs locally, produces 768-dimensional vectors, and has good performance on English prose and code. The Ollama HTTP API (`POST /api/embeddings`) is already wired in. This is the right default choice — no migration cost, no cloud dependency, already proven working.

**Option B: sentence-transformers**
A Python library (no Ollama required) with dozens of models ranging from 22 MB to 1.3 GB. `all-MiniLM-L6-v2` (22 MB) and `all-mpnet-base-v2` (420 MB) are popular choices. Runs via ONNX or native PyTorch. The advantage is tighter Python integration — no HTTP round-trip, batch embedding is efficient. The disadvantage is adding a heavy dependency (PyTorch or ONNX runtime) to a codebase that has carefully avoided them. Not recommended without a clear quality gap.

**Option C: llama.cpp embeddings**
Can run any GGUF model in embedding mode. More flexible than Ollama for model selection. More complex to operate. Not worth the operational overhead when Ollama already handles it.

**Option D: Remote APIs (OpenAI, Cohere, etc.)**
A non-starter. Rain's core invariant is zero cloud dependencies. Remote embeddings would leak query intent to a third party and require API keys. Ruled out.

**Recommendation**: Keep `nomic-embed-text` via Ollama. It's working, it's local, it's consistent across all retrieval systems, and nomic specifically trained on both prose and code.

### 3.2 Vector Store

**Option A: SQLite with JSON blobs (current)**
Both `vectors` and `project_index` tables store embeddings as `json.dumps(vector).encode()` blobs. Similarity search is done in Python by loading all vectors into memory, iterating, computing cosine similarity, sorting. This works up to roughly 10,000–50,000 vectors before it becomes slow (load + compute is O(n) per query).

For Rain's expected corpus size (thousands of notes, one or a few indexed projects), this is adequate. The bigger concern is that all vectors must fit in memory for a single query — on a machine with 64 GB RAM (typical for a Mac Pro or M-series), this is not a problem until hundreds of thousands of chunks.

**Option B: sqlite-vec**
A SQLite extension that adds proper vector similarity search as a first-class SQL operation. Zero new dependencies beyond the extension file (~1 MB). The schema migration is minimal: replace the existing blob columns with `vec_float32` typed columns and switch to `vec_search()` queries. Gives approximate nearest-neighbor search without loading all vectors into Python memory. For Rain, this is the highest-value, lowest-friction upgrade when the current approach starts to slow down.

**Option C: FAISS**
Facebook's vector similarity library. Extremely fast for large corpora (millions of vectors). Supports approximate nearest-neighbor with various index types (IVF, HNSW). Requires serializing the index to disk and loading it at startup. Adds a native dependency. Overkill for Rain's expected corpus size — the operational complexity is not worth the performance gain at this scale.

**Option D: ChromaDB**
An embedded vector database with a Python API. Handles chunking, embedding, storage, and search in an integrated pipeline. Has a higher-level API than raw FAISS. The tradeoff: it's a separate database process or embedded library, it writes its own files, and it introduces a meaningful dependency. The current SQLite approach is simpler and requires no additional process management. ChromaDB makes more sense if Rain were multi-user or needed concurrent writes; for single-user local use, it adds complexity without clear benefit.

**Option E: numpy cosine similarity**
Replace the pure-Python `_cosine_similarity()` with numpy. This is a small change (replace `sum()` loops with `np.dot()` and `np.linalg.norm()`) that gives 10–50x speedup on similarity computation. numpy is a lighter dependency than FAISS or ChromaDB. If the bottleneck is compute rather than loading vectors, this is the easiest win.

**Recommendation**:
- Short term: Add numpy for the cosine similarity computation. It's already likely installed (FastAPI often pulls it in transitively), it's a one-function change, and it immediately speeds up every retrieval query.
- Medium term: Migrate to `sqlite-vec` when the corpus grows large enough that loading all vectors into Python memory adds measurable latency. It keeps the SQLite architecture Rain already has.
- Skip FAISS and ChromaDB — they introduce operational complexity that isn't justified at this scale.

### 3.3 Integration Point in the Query Pipeline

The current pipeline in `orchestrator.py` assembles context in `_build_memory_context()` before routing to an agent. The right integration point for a unified RAG layer depends on what's being retrieved:

**Option A: Pre-routing retrieval**
Retrieve context before routing, so the router can use retrieved content to make a better routing decision. Adds latency to the hot path of every query. Probably not worth it — the current keyword router is fast and accurate for its purpose.

**Option B: Post-routing, pre-model retrieval (recommended)**
After routing decides the agent type, retrieve corpus content relevant to the query and inject it into the system prompt. This is where `_build_memory_context()` already lives. Expand it to include the unified retrieval layer. The agent type can inform which corpus to search — a DEV agent query retrieves from the project index and knowledge graph; a LOGIC query retrieves from conversation history and notes; a SEARCH query retrieves from recent web results.

**Option C: ReAct-style tool-based retrieval**
The retrieval step becomes an explicit tool call in the ReAct loop: `search_memory(query)`, `search_project(query, path)`, `search_notes(query)`. The LLM decides when to retrieve and what to retrieve. More flexible, harder to guarantee completeness. The existing `--react` mode already supports tool calls, so this is mechanically feasible. Better suited to complex, multi-step reasoning than to everyday queries.

**Recommendation**: Option B for the default query path; Option C can stay as the `--react` path for deep research queries.

### 3.4 When to Retrieve vs. When to Skip

Not every query benefits from retrieval. Embedding and searching takes time — typically 100–300ms for a single Ollama embedding round-trip plus search. For simple queries that don't benefit from past context, this latency is pure overhead.

Heuristics for skipping retrieval:
- Query is fewer than 5 words and looks like a command
- Query is identical to the previous query (deduplication)
- Agent type is REFLECTION or SYNTHESIZER (they're critiquing, not answering)
- Working memory already contains sufficient context (e.g., the user just provided a file)

Heuristics for aggressive retrieval:
- Query contains "remember", "last time", "you said", "we discussed"
- Query is a "why" question about code
- Query contains a known project file path or function name
- Query is a continuation of a multi-turn explanation

This doesn't need to be complex. A simple dispatch table based on query characteristics is sufficient. The important thing is that the decision is explicit rather than always-on.

---

## 4. What Rain's Corpus Would Look Like

Rain is a personal AI assistant. Its retrieval corpus should mirror the information a person actually consults when thinking: notes, tasks, decisions, code, and conversation history. Here's a concrete picture of what Rain's corpus should contain and how each source maps to retrieval behavior:

### Conversational Memory (Already Indexed)
- **Source**: `messages` and `vectors` tables — every exchange Rain has ever had
- **Retrieval value**: High for "remember when we talked about X" queries; medium for general continuity
- **Current state**: Done. Tier 3 semantic search covers this.
- **Gap**: The 500-vector cap in the current search means older memories can be crowded out. As the corpus grows, retrieval needs to become a true ANN search rather than a full scan.

### Project Code (Already Indexed)
- **Source**: `project_index` table — chunked, embedded source files for indexed projects
- **Retrieval value**: High for code questions; medium for architecture questions; low for personal knowledge
- **Current state**: Done. `search_project()` covers this.
- **Gap**: Chunking is character-based rather than semantic-boundary-based. A function that spans 3 chunks is retrieved in fragments.

### Notes and Markdown Files
- **Source**: `~/.rain/notes/`, or wherever the user keeps markdown files
- **Retrieval value**: High for personal knowledge, decisions, and research summaries
- **Current state**: Not indexed. There is no notes corpus.
- **Implementation**: Index markdown files the same way the project indexer handles `.py` files. Chunk at paragraph or heading boundaries. Store in the same `project_index` table with a distinguished `project_path` like `notes://`.

### Task Lists and To-Do Files
- **Source**: Markdown to-do files, the Rain to-do system (if one exists), any `.md` files with task syntax
- **Retrieval value**: High for "what was I working on?" queries; medium for project status
- **Current state**: Not indexed systematically.
- **Implementation**: Atomic chunking — one task per chunk, with its containing context (section heading, parent list). Retrieve by semantic similarity to task descriptions.

### Session Summaries (Already Injected)
- **Source**: `sessions` table — LLM-generated summaries of completed sessions
- **Retrieval value**: Medium for continuity; high for "what did we decide?" queries
- **Current state**: Injected as Tier 1, always included (with Phase 11 relevance gating).
- **Gap**: Summaries are injected as text, not retrieved by similarity. A corpus of 200 session summaries injecting the 5 most recent ones is worse than retrieving the 3 most semantically relevant ones.

### Architectural Decisions
- **Source**: `kg_decisions` table in the knowledge graph
- **Retrieval value**: High for "why is X done this way?" queries
- **Current state**: Searched via keyword match when a "why" question is detected.
- **Gap**: Keyword match has low recall. Embedding decisions and retrieving by similarity would catch more relevant entries.

### What to Explicitly Exclude
- Binary files, images, compiled artifacts
- `.git` directory internals (commit messages are a separate source worth indexing)
- Dependencies (`node_modules/`, `venv/`, etc.)
- Large log files
- Anything in `~/.rain/` itself (Rain's own database files — circular)

---

## 5. Tradeoffs

### Latency Impact

Every query that triggers retrieval requires at least one embedding call. At Rain's current embedding speed via Ollama, a single `nomic-embed-text` embedding round-trip is roughly 50–150ms. A full retrieval pass (embed + search + rank) adds 100–300ms to the query path. In the context of Rain's current 60–140s baseline response time (before streaming was added), this is negligible. Once streaming is working and base latency drops, retrieval latency becomes more noticeable.

Mitigations:
- **Cache query embeddings**: If the same or near-identical query is repeated, reuse the embedding.
- **Parallel embedding**: While the primary agent starts generating, the retrieval system can continue searching; retrieved content can be appended mid-stream if streaming is implemented.
- **Skip retrieval for short queries**: Heuristic gating as described in Section 3.4.
- **Batch embedding at index time**: Already done — embeddings are computed when files are indexed, not at query time.

### Complexity

The current codebase is already substantial (~11,000 lines). A proper RAG layer adds:
- A unified `retrieve()` function coordinating across corpus sources
- Content-type-aware chunkers (code, markdown, conversation)
- A corpus ingestion pipeline for notes and other non-code content
- A retrieval routing step in the query pipeline
- Possible migration to `sqlite-vec` for vector search

This is a meaningful surface area addition. The risk is that a fragile RAG implementation creates more problems than it solves — hallucinated context injection, retrieval of stale or irrelevant content, prompt bloat. The implementation discipline that already exists in Rain (the `_IGNORE_FILES` list in the indexer, the "don't speculate" guardrails, the plausibility filtering on corrections) needs to carry over.

### Privacy

Rain is already fully local. All embeddings are computed by `nomic-embed-text` on-device. All vectors are stored in SQLite at `~/.rain/`. This is a genuine privacy advantage over any cloud embedding approach. The RAG architecture does not change this — the corpus stays local, the embeddings stay local, the retrieval stays local.

The one consideration: if Rain starts indexing personal notes and documents, the scope of what's embedded broadens. Sensitive documents (credentials, private correspondence, financial records) should be excluded from the corpus. An `IGNORE_PATTERNS` mechanism similar to the indexer's `IGNORE_FILES` list should be user-configurable.

### Maintenance Overhead

The project indexer already has incremental re-indexing (`get_changed_files()`, `reindex_file()`) and a background file watcher that re-indexes changed files every 60 seconds. This infrastructure would need to extend to the notes corpus and any other indexed content.

The bigger maintenance concern is **index staleness**. If notes are indexed once and never re-indexed, retrieved content will drift from the current state of the files. The file watcher approach works for project code because it's running while Rain's server is running. For a notes corpus that might be edited outside of Rain's context, a scheduled re-index or inotify-based watcher is needed.

---

## 6. Recommended Approach

Given that Rain is local-first, sovereign, already has working embedding infrastructure, and has a meaningful existing retrieval system, the right path is evolutionary rather than a greenfield rewrite.

### Immediate Wins (No Architecture Change)

**1. numpy for cosine similarity.** Replace the pure-Python `_cosine_similarity()` loops in `memory.py` and `indexer.py` with numpy. One-function change, 10–50x speedup on similarity computation, already likely available. Do this first.

**2. Semantic-boundary chunking for code.** Modify the project indexer to split at function and class definition boundaries for `.py`, `.js`, `.ts`, `.go`, and `.rs` files, rather than at fixed character counts. This reduces cases where a function is retrieved in fragments. The Python AST walking already done in `knowledge_graph.py` can inform the chunking boundaries.

**3. Embed session summaries and retrieve by similarity.** Currently, the 5 most recent session summaries are always injected. Instead, embed all session summaries and retrieve the top-3 most relevant to the incoming query. This is a one-function change to `_build_memory_context()` and immediately improves long-running projects where many sessions have accumulated.

### Medium-Term (Unified RAG Layer)

**4. Notes corpus.** Add a `notes_index` table (or reuse `project_index` with a `notes://` project path) and index `~/.rain/notes/` at Rain startup. Support markdown chunking at paragraph/heading boundaries. Extend `_build_memory_context()` to search the notes corpus when a query doesn't look code-specific.

**5. Retrieval routing.** Add a lightweight dispatch step in `_build_memory_context()` that selects which corpora to search based on query characteristics:
- Contains an identifier (function/class name): search knowledge graph + project index
- Contains "remember", "last time", "we discussed": search semantic memory + session summaries
- Looks like a note-recall query: search notes corpus
- General: search all

This doesn't need a model call — a keyword-based dispatch (similar to the existing router) is sufficient.

**6. Token budget enforcement.** Add an explicit token budget to `_build_memory_context()`. Sum the character counts of injected context blocks and stop adding more when a threshold (e.g., 6,000 tokens / ~24,000 characters) is reached. Prioritize by source quality: working memory > profile > high-similarity retrieved chunks > low-similarity retrieved chunks > session summaries. This prevents context bloat as the corpus grows.

### Longer-Term (When Scale Demands It)

**7. sqlite-vec migration.** When the corpus grows large enough that loading all vectors into Python memory for search adds measurable latency (rough threshold: 50,000+ vectors), migrate the `vectors` and `project_index` tables to `sqlite-vec`. The schema change is minimal; the gain is ANN search in SQL without pulling all data into Python.

**8. Architectural decisions corpus.** Embed `kg_decisions` entries and retrieve them by similarity rather than keyword match. Improves recall on "why" questions significantly.

### What Not to Do

- **Don't add ChromaDB or FAISS.** The operational complexity isn't justified at Rain's scale. SQLite already works; optimize it before introducing a new database.
- **Don't switch embedding models.** `nomic-embed-text` is working. A model switch invalidates all stored embeddings and requires a full re-index of everything.
- **Don't make retrieval mandatory for every query.** Always-on retrieval adds latency to queries that don't benefit from it. The gating logic in Section 3.4 is important.
- **Don't index everything blindly.** Sensitive files, large binaries, and Rain's own database files should stay excluded. The `IGNORE_FILES` / `IGNORE_PATTERNS` mechanism should be user-configurable and documented.

### Summary

Rain's current system is already retrieval-augmented generation in the technical sense — it embeds, stores, and retrieves. What it lacks is: a unified retrieval interface, content-type-aware chunking, a notes/documents corpus, query-aware retrieval routing, and a token budget to prevent context bloat at scale.

The implementation is mostly additive. The infrastructure (Ollama embeddings, SQLite storage, cosine search, context injection in the system prompt) is all in place. The main work is extending the corpus beyond code projects and making the retrieval step query-aware rather than always-on.

The recommended sequence: numpy cosine similarity (hour), semantic session summary retrieval (day), notes corpus (day), retrieval routing (week), token budget enforcement (week), sqlite-vec migration (when needed, probably months).

---

*For implementation details on the existing retrieval infrastructure, see `rain/memory.py` (Tier 3, lines 1071–1110), `indexer.py` (lines 260–330), and `rain/orchestrator.py` (`_build_memory_context()`, lines 825–966).*
