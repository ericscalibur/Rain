#!/usr/bin/env python3
"""
Rain ⛈️ - Web Server (Phase 4)

FastAPI backend serving Rain's multi-agent capabilities via HTTP.
Runs entirely locally at http://localhost:7734
Zero cloud. Zero tracking. Fully sovereign.

Usage:
    .venv/bin/uvicorn server:app --host 127.0.0.1 --port 7734 --reload
"""

import asyncio
import json
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add Rain's directory to path so we can import rain.py
sys.path.insert(0, str(Path(__file__).parent))

from rain import (
    MultiAgentOrchestrator,
    RainMemory,
    AgentType,
    CodeSandbox,
)

# ── App ────────────────────────────────────────────────────────────────

# ── Global orchestrator (initialized once on startup) ──────────────────

_orchestrator: Optional[MultiAgentOrchestrator] = None
_memory: Optional[RainMemory] = None
_custom_agents: list = []  # [{id, name, prompt}]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Rain on startup, clean up on shutdown."""
    global _orchestrator, _memory
    _memory = RainMemory()
    _memory.start_session(model="llama3.1")
    _orchestrator = MultiAgentOrchestrator(
        default_model="llama3.1",
        memory=_memory,
        sandbox_enabled=False,  # user can toggle per request
    )
    yield
    if _memory:
        summary = _memory.generate_summary()
        _memory.end_session()
        if summary:
            _memory.update_summary(summary)


app = FastAPI(
    title="Rain",
    description="Sovereign AI Ecosystem — local web interface",
    version="0.4.0",
    docs_url=None,   # disable swagger UI (keep it minimal)
    redoc_url=None,
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    sandbox: bool = False
    sandbox_timeout: int = 10
    verbose: bool = False

    web_search: bool = False


class CustomAgentRequest(BaseModel):
    name: str
    prompt: str


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: str  # 'good' or 'bad'
    correction: Optional[str] = None


def _duckduckgo_search(query: str, max_results: int = 5) -> list:
    """
    Search DuckDuckGo using their free HTML endpoint. No API key required.
    Returns a list of {title, snippet, url} dicts.
    """
    import urllib.request
    import urllib.parse
    import re

    try:
        params = urllib.parse.urlencode({"q": query, "kl": "us-en"})
        url = f"https://html.duckduckgo.com/html/?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Rain/1.0)",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        results = []
        # Extract result blocks
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )
        for url_raw, title_raw, snippet_raw in blocks[:max_results]:
            title = re.sub(r"<[^>]+>", "", title_raw).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet_raw).strip()
            # DuckDuckGo wraps URLs — extract the real one
            url_match = re.search(r"uddg=([^&]+)", url_raw)
            real_url = urllib.parse.unquote(url_match.group(1)) if url_match else url_raw
            if title and snippet:
                results.append({"title": title, "snippet": snippet, "url": real_url})

        return results
    except Exception as e:
        print(f"[web search] failed: {e}")
        return []


class SessionSummary(BaseModel):
    id: str
    started_at: str
    ended_at: Optional[str]
    summary: Optional[str]
    model: Optional[str]
    message_count: int


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the single-page UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI not found — static/index.html missing")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    """Health check — confirms server and Ollama are reachable."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        ollama_ok = result.returncode == 0
    except Exception:
        ollama_ok = False

    return {
        "status": "ok",
        "ollama": ollama_ok,
        "model": _orchestrator.default_model if _orchestrator else None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/models")
async def list_models():
    """Return installed Ollama models."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            if line.strip():
                parts = line.split()
                models.append({
                    "name": parts[0],
                    "size": parts[2] if len(parts) > 2 else "",
                })
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents")
async def get_agents():
    """Return the current agent roster."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    roster = []
    for agent_type in [AgentType.DEV, AgentType.LOGIC, AgentType.DOMAIN,
                       AgentType.REFLECTION, AgentType.SYNTHESIZER]:
        agent = _orchestrator.agents[agent_type]
        specialized = not agent.model_name.startswith(
            _orchestrator.default_model.split(":")[0]
        )
        roster.append({
            "type": agent.agent_type.value,
            "description": agent.description,
            "model": agent.model_name,
            "specialized": specialized,
        })

    missing = []
    from rain import AGENT_PREFERRED_MODELS
    for agent_type, preferred in AGENT_PREFERRED_MODELS.items():
        if agent_type in (AgentType.REFLECTION, AgentType.SYNTHESIZER, AgentType.GENERAL):
            continue
        best = preferred[0]
        if not any(m.startswith(best.split(":")[0])
                   for m in _orchestrator._installed_models):
            missing.append(best)

    return {
        "roster": roster,
        "suggestions": list(dict.fromkeys(missing)),
        "custom_agents": _custom_agents,
    }


@app.post("/api/agents/custom")
async def add_custom_agent(req: CustomAgentRequest):
    """Register a new custom agent with a user-defined system prompt."""
    global _custom_agents, _orchestrator
    if not req.name.strip() or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Name and prompt are required")

    agent_id = str(uuid.uuid4())[:8]
    entry = {"id": agent_id, "name": req.name.strip(), "prompt": req.prompt.strip()}
    _custom_agents.append(entry)

    # Register with orchestrator so it can be routed to
    if _orchestrator:
        from rain import Agent, AgentType
        agent = Agent(
            agent_type=AgentType.GENERAL,
            model_name=_orchestrator.default_model,
            system_prompt=req.prompt.strip(),
            description=req.name.strip(),
        )
        _orchestrator.custom_agents[agent_id] = agent

    return {"status": "ok", "id": agent_id}


@app.delete("/api/agents/custom/{agent_id}")
async def remove_custom_agent(agent_id: str):
    """Remove a custom agent by ID."""
    global _custom_agents, _orchestrator
    _custom_agents = [a for a in _custom_agents if a["id"] != agent_id]
    if _orchestrator and hasattr(_orchestrator, "custom_agents"):
        _orchestrator.custom_agents.pop(agent_id, None)
    return {"status": "ok"}


@app.get("/api/sessions")
async def get_sessions():
    """Return recent session history."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    sessions = _memory.get_recent_sessions(limit=20)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Return all messages for a given session."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    import sqlite3
    with sqlite3.connect(_memory.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT role, content, timestamp, is_code, confidence
               FROM messages WHERE session_id = ?
               ORDER BY timestamp ASC""",
            (session_id,),
        ).fetchall()
    return {"messages": [dict(r) for r in rows]}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    import sqlite3
    with sqlite3.connect(_memory.db_path) as conn:
        conn.execute("DELETE FROM vectors WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return {"status": "ok"}


@app.post("/api/new-session")
async def new_session():
    """End the current session and start a fresh one."""
    global _memory, _orchestrator
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    import threading

    # Hold a reference to the ending session, end it synchronously so it
    # appears in the sidebar immediately (ended_at IS NOT NULL filter)
    ending_memory = _memory
    ending_memory.end_session()

    # Create a brand-new RainMemory — generates a fresh UUID
    _memory = RainMemory()
    _memory.start_session(model=_orchestrator.default_model if _orchestrator else "llama3.1")
    if _orchestrator:
        _orchestrator.memory = _memory

    # Summarize the ended session in the background
    def _summarize_in_background():
        summary = ending_memory.generate_summary()
        if summary:
            ending_memory.update_summary(summary)
    threading.Thread(target=_summarize_in_background, daemon=True).start()

    return {"status": "ok", "session_id": _memory.session_id}


@app.get("/api/feedback/stats")
async def feedback_stats():
    """Return feedback counts and training readiness."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    import sqlite3 as _sqlite3
    with _sqlite3.connect(_memory.db_path) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
        ).fetchone()
        if not exists:
            return {"total": 0, "good": 0, "bad": 0, "corrections": 0, "ready": False}
        row = conn.execute("""
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN rating='good' THEN 1 ELSE 0 END), 0) AS good,
                COALESCE(SUM(CASE WHEN rating='bad'  THEN 1 ELSE 0 END), 0) AS bad,
                COALESCE(SUM(CASE WHEN rating='bad' AND correction IS NOT NULL
                              AND correction != '' THEN 1 ELSE 0 END), 0) AS corrections
            FROM feedback
        """).fetchone()
        corrections = row[3] if row else 0
        return {
            "total":       row[0] if row else 0,
            "good":        row[1] if row else 0,
            "bad":         row[2] if row else 0,
            "corrections": corrections,
            "ready":       corrections >= 10,
        }


@app.post("/api/finetune/export")
async def finetune_export():
    """
    Export the corrections table to JSONL and ChatML training files.
    Returns paths and counts. Non-blocking — runs synchronously (fast).
    """
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    import sqlite3 as _sqlite3
    import json as _json

    with _sqlite3.connect(_memory.db_path) as conn:
        rows = conn.execute("""
            SELECT query, response, correction FROM feedback
            WHERE rating = 'bad' AND correction IS NOT NULL AND correction != ''
            ORDER BY id ASC
        """).fetchall()

    if not rows:
        raise HTTPException(status_code=400, detail="No corrections to export.")

    training_dir = Path.home() / ".rain" / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    SYSTEM = (
        "You are Rain, a sovereign AI assistant running locally on the user's computer. "
        "You are direct, precise, and honest about uncertainty. "
        "You never use third-party Python packages when stdlib alternatives exist. "
        "For Bitcoin and blockchain data you use the mempool.space public REST API. "
        "You never output HTML tags or markup inside code blocks."
    )

    # Alpaca JSONL
    jsonl_path = training_dir / "corrections.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for query, response, correction in rows:
            f.write(_json.dumps({
                "instruction": query.strip(),
                "input": "",
                "output": correction.strip(),
                "system": SYSTEM,
            }, ensure_ascii=False) + "\n")

    # ChatML for llama.cpp
    chatml_path = training_dir / "corrections.chatml.txt"
    with open(chatml_path, "w", encoding="utf-8") as f:
        for query, response, correction in rows:
            f.write(f"<|im_start|>system\n{SYSTEM}\n<|im_end|>\n")
            f.write(f"<|im_start|>user\n{query.strip()}\n<|im_end|>\n")
            f.write(f"<|im_start|>assistant\n{correction.strip()}\n<|im_end|>\n\n")

    return {
        "status":      "ok",
        "corrections": len(rows),
        "jsonl_path":  str(jsonl_path),
        "chatml_path": str(chatml_path),
    }


@app.post("/api/feedback")
async def save_feedback(req: FeedbackRequest):
    """
    Save user feedback on a Rain response.
    rating must be 'good' or 'bad'.
    correction is optional free-text — what the answer should have been.
    Corrections are embedded in background and injected into future relevant prompts.
    """
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    if req.rating not in ("good", "bad"):
        raise HTTPException(status_code=400, detail="rating must be 'good' or 'bad'")
    _memory.save_feedback(
        query=req.query,
        response=req.response,
        rating=req.rating,
        correction=req.correction,
    )
    return {"status": "ok"}


@app.post("/api/forget")
async def forget():
    """Wipe all memory."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    _memory.forget_all()
    return {"status": "ok", "message": "All memory wiped."}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Returns a Server-Sent Events stream.

    Each event is a JSON object with a `type` field:
      { type: "routing",   agent: "Domain Expert ..." }
      { type: "progress",  message: "Primary response ready..." }
      { type: "progress",  message: "Reflection complete (rating: GOOD)" }
      { type: "progress",  message: "Synthesizing improvements..." }
      { type: "sandbox",   message: "Testing block 1/2...", status: "running" }
      { type: "sandbox",   message: "✅ verified", status: "ok" }
      { type: "done",      content: "...", confidence: 0.75, duration: 12.3,
                           sandbox_verified: true, agent: "dev" }
      { type: "error",     message: "..." }
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    return StreamingResponse(
        _stream_chat(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_chat(req: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Run the multi-agent pipeline in a thread and stream SSE events back
    as each stage completes. Uses a queue to bridge the sync orchestrator
    and the async SSE stream.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def emit(event: dict):
        """Thread-safe event emitter — puts onto the asyncio queue."""
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def run_pipeline():
        """Runs in a background thread — calls the synchronous orchestrator."""
        try:
            # Temporarily patch the orchestrator to emit SSE events at each stage
            original_route = _orchestrator.router.route

            def patched_route(query):
                result = original_route(query)
                emit({
                    "type": "routing",
                    "agent": _orchestrator.router.explain(result),
                    "agent_type": result.value,
                })
                return result

            _orchestrator.router.route = patched_route

            # Configure sandbox for this request
            _orchestrator.sandbox_enabled = req.sandbox
            if req.sandbox:
                _orchestrator.sandbox = CodeSandbox(timeout=req.sandbox_timeout)
            else:
                _orchestrator.sandbox = None

            # Monkey-patch _query_agent to emit progress events
            original_query_agent = _orchestrator._query_agent

            def patched_query_agent(agent, prompt, label=None, include_memory=True):
                emit({"type": "progress", "message": f"💭 {label or agent.description + ' thinking...'}"})
                result = original_query_agent(agent, prompt, label=label, include_memory=include_memory)
                return result

            _orchestrator._query_agent = patched_query_agent

            # Monkey-patch reflection rating emit
            original_parse_rating = _orchestrator._parse_reflection_rating

            def patched_parse_rating(critique):
                rating = original_parse_rating(critique)
                emit({"type": "progress", "message": f"🔍 Reflection complete (rating: {rating})"})
                return rating

            _orchestrator._parse_reflection_rating = patched_parse_rating

            # ── Web search (if enabled) ───────────────────────────────────────
            search_results_count = 0
            search_augmented_message = None
            if req.web_search:
                emit({"type": "progress", "message": "🌐 Searching the web..."})
                search_results = _duckduckgo_search(req.message, max_results=5)
                search_results_count = len(search_results)
                if search_results:
                    snippets = "\n\n".join(
                        f"[{r['title']}]\n{r['snippet']}\nSource: {r['url']}"
                        for r in search_results
                    )
                    emit({"type": "progress", "message": f"🌐 {search_results_count} results retrieved"})
                    search_augmented_message = (
                        f"[Web search results for: {req.message}]\n\n"
                        f"{snippets}\n\n"
                        f"---\n"
                        f"Using the above search results as context, answer this question accurately. "
                        f"Cite sources where relevant. If the search results don't contain enough "
                        f"information, say so.\n\n"
                        f"Question: {req.message}"
                    )
                else:
                    emit({"type": "progress", "message": "🌐 No results found — using local knowledge"})

            # Memory system (working memory = last 20 messages) already handles
            # continuity — no additional history injection needed here.
            query = search_augmented_message if search_augmented_message else req.message

            # Run the full pipeline
            result = _orchestrator.recursive_reflect(query, verbose=req.verbose)

            # Restore originals
            _orchestrator.router.route = original_route
            _orchestrator._query_agent = original_query_agent
            _orchestrator._parse_reflection_rating = original_parse_rating

            if result:
                sandbox_summary = None
                if result.sandbox_results:
                    def _is_network(r):
                        err = (r.stderr + (r.error_message or "")).lower()
                        return not r.success and any(x in err for x in [
                            "urlerror", "connectionrefused", "nodename nor servname",
                            "name or service not known", "network is unreachable",
                            "connection timed out", "urlopen error", "ssl",
                        ])
                    def _is_long_running(r):
                        err = (r.error_message or "").lower()
                        return not r.success and "long-running" in err

                    skipped = [r for r in result.sandbox_results if _is_network(r) or _is_long_running(r)]
                    testable = [r for r in result.sandbox_results if r not in skipped]
                    verified = sum(1 for r in testable if r.success)
                    total = len(testable)
                    sandbox_summary = {
                        "verified": verified,
                        "total": total,
                        "all_ok": verified == total if total > 0 else True,
                        "network": any(_is_network(r) for r in skipped),
                        "long_running": any(_is_long_running(r) for r in skipped),
                    }

                emit({
                    "type": "done",
                    "content": result.content,
                    "confidence": round(result.confidence, 2),
                    "duration": round(result.duration_seconds, 1),
                    "iterations": result.iteration,
                    "improvements": result.improvements,
                    "sandbox": sandbox_summary,
                    "search_results": search_results_count if req.web_search else 0,
                })
            else:
                emit({"type": "error", "message": "No response from model."})

        except Exception as e:
            emit({"type": "error", "message": str(e)})
        finally:
            emit({"type": "_done_sentinel"})

    # Start pipeline in background thread
    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    # Stream events from queue until sentinel received
    while True:
        event = await queue.get()
        if event.get("type") == "_done_sentinel":
            break
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0)  # yield control to event loop


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=7734,
        reload=True,
        log_level="warning",
    )
