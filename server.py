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
    history: list = []  # recent messages [{role, content}] from the browser


class CustomAgentRequest(BaseModel):
    name: str
    prompt: str


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

            def patched_query_agent(agent, prompt, label=None):
                emit({"type": "progress", "message": f"💭 {label or agent.description + ' thinking...'}"})
                result = original_query_agent(agent, prompt, label=label)
                return result

            _orchestrator._query_agent = patched_query_agent

            # Monkey-patch reflection rating emit
            original_parse_rating = _orchestrator._parse_reflection_rating

            def patched_parse_rating(critique):
                rating = original_parse_rating(critique)
                emit({"type": "progress", "message": f"🔍 Reflection complete (rating: {rating})"})
                return rating

            _orchestrator._parse_reflection_rating = patched_parse_rating

            # Build context-aware query from recent history so short
            # follow-ups like "yes" carry full conversation context.
            # Strip code blocks from history to avoid poisoning non-code queries.
            query = req.message
            if req.history and len(req.history) > 0:
                import re as _re
                recent = req.history[-6:]  # last 3 exchanges (6 messages)
                context_lines = []
                for msg in recent:
                    role = "User" if msg.get("role") == "user" else "Rain"
                    content = msg.get("content", "")
                    # Strip fenced code blocks — keep prose only
                    content = _re.sub(r'```[\s\S]*?```', '[code block]', content)
                    content = content.strip()[:300]
                    if content:
                        context_lines.append(f"{role}: {content}")
                if context_lines:
                    context_block = "\n".join(context_lines)
                    query = f"[Recent conversation:\n{context_block}\n]\n\nCurrent question: {req.message}"

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
