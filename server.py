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
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

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
    _SKILLS_AVAILABLE,
    auto_pick_default_model,
)

# Phase 7: Project indexer — lazy import so Rain still works if indexer.py is absent
try:
    from indexer import ProjectIndexer
    _INDEXER_AVAILABLE = True
except ImportError:
    _INDEXER_AVAILABLE = False

# Phase 10: Knowledge graph — lazy import so Rain still works if knowledge_graph.py is absent
try:
    from knowledge_graph import KnowledgeGraph
    _KG_AVAILABLE = True
except ImportError:
    _KG_AVAILABLE = False

# Phase 6: Skills runtime
if _SKILLS_AVAILABLE:
    from skills import SkillLoader

# ── App ────────────────────────────────────────────────────────────────

# ── Global orchestrator (initialized once on startup) ──────────────────

_orchestrator: Optional[MultiAgentOrchestrator] = None
_memory: Optional[RainMemory] = None
_custom_agents: list = []  # [{id, name, prompt}]
_skill_loader: Optional['SkillLoader'] = None  # Phase 6


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Rain on startup, clean up on shutdown."""
    global _orchestrator, _memory, _skill_loader
    _default_model = auto_pick_default_model()
    _memory = RainMemory()
    _memory.start_session(model=_default_model)
    _orchestrator = MultiAgentOrchestrator(
        default_model=_default_model,
        memory=_memory,
        sandbox_enabled=False,  # user can toggle per request
    )
    # Phase 6: load skills (graceful — missing skills dir is fine)
    if _SKILLS_AVAILABLE:
        try:
            _skill_loader = SkillLoader()
            _skill_loader.load()
        except Exception:
            _skill_loader = None
    # Phase 7C: start background file watcher for indexed projects
    if _INDEXER_AVAILABLE:
        _start_file_watcher()
    yield
    # Phase 7C: stop file watcher
    _stop_file_watcher()
    if _memory:
        summary = _memory.generate_summary()
        _memory.end_session()
        if summary:
            _memory.update_summary(summary)
        # Phase 7: extract and persist structured facts from this session in background
        try:
            facts = _memory.extract_session_facts()
            if facts:
                _memory.save_session_facts(facts)
        except Exception:
            pass
        # Phase 10: extract architectural decisions from the session transcript
        if _KG_AVAILABLE:
            try:
                with sqlite3.connect(_memory.db_path) as _conn:
                    _conn.row_factory = sqlite3.Row
                    _rows = _conn.execute(
                        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                        (_memory.session_id,),
                    ).fetchall()
                if _rows:
                    _transcript = "\n".join(
                        f"{'User' if r['role'] == 'user' else 'Rain'}: {r['content'][:250]}"
                        for r in _rows
                    )
                    _kg = KnowledgeGraph()
                    _kg.extract_decisions_from_transcript(
                        _transcript,
                        session_id=_memory.session_id,
                    )
            except Exception:
                pass


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
    image_b64: Optional[str] = None      # base64-encoded image for multimodal queries (Phase 9)
    project_path: Optional[str] = None   # Phase 7: inject indexed project context into every agent


class CustomAgentRequest(BaseModel):
    name: str
    prompt: str


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: str  # 'good' or 'bad'
    correction: Optional[str] = None
    agent_type: Optional[str] = None   # which agent produced this response
    confidence: Optional[float] = None  # confidence score at time of response


# ── OpenAI-compatible request model (Phase 7 — IDE integration) ────────
class OpenAIMessage(BaseModel):
    role: str
    content: Optional[str] = None
    # Tool calling fields (OpenAI spec)
    tool_calls: Optional[List[dict]] = None      # assistant turn: list of tool calls requested
    tool_call_id: Optional[str] = None           # tool turn: which call this result belongs to
    name: Optional[str] = None                   # tool turn: tool name


class OpenAIChatRequest(BaseModel):
    """
    Subset of the OpenAI /v1/chat/completions request schema.

    Accepted by ZED, Continue.dev, Aider, and any tool that supports
    a custom OpenAI-compatible endpoint.  Set the base URL to:
        http://localhost:7734
    Any API key is accepted and silently ignored — Rain is local and free.
    """
    model: str = "rain"
    messages: List[OpenAIMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Tool calling (OpenAI spec) — when present, routes to Ollama's native
    # function-calling API instead of Rain's multi-agent pipeline
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None
    # Rain-specific extensions — non-standard clients ignore these
    web_search: bool = False
    project_path: Optional[str] = None


class IndexProjectRequest(BaseModel):
    """Request body for /api/index-project."""
    project_path: str
    force: bool = False   # if True, re-index files already in the DB


# ── Phase 7B/7C: Live data feeds (no API keys required) ───────────────

_MEMPOOL_FEE_KEYWORDS = frozenset({
    'mempool fee', 'fee rate', 'sat/vb', 'sat/byte', 'feerate',
    'transaction fee', 'mining fee', 'priority fee',
    'mempool', 'current fee', 'fastest fee', 'recommended fee',
})

_BTC_PRICE_KEYWORDS = frozenset({
    'bitcoin price', 'btc price', 'btc usd', 'bitcoin usd',
    'how much is bitcoin', 'how much is btc', 'bitcoin worth',
    'btc worth', 'bitcoin value', 'btc value', 'price of bitcoin',
    'price of btc', 'exchange rate', 'market price',
})


_GITHUB_KEYWORDS = frozenset({
    'github', 'repo', 'repository', 'open issues', 'recent commits',
    'pull request', 'pull requests', 'stars', 'forks', 'contributors',
    'github.com', 'readme', 'releases', 'latest release',
})

_GITHUB_REPO_RE = None  # compiled lazily


def _extract_github_repo(query: str) -> str | None:
    """
    Try to extract an owner/repo slug from the query.

    Matches patterns like:
      - github.com/owner/repo
      - owner/repo  (when surrounded by github keywords)
      - gh:owner/repo
    Returns 'owner/repo' or None.
    """
    import re
    global _GITHUB_REPO_RE
    if _GITHUB_REPO_RE is None:
        _GITHUB_REPO_RE = re.compile(
            r'(?:github\.com/|gh:)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)'
            r'|(?:^|\s)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:\s|$)',
        )
    m = _GITHUB_REPO_RE.search(query)
    if not m:
        return None
    slug = (m.group(1) or m.group(2) or "").strip().rstrip("/.")
    # Sanity: both parts must be non-empty, not a file path with extension
    parts = slug.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    # Reject things that look like file paths (e.g. "server.py/something")
    if "." in parts[0] and not parts[0].startswith("."):
        return None
    return slug


def _fetch_github_data(query: str) -> str:
    """
    Phase 7C: Fetch public repo data from the GitHub REST API.
    No API key required for public repos (rate limit: 60 req/hr per IP).

    Returns a formatted [GITHUB DATA] block or empty string.
    """
    import urllib.request
    import json as _json

    q = query.lower()

    # Only fire if the query mentions GitHub-related terms
    if not any(kw in q for kw in _GITHUB_KEYWORDS):
        return ""

    slug = _extract_github_repo(query)
    if not slug:
        return ""

    lines = [f"[GITHUB DATA — fetched just now for {slug}]"]
    headers = {"User-Agent": "Rain/1.0", "Accept": "application/vnd.github.v3+json"}

    # ── Repo metadata ──────────────────────────────────────────────
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{slug}",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            repo = _json.loads(resp.read().decode("utf-8"))
        lines.append(
            f"Repository: {repo.get('full_name', slug)}\n"
            f"  Description: {repo.get('description') or '(none)'}\n"
            f"  Language: {repo.get('language') or '?'}\n"
            f"  Stars: {repo.get('stargazers_count', '?'):,}  ·  Forks: {repo.get('forks_count', '?'):,}\n"
            f"  Open issues: {repo.get('open_issues_count', '?'):,}\n"
            f"  Default branch: {repo.get('default_branch', '?')}\n"
            f"  Created: {repo.get('created_at', '?')[:10]}  ·  Updated: {repo.get('updated_at', '?')[:10]}\n"
            f"  License: {(repo.get('license') or {}).get('spdx_id', 'none')}\n"
            f"  URL: https://github.com/{slug}"
        )
    except Exception as e:
        lines.append(f"Repo lookup failed: {e}")
        return ""  # if repo doesn't exist, bail out entirely

    # ── Recent open issues (top 5) ─────────────────────────────────
    want_issues = any(kw in q for kw in ('issue', 'issues', 'bug', 'bugs', 'problem'))
    if want_issues:
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{slug}/issues?state=open&per_page=5&sort=updated",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                issues = _json.loads(resp.read().decode("utf-8"))
            if issues:
                issue_lines = ["Recent open issues:"]
                for iss in issues:
                    if iss.get("pull_request"):
                        continue  # skip PRs that show up in /issues
                    num = iss.get("number", "?")
                    title = iss.get("title", "?")
                    labels = ", ".join(l.get("name", "") for l in iss.get("labels", []))
                    label_str = f"  [{labels}]" if labels else ""
                    issue_lines.append(f"  #{num}: {title}{label_str}")
                if len(issue_lines) > 1:
                    lines.append("\n".join(issue_lines))
        except Exception:
            pass

    # ── Recent commits (top 5) ─────────────────────────────────────
    want_commits = any(kw in q for kw in ('commit', 'commits', 'recent', 'latest', 'history', 'activity'))
    if want_commits:
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{slug}/commits?per_page=5",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                commits = _json.loads(resp.read().decode("utf-8"))
            if commits:
                commit_lines = ["Recent commits:"]
                for c in commits:
                    sha = c.get("sha", "?")[:7]
                    msg = (c.get("commit", {}).get("message", "") or "").split("\n")[0][:80]
                    author = (c.get("commit", {}).get("author", {}) or {}).get("name", "?")
                    date = ((c.get("commit", {}).get("author", {}) or {}).get("date", "") or "")[:10]
                    commit_lines.append(f"  {sha} {date} ({author}): {msg}")
                lines.append("\n".join(commit_lines))
        except Exception:
            pass

    # ── Pull requests (if asked) ───────────────────────────────────
    want_prs = any(kw in q for kw in ('pull request', 'pull requests', 'pr', 'prs', 'merge'))
    if want_prs:
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{slug}/pulls?state=open&per_page=5&sort=updated",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                prs = _json.loads(resp.read().decode("utf-8"))
            if prs:
                pr_lines = ["Open pull requests:"]
                for pr in prs:
                    num = pr.get("number", "?")
                    title = pr.get("title", "?")
                    user = (pr.get("user") or {}).get("login", "?")
                    pr_lines.append(f"  #{num}: {title} (by {user})")
                lines.append("\n".join(pr_lines))
        except Exception:
            pass

    # ── Latest release (if asked) ──────────────────────────────────
    want_release = any(kw in q for kw in ('release', 'releases', 'version', 'latest version', 'tag'))
    if want_release:
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{slug}/releases/latest",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                rel = _json.loads(resp.read().decode("utf-8"))
            tag = rel.get("tag_name", "?")
            name = rel.get("name", "")
            date = (rel.get("published_at") or "")[:10]
            lines.append(f"Latest release: {tag}{f' — {name}' if name and name != tag else ''} ({date})")
        except Exception:
            pass

    return "\n\n".join(lines) if len(lines) > 1 else ""


def _fetch_live_data(query: str) -> str:
    """
    Phase 7B/7C: Fetch live data from public APIs (no API keys) when the query
    is about something that changes in real time.

    Currently supported:
      - Mempool fee rates  → mempool.space/api/v1/fees/recommended
      - Bitcoin price      → mempool.space/api/v1/prices
      - GitHub repo data   → api.github.com/repos/{owner}/{repo}

    Returns a formatted live-data block ready to prepend to the agent context,
    or an empty string if no live feed matches or the request fails.
    """
    import urllib.request

    q = query.lower()

    want_fees  = any(kw in q for kw in _MEMPOOL_FEE_KEYWORDS)
    want_price = any(kw in q for kw in _BTC_PRICE_KEYWORDS)

    # Phase 7C: GitHub data
    github_block = _fetch_github_data(query)

    if not (want_fees or want_price) and not github_block:
        return ""

    # Build a descriptive header based on which sources we're pulling from
    sources = []
    if want_fees or want_price:
        sources.append("mempool.space")
    if github_block:
        sources.append("GitHub API")
    lines = [f"[LIVE DATA — fetched just now from {' + '.join(sources)}]"]

    if want_fees:
        try:
            req = urllib.request.Request(
                "https://mempool.space/api/v1/fees/recommended",
                headers={"User-Agent": "Rain/1.0"},
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                import json as _json
                data = _json.loads(resp.read().decode("utf-8"))
            lines.append(
                f"Current Bitcoin mempool fee rates (sat/vB):\n"
                f"  Fastest (next block): {data.get('fastestFee', '?')} sat/vB\n"
                f"  Half-hour:            {data.get('halfHourFee', '?')} sat/vB\n"
                f"  One hour:             {data.get('hourFee', '?')} sat/vB\n"
                f"  Economy:              {data.get('economyFee', '?')} sat/vB\n"
                f"  Minimum:              {data.get('minimumFee', '?')} sat/vB\n"
                f"Source: mempool.space/api/v1/fees/recommended"
            )
        except Exception as e:
            lines.append(f"Fee rate lookup failed: {e} — answer from training data instead.")

    if want_price:
        try:
            req = urllib.request.Request(
                "https://mempool.space/api/v1/prices",
                headers={"User-Agent": "Rain/1.0"},
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                import json as _json
                data = _json.loads(resp.read().decode("utf-8"))
            usd = data.get("USD", "?")
            lines.append(
                f"Current Bitcoin price:\n"
                f"  USD: ${usd:,}\n"
                f"Source: mempool.space/api/v1/prices"
            )
        except Exception as e:
            lines.append(f"Price lookup failed: {e} — answer from training data instead.")

    # Phase 7C: append GitHub data if present
    if github_block:
        lines.append(github_block)

    return "\n\n".join(lines) if len(lines) > 1 else ""


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


@app.get("/api/skills")
async def get_skills():
    """
    Phase 6: List all installed skills.
    Returns skill metadata for the web UI skills panel.
    """
    if not _SKILLS_AVAILABLE or not _skill_loader:
        return {"skills": [], "count": 0, "skills_dir": str(SkillLoader.GLOBAL_SKILLS_DIR) if _SKILLS_AVAILABLE else "~/.rain/skills"}
    skills = []
    for s in _skill_loader.skills:
        skills.append({
            "name": s.name,
            "slug": s.slug,
            "description": s.description,
            "tags": s.tags,
            "env_satisfied": s.env_satisfied,
            "primary_env": s.primary_env,
            "source": s.source_label,
        })
    return {
        "skills": skills,
        "count": len(skills),
        "skills_dir": str(SkillLoader.GLOBAL_SKILLS_DIR),
    }


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

    # Summarize and extract facts from the ended session in the background
    def _summarize_in_background():
        summary = ending_memory.generate_summary()
        if summary:
            ending_memory.update_summary(summary)
        try:
            facts = ending_memory.extract_session_facts()
            if facts:
                ending_memory.save_session_facts(facts)
        except Exception:
            pass
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
        agent_type=req.agent_type,
        confidence=req.confidence,
    )
    # Attach rating to synthesis_log if this query was synthesized — lets us
    # track whether the synthesizer is actually improving responses over time.
    try:
        import hashlib
        qhash = hashlib.md5(req.query.encode()).hexdigest()
        _memory.update_synthesis_rating(qhash, req.rating)
    except Exception:
        pass
    # Refresh calibration factors so the next request benefits immediately
    if _orchestrator:
        try:
            _orchestrator._calibration_factors = _memory.get_calibration_factors()
        except Exception:
            pass
    return {"status": "ok"}


@app.post("/api/forget")
async def forget():
    """Wipe all memory."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    _memory.forget_all()
    return {"status": "ok", "message": "All memory wiped."}


# ── Phase 7: Project indexing ──────────────────────────────────────────

# ── Phase 7C: Background file watcher ─────────────────────────────────

_watcher_thread: Optional[threading.Thread] = None
_watcher_stop = threading.Event()
_WATCHER_INTERVAL = 60  # seconds between checks


def _file_watcher_loop():
    """
    Background thread that checks indexed projects for changed files
    and re-indexes them automatically.
    """
    import os
    from datetime import datetime as _dt

    while not _watcher_stop.is_set():
        _watcher_stop.wait(_WATCHER_INTERVAL)
        if _watcher_stop.is_set():
            break
        if not _INDEXER_AVAILABLE:
            continue
        try:
            idx = ProjectIndexer()
            projects = idx.list_indexed_projects()
            for proj in projects:
                ppath = proj["project_path"]
                if not Path(ppath).is_dir():
                    continue
                changed = idx.get_changed_files(ppath)
                for fpath in changed:
                    try:
                        idx.reindex_file(ppath, fpath)
                    except Exception:
                        pass
        except Exception:
            pass


def _start_file_watcher():
    """Start the background file watcher thread (idempotent)."""
    global _watcher_thread
    if _watcher_thread and _watcher_thread.is_alive():
        return
    _watcher_stop.clear()
    _watcher_thread = threading.Thread(target=_file_watcher_loop, daemon=True)
    _watcher_thread.start()


def _stop_file_watcher():
    """Signal the file watcher to stop."""
    _watcher_stop.set()


@app.post("/api/index-project")
async def index_project(req: IndexProjectRequest):
    """
    Index a project directory for semantic search (Phase 7).

    Walks the project tree, splits files into chunks, embeds each chunk
    with nomic-embed-text, and stores the vectors in the project_index
    table of Rain's SQLite database.

    This can take a few minutes for large projects.  The endpoint blocks
    until indexing is complete and returns a summary of what was indexed.
    For very large codebases, run the CLI indexer in a terminal instead:
        python3 indexer.py --index /path/to/project
    """
    if not _INDEXER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="indexer.py not found in Rain directory. "
                   "Make sure indexer.py is alongside server.py.",
        )
    try:
        loop = asyncio.get_event_loop()
        idx = ProjectIndexer()
        stats = await loop.run_in_executor(
            None,
            lambda: idx.index_project(req.project_path, force=req.force),
        )
        if "error" in stats:
            raise HTTPException(status_code=400, detail=stats["error"])
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/indexed-projects")
async def get_indexed_projects():
    """Return metadata for all projects that have been indexed."""
    if not _INDEXER_AVAILABLE:
        return {"projects": [], "indexer_available": False}
    try:
        idx = ProjectIndexer()
        projects = idx.list_indexed_projects()
        return {"projects": projects, "indexer_available": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/indexed-projects")
async def remove_indexed_project(project_path: str):
    """Remove a project from the semantic index."""
    if not _INDEXER_AVAILABLE:
        raise HTTPException(status_code=503, detail="indexer.py not available")
    try:
        idx = ProjectIndexer()
        n = idx.remove_project(project_path)
        return {"status": "ok", "chunks_removed": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/indexed-projects/{project_path:path}/changed")
async def get_changed_files(project_path: str):
    """Phase 7C: Return list of files that have changed since last index."""
    if not _INDEXER_AVAILABLE:
        raise HTTPException(status_code=503, detail="indexer.py not available")
    try:
        idx = ProjectIndexer()
        resolved = str(Path(project_path).resolve())
        changed = idx.get_changed_files(resolved)
        return {"project_path": resolved, "changed_files": changed, "count": len(changed)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 10: Knowledge Graph & Deep Project Intelligence ─────────────

class BuildGraphRequest(BaseModel):
    """Request body for /api/build-graph."""
    project_path: str
    force: bool = False


class LogDecisionRequest(BaseModel):
    """Request body for /api/decisions."""
    title: str
    description: str = ""
    project_path: Optional[str] = None
    context: Optional[str] = None
    alternatives: Optional[str] = None
    rationale: Optional[str] = None
    tags: Optional[str] = None


@app.post("/api/build-graph")
async def build_graph(req: BuildGraphRequest):
    """
    Phase 10: Build a knowledge graph for a project directory.

    Parses every code file (Python via AST, JS/TS/Rust/Go via regex),
    extracts functions, classes, methods, imports, and call relationships,
    and stores them in the kg_nodes / kg_edges tables in Rain's SQLite DB.
    Also indexes git history if the project is a git repo.
    """
    if not _KG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="knowledge_graph.py not found. Make sure it is alongside server.py.",
        )
    try:
        loop = asyncio.get_event_loop()
        kg = KnowledgeGraph()
        stats = await loop.run_in_executor(
            None,
            lambda: kg.build_graph(req.project_path, force=req.force),
        )
        if "error" in stats:
            raise HTTPException(status_code=400, detail=stats["error"])
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/onboard-project")
async def onboard_project_full(req: BuildGraphRequest):
    """
    Phase 10: Full project onboarding — build graph + generate LLM summary.
    This is the 'drop a new project path and Rain understands it' feature.
    """
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        loop = asyncio.get_event_loop()
        kg = KnowledgeGraph()
        result = await loop.run_in_executor(
            None,
            lambda: kg.onboard_project(req.project_path, force=req.force),
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/stats")
async def graph_stats(project_path: str):
    """Phase 10: Return knowledge graph statistics for a project."""
    if not _KG_AVAILABLE:
        return {"error": "knowledge_graph.py not available", "kg_available": False}
    try:
        kg = KnowledgeGraph()
        stats = kg.get_project_stats(project_path)
        stats["kg_available"] = True
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/summary")
async def graph_summary(project_path: str):
    """Phase 10: Return the stored LLM-generated project summary."""
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        kg = KnowledgeGraph()
        summary = kg.get_project_summary(project_path)
        return {"project_path": project_path, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/find")
async def graph_find(project_path: str, name: Optional[str] = None, node_type: Optional[str] = None):
    """Phase 10: Find nodes in the knowledge graph by name and/or type."""
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        kg = KnowledgeGraph()
        nodes = kg.find_nodes(project_path, name=name, node_type=node_type)
        return {"nodes": nodes, "count": len(nodes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/callers")
async def graph_callers(project_path: str, function_name: str):
    """Phase 10: Find all callers of a function."""
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        kg = KnowledgeGraph()
        callers = kg.get_callers(project_path, function_name)
        return {"function": function_name, "callers": callers, "count": len(callers)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/callees")
async def graph_callees(project_path: str, function_name: str):
    """Phase 10: Find all functions called by a given function."""
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        kg = KnowledgeGraph()
        callees = kg.get_callees(project_path, function_name)
        return {"function": function_name, "callees": callees, "count": len(callees)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/file-structure")
async def graph_file_structure(project_path: str, file_path: str):
    """Phase 10: Return structure summary of a file (functions, classes, imports)."""
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        kg = KnowledgeGraph()
        return kg.get_file_structure(project_path, file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/history")
async def graph_git_history(project_path: str, file_path: Optional[str] = None, n: int = 20):
    """Phase 10: Return git commit history for a project or specific file."""
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        kg = KnowledgeGraph()
        commits = kg.get_git_history(project_path, file_path=file_path, n=n)
        return {"commits": commits, "count": len(commits)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions")
async def list_decisions(project_path: Optional[str] = None):
    """Phase 10: List architectural decisions, optionally filtered by project."""
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        kg = KnowledgeGraph()
        decisions = kg.list_decisions(project_path=project_path)
        return {"decisions": decisions, "count": len(decisions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decisions")
async def create_decision(req: LogDecisionRequest):
    """Phase 10: Log a new architectural decision."""
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        kg = KnowledgeGraph()
        dec_id = kg.log_decision(
            title=req.title,
            description=req.description,
            project_path=req.project_path,
            context=req.context,
            alternatives=req.alternatives,
            rationale=req.rationale,
            tags=req.tags,
        )
        return {"status": "ok", "id": dec_id, "title": req.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions/search")
async def search_decisions_endpoint(query: str, project_path: Optional[str] = None):
    """Phase 10: Search decisions by keyword."""
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        kg = KnowledgeGraph()
        results = kg.search_decisions(query, project_path=project_path)
        return {"decisions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/cross-project")
async def cross_project_search(query: str, exclude_project: Optional[str] = None):
    """Phase 10: Search across all projects for similar patterns."""
    if not _KG_AVAILABLE:
        raise HTTPException(status_code=503, detail="knowledge_graph.py not available")
    try:
        kg = KnowledgeGraph()
        results = kg.find_similar_patterns(query, exclude_project=exclude_project)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 7: OpenAI-compatible endpoint (IDE integration) ─────────────

@app.post("/v1/chat/completions")
async def openai_chat_completions(req: OpenAIChatRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Makes Rain usable as a custom AI provider in any tool that supports
    a custom OpenAI-compatible API base URL.

    Configuration for popular tools:

    ZED (settings.json):
        "assistant": {
            "version": "2",
            "default_model": {
                "provider": "openai",
                "model": "rain"
            },
            "openai_api_url": "http://localhost:7734/v1"
        }

    Continue.dev (~/.continue/config.json):
        "models": [{
            "title": "Rain",
            "provider": "openai",
            "model": "rain",
            "apiBase": "http://localhost:7734/v1",
            "apiKey": "local"
        }]

    Aider (terminal):
        aider --openai-api-base http://localhost:7734/v1 --model rain

    OpenAI Python SDK:
        import openai
        client = openai.OpenAI(base_url="http://localhost:7734/v1", api_key="local")

    API key: set anything — it is accepted and silently ignored.
    Rain is local and free.
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    # Pull last user message and optional system message
    user_message = ""
    for msg in req.messages:
        if msg.role == "user":
            user_message = msg.content  # keep iterating — use the last one

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in messages array")

    # ── Context injection ─────────────────────────────────────────────
    # Two paths, same goal: enrich the query with project index + memory
    # before it hits Rain's local pipeline.
    #
    # Path A — client sent MCP tool definitions (ZED agent, Continue.dev
    #           with tools configured): execute them proactively and inject
    #           results.  No tool_calls round-trip needed.
    #
    # Path B — plain chat request, no tool definitions (ZED inline AI,
    #           most IDE integrations): auto-inject context directly so
    #           the local model isn't answering blind.
    if req.tools:
        user_message = _execute_rain_tools(req.tools, user_message)
    else:
        user_message = _auto_inject_project_context(user_message)

    if req.stream:
        return StreamingResponse(
            _openai_stream(user_message, req.web_search),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Non-streaming ─────────────────────────────────────────────────
    try:
        loop = asyncio.get_event_loop()
        query = _maybe_augment_with_search(user_message, req.web_search)
        result = await loop.run_in_executor(
            None,
            lambda: _orchestrator.recursive_reflect(query),
        )
        content = result.content if result else "I was unable to generate a response."
        created = int(time.time())
        return {
            "id": f"chatcmpl-rain-{created}",
            "object": "chat.completion",
            "created": created,
            "model": "rain",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens":     len(user_message.split()),
                "completion_tokens": len(content.split()) if content else 0,
                "total_tokens":      len(user_message.split()) + (len(content.split()) if content else 0),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _ollama_tool_call(req: "OpenAIChatRequest") -> dict:
    """
    Handle OpenAI tool-calling requests by forwarding to Ollama's native
    function-calling API. Ollama supports this for llama3.1, llama3.2,
    mistral, qwen2.5, and other models that understand function call syntax.

    Flow:
      Turn 1 — user message + tools → Ollama may return tool_calls
      Turn 2 — tool results added   → Ollama generates final answer

    Returns a complete OpenAI-format chat.completion object.
    """
    import urllib.request as _urllib
    import json as _json

    created  = int(time.time())
    chat_id  = f"chatcmpl-rain-{created}"
    model    = _orchestrator.default_model if _orchestrator else "llama3.2"

    # Convert OpenAI messages → Ollama format
    ollama_messages = []
    for msg in req.messages:
        if msg.role == "tool":
            # Tool result turn
            ollama_messages.append({
                "role":         "tool",
                "content":      msg.content or "",
                "tool_call_id": msg.tool_call_id or "",
            })
        elif msg.tool_calls:
            # Assistant turn that requested tool calls
            ollama_messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": msg.tool_calls,
            })
        else:
            ollama_messages.append({
                "role":    msg.role,
                "content": msg.content or "",
            })

    # Convert OpenAI tool schema → Ollama tool schema (identical structure)
    ollama_tools = req.tools or []

    # Inject a system message that forces structured tool_calls output.
    # Small models (llama3.2 7B) tend to describe tool calls in prose instead
    # of emitting the structured format — this instruction overrides that.
    tool_names = [t.get("function", {}).get("name", "") for t in ollama_tools]
    tool_instruction = {
        "role": "system",
        "content": (
            "You have access to the following tools: "
            + ", ".join(tool_names)
            + ". When the user's request requires a tool, you MUST call it using "
            "the structured tool_calls format. Do NOT describe what you would do "
            "or show JSON code blocks — invoke the tool directly. "
            "If no tool is needed, answer normally."
        ),
    }
    # Prepend instruction only if no system message already exists
    has_system = any(m.get("role") == "system" for m in ollama_messages)
    if not has_system:
        ollama_messages = [tool_instruction] + ollama_messages

    payload = _json.dumps({
        "model":    model,
        "messages": ollama_messages,
        "tools":    ollama_tools,
        "stream":   False,
        "options":  {"temperature": 0.1, "num_ctx": 16384},
    }).encode()

    def _call():
        r = _urllib.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urllib.urlopen(r, timeout=120) as resp:
            return _json.loads(resp.read().decode())

    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, _call)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    msg = data.get("message", {})

    # ── Prose fallback — model described the call instead of using tool_calls ──
    # Some small models emit JSON in a code block instead of structured output.
    # Parse it out and construct the tool_calls response manually.
    if not msg.get("tool_calls") and msg.get("content"):
        content_text = msg["content"]
        # Look for a JSON block that names one of our tools
        import re as _re
        json_match = _re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', content_text, _re.DOTALL)
        if not json_match:
            # Also try bare JSON object
            json_match = _re.search(r'(\{[^}]*"name"\s*:\s*"[^"]+[^}]*\})', content_text, _re.DOTALL)
        if json_match:
            try:
                parsed = _json.loads(json_match.group(1))
                fn_name = parsed.get("name") or parsed.get("function", {}).get("name", "")
                fn_args = parsed.get("parameters") or parsed.get("arguments") or parsed.get("args") or {}
                known_names = {t.get("function", {}).get("name", "") for t in ollama_tools}
                if fn_name in known_names:
                    msg = {
                        "tool_calls": [{
                            "function": {"name": fn_name, "arguments": fn_args}
                        }]
                    }
            except (_json.JSONDecodeError, AttributeError):
                pass

    # ── Tool calls requested by the model ────────────────────────────
    if msg.get("tool_calls"):
        openai_tool_calls = []
        for i, tc in enumerate(msg["tool_calls"]):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            openai_tool_calls.append({
                "id":       f"call_{i}_{created}",
                "type":     "function",
                "function": {
                    "name":      fn.get("name", ""),
                    "arguments": _json.dumps(args) if isinstance(args, dict) else str(args),
                },
            })
        return {
            "id": chat_id, "object": "chat.completion",
            "created": created, "model": "rain",
            "choices": [{
                "index": 0,
                "message": {
                    "role":       "assistant",
                    "content":    None,
                    "tool_calls": openai_tool_calls,
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # ── Final answer (no tool calls — either direct or after tool results) ──
    content = msg.get("content", "")
    return {
        "id": chat_id, "object": "chat.completion",
        "created": created, "model": "rain",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     sum(len((m.content or "").split()) for m in req.messages),
            "completion_tokens": len(content.split()),
            "total_tokens":      sum(len((m.content or "").split()) for m in req.messages) + len(content.split()),
        },
    }


async def _ollama_tool_call_stream(req: "OpenAIChatRequest") -> AsyncGenerator[str, None]:
    """
    Streaming variant of _ollama_tool_call.
    For tool_calls turns: emits the tool call as compact SSE chunks then [DONE].
    For final answer turns: streams content character-by-character.
    """
    result = await _ollama_tool_call(req)
    created = result["created"]
    chat_id = result["id"]
    choice  = result["choices"][0]

    if choice["finish_reason"] == "tool_calls":
        # Emit role chunk
        yield f'data: {json.dumps({"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": "rain", "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]})}\n\n'
        # Emit each tool call as a chunk
        for tc in choice["message"].get("tool_calls", []):
            yield f'data: {json.dumps({"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": "rain", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": tc["id"].split("_")[1], "id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}]}, "finish_reason": None}]})}\n\n'
        # Final stop chunk
        yield f'data: {json.dumps({"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": "rain", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})}\n\n'
    else:
        # Stream final content in pieces
        content   = choice["message"].get("content", "")
        chunk_size = 12
        yield f'data: {json.dumps({"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": "rain", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})}\n\n'
        for i in range(0, len(content), chunk_size):
            piece = content[i:i + chunk_size]
            yield f'data: {json.dumps({"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": "rain", "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})}\n\n'
        yield f'data: {json.dumps({"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": "rain", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n'

    yield "data: [DONE]\n\n"


# ── Known Rain MCP tool names ─────────────────────────────────────────────
_RAIN_TOOL_NAMES = {"search_project", "get_user_memory", "index_project", "list_indexed_projects"}


def _auto_inject_project_context(user_message: str) -> str:
    """
    Always enrich the user message with project index context and user memory
    before sending to Rain's local pipeline via the OpenAI endpoint.

    This fires when the client (ZED inline AI, Continue.dev, etc.) sends a
    plain chat request WITHOUT MCP tool definitions — meaning _execute_rain_tools
    never runs.  Without this, the local model has no project knowledge and
    guesses blindly.

    Injects:
      1. Tier 5 user memory facts (what Rain knows about you)
      2. Top-6 semantically relevant project index chunks for the query
    """
    if not _INDEXER_AVAILABLE:
        return user_message

    context_parts = []

    # ── User memory (Tier 5) ──────────────────────────────────────────
    if _memory:
        try:
            facts = _memory.get_fact_context()
            if facts and facts.strip():
                context_parts.append(facts.strip())
        except Exception:
            pass

    # ── Project index semantic search ─────────────────────────────────
    try:
        idx = ProjectIndexer()
        projects = idx.list_indexed_projects()
        project_path = None

        if len(projects) == 1:
            project_path = projects[0]["project_path"]
        elif len(projects) > 1:
            # Try to find a path hint in the user message
            import re as _re
            m = _re.search(r"(/(?:Users|home|workspace|projects?|code)/[^\s\"']+)", user_message)
            if m:
                project_path = m.group(1)

        if project_path:
            block = idx.build_context_block(user_message, project_path, top_k=6)
            if block:
                context_parts.append(block)
    except Exception:
        pass

    if not context_parts:
        return user_message

    injected = "\n\n".join(context_parts)
    return (
        f"{injected}\n\n"
        f"Answer the following question using only what the above context actually states. "
        f"Do not speculate about features that aren't mentioned. "
        f"If the context covers the topic, cite specific details. "
        f"If the context doesn't cover something, say so rather than guessing.\n\n"
        f"---\n\n"
        f"{user_message}"
    )


def _execute_rain_tools(tools: List[dict], user_message: str) -> str:
    """
    When Rain's own MCP tools appear in an OpenAI tools request, execute them
    directly and inject results into the user message as context.

    This replaces the tool_calls round-trip entirely:
    - No SSE format issues (ZED's parser rejects tool_calls chunks)
    - No model reliability issues (small models describe calls instead of making them)
    - Results arrive in the same response, not a second round trip

    Only Rain's known tools are executed. Unknown tools are ignored.
    """
    request_tool_names = {
        t.get("function", {}).get("name", "")
        for t in (tools or [])
    }
    active = request_tool_names & _RAIN_TOOL_NAMES
    if not active:
        return user_message

    context_parts = []

    # get_user_memory — pull stored profile facts from RainMemory
    if "get_user_memory" in active and _memory:
        try:
            facts = _memory.get_fact_context()
            if facts and facts.strip():
                context_parts.append(facts.strip())
        except Exception:
            pass

    # list_indexed_projects — return indexed project metadata
    if "list_indexed_projects" in active and _INDEXER_AVAILABLE:
        try:
            idx = ProjectIndexer()
            projects = idx.list_indexed_projects()
            if projects:
                lines = [
                    f"  • {p['project_path']}  ({p['file_count']} files, {p['chunk_count']} chunks)"
                    for p in projects
                ]
                context_parts.append("[Indexed projects]\n" + "\n".join(lines))
        except Exception:
            pass

    # search_project — semantic search if a project is already indexed
    # Only fires when there is exactly one indexed project (unambiguous) or
    # when the user message contains a path hint.
    if "search_project" in active and _INDEXER_AVAILABLE:
        try:
            idx = ProjectIndexer()
            projects = idx.list_indexed_projects()
            project_path = None
            if len(projects) == 1:
                project_path = projects[0]["project_path"]
            else:
                # Try to find a path hint in the user message
                import re as _re
                m = _re.search(r"(/(?:Users|home|workspace|projects?|code)/[^\s\"']+)", user_message)
                if m:
                    project_path = m.group(1)
            if project_path:
                block = idx.build_context_block(user_message, project_path, top_k=4)
                if block:
                    context_parts.append(block)
        except Exception:
            pass

    if not context_parts:
        return user_message

    injected = "\n\n".join(context_parts)
    return (
        f"{injected}\n\n"
        f"Using the above context, answer this question accurately and specifically:\n\n"
        f"---\n\n"
        f"{user_message}"
    )


def _maybe_augment_with_search(message: str, web_search: bool) -> str:
    """
    If web_search is True, fetch live data + DuckDuckGo results and prepend
    them to the message so the Search Agent can synthesize a grounded answer.
    Phase 7B: live data feeds checked first for structured real-time numbers.
    """
    if not web_search:
        return message

    live_block = _fetch_live_data(message)
    results = _duckduckgo_search(message, max_results=5)

    if not live_block and not results:
        return message

    context_parts = []
    if live_block:
        context_parts.append(live_block)
    if results:
        snippets = "\n\n".join(
            f"[{r['title']}]\n{r['snippet']}\nSource: {r['url']}"
            for r in results
        )
        context_parts.append(f"[Web search results for: {message}]\n\n{snippets}")

    combined = "\n\n".join(context_parts)
    return (
        f"{combined}\n\n---\n"
        f"Using the above live data and search results as context, answer accurately. "
        f"The LIVE DATA block contains real-time numbers — use those figures directly. "
        f"Cite sources where relevant.\n\nQuestion: {message}"
    )


async def _openai_stream(
    user_message: str,
    web_search: bool,
) -> AsyncGenerator[str, None]:
    """
    Run Rain's pipeline and stream the final response content in OpenAI SSE
    format.  IDEs like ZED and Continue.dev consume this for live typewriter
    output.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    chat_id = f"chatcmpl-rain-{int(time.time())}"
    created = int(time.time())

    def run():
        try:
            query = _maybe_augment_with_search(user_message, web_search)
            result = _orchestrator.recursive_reflect(query)
            content = result.content if result else ""
            loop.call_soon_threadsafe(queue.put_nowait, ("content", content))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    threading.Thread(target=run, daemon=True).start()

    # Send the role delta first (OpenAI convention)
    role_chunk = {
        "id": chat_id, "object": "chat.completion.chunk",
        "created": created, "model": "rain",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    while True:
        kind, payload = await queue.get()
        if kind == "done":
            break
        if kind == "error":
            err = {
                "id": chat_id, "object": "chat.completion.chunk",
                "created": created, "model": "rain",
                "choices": [{"index": 0, "delta": {"content": f"\n\n[Error: {payload}]"}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(err)}\n\n"
            break
        if kind == "content" and payload:
            # Emit content in small pieces for a natural typewriter feel
            chunk_size = 12
            for i in range(0, len(payload), chunk_size):
                piece = payload[i:i + chunk_size]
                chunk = {
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": "rain",
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)

    # Final stop chunk
    stop_chunk = {
        "id": chat_id, "object": "chat.completion.chunk",
        "created": created, "model": "rain",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


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

            def patched_query_agent(agent, prompt, label=None, include_memory=True, image_b64=None):
                emit({"type": "progress", "message": f"💭 {label or agent.description + ' thinking...'}"})
                result = original_query_agent(agent, prompt, label=label, include_memory=include_memory, image_b64=image_b64)
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
            live_block = ""
            if req.web_search:
                emit({"type": "progress", "message": "🌐 Searching the web..."})

                # Phase 7B: try live data feeds first — these return structured
                # real-time numbers that DuckDuckGo snippets can never provide.
                live_block = _fetch_live_data(req.message)
                if live_block:
                    if "GITHUB DATA" in live_block and "mempool" not in live_block.lower().split("github")[0]:
                        emit({"type": "progress", "message": "⚡ Live data retrieved from GitHub API"})
                    elif "GITHUB DATA" in live_block:
                        emit({"type": "progress", "message": "⚡ Live data retrieved from mempool.space + GitHub API"})
                    else:
                        emit({"type": "progress", "message": "⚡ Live data retrieved from mempool.space"})

                search_results = _duckduckgo_search(req.message, max_results=5)
                search_results_count = len(search_results)
                if search_results or live_block:
                    snippets = "\n\n".join(
                        f"[{r['title']}]\n{r['snippet']}\nSource: {r['url']}"
                        for r in search_results
                    )
                    if search_results:
                        emit({"type": "progress", "message": f"🌐 {search_results_count} results retrieved"})

                    # Live data block goes first — it's more precise than snippets
                    context_parts = []
                    if live_block:
                        context_parts.append(live_block)
                    if snippets:
                        context_parts.append(f"[Web search results for: {req.message}]\n\n{snippets}")

                    combined_context = "\n\n".join(context_parts)
                    search_augmented_message = (
                        f"{combined_context}\n\n"
                        f"---\n"
                        f"Using the above live data and search results as context, answer this question accurately. "
                        f"Cite sources where relevant. The LIVE DATA block contains real-time numbers — "
                        f"use those figures directly rather than saying you don't know the current value.\n\n"
                        f"Question: {req.message}"
                    )
                else:
                    emit({"type": "progress", "message": "🌐 No results found — using local knowledge"})

            # ── Project context (Phase 7) ─────────────────────────────────────
            project_context_block = ""
            if req.project_path and _INDEXER_AVAILABLE:
                try:
                    emit({"type": "progress", "message": f"📂 Searching project index: {req.project_path.split('/')[-1]}..."})
                    _idx = ProjectIndexer()
                    project_context_block = _idx.build_context_block(
                        req.message, req.project_path, top_k=4
                    )
                    if project_context_block:
                        emit({"type": "progress", "message": "📂 Project context injected"})
                    else:
                        emit({"type": "progress", "message": "📂 No relevant project context found — index the project first"})
                except Exception as _idx_err:
                    emit({"type": "progress", "message": f"📂 Project index error: {_idx_err}"})

            # ── Knowledge graph context (Phase 10) ────────────────────────────
            kg_context_block = ""
            if req.project_path and _KG_AVAILABLE:
                try:
                    _kg = KnowledgeGraph()
                    kg_context_block = _kg.build_context_block(req.message, req.project_path)
                    if kg_context_block:
                        emit({"type": "progress", "message": "🧠 Knowledge graph context injected"})
                except Exception:
                    pass

            # Memory system (working memory = last 20 messages) already handles
            # continuity — no additional history injection needed here.
            base_message = search_augmented_message if search_augmented_message else req.message

            # Combine all context blocks: project index + knowledge graph
            combined_context = ""
            if project_context_block:
                combined_context += project_context_block
            if kg_context_block:
                if combined_context:
                    combined_context += "\n\n"
                combined_context += kg_context_block

            if combined_context:
                query = f"{combined_context}\n\n---\n\n{base_message}"
            else:
                query = base_message

            # Emit vision notice if image is attached
            if req.image_b64:
                vision_model = _orchestrator._best_vision_model()
                if vision_model:
                    is_large = any(x in vision_model for x in ['llama3.2-vision', 'llava:13b', 'llava:34b', 'minicpm', 'qwen'])
                    slow_note = " (this may take 1–3 min on first run)" if is_large else ""
                    emit({"type": "progress", "message": f"👁️ Analysing image with {vision_model}...{slow_note}"})
                else:
                    emit({"type": "progress", "message": "⚠️ No vision model installed — run: ollama pull llama3.2-vision  (or: ollama pull llava:7b for a lighter option)"})

            # Propagate active project path onto the orchestrator so
            # _build_memory_context can proactively query the knowledge graph.
            if req.project_path:
                _orchestrator.project_path = req.project_path

            # Auto-detect react vs reflect — ReAct wins when the query signals
            # real-world discovery (filesystem, git, logs) AND tools are loaded.
            _mode = _orchestrator._auto_mode(query, image_b64=req.image_b64)
            if _mode == 'react':
                emit({"type": "progress", "message": f"⚡ Auto-selected ReAct mode (discovery query detected)"})
                result = _orchestrator.react_loop(query, verbose=req.verbose)
            else:
                result = _orchestrator.recursive_reflect(query, verbose=req.verbose, image_b64=req.image_b64)

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

                # Phase 7C/10: build data_sources list for freshness indicators
                data_sources = []
                if req.web_search and search_results_count:
                    data_sources.append("web_search")
                if req.web_search and live_block:
                    data_sources.append("live_api")
                if project_context_block:
                    data_sources.append("project_index")
                if kg_context_block:
                    data_sources.append("knowledge_graph")
                if req.image_b64:
                    data_sources.append("vision")
                if not data_sources:
                    data_sources.append("training_data")

                emit({
                    "type": "done",
                    "content": result.content,
                    "confidence": round(result.confidence, 2),
                    "duration": round(result.duration_seconds, 1),
                    "iterations": result.iteration,
                    "improvements": result.improvements,
                    "sandbox": sandbox_summary,
                    "search_results": search_results_count if req.web_search else 0,
                    "vision_used": bool(req.image_b64),
                    "data_sources": data_sources,
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
