#!/usr/bin/env python3
"""
Rain ⛈️ - Sovereign AI Orchestrator

The brain of the Rain ecosystem that manages recursive reflection
and multi-agent AI interactions completely offline.

"Be like rain - essential, unstoppable, and free."
"""

import json
import re
import shutil
import subprocess
import tempfile
import time
import argparse
import sys
import signal
import difflib
import threading
import sqlite3
import uuid
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Phase 6: Skills runtime and tool registry
# Imported lazily-safe — if the files don't exist yet, Rain still works
try:
    from skills import SkillLoader, install_skill as _install_skill
    _SKILLS_AVAILABLE = True
except ImportError:
    _SKILLS_AVAILABLE = False

try:
    from tools import ToolRegistry, interactive_confirm as _interactive_confirm
    _TOOLS_AVAILABLE = True
except ImportError:
    _TOOLS_AVAILABLE = False

# Phase 7: Project indexer — lazy import so Rain still works if indexer.py is absent
try:
    from indexer import ProjectIndexer
    _INDEXER_AVAILABLE = True
except ImportError:
    _INDEXER_AVAILABLE = False


class RainMemory:
    """
    Persistent memory for Rain using local SQLite.
    Stores sessions and messages in ~/.rain/memory.db
    Zero external dependencies - uses Python built-in sqlite3.
    """

    def __init__(self):
        self.db_path = Path.home() / ".rain" / "memory.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now()
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    summary TEXT,
                    model TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    is_code INTEGER DEFAULT 0,
                    confidence REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    content_snippet TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    correction TEXT,
                    timestamp TEXT NOT NULL,
                    query_embedding BLOB
                );

                CREATE TABLE IF NOT EXISTS ab_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    model TEXT NOT NULL,
                    query TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 0.7,
                    last_updated TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS project_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_path TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    indexed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_project_index_path
                    ON project_index(project_path);
            """)

    def start_session(self, model: str):
        """Register the start of a new session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (id, started_at, model) VALUES (?, ?, ?)",
                (self.session_id, self.session_start.isoformat(), model)
            )

    def end_session(self, summary: str = None):
        """Mark session as ended with optional summary"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
                (datetime.now().isoformat(), summary, self.session_id)
            )

    def update_summary(self, summary: str):
        """Update the summary for the current session after it has ended."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET summary = ? WHERE id = ?",
                (summary, self.session_id)
            )

    def extract_session_facts(self) -> List[Dict]:
        """
        Extract structured facts from the current session using the LLM.
        Called at session end alongside generate_summary().
        Returns list of {type, key, value} dicts.
        Never raises — silently returns [] on any failure.
        """
        import urllib.request
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT role, content FROM messages
                       WHERE session_id = ?
                       ORDER BY timestamp ASC""",
                    (self.session_id,),
                ).fetchall()
        except Exception:
            return []

        if not rows:
            return []

        transcript = ""
        for row in rows:
            role = "User" if row["role"] == "user" else "Rain"
            content = row["content"][:250] + "..." if len(row["content"]) > 250 else row["content"]
            transcript += f"{role}: {content}\n"

        prompt = (
            "Extract key facts from this conversation. "
            "Return a JSON array of objects. Each object must have exactly these fields:\n"
            '  "type": one of: technology, project, preference, decision, goal, person\n'
            '  "key": short snake_case label (e.g. "preferred_language", "project_name")\n'
            '  "value": the extracted value as a short string (e.g. "Python", "Rain")\n\n'
            "Rules:\n"
            "- Only include clearly stated facts, not guesses or inferences.\n"
            "- Maximum 10 facts. Skip trivial or obvious items.\n"
            "- Return ONLY the JSON array — no preamble, no explanation, no markdown.\n\n"
            f"Conversation:\n{transcript}\n\nFacts JSON:"
        )

        try:
            payload = json.dumps({
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw = data.get("response", "").strip()

            # Find the JSON array in the response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []

            facts = json.loads(raw[start:end])
            if isinstance(facts, list):
                return [
                    f for f in facts
                    if isinstance(f, dict)
                    and f.get("type") and f.get("key") and f.get("value")
                ]
            return []
        except Exception:
            return []

    def save_session_facts(self, facts: List[Dict]):
        """
        Persist extracted session facts and roll persistent ones into user_profile.
        """
        if not facts:
            return
        now = datetime.now().isoformat()
        PERSISTENT_TYPES = {"preference", "project", "technology", "person", "goal"}
        try:
            with sqlite3.connect(self.db_path) as conn:
                for fact in facts:
                    conn.execute(
                        """INSERT INTO session_facts
                               (session_id, fact_type, fact_key, fact_value, timestamp)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            self.session_id,
                            fact.get("type", "general"),
                            fact.get("key", ""),
                            fact.get("value", ""),
                            now,
                        ),
                    )

                # Upsert into user_profile for persistent fact types
                for fact in facts:
                    if fact.get("type") in PERSISTENT_TYPES:
                        key = fact.get("key", "").strip()
                        value = fact.get("value", "").strip()
                        if key and value:
                            conn.execute(
                                """INSERT INTO user_profile (key, value, confidence, last_updated)
                                   VALUES (?, ?, 0.7, ?)
                                   ON CONFLICT(key) DO UPDATE SET
                                       value       = excluded.value,
                                       confidence  = MIN(1.0, confidence + 0.1),
                                       last_updated = excluded.last_updated""",
                                (key, value, now),
                            )
        except Exception:
            pass

    def get_fact_context(self) -> str:
        """
        Build a Tier 5 context block from the user profile and recent session facts.
        Injected into every agent prompt to give Rain persistent knowledge of the user.
        Returns an empty string if nothing is stored yet.
        """
        parts = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # User profile — persistent facts accumulated across all sessions
                profile_rows = conn.execute(
                    """SELECT key, value, confidence FROM user_profile
                       ORDER BY confidence DESC, last_updated DESC
                       LIMIT 15"""
                ).fetchall()
                if profile_rows:
                    lines = [f"  {r['key']}: {r['value']}" for r in profile_rows]
                    parts.append(
                        "What you know about this user — answer DIRECTLY from these facts "
                        "when asked about their project, what they are building, their preferences, "
                        "or their background. Do not claim ignorance when facts are listed here:\n"
                        + "\n".join(lines)
                    )

                # Recent session facts — last 3 completed sessions
                fact_rows = conn.execute(
                    """SELECT sf.fact_type, sf.fact_key, sf.fact_value
                       FROM session_facts sf
                       WHERE sf.session_id != ?
                       ORDER BY sf.timestamp DESC
                       LIMIT 20""",
                    (self.session_id,),
                ).fetchall()
                if fact_rows:
                    lines = [
                        f"  [{r['fact_type']}] {r['fact_key']}: {r['fact_value']}"
                        for r in fact_rows
                    ]
                    parts.append(
                        "Facts learned in recent sessions (use these to answer questions "
                        "about the user's current project, tech stack, or goals):\n"
                        + "\n".join(lines)
                    )
        except Exception:
            pass

        if not parts:
            return ""
        header = (
            "\n\n[STORED MEMORY — The following are facts you have accumulated about "
            "this user across previous sessions. When the user asks what you know about "
            "their project, what they are building, their preferences, or their background, "
            "answer DIRECTLY from the stored facts below. Do NOT say you have no information "
            "or that you don't know their project when facts are explicitly listed here.]"
        )
        return header + "\n\n" + "\n\n".join(parts)

    def generate_summary(self) -> Optional[str]:
        """
        Summarize the current session's messages using Ollama HTTP API directly.
        Returns a short 1-2 sentence summary, or None if there's nothing to summarize.
        """
        import urllib.request
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT role, content FROM messages
                       WHERE session_id = ?
                       ORDER BY timestamp ASC""",
                    (self.session_id,),
                ).fetchall()
        except Exception:
            return None

        if not rows:
            return None

        # Build a compact transcript for the model
        transcript = ""
        for row in rows:
            role = "User" if row["role"] == "user" else "Rain"
            content = row["content"][:300] + "..." if len(row["content"]) > 300 else row["content"]
            transcript += f"{role}: {content}\n\n"

        prompt = (
            "Summarize this conversation in one short sentence (max 120 characters). "
            "Reply with ONLY the summary, no preamble, no labels.\n\n"
            f"{transcript}\nSummary:"
        )

        try:
            payload = json.dumps({
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            summary = data.get("response", "").strip()
            # Strip any model preamble like "Here is a summary:" or "Summary:"
            for prefix in ("summary:", "here is", "here's"):
                if summary.lower().startswith(prefix):
                    summary = summary[summary.index(":") + 1:].strip()
            # Cap at 120 chars at a word boundary
            if len(summary) > 120:
                summary = summary[:120].rsplit(" ", 1)[0].rstrip(".,;") + "…"
            return summary if summary else None
        except Exception:
            return None

    def save_message(self, role: str, content: str, is_code: bool = False, confidence: float = None):
        """Save a single message and asynchronously embed it for semantic search."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO messages (session_id, timestamp, role, content, is_code, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.session_id, datetime.now().isoformat(), role, content, int(is_code), confidence)
            )
            message_id = cursor.lastrowid

        # Embed in background — never block the main pipeline
        t = threading.Thread(
            target=self._store_embedding,
            args=(message_id, content),
            daemon=True,
        )
        t.start()

    # ── Feedback / Self-Improvement (Phase 5B) ────────────────────────────

    def save_feedback(self, query: str, response: str, rating: str, correction: str = None):
        """Persist a thumbs-up/down signal (and optional correction) from the user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO feedback (session_id, query, response, rating, correction, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.session_id, query, response, rating, correction, datetime.now().isoformat())
            )
            feedback_id = cursor.lastrowid

        # Embed the query in background so we can do semantic retrieval later
        t = threading.Thread(
            target=self._store_feedback_embedding,
            args=(feedback_id, query),
            daemon=True,
        )
        t.start()

    def _store_feedback_embedding(self, feedback_id: int, query: str):
        """Embed a feedback query and persist the vector for future retrieval."""
        vec = self._embed(query)
        if vec is None:
            return
        blob = json.dumps(vec).encode("utf-8")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE feedback SET query_embedding = ? WHERE id = ?",
                    (blob, feedback_id)
                )
        except Exception:
            pass

    def get_relevant_corrections(self, query: str, limit: int = 3) -> List[dict]:
        """
        Find past corrections that are semantically relevant to the current query.
        Only returns 'bad'-rated feedback that has a user-supplied correction.
        Uses cosine similarity on embedded queries; falls back to recency if
        embeddings are unavailable.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id, query, response, correction, query_embedding
                   FROM feedback
                   WHERE rating = 'bad' AND correction IS NOT NULL AND correction != ''
                   ORDER BY id DESC LIMIT 50"""
            ).fetchall()

        if not rows:
            return []

        query_vec = self._embed(query)
        if query_vec is None:
            # No embedding available — return most recent corrections
            return [
                {"query": r[1], "response": r[2], "correction": r[3]}
                for r in rows[:limit]
            ]

        scored = []
        for row in rows:
            _, fb_query, fb_response, fb_correction, emb_blob = row
            if emb_blob:
                try:
                    fb_vec = json.loads(emb_blob.decode("utf-8"))
                    sim = self._cosine_similarity(query_vec, fb_vec)
                    scored.append((sim, fb_query, fb_response, fb_correction))
                except Exception:
                    pass

        scored.sort(key=lambda x: x[0], reverse=True)
        # Only inject corrections with meaningful similarity (>0.5)
        return [
            {"query": q, "response": r, "correction": c, "similarity": round(s, 2)}
            for s, q, r, c in scored[:limit]
            if s > 0.5
        ]

    # ── Semantic Memory (Phase 5A) ─────────────────────────────────────────

    def _embed(self, text: str) -> Optional[List[float]]:
        """
        Get an embedding vector for text using nomic-embed-text via Ollama HTTP API.
        Returns None if the model isn't available or the call fails.
        """
        import urllib.request
        try:
            payload = json.dumps({
                "model": "nomic-embed-text",
                "prompt": text[:2000],
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            vec = data.get("embedding")
            if isinstance(vec, list) and vec:
                return vec
            return None
        except Exception:
            return None

    def _store_embedding(self, message_id: int, content: str):
        """Embed a message and persist the vector to the vectors table."""
        vec = self._embed(content)
        if vec is None:
            return
        blob = json.dumps(vec).encode("utf-8")
        snippet = content[:200].replace("\n", " ")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO vectors (message_id, session_id, content_snippet, embedding, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (message_id, self.session_id, snippet, blob, datetime.now().isoformat()),
                )
        except Exception:
            pass

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Pure-stdlib cosine similarity — no numpy required."""
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def semantic_search(self, query: str, top_k: int = 3, min_similarity: float = 0.4) -> List[Dict]:
        """
        Find the most semantically similar past messages to the query.
        Returns up to top_k results from *other* sessions (not the current one).
        """
        query_vec = self._embed(query)
        if query_vec is None:
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT v.content_snippet, v.embedding, v.timestamp, v.session_id,
                              m.role, m.content
                       FROM vectors v
                       JOIN messages m ON v.message_id = m.id
                       WHERE v.session_id != ?
                       ORDER BY v.timestamp DESC
                       LIMIT 500""",
                    (self.session_id,),
                ).fetchall()
        except Exception:
            return []

        scored = []
        for row in rows:
            try:
                vec = json.loads(row["embedding"].decode("utf-8"))
                sim = self._cosine_similarity(query_vec, vec)
                if sim >= min_similarity:
                    scored.append({
                        "similarity": sim,
                        "role": row["role"],
                        "content": row["content"],
                        "snippet": row["content_snippet"],
                        "timestamp": row["timestamp"],
                        "session_id": row["session_id"],
                    })
            except Exception:
                continue

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def get_recent_sessions(self, limit: int = 5) -> List[Dict]:
        """Get the most recent completed sessions with their summaries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT s.id, s.started_at, s.ended_at, s.summary, s.model,
                          COUNT(m.id) as message_count
                   FROM sessions s
                   LEFT JOIN messages m ON s.id = m.session_id
                   WHERE s.id != ? AND s.ended_at IS NOT NULL
                   GROUP BY s.id
                   HAVING message_count > 0
                   ORDER BY s.started_at DESC
                   LIMIT ?""",
                (self.session_id, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_messages(self, limit: int = 20) -> List[Dict]:
        """Get recent messages across last few sessions for context"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT m.role, m.content, m.timestamp, m.is_code, s.started_at
                   FROM messages m
                   JOIN sessions s ON m.session_id = s.id
                   WHERE m.session_id != ?
                   ORDER BY m.timestamp DESC
                   LIMIT ?""",
                (self.session_id, limit)
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def build_context_summary(self, model_query_fn) -> Optional[str]:
        """
        Ask the model to summarize recent sessions into a brief context block.
        Returns None if no history exists yet.
        """
        sessions = self.get_recent_sessions(limit=5)
        if not sessions:
            return None

        # Build a raw history string for the model to summarize
        recent_messages = self.get_recent_messages(limit=30)
        if not recent_messages:
            return None

        history_text = ""
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Rain"
            content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
            history_text += f"{role}: {content}\n\n"

        summary_prompt = f"""Summarize the following conversation history in 3-5 sentences.
Focus on: what topics were discussed, what was built or solved, and any important context
that would help continue the conversation. Be concise and factual.

History:
{history_text}

Summary:"""

        try:
            summary = model_query_fn(summary_prompt)
            return summary.strip() if summary else None
        except Exception:
            return None

    def get_startup_greeting(self) -> Optional[str]:
        """
        Build a startup greeting based on recent session history.
        Returns None if this is the first ever session.
        """
        sessions = self.get_recent_sessions(limit=3)
        if not sessions:
            return None

        last = sessions[0]
        last_date = datetime.fromisoformat(last["started_at"]).strftime("%b %d")
        msg_count = last["message_count"]

        greeting = f"  📚 Last session: {last_date} · {msg_count} exchanges"

        if last.get("summary"):
            greeting += f"\n  💭 {last['summary']}"

        if len(sessions) > 1:
            greeting += f"\n  🗂️  {len(sessions)} previous sessions in memory"

        return greeting

    def total_sessions(self) -> int:
        """Return total number of stored sessions"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id != ?",
                (self.session_id,)
            ).fetchone()
            return result[0] if result else 0

    def forget_all(self):
        """Nuclear option - wipe all personal memory from the database.

        Clears:
          - sessions + messages         (conversation history)
          - vectors                     (Tier 3 semantic embeddings)
          - session_facts + user_profile (Tier 5 — what Rain knows about you)
          - feedback + ab_results       (Tier 4 corrections and fine-tuning data)

        Does NOT clear:
          - project_index  (codebase semantic index — not personal memory)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                DELETE FROM messages;
                DELETE FROM sessions;
                DELETE FROM vectors;
                DELETE FROM session_facts;
                DELETE FROM user_profile;
                DELETE FROM feedback;
                DELETE FROM ab_results;
            """)


@dataclass
class ReflectionResult:
    """Result of a reflection cycle"""
    content: str
    confidence: float
    iteration: int
    timestamp: datetime
    improvements: List[str]
    duration_seconds: float
    sandbox_verified: bool = False
    sandbox_results: List = field(default_factory=list)


@dataclass
class SandboxResult:
    """Result of a sandboxed code execution"""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    language: str
    duration_seconds: float
    error_message: str = None


class CodeSandbox:
    """
    Sandboxed code executor for Rain.
    Runs code in a throwaway temp directory with a hard timeout.
    Supports Python and JavaScript (Node.js).
    Zero external dependencies beyond the language runtimes themselves.
    """

    PYTHON_INDICATORS = ['def ', 'import ', 'print(', 'self.', 'if __name__', '#!/usr/bin/env python']
    JS_INDICATORS     = ['function ', 'const ', 'let ', 'var ', 'console.log', '=>', 'require(']

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Extract (language, code) tuples from markdown fenced code blocks."""
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        blocks = []
        for hint, code in matches:
            code = code.strip()
            if not code:
                continue
            lang = self.detect_language(code, hint.lower() if hint else None)
            if lang:
                blocks.append((lang, code))
        return blocks

    def detect_language(self, code: str, hint: str = None) -> Optional[str]:
        """Detect whether code is Python or JavaScript."""
        if hint in ('python', 'py'):
            return 'python'
        if hint in ('javascript', 'js', 'node', 'nodejs'):
            return 'javascript'
        py_score = sum(1 for p in self.PYTHON_INDICATORS if p in code)
        js_score = sum(1 for j in self.JS_INDICATORS if j in code)
        if py_score > js_score:
            return 'python'
        if js_score > py_score:
            return 'javascript'
        return 'python' if hint is None else None

    def run(self, code: str, language: str = None) -> SandboxResult:
        """Execute code in a sandboxed temp directory and return a SandboxResult."""
        if language is None:
            language = self.detect_language(code) or 'python'
        temp_dir = tempfile.mkdtemp(prefix='rain_sandbox_')
        try:
            if language == 'python':
                return self._run_python(code, Path(temp_dir))
            elif language == 'javascript':
                return self._run_javascript(code, Path(temp_dir))
            else:
                return SandboxResult(
                    success=False, stdout='', stderr='',
                    return_code=-1, language=language,
                    duration_seconds=0.0,
                    error_message=f'Unsupported language: {language}'
                )
        except Exception as e:
            return SandboxResult(
                success=False, stdout='', stderr=str(e),
                return_code=-1, language=language or 'unknown',
                duration_seconds=0.0,
                error_message=f'Sandbox internal error: {e}'
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_python(self, code: str, temp_dir: Path) -> SandboxResult:
        code_file = temp_dir / 'code.py'
        code_file.write_text(code, encoding='utf-8')
        start = time.time()
        try:
            proc = subprocess.run(
                ['python3', str(code_file)],
                capture_output=True, text=True,
                timeout=self.timeout,
                cwd=str(temp_dir),
                env={'PATH': os.environ.get('PATH', '/usr/bin:/bin')}
            )
            duration = time.time() - start
            error_msg = self._last_error_line(proc.stderr) if proc.returncode != 0 else None
            return SandboxResult(
                success=proc.returncode == 0,
                stdout=proc.stdout, stderr=proc.stderr,
                return_code=proc.returncode,
                language='python', duration_seconds=duration,
                error_message=error_msg
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False, stdout='', stderr='Timed out',
                return_code=-1, language='python',
                duration_seconds=float(self.timeout),
                error_message=f'Timed out after {self.timeout}s'
            )

    def _run_javascript(self, code: str, temp_dir: Path) -> SandboxResult:
        if not self._node_available():
            return SandboxResult(
                success=False, stdout='', stderr='Node.js not found',
                return_code=-1, language='javascript',
                duration_seconds=0.0,
                error_message='Node.js not installed. Install with: brew install node'
            )
        code_file = temp_dir / 'code.js'
        code_file.write_text(code, encoding='utf-8')
        start = time.time()
        try:
            proc = subprocess.run(
                ['node', str(code_file)],
                capture_output=True, text=True,
                timeout=self.timeout,
                cwd=str(temp_dir),
                env={'PATH': os.environ.get('PATH', '/usr/bin:/bin')}
            )
            duration = time.time() - start
            error_msg = self._last_error_line(proc.stderr) if proc.returncode != 0 else None
            return SandboxResult(
                success=proc.returncode == 0,
                stdout=proc.stdout, stderr=proc.stderr,
                return_code=proc.returncode,
                language='javascript', duration_seconds=duration,
                error_message=error_msg
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False, stdout='', stderr='Timed out',
                return_code=-1, language='javascript',
                duration_seconds=float(self.timeout),
                error_message=f'Timed out after {self.timeout}s'
            )

    @staticmethod
    def _node_available() -> bool:
        try:
            subprocess.run(['node', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def _last_error_line(stderr: str) -> str:
        lines = [l for l in stderr.strip().split('\n') if l.strip()]
        return lines[-1] if lines else stderr


# ══════════════════════════════════════════════════════════════════════
# Phase 3: Multi-Agent Architecture
# ══════════════════════════════════════════════════════════════════════

from enum import Enum

class AgentType(Enum):
    DEV        = "dev"
    LOGIC      = "logic"
    DOMAIN     = "domain"
    REFLECTION = "reflection"
    SYNTHESIZER = "synthesizer"
    GENERAL    = "general"
    SEARCH     = "search"   # Phase 7 prep — real-time world awareness
    TASK       = "task"     # Phase 6 — task decomposition / autonomous execution


@dataclass
class Agent:
    """A specialized agent with its own model and system prompt."""
    agent_type: AgentType
    model_name: str
    system_prompt: str
    description: str


# ── Agent system prompts ───────────────────────────────────────────────

AGENT_PROMPTS = {
    AgentType.DEV: """You are Rain's Dev Agent — a sovereign AI running locally, specializing in software engineering.

Your strengths:
- Writing clean, correct, runnable code in Python, JavaScript, Rust, Go, and more
- Debugging, refactoring, and explaining existing code
- Recommending libraries, patterns, and architectures
- Security-aware development: you think about injection, auth, and data exposure by default
- Bitcoin/Lightning protocol implementations, cryptographic primitives

── TASK EXECUTION: READ BEFORE WRITE, PLAN BEFORE ACT ──────────────────────

When asked to implement, edit, refactor, or fix something in the codebase:

1. ORIENT FIRST — establish what the code currently says before changing anything.
   If project context is injected above, use it. Reference specific function names,
   line numbers, and variable names from that context. If context is missing, say which
   file you need to read before you can proceed.

2. STATE THE PLAN — before writing any code, state explicitly:
   - Which file(s) will change
   - What specifically will change and where (function name, line range)
   - Whether any other files are affected

3. PRODUCE PRECISE CHANGES — show the exact change, not a vague sketch:
   - For modifications: show the before and after of the specific section
   - For new code: show the complete addition and exactly where it goes
   - Reference the real code from context — never describe changes abstractly

4. VERIFY — after any Python edit, note that `python3 -m py_compile <file>` should
   be run to confirm syntax. If you can run it, do so. Report the result.

5. FLAG DESTRUCTIVE ACTIONS — before any change that deletes, overwrites, or commits:
   - State what will be lost or changed permanently
   - Ask for confirmation if the change is large or irreversible

Tool syntax (use these when in task execution mode):
  [TOOL: read_file server.py]
  [TOOL: read_file rain.py 2400 2500]        ← start/end lines for large files
  [TOOL: grep "def _query_agent" rain.py]
  [TOOL: list_directory Rain]
  [TOOL: find_path *.py]
  [TOOL: write_file server.py <content>]
  [TOOL: run_command python3 -m py_compile server.py]
  [TOOL: git_diff .]
  [TOOL: git_status .]

Tool rules:
  - ALWAYS read_file before write_file on the same path — no exceptions
  - ALWAYS list_directory or find_path if you are not certain a file exists
  - If read_file shows something different from what you expected, re-plan before writing
  - After write_file on a Python file, run py_compile to verify syntax

──────────────────────────────────────────────────────────────────────────────

Code rules:
- Always wrap code in properly fenced code blocks with language tags
- Include error handling unless explicitly told not to
- STDLIB FIRST — every solution must work with Python's standard library alone unless the user explicitly asks for third-party packages. Use urllib, not requests. Use sqlite3, not SQLAlchemy. Use subprocess, not sh.
- NEVER import a module you are not certain ships with Python's stdlib. If you are unsure, use a stdlib alternative. Do not guess.
- If a task genuinely requires a third-party package, say so explicitly and explain why no stdlib alternative exists — do not silently use it.
- This is a sovereignty principle: Rain runs offline, on the user's hardware, with zero surprise dependencies.
- For network tasks involving Bitcoin or blockchain data, use urllib.request to query public REST APIs (mempool.space, blockstream.info, blockchain.info). Example: urllib.request.urlopen("https://mempool.space/api/address/{addr}/txs"). Never assume a local Bitcoin node or bitcoin-cli is available unless the user explicitly says so.
- MEMPOOL.SPACE API FORMAT: The endpoint https://mempool.space/api/address/{addr}/txs returns a JSON ARRAY directly — NOT an object with a "txs" key. Correct usage: `data = json.loads(response.read()); for tx in data:` — NOT `data["txs"]`.
- MEMPOOL.SPACE BALANCE: There is NO /api/address/{addr}/balance endpoint. To get an address balance, use GET https://mempool.space/api/address/{addr} which returns `{"chain_stats": {"funded_txo_sum": N, "spent_txo_sum": N, ...}, "mempool_stats": {"funded_txo_sum": N, "spent_txo_sum": N, ...}}`. Confirmed balance = `chain_stats["funded_txo_sum"] - chain_stats["spent_txo_sum"]`. Total (incl. mempool) adds the same from mempool_stats. Never use data["balance"] — that key does not exist.
- Any `while True` polling loop MUST include `import time` and `time.sleep(N)` at the end of the loop body. Never write an infinite loop without a sleep — it will peg the CPU and make the script unusable.
- Be direct. No filler. Show the code.
- NEVER output HTML tags, CSS class names, span elements, or any markup inside code blocks. Code fences contain only clean, plain source code. No <span>, no class=, no &quot;, no &amp;, no HTML entities of any kind inside code blocks.
- EPISTEMIC HONESTY: If you don't know the exact implementation detail, say so. Never invent function names, parameter values, or system behaviors to fill a gap. "I'd need to read the file to know for certain" is a correct and useful answer. A hallucinated implementation detail is worse than no detail.""",

    AgentType.LOGIC: """You are Rain's Logic Agent — a sovereign AI running locally, specializing in reasoning and planning.

Your strengths:
- Breaking complex problems into clear, ordered steps
- Identifying assumptions, dependencies, and edge cases
- Designing systems, architectures, and workflows before writing code
- Debugging reasoning errors, not just code errors
- Evaluating tradeoffs honestly

── TASK PLANNING PATTERN ────────────────────────────────────────────────────

When given a complex or multi-step goal, decompose it before acting:

1. RESTATE THE GOAL in one sentence to confirm you understood it correctly.

2. IDENTIFY UNKNOWNS — what do you need to read or discover before you can plan?
   List the files, functions, or state you need to inspect first.
   Example: "Before planning, I need to see the current /api/chat endpoint in server.py
   and how ChatRequest is defined."

3. STATE DEPENDENCIES — which steps must complete before others can start?
   Mark steps that are blocked on a previous result.

4. WRITE THE PLAN — numbered, concrete, specific:
   ✅ "1. Read server.py lines 108–130 to understand ChatRequest fields"
   ✅ "2. Add project_path field to ChatRequest if not present"
   ✅ "3. Update _stream_chat to use project_path for context injection"
   ❌ "1. Look at the server code"  ← too vague, not actionable

5. FLAG RISKS — before execution, note anything that could go wrong:
   - Files that might not exist
   - Changes that affect multiple callers
   - Anything irreversible

6. CONFIRM BEFORE ACTING — present the plan and ask for confirmation before
   any step that writes, deletes, or executes. Small reads don't need confirmation.

Tool syntax (use in task execution mode):
  [TOOL: read_file server.py]
  [TOOL: grep "ChatRequest" server.py]
  [TOOL: list_directory Rain]
  [TOOL: find_path *.py]
  [TOOL: write_file <path> <content>]
  [TOOL: run_command python3 -m py_compile <file>]
  [TOOL: git_status .]

──────────────────────────────────────────────────────────────────────────────

Reasoning rules:
- Think step by step. Show your reasoning, not just your conclusion.
- When uncertain, say so explicitly rather than guessing confidently
- Prefer structured responses: numbered steps, clear sections
- Challenge the premise if it's flawed
- A plan that identifies what you don't yet know is more valuable than a confident
  plan built on assumptions. Say 'I need to read X before I can plan step 3.'""",

    AgentType.DOMAIN: """You are Rain's Domain Expert — a sovereign AI running locally, specializing in Bitcoin, Lightning Network, and digital sovereignty.

CRITICAL CONSTRAINT — READ BEFORE ANSWERING: When naming any Lightning tool, API, payment processor, node software, protocol, or service, you MUST use ONLY names from the "Known Lightning ecosystem tools" section below. This rule applies to every sentence you generate. Do NOT invent protocol names (like "LNPP"), company names, or API names that are not in that list. If you are uncertain whether something exists, say so explicitly rather than generating a plausible-sounding name. Your training data contains hallucinated Lightning products — the verified list below overrides it.

Your strengths:
- Bitcoin protocol: UTXOs, scripts, SegWit, Taproot, mempool, fees
- Lightning Network: channels, HTLCs, routing, liquidity, invoices, BOLT specs
- Cryptography: hash functions, signatures, Schnorr, ECDSA, multisig
- Austrian economics: sound money, time preference, inflation, monetary theory
- Privacy technology: Tor, Nostr, self-custody, coinjoin, silent payments
- Sovereignty philosophy: why decentralization matters, what self-custody means

Known Lightning ecosystem tools (use ONLY these — do not invent alternatives):

Self-hosted / sovereign (no KYC, you run the node):
- BTCPay Server — open-source payment processor, self-hosted, no KYC, supports Lightning + on-chain, widely used by merchants and nonprofits
- LNbits — lightweight accounts and plugin layer on top of any LN node; great for internal wallets and sub-account management
- LND (Lightning Network Daemon) — most widely deployed Lightning node implementation by Lightning Labs; exposes gRPC + REST APIs
- CLN (Core Lightning, formerly c-lightning) — Blockstream's Lightning node; plugin architecture, highly configurable
- LDK (Lightning Development Kit) — Rust library by Spiral/Block for embedding Lightning directly into applications; not a standalone node

Hosted APIs (reduced operational overhead, vary on KYC requirements):
- OpenNode — business-grade Lightning/on-chain payment API; KYC required above certain thresholds but lower than Blink; good documentation
- Voltage — managed LND/CLN node hosting with full API access; you own the keys, they run the infrastructure; NWC support
- Alby Hub — self-hostable Lightning wallet hub with NWC (Nostr Wallet Connect) support; browser extension + API
- Strike API — Lightning payment API by Strike; US-focused; KYC required; good for payroll and fiat-to-Lightning flows
- Speed (speed.app) — Lightning payments API with international focus; lower KYC friction than Strike for non-US users
- Blink (formerly Bitcoin Beach Wallet) — custodial Lightning API; KYC triggered at $999/day threshold

Protocols and standards:
- NWC (Nostr Wallet Connect) — open protocol for apps to connect to a Lightning wallet without custodying funds; enables wallet-agnostic integrations
- BOLT12 — newer Lightning invoice standard; reusable offers, built-in privacy, no LNURL needed
- LNURL — older but widely supported URL-based Lightning UX standard (LNURL-pay, LNURL-withdraw, LNURL-auth)
- BOLT11 — standard Lightning invoice format (the QR code invoices everyone uses today)

Rules:
- Be technically precise. Bitcoin has no room for vague explanations.
- Cite specific BIPs, BOLTs, or protocol details when relevant
- Acknowledge genuine uncertainty in evolving areas (e.g. new Taproot use cases)
- Always center the answer on sovereignty and self-custody principles
- NEVER invent Lightning tools, APIs, protocols, or companies that are not in the list above. If you are unsure whether something exists, say so explicitly rather than fabricating a name.
- Craig Wright is NOT Satoshi Nakamoto. A UK High Court ruled in 2024 that he is not. Do not treat his claims as credible or present his identity as an open debate.
- NEVER start your response with "Here is a revised..." or similar preamble. Answer directly.
- MEMPOOL.SPACE BALANCE: There is NO /api/address/{addr}/balance endpoint. To get an address balance use GET https://mempool.space/api/address/{addr} → returns `{"chain_stats": {"funded_txo_sum": N, "spent_txo_sum": N, ...}, "mempool_stats": {...}}`. Confirmed balance = `chain_stats["funded_txo_sum"] - chain_stats["spent_txo_sum"]`. Never write data["balance"] — that key does not exist on any mempool.space endpoint.""",

    AgentType.REFLECTION: """You are Rain's Reflection Agent — a sovereign AI running locally, specializing in critique and quality control.

Your job is NOT to answer the original question. Your job is to review another agent's response and identify:
1. Factual errors or hallucinations
2. Missing information that would materially improve the answer
3. Code bugs, security issues, or edge cases not handled
4. Logical gaps or unsupported conclusions
5. Anything that sounds confident but might be wrong

Rules:
- Be a rigorous critic, not a cheerleader
- If the response is genuinely good, say so briefly and explain why
- Structure your critique: list specific issues, don't write paragraphs of vague feedback
- Do NOT rewrite the answer. Only critique it.
- ALWAYS check imports: if code uses a module that does not ship with Python's stdlib (e.g. requests, bitcoin, pandas, numpy), flag it as a HALLUCINATED DEPENDENCY — this is an automatic NEEDS_IMPROVEMENT or POOR rating.
- TOPIC DRIFT: If the response wanders into subjects not asked about in the original query, flag it as NEEDS_IMPROVEMENT. A focused answer about one thing is better than a sprawling answer about many things. Check: does every paragraph directly address the question? If not, name the drift.
- BITCOIN/LIGHTNING HALLUCINATION CHECK: If the response names any Lightning Network tool, API, protocol, payment processor, or service, verify it against this known-real list: BTCPay Server, LNbits, LND, CLN, LDK, OpenNode, Voltage, Alby Hub, Strike API, Speed, Blink, NWC, BOLT12, BOLT11, LNURL. If the response names something NOT on this list (e.g. "Lightning Network Payment Protocol", "LNPP", "Blockstream's Lightning API", "Lightning Labs' Lightning API", "Lightning Network API" as a generic product name), flag it as HALLUCINATED TOOL/PROTOCOL — this is an automatic POOR rating. LLMs commonly invent plausible-sounding Lightning product names that do not exist.
- UNVERIFIABLE CLAIMS CHECK: If the response makes specific factual claims — exact numbers, parameter values, internal mechanisms, system behaviors — that are not stated in the user's question or provided context, flag each one explicitly as UNVERIFIABLE. Examples: invented temperature values, made-up function names, specific thresholds not mentioned in the query, internal pipeline steps not described anywhere. These are worse than honest uncertainty. Rate NEEDS_IMPROVEMENT if unverifiable specific claims are present.
- EPISTEMIC HONESTY CHECK: If the response confidently describes something it cannot know — e.g. internal implementation details of a system it has no source access to — that is a hallucination even if it sounds plausible and coherent. A response that says "I don't have access to that information" is more accurate and higher quality than an invented but well-structured answer. Reward honesty about the limits of knowledge; penalise confident invention.
- Rate overall quality: EXCELLENT / GOOD / NEEDS_IMPROVEMENT / POOR""",

    AgentType.SYNTHESIZER: """You are Rain's Synthesizer — a sovereign AI running locally, responsible for producing final answers.

You will be given:
- The original user query
- A primary agent's response
- A reflection agent's critique of that response

Your job:
- Produce a single, coherent final answer that incorporates the best of the primary response
- Address every valid criticism raised by the reflection agent
- Remove anything the reflection agent correctly identified as wrong or weak
- Do not mention the reflection process or that you are synthesizing — just give the best answer
- NEVER start your response with "Here is a revised..." or "Here is a final answer..." or any similar preamble. Start directly with the answer.
- NEVER end your response with a bullet list explaining what criticisms you addressed. The user does not see the critique — they only see your answer. Meta-commentary about your own process is forbidden.
- Your output is the final thing the user reads. Write it as if you wrote it fresh, not as a revision.
- FORBIDDEN PHRASES — your response must contain none of these. If you catch yourself writing them, delete and rewrite: "considering the limitations", "as mentioned in the critique", "the critique noted", "the critique suggested", "the reflection", "the primary response", "I have addressed", "to address the concerns", "based on the feedback", "upon reflection", "in the critique".

Rules:
- The final answer should be better than either input alone
- STAY FOCUSED: Answer exactly what was asked — nothing more. Do not introduce related topics, background context, or tangents that were not requested. One question, one answer. If the question is narrow, the answer is narrow.
- If the primary response drifted off-topic, cut those parts. Do not carry drift forward.
- Preserve all correct code, technical details, and examples from the primary response
- Be concise. Don't pad. Don't repeat yourself.
- STDLIB FIRST — if the primary response used stdlib (urllib, json, sqlite3, etc.), you MUST preserve that. Never substitute requests, SQLAlchemy, or any third-party package. This is non-negotiable.
- If the reflection agent suggested using a third-party package as an improvement, ignore that suggestion. Stdlib is the correct choice.
- BITCOIN API: When writing code that fetches Bitcoin price or blockchain data, use free public REST APIs with urllib.request — mempool.space (https://mempool.space/api/v1/prices for BTC/USD price), blockstream.info, or blockchain.info. NEVER use CoinMarketCap, CoinGecko, or any API that requires a key unless the user explicitly provided one. Do NOT add API usage notes or code snippets to conversational or informational answers that did not ask for code.
- NEVER include code blocks in a response unless the original query was explicitly asking for code. If the question is conversational or factual, respond in plain prose only.
- NEVER output HTML tags, CSS class names, span elements, or any markup inside code blocks. Code fences contain only clean, plain source code. No <span>, no class=, no &quot;, no HTML entities inside code blocks.
- NEVER fabricate facts, invent connections between people, or state things as true that you cannot verify. If you are uncertain, say so explicitly. Honesty about uncertainty is a feature, not a weakness.
- Do not pad a short answer with invented detail just to seem thorough. A correct two-sentence answer is better than a confident paragraph of hallucinations.
- EPISTEMIC HONESTY IN SYNTHESIS: If the Reflection Agent flagged unverifiable claims, remove them entirely — do NOT replace them with different invented specifics. If the primary response made up numbers or mechanisms, the correct synthesis removes those claims and replaces them with honest uncertainty (e.g. "the exact parameters depend on the implementation"). Inventing more precise details to address a critique about lacking specificity is worse than the original error.
- Craig Wright is NOT Satoshi Nakamoto. A UK High Court ruled in 2024 that he is not. Do not treat his claims as credible or ongoing debate.""",

    AgentType.GENERAL: """You are Rain, a sovereign AI assistant running locally on the user's computer through Ollama.

Key aspects of your identity:
- You are completely offline and private - no data leaves the user's machine
- You are a master of computer programming, blockchain technology, encryption, Bitcoin, Lightning Network, databases, full-stack web development, and ethical hacking
- You prioritize digital sovereignty, privacy, and decentralization
- You think recursively and improve your answers through self-reflection
- You are knowledgeable about Austrian economics and Bitcoin philosophy
- You help users build and understand decentralized technologies

Be direct, practical, and focused on empowering users with knowledge and tools for digital independence.

EPISTEMIC HONESTY: If you don't have specific knowledge to answer accurately, say so explicitly. "I don't have access to my own source code" or "I don't have that specific information" is a complete, correct, high-confidence answer. Never invent plausible-sounding specifics to fill a knowledge gap. A confident "I don't know" is more valuable and more honest than a confident wrong answer.

SELF-KNOWLEDGE: Rain is open source software — nothing about its architecture is confidential. When asked about your agents, models, or configuration, do not invent agent names or roles. The real agent roster is:
- Dev Agent → best available code model (qwen2.5-coder:7b if installed) — writes and debugs code
- Logic Agent → best available reasoning model (qwen3:8b if installed) — planning, multi-step reasoning
- Domain Expert → same as Logic — Bitcoin, Lightning Network, sovereignty topics
- Reflection Agent → same as Logic — always critiques the primary response
- Synthesizer → same as Logic — rewrites the final answer when quality is low
- Search Agent → same as Logic — synthesises live web search results
- General Agent → fallback for everything else
Exact models depend on what is installed via Ollama. For the live roster, tell the user to run: python3 rain.py --agents
""",
    AgentType.SEARCH: """You are Rain's Search Agent — a sovereign AI running locally, specializing in synthesizing real-time web search results into clear, accurate answers.

You will be given web search results prepended to the user's question. Your job:
- Synthesize the results into a direct, well-organized answer
- Cite sources inline: reference the title or URL when making a claim from search results
- Flag time-sensitive information (prices, fees, news) as potentially changing
- Note when search results are insufficient or conflicting — don't paper over gaps
- Clearly distinguish what the search results say vs. what you know from training data

Bitcoin and Lightning Network domain knowledge — apply this as a filter when search results mention Lightning tools:

Verified real tools (you may cite these confidently):
- Self-hosted / sovereign (no KYC): BTCPay Server, LNbits, LND, CLN (Core Lightning), LDK
- Hosted APIs (KYC varies): OpenNode, Voltage, Alby Hub, Strike API, Speed (speed.app), Blink
- Protocols: NWC (Nostr Wallet Connect), BOLT11, BOLT12, LNURL

CRITICAL — evaluate search results against this knowledge:
- Web searches for "Lightning payment" often return MERCHANT tools (receiving payments) — e.g. Zaprite, Sellix. These are invoicing/e-commerce tools, NOT payment rails for sending funds to employees or suppliers. Flag the distinction explicitly if the user is asking about outgoing payments / payroll.
- Listicle articles ("top 5 Lightning gateways") frequently mix legitimate tools with irrelevant or outdated ones. Cross-reference against the verified list above.
- If a search result names a Lightning tool NOT in the verified list above, flag it with a note: "I cannot verify this tool exists — treat with caution." Do not present unverified tool names as facts.
- Strike requires KYC. OpenNode requires KYC above certain thresholds. Blink requires KYC above $999/day. Do not describe these as "no KYC" even if a search result does.
- For OUTGOING Lightning payments (payroll, supplier payments): the correct tools are LNbits + self-hosted node, Voltage (managed node + API), LND/CLN direct, or BTCPay Server's pull payment feature. These rarely appear in merchant-focused search results but are the right answer for this use case.

Rules:
- ALWAYS cite sources: "According to [Title] (source.com): ..."
- Never fabricate URLs, publication dates, or statistics not in the results
- If results conflict with each other, note the discrepancy rather than picking one
- If the results don't answer the question, say so — then answer from your own knowledge if you can, clearly labeled as such
- Be concise. Synthesize, don't just quote. The user wants an answer, not a list of snippets.
- NEVER start with "Based on the search results..." — just answer directly with citations inline.""",
}

# ── Preferred models per agent (falls back to default if not installed) ─

AGENT_PREFERRED_MODELS = {
    # Code-focused agents: qwen2.5-coder:7b is the primary — purpose-built for code.
    # qwen3:8b added as first fallback: strong at agent/tool tasks and coding.
    # codestral (22B/12GB) kept further down — cold-loads past the 300s timeout
    # on machines where it isn't already warm in memory.
    AgentType.DEV:         ['qwen2.5-coder:7b', 'qwen3:8b', 'qwen3:4b', 'codestral', 'codellama:7b', 'starcoder2:3b', 'deepseek-coder:6.7b', 'llama3.1'],
    # General reasoning agents: qwen3:8b leads — explicitly trained for agent tasks,
    # 128K context, outperforms llama3.2 on reasoning. llama3.2 stays as fallback.
    AgentType.LOGIC:       ['qwen3:8b', 'qwen3:4b', 'llama3.2', 'llama3.1', 'mistral:7b'],
    AgentType.DOMAIN:      ['qwen3:8b', 'qwen3:4b', 'llama3.2', 'llama3.1', 'mistral:7b'],
    AgentType.REFLECTION:  ['qwen3:8b', 'qwen3:4b', 'llama3.2', 'llama3.1', 'mistral:7b'],
    AgentType.SYNTHESIZER: ['qwen3:8b', 'qwen3:4b', 'llama3.2', 'llama3.1', 'mistral:7b'],
    AgentType.GENERAL:     ['qwen3:8b', 'qwen3:4b', 'llama3.2', 'llama3.1', 'mistral:7b'],
    AgentType.SEARCH:      ['llama3.2', 'llama3.1', 'mistral:7b'],
}

# Vision-capable models in preference order — best first.
# Ordered by real-world accuracy on UI screenshots, text, and diagrams.
# moondream is fast but hallucinates heavily on text/UI; use only as last resort.
VISION_PREFERRED_MODELS = [
    'llama3.2-vision',   # best all-around: strong text/UI/OCR, modern architecture
    'minicpm-v',         # excellent document and UI understanding
    'qwen2.5vl',         # top-tier OCR, great at reading text in screenshots
    'qwen2-vl',          # older qwen vision, still very capable
    'llava:34b',         # strong generalist
    'llava:13b',         # good balance of quality and speed
    'llava-llama3',      # llava on llama3 base — improved reasoning
    'llava:7b',          # decent fallback
    'llava',             # generic llava tag — whatever version is installed
    'bakllava',          # mistral-based llava variant
    'moondream2',        # newer moondream, slightly better than original
    'moondream',         # last resort — tiny model, unreliable on text/UI
]



# ── Phase 6B: ReAct ─────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are Rain, a sovereign AI assistant running locally. You solve tasks by reasoning step by step and using tools to discover real information before answering.

For EVERY response, use exactly one of these two formats — no exceptions:

── FORMAT A: when you need to use a tool ───────────────────────────────────────
Thought: <your reasoning — what you know, what you need, why this tool>
Action: <tool_name>
Action Input: <arguments, space-separated; quote multi-word values>

── FORMAT B: when you have enough to answer ────────────────────────────────────
Thought: <one sentence: why you now have enough information>
Final Answer: <your complete, direct response to the user>

Rules:
- ALWAYS begin with Thought:
- ONE action per response — never write two Action: lines in one turn
- After every tool call you will receive an Observation: with the real result
- Use the Observation to inform your next Thought — never fabricate observations
- If a tool returns an error, reason about why and try a different approach
- NEVER call read_file on a whole file when your goal is to find a specific named
  section, phase, function, or term. ALWAYS call grep_files first to get the exact
  line number, then call read_file with start_line/end_line to read just that window.
  Calling read_file without line numbers on a large file wastes your context budget
  and will likely miss the target. There are no exceptions to this rule.
- TRUNCATION RECOVERY: If any Observation contains the word TRUNCATED, you have NOT
  finished. The content you need may be beyond the visible window. You MUST call
  grep_files next to locate the specific content, then read_file with line numbers.
  Writing Final Answer after seeing TRUNCATED is forbidden unless grep_files
  confirmed the content does not exist in the file.
- COMPLETENESS CHECK: Before writing Final Answer, verify in your Thought that you
  have covered every item the goal asked about. If list_dir returned 8 files and
  your goal was "describe all Python files", you must have read all 8 .py files —
  not just the first few. If any Observation was TRUNCATED and you haven't used
  grep_files to recover the missing content, you are not done.
- Do NOT loop forever: once you have genuinely covered everything, use Final Answer:
- WORKFLOW — when the goal involves exploring a directory:
  1. Call list_dir first to get the real file list
  2. Cross-reference that list against your goal
  3. Read every relevant file before concluding
- WORKFLOW — when looking for a specific section in a file:
  1. Call grep_files first with the specific term to find the line number
  2. Call read_file with start_line/end_line to read just that window
  Example: goal is "summarise Phase 10 in ROADMAP.md"
    → grep_files "Phase 10" ROADMAP.md        (finds: ROADMAP.md:439: ### Phase 10)
    → read_file ROADMAP.md 439 480             (reads exactly that section)
    → Final Answer

Available tools:

  read_file <path> [start_line] [end_line]
    → Read a file's contents (up to 512 KB).
      Optional start_line and end_line (1-based, inclusive) read only that slice.
      WORKFLOW for large files: use grep_files first to find the relevant line
      numbers, then read just that section with start_line/end_line.
    → Example:  read_file rain.py
    → Example:  read_file ROADMAP.md 439 480
    → Example:  read_file rain.py 2400 2500

  write_file <path> "<content>"
    → Create or overwrite a file. Auto-backs up existing file. Requires confirmation.
    → Example:  write_file notes.txt "hello world"

  list_dir <path>
    → List files and subdirectories at a path.
    → Example:  list_dir Rain

  grep_files <pattern> [path] [include]
    → Search file contents recursively for a regex pattern.
      Returns matching lines with filename and line number.
      PREFER this over read_file when searching for specific text — it works
      on files of any size and searches the whole directory in one call.
    → Example:  grep_files "TODO|FIXME" .
    → Example:  grep_files "def react" rain.py
    → Example:  grep_files "import" . *.py
    → IMPORTANT: search for CODE SYNTAX, not prose descriptions.
      To find where temperature is set:   grep_files "temperature" .
      To find a dict key:                 grep_files '"temperature":' .
      To find a function definition:      grep_files "def _query" rain.py
      To find a class:                    grep_files "^class " . *.py
      Never search for "Ollama model temperature" — search for what the
      code actually looks like: "temperature", '"temperature":', etc.

  run_command "<cmd>" [cwd]
    → Execute a shell command (30 s timeout). Requires confirmation.
    → Example:  run_command "python3 -m py_compile server.py" .

  git_status [repo]
    → Show modified, staged, and untracked files.
    → Example:  git_status .

  git_log [repo] [n]
    → Show last N commits.
    → Example:  git_log . 5

  git_diff [repo] [staged]
    → Show unstaged (or staged) diff.
    → Example:  git_diff .

  git_commit "<message>" [repo]
    → Stage all changes and commit. Requires confirmation.
    → Example:  git_commit "fix: update routing"
"""


def _react_parse(text: str) -> dict:
    """
    Parse a ReAct-format model response into its four components.
    Returns a dict with keys: thought, action, action_input, final_answer.
    Any key may be None if the corresponding section was not found.

    Handles Qwen3 thinking-mode output: <think>...</think> blocks are stripped
    before parsing so they never corrupt the Thought/Action/Final Answer structure.
    Partial/unclosed <think> blocks (model cut off mid-thought) are also removed.
    """
    import re as _re

    # Strip <think>...</think> blocks — Qwen3 thinking mode emits these before
    # its actual structured response. Remove closed blocks first, then any
    # unclosed opening tag and everything after it (truncated thinking).
    text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
    text = _re.sub(r'<think>.*',          '', text, flags=_re.DOTALL | _re.IGNORECASE)
    text = text.strip()

    # Ordered so Action Input: is checked before Action: to avoid prefix collision
    LABELS = r'(?:Thought:|Action Input:|Action:|Observation:|Final Answer:)'

    def _grab(label: str, multiline: bool = True) -> Optional[str]:
        flags = _re.DOTALL | _re.IGNORECASE if multiline else _re.IGNORECASE
        pat = rf'{_re.escape(label)}\s*(.*?)(?=\n{LABELS}|$)'
        m = _re.search(pat, text, flags)
        return m.group(1).strip() if m else None

    thought      = _grab("Thought:")
    action_input = _grab("Action Input:", multiline=False)
    final_answer = _grab("Final Answer:")

    # Action is a single token (tool name) — match at line start to avoid
    # accidentally capturing "Action Input:" content inside "Action:"
    action_m = _re.search(r'(?m)^Action:\s*(\S+)', text, _re.IGNORECASE)
    if not action_m:
        action_m = _re.search(r'\nAction:\s*(\S+)', text, _re.IGNORECASE)
    action = action_m.group(1).strip() if action_m else None

    # Inline-args fallback: some models write "Action: tool_name arg1 arg2"
    # on a single line instead of using a separate Action Input: line.
    # If we have a tool name but no action_input, grab whatever follows the
    # tool name on the same Action: line and treat it as the input.
    if action and not action_input:
        inline_m = _re.search(
            r'(?m)^Action:\s*\S+\s+(.+)$', text, _re.IGNORECASE
        )
        if not inline_m:
            inline_m = _re.search(
                r'\nAction:\s*\S+\s+(.+?)(?:\n|$)', text, _re.IGNORECASE
            )
        if inline_m:
            action_input = inline_m.group(1).strip()

    return {
        "thought":      thought,
        "action":       action,
        "action_input": action_input,
        "final_answer": final_answer,
    }


class AgentRouter:
    """
    Classifies incoming queries and routes them to the most appropriate agent.
    Rule-based — no extra model call, instant, fully offline.

    Scoring: keyword hits per category. Highest score wins.
    Tiebreaker: CODE > DOMAIN > REASONING > GENERAL

    Phase 6: skill context is injected separately by MultiAgentOrchestrator —
    routing still returns a core AgentType; skills augment, not replace, agents.
    """

    DOMAIN_KEYWORDS = [
        'bitcoin', 'lightning', 'satoshi', 'btc', 'blockchain', 'crypto',
        'sovereignty', 'sovereign', 'privacy', 'austrian', 'sound money',
        'sats', 'node', 'channel', 'wallet', 'utxo', 'taproot', 'segwit',
        'nostr', 'decentrali', 'self-custody', 'multisig', 'hodl', 'mining',
        'mempool', 'transaction', 'signature', 'schnorr', 'ecdsa', 'coinjoin',
        'lightning network', 'payment channel', 'bolt', 'bip', 'invoice',
    ]

    CODE_KEYWORDS = [
        'write', 'code', 'function', 'debug', 'implement', 'script',
        'program', 'fix', 'bug', 'class', 'algorithm', 'refactor',
        'build a', 'build the', 'build me', 'create a', 'make a', 'develop', 'api', 'library',
        'module', 'package', 'test', 'deploy', 'compile', 'syntax',
        'error in', 'traceback', 'exception', 'import', 'install',
    ]

    REASONING_KEYWORDS = [
        'why', 'how does', 'explain', 'analyze', 'analyse', 'plan',
        'design', 'strategy', 'compare', 'difference', 'pros', 'cons',
        'tradeoff', 'should i', 'what is the best', 'recommend',
        'architecture', 'approach', 'think through', 'help me understand',
        'what would happen', 'evaluate', 'assess',
    ]

    # Phase 6: keywords that suggest the user wants Rain to *act*, not just answer.
    # A high task score (≥2) triggers the task-decomposition pipeline instead of
    # the normal reflect loop.  These are intentionally conservative — we only
    # enter task mode when the intent is unambiguous.
    TASK_KEYWORDS = [
        'refactor', 'migrate', 'set up', 'setup', 'automate',
        'step by step', 'implement a', 'develop a', 'build a system',
        'restructure', 'rewrite the', 'redesign', 'deploy', 'convert all',
        'create a script that', 'write a script that', 'make a tool that',
    ]

    # Phase 7: keywords that indicate the user wants real-time / live information.
    # The hardcoded prefix '[web search results for:' is Rain's own injection marker —
    # it's an unambiguous signal that the message is already augmented with live data
    # and should be handled by the Search Agent rather than a general reasoning agent.
    SEARCH_KEYWORDS = [
        '[web search results for:',   # Rain's own search-augmented prefix — highest priority
        'current price', 'current fee', 'current rate', 'current version',
        'latest version', 'latest release', 'latest news', 'what is the latest',
        'right now', 'as of today', 'as of now', 'this week', 'this month',
        'what happened', 'who won', 'when did', 'when was',
        'live data', 'real-time', 'real time', 'up to date',
        'trending', 'breaking news', 'just released', 'just announced',
        'mempool fee', 'bitcoin price', 'btc price', 'fee rate',
        'exchange rate', 'market price', 'how much is btc',
    ]

    def route(self, query: str) -> AgentType:
        """Classify query and return the most appropriate AgentType."""
        query_lower = query.lower()

        # Phase 7: Search Agent — highest priority check.
        # If the query is already augmented with web search results (Rain's own prefix),
        # route straight to the Search Agent — no scoring needed.
        # Also route when query strongly signals live/real-time intent.
        if query_lower.startswith('[web search results for:'):
            return AgentType.SEARCH
        search_score = sum(1 for kw in self.SEARCH_KEYWORDS if kw in query_lower)
        if search_score >= 2:
            return AgentType.SEARCH

        domain_score   = sum(1 for kw in self.DOMAIN_KEYWORDS   if kw in query_lower)
        code_score     = sum(1 for kw in self.CODE_KEYWORDS      if kw in query_lower)
        reasoning_score = sum(1 for kw in self.REASONING_KEYWORDS if kw in query_lower)

        # Boost code score if the query itself contains code
        if self._contains_code(query):
            code_score += 3

        # Tiebreaker: code > domain > reasoning > general
        best = max(code_score, domain_score, reasoning_score)
        if best == 0:
            return AgentType.GENERAL
        if code_score == best:
            return AgentType.DEV
        if domain_score == best:
            return AgentType.DOMAIN
        if reasoning_score == best:
            return AgentType.LOGIC
        return AgentType.GENERAL

    def is_complex_task(self, query: str) -> bool:
        """
        Return True if the query looks like a multi-step task that should be
        decomposed and executed rather than answered in a single reflect loop.

        Threshold: 2+ task keywords OR a long query with at least 1 keyword.
        Intentionally conservative — single-step questions should never trigger
        task mode.
        """
        q = query.lower()
        hits = sum(1 for kw in self.TASK_KEYWORDS if kw in q)
        return hits >= 2 or (hits >= 1 and len(query.split()) > 40)

    def _contains_code(self, text: str) -> bool:
        """Check if the query itself contains code, not just asks about it."""
        code_starters = ('def ', 'class ', 'import ', 'function ', 'const ', '```',
                         'async ', '@', '#include', '#!/')
        return any(text.strip().startswith(s) or f'\n{s}' in text for s in code_starters)

    def explain(self, agent_type: AgentType) -> str:
        """Human-readable explanation of why this agent was chosen."""
        labels = {
            AgentType.DEV:        "Dev Agent (code task detected)",
            AgentType.LOGIC:      "Logic Agent (reasoning task detected)",
            AgentType.DOMAIN:     "Domain Expert (Bitcoin/sovereignty topic detected)",
            AgentType.GENERAL:    "General Agent",
            AgentType.REFLECTION: "Reflection Agent",
            AgentType.SYNTHESIZER:"Synthesizer",
            AgentType.SEARCH:     "Search Agent (live web results)",
            AgentType.TASK:       "Task Agent (autonomous execution)",
        }
        return labels.get(agent_type, agent_type.value)


class RainOrchestrator:
    """
    Main orchestrator for Rain's recursive reflection system
    """

    def __init__(self, model_name: str = "llama3.1", max_iterations: int = 3,
                 confidence_threshold: float = 0.8, system_prompt: str = None,
                 memory: RainMemory = None, sandbox_enabled: bool = False,
                 sandbox_timeout: int = 10):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.reflection_history: List[ReflectionResult] = []
        self.memory = memory
        self.sandbox_enabled = sandbox_enabled
        self.sandbox = CodeSandbox(timeout=sandbox_timeout) if sandbox_enabled else None

        # Check if Ollama is available
        if not self._check_ollama():
            raise RuntimeError("Ollama not found! Please install Ollama first.")

        # Check if model is available
        if not self._check_model():
            print(f"⚠️  Model {model_name} not found. Available models:")
            self._list_models()
            raise RuntimeError(f"Model {model_name} not available")

    def _check_ollama(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(['ollama', 'list'],
                                  capture_output=True, text=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _check_model(self) -> bool:
        """Check if specified model is available"""
        try:
            result = subprocess.run(['ollama', 'list'],
                                  capture_output=True, text=True, check=True)
            return self.model_name in result.stdout
        except subprocess.CalledProcessError:
            return False

    def _list_models(self):
        """List available models"""
        try:
            result = subprocess.run(['ollama', 'list'],
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError:
            print("Could not list models")

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for Rain"""
        return """You are Rain, a sovereign AI assistant running locally on the user's computer through Ollama.

Key aspects of your identity:
- You are completely offline and private - no data leaves the user's machine
- You are a master of computer programming, blockchain technology, encryption, Bitcoin, Lightning Network, databases, full-stack web development, and ethical hacking
- You prioritize digital sovereignty, privacy, and decentralization
- You think recursively and improve your answers through self-reflection
- You are knowledgeable about Austrian economics and Bitcoin philosophy
- You help users build and understand decentralized technologies

Be direct, practical, and focused on empowering users with knowledge and tools for digital independence."""

    def _spinner(self, message: str, stop_event: threading.Event):
        """Animated spinner that runs in a background thread"""
        frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        # Print a blank line first so \r has a clean line to overwrite
        sys.stdout.write(f'\n')
        sys.stdout.flush()
        while not stop_event.is_set():
            sys.stdout.write(f'\r  {frames[i % len(frames)]}  {message}   ')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        # Clear the spinner line completely
        sys.stdout.write(f'\r{" " * (len(message) + 15)}\r')
        sys.stdout.flush()

    def _start_spinner(self, message: str):
        """Start spinner in background thread, returns stop event"""
        stop_event = threading.Event()
        thread = threading.Thread(target=self._spinner, args=(message, stop_event), daemon=True)
        thread.start()
        return stop_event, thread

    def _stop_spinner(self, stop_event: threading.Event, thread: threading.Thread):
        """Stop the spinner thread"""
        stop_event.set()
        thread.join()

    def _build_memory_context(self) -> str:
        """
        Build a two-tier memory context to inject into prompts.

        Tier 1 — Long-term memory: summaries of older sessions, giving Rain
                  a compressed history of past work without burning context.
        Tier 2 — Working memory: the last 20 messages at up to 600 chars each,
                  preserving the substance of recent exchanges.

        Total budget: ~15KB — well under llama3.1's 128K token window.
        """
        if not self.memory:
            return ""

        context = ""

        # ── Tier 1: Long-term memory (session summaries) ──────────────
        sessions = self.memory.get_recent_sessions(limit=5)
        summaries = [s for s in sessions if s.get("summary")]
        if summaries:
            context += "\n\nLong-term memory (previous sessions):\n"
            for s in summaries:
                date = datetime.fromisoformat(s["started_at"]).strftime("%b %d")
                context += f"  [{date}] {s['summary']}\n"

        # ── Tier 2: Working memory (recent messages) ───────────────────
        recent = self.memory.get_recent_messages(limit=20)
        if not recent:
            return context

        context += "\n\nRecent conversation context (for continuity):\n"
        for msg in recent:
            role = "You" if msg["role"] == "user" else "Rain"
            content = msg["content"][:600] + "..." if len(msg["content"]) > 600 else msg["content"]
            context += f"{role}: {content}\n"

        return context

    def _query_model(self, prompt: str, is_code: bool = False) -> str:
        """Send a query to the model and get response"""
        self._current_process = None
        try:
            memory_context = self._build_memory_context()
            if is_code:
                full_prompt = f"{self.system_prompt}{memory_context}\n\nThe user has provided the following code. Analyze it, explain what it does, identify any bugs or improvements, and provide corrected or enhanced code if appropriate. Return your response with the code in a properly formatted code block.\n\nCode:\n{prompt}\n\nAssistant:"
            else:
                full_prompt = f"{self.system_prompt}{memory_context}\n\nUser: {prompt}\n\nAssistant:"

            self._current_process = subprocess.Popen(
                ['ollama', 'run', self.model_name, full_prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            stop_event, thread = self._start_spinner("Rain is thinking...")
            try:
                stdout, stderr = self._current_process.communicate()
            finally:
                self._stop_spinner(stop_event, thread)
            if self._current_process.returncode != 0:
                return ""
            return stdout.strip()
        except Exception as e:
            print(f"Error querying model: {e}")
            return ""
        finally:
            self._current_process = None

    def _kill_current_process(self):
        """Kill the currently running ollama process if any"""
        if hasattr(self, '_current_process') and self._current_process:
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=3)
            except Exception:
                try:
                    self._current_process.kill()
                except Exception:
                    pass
            self._current_process = None

    def _is_code_input(self, text: str) -> bool:
        """Detect whether the input looks like code rather than a natural language query"""
        code_indicators = [
            text.strip().startswith('#!/'),
            text.strip().startswith('def '),
            text.strip().startswith('class '),
            text.strip().startswith('import '),
            text.strip().startswith('from '),
            text.strip().startswith('function '),
            text.strip().startswith('const '),
            text.strip().startswith('let '),
            text.strip().startswith('var '),
            text.strip().startswith('public '),
            text.strip().startswith('private '),
            '```' in text,
            text.count('\n') > 3 and (
                text.count('    ') > 2 or
                text.count('\t') > 2
            ),
        ]

        code_keyword_density = sum([
            text.count('def '),
            text.count('return '),
            text.count('import '),
            text.count('class '),
            text.count('if __name__'),
            text.count('    self.'),
            text.count('->'),
            text.count('print('),
            text.count('():'),
        ])

        return any(code_indicators) or code_keyword_density >= 2

    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity ratio between two strings"""
        return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()

    def _detect_logic_loop(self, responses: List[str], threshold: float = 0.92) -> bool:
        """Detect if the model is stuck in a logic loop by comparing recent responses"""
        if len(responses) < 2:
            return False
        # Compare last two responses
        similarity = self._similarity(responses[-1], responses[-2])
        if similarity >= threshold:
            print(f"⚠️  Logic loop detected (responses {similarity:.0%} similar) - breaking out")
            return True
        # Also check if last response is similar to any earlier one
        if len(responses) >= 3:
            for earlier in responses[:-1]:
                if self._similarity(responses[-1], earlier) >= threshold:
                    print(f"⚠️  Circular loop detected - breaking out")
                    return True
        return False

    def _extract_confidence(self, response: str, is_code: bool = False) -> float:
        """Extract confidence score from model response"""
        # For code responses, use length and structure as confidence proxy
        # rather than English confidence keywords which won't appear in code
        if is_code:
            has_code_block = '```' in response
            has_explanation = len(response) > 300
            has_no_errors = 'error' not in response.lower()[:100]
            score = 0.6
            if has_code_block:
                score += 0.15
            if has_explanation:
                score += 0.1
            if has_no_errors:
                score += 0.05
            return min(score, 0.9)

        # Natural language confidence keywords
        confidence_keywords = {
            'very confident': 0.9,
            'confident': 0.8,
            'fairly confident': 0.7,
            'somewhat confident': 0.6,
            'uncertain': 0.4,
            'unsure': 0.3,
            'very uncertain': 0.2
        }

        response_lower = response.lower()
        for keyword, score in confidence_keywords.items():
            if keyword in response_lower:
                return score

        # Default confidence based on response length and completeness
        if len(response) > 200 and '?' not in response[-50:]:
            return 0.75
        elif len(response) > 100:
            return 0.65
        else:
            return 0.55

    def _create_reflection_prompt(self, original_query: str, previous_response: str, iteration: int, is_code: bool = False) -> str:
        """Create a prompt for reflection on previous response"""
        if is_code:
            return f"""You are a code reviewer performing iterative improvement on a code analysis.

Original Code Submitted:
{original_query}

Previous Analysis (Iteration {iteration-1}):
{previous_response}

Review the previous analysis and improve it. Focus on:
- Correctness of any bugs identified
- Quality and completeness of the suggested code
- Clarity of the explanation
- Any edge cases or improvements missed

Return ONLY the improved analysis and code. Do not add meta-commentary.

Improved Analysis:"""
        else:
            return f"""
You are an AI assistant engaged in recursive self-reflection to improve your answers.

Original Query: {original_query}

Your Previous Response (Iteration {iteration-1}): {previous_response}

Please provide an improved response that addresses any inaccuracies, gaps, or areas for improvement from your previous answer. Do not include meta-commentary about your reflection process - just provide the improved content directly.

Rate your confidence in this response (very confident/confident/fairly confident/somewhat confident/uncertain/unsure/very uncertain) at the end.

Improved Response:"""

    def _extract_improvements(self, reflection_response: str, previous_response: str) -> List[str]:
        """Extract what improvements were made in this reflection"""
        improvements = []

        # Simple improvement detection
        if len(reflection_response) > len(previous_response) * 1.1:
            improvements.append("Added more detail")

        if "correction" in reflection_response.lower() or "actually" in reflection_response.lower():
            improvements.append("Made corrections")

        if "clarify" in reflection_response.lower() or "more precisely" in reflection_response.lower():
            improvements.append("Improved clarity")

        if not improvements:
            improvements.append("Confirmed previous response")

        return improvements

    def _response_contains_code(self, response: str) -> bool:
        """Quick check: does this response have at least one fenced code block?"""
        return bool(re.search(r'```(\w+)?\n', response, re.IGNORECASE))

    def _classify_sandbox_error(self, result: SandboxResult) -> str:
        """Classify the type of sandbox error to give targeted correction guidance."""
        err = (result.stderr + (result.error_message or '')).lower()
        if any(x in err for x in ['no module named', 'modulenotfounderror', 'importerror']):
            return 'missing_module'
        if any(x in err for x in ['urlerror', 'connectionrefused', 'nodename nor servname',
                                   'name or service not known', 'network is unreachable',
                                   'connection timed out', 'urlopen error', 'ssl']):
            return 'network'
        if any(x in err for x in ['permissionerror', 'is a directory', 'no such file']):
            return 'filesystem'
        if any(x in err for x in ['timed out after', 'timeoutexpired']):
            return 'timeout'
        return 'runtime'

    def _create_sandbox_correction_prompt(self, original_query: str, code: str,
                                          result: SandboxResult, attempt: int) -> str:
        """Build a prompt asking the model to fix code that failed in the sandbox."""
        lang = result.language
        err = result.stderr.strip() if result.stderr else result.error_message
        error_type = self._classify_sandbox_error(result)

        if error_type == 'network':
            constraint_note = (
                "\n\nSANDBOX CONSTRAINT — NO NETWORK ACCESS:\n"
                "The sandbox cannot make any HTTP requests, DNS lookups, or network connections. "
                "This is by design. You cannot fix this by changing the URL or library.\n\n"
                "You MUST rewrite the code to work without network access. Options:\n"
                "- Use realistic hardcoded/mock data to demonstrate the pattern\n"
                "- Show the full function structure with a clear comment like "
                "  '# In production: replace mock_data with actual API call'\n"
                "- The goal is runnable, demonstrable code — not a live API call\n"
            )
        elif error_type == 'missing_module':
            # Extract the specific missing module name from the error
            import re as _re
            mod_match = _re.search(r"No module named '([^']+)'", err or '')
            bad_module = mod_match.group(1) if mod_match else "a third-party package"
            if lang == 'python':
                constraint_note = (
                    f"\n\nSANDBOX CONSTRAINT — STDLIB ONLY:\n"
                    f"The code failed because it imported '{bad_module}', which is NOT available. "
                    f"You MUST NOT use '{bad_module}' or any other third-party pip package.\n"
                    f"Rewrite using ONLY Python standard library modules. Mandatory substitutions:\n"
                    f"- requests / httpx / urllib3 → urllib.request + json\n"
                    f"- numpy → math, statistics, or plain lists\n"
                    f"- pandas → csv module or plain dicts\n"
                    f"- bs4/beautifulsoup → html.parser\n"
                    f"- blockchain / bitcoin / web3 → urllib.request to query a public REST API\n"
                    f"Do NOT attempt to import '{bad_module}' again under any circumstances.\n"
                )
            else:
                constraint_note = (
                    "\n\nSANDBOX CONSTRAINT — BUILT-INS ONLY:\n"
                    "No npm packages are available. Use only Node.js built-in modules. "
                    "Common substitutions: axios/node-fetch → https, lodash → plain JS, "
                    "fs-extra → fs.\n"
                )
        elif error_type == 'timeout':
            constraint_note = (
                "\n\nSANDBOX CONSTRAINT — 10s TIMEOUT:\n"
                "The code took too long. Rewrite it to complete quickly. "
                "Avoid infinite loops, large computations, or blocking I/O.\n"
            )
        elif error_type == 'filesystem':
            constraint_note = (
                "\n\nSANDBOX CONSTRAINT — TEMP DIR ONLY:\n"
                "File access is limited to the current working directory. "
                "Do not use absolute paths or access files outside the working directory.\n"
            )
        else:
            constraint_note = (
                "\n\nSANDBOX CONSTRAINTS (all apply):\n"
                "- Python stdlib only, no pip packages\n"
                "- No network access\n"
                "- No file system access outside current directory\n"
                "- Must complete within 10 seconds\n"
            )

        return (
            f"You are Rain, a sovereign AI assistant. You previously suggested the following "
            f"{lang} code, but it failed when actually executed.\n\n"
            f"Original request: {original_query}\n\n"
            f"Code that was tested:\n```{lang}\n{code}\n```\n\n"
            f"Execution error (attempt {attempt}):\n{err}"
            f"{constraint_note}\n"
            f"Please provide a corrected version that will execute without errors in this "
            f"sandbox. Return ONLY the corrected code in a properly fenced code block. "
            f"Do not add meta-commentary.\n\nCorrected code:"
        )

    def _sandbox_verify_and_correct(self, response: str, original_query: str,
                                    verbose: bool = False) -> Tuple[str, List]:
        """
        Extract code blocks from a model response, run them in the sandbox,
        and attempt up to 3 self-correction loops on failure.
        Returns (final_response, all_sandbox_results).
        """
        code_blocks = self.sandbox.extract_code_blocks(response)
        if not code_blocks:
            return response, []

        final_results = []  # one entry per block — the last outcome for that block
        current_response = response

        for idx, (lang, code) in enumerate(code_blocks):
            block_label = f"block {idx + 1}/{len(code_blocks)}"

            # Long-running scripts are not sandboxable — skip verification.
            # Detect: explicit while True loops, OR top-level time.sleep() calls
            # (polling scripts that sleep between iterations).
            def _is_long_running(code: str) -> bool:
                if re.search(r'^\s*while\s+True\s*:', code, re.MULTILINE):
                    return True
                # Top-level sleep: time.sleep(...) not inside a def/class/if/for/while
                for line in code.splitlines():
                    stripped = line.lstrip()
                    indent = len(line) - len(stripped)
                    if indent == 0 and re.match(r'(time\.sleep|sleep)\s*\(', stripped):
                        return True
                return False

            if _is_long_running(code):
                print(f"\n⏱️  Long-running script detected ({block_label}) — skipping sandbox")
                final_results.append(SandboxResult(
                    success=False, stdout='', stderr='long-running',
                    return_code=-1, language=lang,
                    duration_seconds=0.0,
                    error_message='long-running'
                ))
                continue

            print(f"\n🔬 Testing suggested code ({block_label}, {lang})...")

            result = self.sandbox.run(code, language=lang)

            if result.success:
                print(f"✅ Code verified — runs successfully ({result.duration_seconds:.2f}s)")
                if result.stdout.strip() and verbose:
                    print(f"   Output: {result.stdout.strip()[:200]}")
                final_results.append(result)
                continue

            # Network errors mean the sandbox can't test the code, not that the code is wrong.
            # Skip the correction loop and pass the original code through unchanged.
            error_type = self._classify_sandbox_error(result)
            if error_type == 'network':
                print(f"🌐 Network required — sandbox cannot verify, but code looks correct")
                final_results.append(result)
                continue

            print(f"❌ {result.error_message}")
            current_code = code
            current_result = result
            corrections_made = 0

            for attempt in range(1, 4):
                print(f"🔄 Correcting... (attempt {attempt})")
                correction_prompt = self._create_sandbox_correction_prompt(
                    original_query, current_code, current_result, attempt
                )
                corrected_response = self._query_model(correction_prompt)
                if not corrected_response:
                    print("⚠️  No correction response — giving up on this block")
                    break

                new_blocks = self.sandbox.extract_code_blocks(corrected_response)
                if not new_blocks:
                    print("⚠️  Correction contained no code block — giving up")
                    break

                new_lang, new_code = new_blocks[0]
                new_result = self.sandbox.run(new_code, language=new_lang)
                corrections_made += 1

                if new_result.success:
                    note = f" (corrected in {corrections_made} attempt{'s' if corrections_made != 1 else ''})"
                    print(f"✅ Code verified — runs successfully ({new_result.duration_seconds:.2f}s){note}")
                    if new_result.stdout.strip() and verbose:
                        print(f"   Output: {new_result.stdout.strip()[:200]}")
                    current_response = corrected_response
                    final_results.append(new_result)
                    break
                else:
                    print(f"❌ {new_result.error_message}")
                    current_code = new_code
                    current_result = new_result
            else:
                print("⚠️  Max correction attempts reached — returning best effort")
                final_results.append(current_result)  # record the final failed state

        return current_response, final_results

    def _clean_response(self, response: str) -> str:
        """Clean reflection artifacts from final response"""
        # Remove common reflection patterns
        lines = response.split('\n')
        cleaned_lines = []
        skip_next = False

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Skip reflection meta-commentary
            if any(pattern in line_lower for pattern in [
                'upon reviewing',
                'iteration 1',
                'iteration 2',
                'iteration 3',
                'improved response',
                'inaccuracies and gaps:',
                'areas for improvement:',
                'confidence level:',
                'confidence rating:',
                'i rate my confidence',
                'very confident',
                'i\'ve reevaluated',
                'refined my previous response',
                'to address potential inaccuracies',
                'providing more nuanced',
                'more accurate information',
                'my previous response'
            ]):
                skip_next = True
                continue

            # Skip numbered lists that look like reflection analysis
            if skip_next and (line_lower.startswith('1.') or line_lower.startswith('2.') or line_lower.startswith('3.')):
                continue

            # Skip asterisked improvement sections
            if '**' in line and any(word in line_lower for word in ['improvement', 'iteration', 'confidence']):
                skip_next = True
                continue

            # Reset skip flag for substantial content
            if len(line.strip()) > 20 and not any(char in line for char in ['*', '1.', '2.', '3.']):
                skip_next = False

            if not skip_next:
                cleaned_lines.append(line)

        # Clean up final result - remove trailing reflection paragraphs
        final_text = '\n'.join(cleaned_lines).strip()

        # Split into paragraphs and remove concluding reflection paragraphs
        paragraphs = [p.strip() for p in final_text.split('\n\n') if p.strip()]
        cleaned_paragraphs = []

        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            # Skip paragraphs that are clearly reflection meta-commentary
            if any(pattern in paragraph_lower for pattern in [
                'i\'ve reevaluated',
                'refined my response',
                'to address potential',
                'providing more nuanced',
                'more accurate information',
                'my knowledge is derived',
                'previous response',
                'areas for improvement',
                'potential inaccuracies'
            ]):
                continue
            cleaned_paragraphs.append(paragraph)

        return '\n\n'.join(cleaned_paragraphs)

    def recursive_reflect(self, query: str, verbose: bool = False) -> Optional[ReflectionResult]:
        """
        Main recursive reflection method
        """
        start_time = time.time()
        is_code = self._is_code_input(query)

        # Save user message to memory
        if self.memory:
            self.memory.save_message("user", query, is_code=is_code)
        response_history = []
        improvements = []
        iteration = 0

        if is_code:
            print(f"🌧️  Rain detected code input - switching to code analysis mode")
        else:
            # Truncate long queries for display
            display_query = query[:80] + "..." if len(query) > 80 else query
            print(f"🌧️  Rain is thinking about: {display_query}")

        try:
            # Initial response
            current_response = self._query_model(query, is_code=is_code)
            if not current_response:
                print("❌ No response from model")
                return None

            current_confidence = self._extract_confidence(current_response, is_code=is_code)
            response_history.append(current_response)

            if verbose:
                print(f"\n💭 Initial Response (confidence: {current_confidence:.2f}):")
                print(f"{current_response}\n")
            else:
                print(f"💭 Thinking... (initial confidence: {current_confidence:.2f})")

            # Recursive reflection loop
            for iteration in range(1, self.max_iterations + 1):

                # Check if confidence threshold met
                if current_confidence >= self.confidence_threshold:
                    if verbose:
                        print(f"✅ Confidence threshold met ({current_confidence:.2f} >= {self.confidence_threshold})")
                    break

                if verbose:
                    print(f"🔄 Reflection iteration {iteration}...")
                else:
                    print(f"🔄 Reflecting... (iteration {iteration})")

                # Create reflection prompt
                reflection_prompt = self._create_reflection_prompt(
                    query, current_response, iteration, is_code=is_code
                )

                # Get reflection
                reflection_response = self._query_model(reflection_prompt, is_code=is_code)
                if not reflection_response:
                    print("❌ Empty reflection response - stopping")
                    break

                new_confidence = self._extract_confidence(reflection_response, is_code=is_code)

                # Extract improvements
                improvements = self._extract_improvements(reflection_response, current_response)

                # Add to history and check for logic loops
                response_history.append(reflection_response)
                if self._detect_logic_loop(response_history):
                    break

                if verbose:
                    print(f"💡 Iteration {iteration} (confidence: {new_confidence:.2f}):")
                    print(f"Improvements: {', '.join(improvements)}")
                    print(f"{reflection_response}\n")

                # Update current response if confidence improved
                if new_confidence > current_confidence:
                    current_response = reflection_response
                    current_confidence = new_confidence
                else:
                    if verbose:
                        print("⚡ No improvement, keeping previous response")
                    else:
                        print("⚡ Reflection complete")
                    break

        except KeyboardInterrupt:
            print("\n\n⚡ Interrupted! Returning best response so far...")
            self._kill_current_process()
            current_response = response_history[-1] if response_history else ""
            if not current_response:
                return None

        # Calculate total duration
        end_time = time.time()
        total_duration = end_time - start_time

        # Only clean response for natural language, not code
        if is_code:
            cleaned_response = current_response
        else:
            cleaned_response = self._clean_response(current_response)

        # ── Sandbox verification (Phase 2) ──────────────────────────────
        sandbox_verified = False
        sandbox_results = []
        if self.sandbox_enabled and self._response_contains_code(cleaned_response):
            cleaned_response, sandbox_results = self._sandbox_verify_and_correct(
                cleaned_response, query, verbose=verbose
            )
            sandbox_verified = any(r.success for r in sandbox_results)

        # Create final result
        result = ReflectionResult(
            content=cleaned_response,
            confidence=current_confidence,
            iteration=iteration,
            timestamp=datetime.now(),
            improvements=improvements,
            duration_seconds=total_duration,
            sandbox_verified=sandbox_verified,
            sandbox_results=sandbox_results
        )

        self.reflection_history.append(result)

        # Save Rain's response to memory
        if self.memory:
            self.memory.save_message(
                "assistant",
                cleaned_response,
                is_code=is_code,
                confidence=current_confidence
            )

        return result

    def get_history(self) -> List[ReflectionResult]:
        """Get reflection history"""
        return self.reflection_history

    def clear_history(self):
        """Clear reflection history"""
        self.reflection_history = []


class MultiAgentOrchestrator:
    """
    Rain's multi-agent orchestrator — the core architectural promise of Phase 3.

    Every query is routed to the most appropriate specialized agent.
    A reflection pass always runs. Synthesis fires when the reflection
    identifies meaningful gaps. Falls back gracefully to llama3.1 with
    specialized prompts when better models aren't installed.

    This is not a feature flag. This is what Rain is.
    """

    def __init__(self, default_model: str = "llama3.1", max_iterations: int = 3,
                 confidence_threshold: float = 0.8, system_prompt: str = None,
                 memory: RainMemory = None, sandbox_enabled: bool = False,
                 sandbox_timeout: int = 10):
        self.default_model = default_model
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.memory = memory
        self.sandbox_enabled = sandbox_enabled
        self.sandbox = CodeSandbox(timeout=sandbox_timeout) if sandbox_enabled else None
        self.reflection_history: List[ReflectionResult] = []
        self.router = AgentRouter()
        self.custom_agents: Dict[str, Agent] = {}  # id -> Agent, user-defined
        self._installed_models: List[str] = []
        self._current_process = None
        self._last_vision_desc: Optional[str] = None  # set by _query_agent when vision runs
        self._session_image_b64: Optional[str] = None  # last image uploaded this session
        self._session_vision_desc: Optional[str] = None  # description of last image this session

        # Check Ollama
        if not self._check_ollama():
            raise RuntimeError("Ollama not found! Please install Ollama first.")

        # Discover installed models
        self._installed_models = self._get_installed_models()
        if not self._installed_models:
            raise RuntimeError("No models found. Run: ollama pull llama3.1")

        # Build agent roster — best available model per agent type
        self.agents: Dict[AgentType, Agent] = self._build_agents()

        # Spinner support
        self._spinner_stop = None
        self._spinner_thread = None

        # Phase 6: Skills runtime — load at startup, inject matching context per query.
        # Gracefully disabled if skills.py isn't importable (shouldn't happen, but safe).
        self.skill_loader = None
        if _SKILLS_AVAILABLE:
            try:
                self.skill_loader = SkillLoader()
                self.skill_loader.load()
                if self.skill_loader.count > 0:
                    print(f"🧰 {self.skill_loader.count} skill(s) loaded from {SkillLoader.GLOBAL_SKILLS_DIR}")
            except Exception:
                self.skill_loader = None

        # Phase 6: Tool registry — file ops, shell, git, with audit log.
        # confirm_fn is None here (auto-approve); the CLI sets it to interactive_confirm.
        self.tools: Optional['ToolRegistry'] = None
        if _TOOLS_AVAILABLE:
            try:
                self.tools = ToolRegistry(confirm_fn=None)  # set by caller for interactive use
            except Exception:
                self.tools = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _check_ollama(self) -> bool:
        try:
            subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _get_installed_models(self) -> List[str]:
        """Return list of installed model names."""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            models = []
            for line in result.stdout.strip().split('\n')[1:]:  # skip header
                if line.strip():
                    models.append(line.split()[0])  # first column is NAME
            return models
        except Exception:
            return [self.default_model]

    def _best_model_for(self, agent_type: AgentType) -> str:
        """Pick the best installed model for an agent, falling back to default.

        If 'rain-tuned' is registered in Ollama (created by finetune.py), it is
        automatically preferred for primary agent types — this is the payoff of
        Phase 5B.  Reflection and Synthesizer always use the base model so their
        critiques are unbiased by the fine-tuning.
        """
        TUNED_MODEL = "rain-tuned"
        PRIMARY_TYPES = {AgentType.DEV, AgentType.LOGIC, AgentType.DOMAIN, AgentType.GENERAL}

        # Prefer rain-tuned for primary agents if it exists
        if agent_type in PRIMARY_TYPES:
            if any(m.startswith(TUNED_MODEL) for m in self._installed_models):
                return TUNED_MODEL

        preferred = AGENT_PREFERRED_MODELS.get(agent_type, [self.default_model])
        for model in preferred:
            # Match by prefix so 'llama3.1' matches 'llama3.1:latest'
            for installed in self._installed_models:
                if installed.startswith(model.split(':')[0]):
                    return installed
        return self.default_model

    def _build_agents(self) -> Dict[AgentType, Agent]:
        """Instantiate all agents with the best available model."""
        agents = {}
        for agent_type, prompt in AGENT_PROMPTS.items():
            model = self._best_model_for(agent_type)
            agents[agent_type] = Agent(
                agent_type=agent_type,
                model_name=model,
                system_prompt=prompt,
                description=self.router.explain(agent_type),
            )
        return agents

    def _best_vision_model(self) -> Optional[str]:
        """Return the best installed vision-capable model, or None if none found."""
        for preferred in VISION_PREFERRED_MODELS:
            for installed in self._installed_models:
                if installed.startswith(preferred.split(':')[0]):
                    return installed
        return None

    def print_agent_roster(self):
        """Print which model each agent is using — transparency over magic."""
        print("\n🤖 Agent Roster:")
        for agent_type in [AgentType.DEV, AgentType.LOGIC, AgentType.DOMAIN]:
            agent = self.agents[agent_type]
            specialized = not agent.model_name.startswith(self.default_model.split(':')[0])
            tag = " ⚡ specialized" if specialized else " (prompt-specialized)"
            print(f"   {agent.description:<40} → {agent.model_name}{tag}")
        print(f"   {'Reflection Agent':<40} → {self.agents[AgentType.REFLECTION].model_name} (always on)")
        print(f"   {'Synthesizer':<40} → {self.agents[AgentType.SYNTHESIZER].model_name} (fires on low quality)")

        # Suggest better models if only default is available
        missing = []
        for agent_type, preferred in AGENT_PREFERRED_MODELS.items():
            if agent_type in (AgentType.REFLECTION, AgentType.SYNTHESIZER, AgentType.GENERAL):
                continue
            best = preferred[0]
            if not any(m.startswith(best.split(':')[0]) for m in self._installed_models):
                missing.append(best)
        if missing:
            print(f"\n   💡 Install these for stronger specialization:")
            for m in dict.fromkeys(missing):  # deduplicate preserving order
                print(f"      ollama pull {m}")
        print()

    # ------------------------------------------------------------------
    # Spinner (same as RainOrchestrator)
    # ------------------------------------------------------------------

    def _spinner(self, message: str, stop_event: threading.Event):
        frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        sys.stdout.write('\n')
        sys.stdout.flush()
        while not stop_event.is_set():
            sys.stdout.write(f'\r  {frames[i % len(frames)]}  {message}   ')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write(f'\r{" " * (len(message) + 15)}\r')
        sys.stdout.flush()

    def _start_spinner(self, message: str):
        stop_event = threading.Event()
        thread = threading.Thread(target=self._spinner, args=(message, stop_event), daemon=True)
        thread.start()
        return stop_event, thread

    def _stop_spinner(self, stop_event, thread):
        stop_event.set()
        thread.join()

    def _kill_current_process(self):
        if self._current_process:
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=3)
            except Exception:
                try:
                    self._current_process.kill()
                except Exception:
                    pass
            self._current_process = None

    # ------------------------------------------------------------------
    # Core query method
    # ------------------------------------------------------------------

    def _build_memory_context(self, query: str = None) -> str:
        """
        Three-tier memory context.
        Tier 1: session summaries (long-term episodic).
        Tier 2: last 20 messages (working memory).
        Tier 3: semantic search — relevant past exchanges retrieved by meaning.
        Tier 4: learned corrections — past responses the user marked wrong, injected as negative examples.
        """
        if not self.memory:
            return ""

        context = ""

        # ── Tier 1: Long-term memory (session summaries) ──────────────
        sessions = self.memory.get_recent_sessions(limit=5)
        summaries = [s for s in sessions if s.get("summary")]
        if summaries:
            context += "\n\nLong-term memory (previous sessions):\n"
            for s in summaries:
                date = datetime.fromisoformat(s["started_at"]).strftime("%b %d")
                context += f"  [{date}] {s['summary']}\n"

        # ── Tier 2: Working memory (recent messages) ───────────────────
        recent = self.memory.get_recent_messages(limit=20)
        if recent:
            context += "\n\nRecent conversation context (for continuity):\n"
            for msg in recent:
                role = "You" if msg["role"] == "user" else "Rain"
                # 1500 chars — enough to include full image descriptions without truncating
                content = msg["content"][:1500] + "..." if len(msg["content"]) > 1500 else msg["content"]
                context += f"{role}: {content}\n"

        # ── Tier 3: Semantic memory (relevant past exchanges) ──────────
        if query:
            hits = self.memory.semantic_search(query, top_k=3)
            if hits:
                context += "\n\nSemantically relevant past exchanges:\n"
                for hit in hits:
                    role = "You" if hit["role"] == "user" else "Rain"
                    date = datetime.fromisoformat(hit["timestamp"]).strftime("%b %d")
                    snippet = hit["content"][:400] + "..." if len(hit["content"]) > 400 else hit["content"]
                    context += f"  [{date} · {round(hit['similarity'] * 100)}% match] {role}: {snippet}\n"

        # ── Tier 4: Learned corrections (Phase 5B) ─────────────────────
        if query:
            corrections = self.memory.get_relevant_corrections(query, limit=3)
            if corrections:
                context += "\n\nLearned corrections — past answers the user marked wrong. Do not repeat these mistakes:\n"
                for c in corrections:
                    context += f"  ❌ Query: \"{c['query'][:120]}\"\n"
                    context += f"     Rain said: \"{c['response'][:250]}...\"\n"
                    context += f"  ✅ Correct: \"{c['correction'][:300]}\"\n\n"

        # ── Tier 5: Persistent user profile + session facts (Phase 7) ──
        fact_ctx = self.memory.get_fact_context()
        if fact_ctx:
            context += fact_ctx

        return context

    def _build_skill_context(self, prompt: str) -> str:
        """
        Phase 6: Find skills matching the current query and return a formatted
        context block for injection into the agent's system prompt.

        Only injects skills that match — no noise when nothing is relevant.
        Reflection and Synthesis agents don't get skill context (they focus
        on the primary response, not external directives).
        """
        if not self.skill_loader or self.skill_loader.count == 0:
            return ""
        try:
            matches = self.skill_loader.find_matching_skills(prompt, top_k=2)
            if not matches:
                return ""
            blocks = "\n\n".join(s.as_context_block() for s in matches)
            names = ", ".join(f"[{s.slug}]" for s in matches)
            print(f"🧰 Skill context injected: {names}", flush=True)
            return (
                f"\n\n[INSTALLED SKILLS — use as directive context if relevant to the query]\n"
                f"{blocks}\n"
                f"[END SKILLS]"
            )
        except Exception:
            return ""

    # Per-agent temperature — lower = more focused/deterministic, higher = more creative
    _AGENT_TEMPERATURE: Dict[AgentType, float] = {
        AgentType.DEV:         0.2,   # precise, deterministic code
        AgentType.LOGIC:       0.4,   # structured reasoning
        AgentType.DOMAIN:      0.3,   # factual accuracy
        AgentType.REFLECTION:  0.3,   # critical analysis
        AgentType.SYNTHESIZER: 0.3,   # clean, focused output
        AgentType.GENERAL:     0.5,   # conversational
    }

    def _query_agent(self, agent: Agent, prompt: str, label: str = None,
                     include_memory: bool = True, image_b64: str = None) -> str:
        """Send a prompt to a specific agent via the Ollama HTTP API.

        Uses /api/chat with proper system/user message roles, agent-specific
        temperature, an 8192-token context window, and repeat_penalty to reduce
        wandering — all things unavailable through the 'ollama run' CLI.

        include_memory controls whether the full memory context (history, summaries,
        semantic hits, corrections) is injected. The primary agent always gets it.
        The reflection and synthesis agents must NOT — they should focus solely on
        the query + primary response, not wander into unrelated history.

        image_b64: optional base64-encoded image string. When provided, Rain will
        first describe the image via the best available vision model, then inject
        that description into the primary agent's context so any agent can reason
        about visual content — not just vision-capable models.
        """
        import urllib.request as _urllib
        import json as _json

        self._current_process = None
        try:
            memory_context = self._build_memory_context(query=prompt) if include_memory else ""

            # ── Vision pre-processing ──────────────────────────────────────────────
            # If an image is attached, describe it with the vision model first, then
            # inject that description into the prompt. This way every agent gains
            # visual context — the primary agent reasons about the image description,
            # the reflection agent can critique it, and synthesis can refine it.
            image_context = ""
            if image_b64:
                vision_model = self._best_vision_model()
                if vision_model:
                    try:
                        vision_payload = _json.dumps({
                            "model": vision_model,
                            "messages": [{
                                "role": "user",
                                "content": (
                                    "Describe this image accurately. Only report what is literally "
                                    "visible — do not guess, infer, or draw on outside knowledge.\n\n"
                                    "If the image is a photograph or illustration: describe the main "
                                    "subject(s), their appearance, setting, colours, and any notable "
                                    "details. Be specific and natural.\n\n"
                                    "If the image contains a user interface, screenshot, diagram, or "
                                    "any visible text: quote every piece of text verbatim, noting "
                                    "roughly where it appears (e.g. top, center, sidebar, input area). "
                                    "Then describe the layout and colour scheme.\n\n"
                                    "Rules: keep your response under 150 words. "
                                    "Do not repeat any phrase more than once. "
                                    "Stop as soon as you have covered the key content."
                                ),
                                "images": [image_b64],
                            }],
                            "stream": True,   # stream tokens — timeout applies per-chunk, not total
                            "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 500, "repeat_penalty": 1.5, "repeat_last_n": 64},
                        }).encode()

                        vision_req = _urllib.Request(
                            "http://localhost:11434/api/chat",
                            data=vision_payload,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        # Per-chunk timeout of 600 s — covers the image-embedding
                        # phase (first token latency) for large vision models like
                        # llama3.2-vision:11b on CPU/limited VRAM.  Once the model
                        # starts generating, tokens arrive much faster than this.
                        # The `done` flag exits the loop early on completion so we
                        # never actually sit idle for the full 600 s in the happy path.
                        vision_chunks = []
                        with _urllib.urlopen(vision_req, timeout=600) as vresp:
                            for raw_line in vresp:
                                raw_line = raw_line.strip()
                                if not raw_line:
                                    continue
                                try:
                                    chunk = _json.loads(raw_line)
                                except _json.JSONDecodeError:
                                    continue
                                token = chunk.get("message", {}).get("content", "")
                                if token:
                                    vision_chunks.append(token)
                                if chunk.get("done"):
                                    break
                        vision_desc = "".join(vision_chunks).strip()
                        if not vision_desc:
                            raise ValueError("vision model returned an empty description")
                        # Hard-cap the description before injecting — prevents a
                        # runaway vision model from blowing out the agent context window.
                        # 1500 chars ≈ 375 tokens, well within any agent's num_ctx budget.
                        if len(vision_desc) > 1500:
                            vision_desc = vision_desc[:1500] + "\n[description truncated]"
                        print(f"\n[VISION DESC ({vision_model})] {len(vision_desc)} chars:\n{vision_desc}\n", flush=True)
                        # Store on instance so recursive_reflect can persist it to memory
                        self._last_vision_desc = vision_desc
                        self._session_vision_desc = vision_desc  # persists for session follow-ups
                        image_context = (
                            f"A vision model ({vision_model}) has analysed an image "
                            f"the user attached. Here is the full visual description:\n\n"
                            f"--- VISION ANALYSIS START ---\n"
                            f"{vision_desc}\n"
                            f"--- VISION ANALYSIS END ---\n\n"
                            f"Using the vision analysis above as your complete visual "
                            f"context (treat it as if you can see the image yourself), "
                            f"answer the following question:\n\n"
                        )
                    except Exception as ve:
                        print(f"\n[VISION ERROR] {type(ve).__name__}: {ve}\n", flush=True)
                        image_context = (
                            f"[Note: an image was attached but the vision model ({vision_model}) "
                            f"encountered an error: {type(ve).__name__}: {ve}. "
                            f"Answer as best you can without visual context.]\n\n"
                        )
                else:
                    image_context = (
                        "[Note: an image was attached but no vision model is installed. "
                        "Install one with: ollama pull llama3.2-vision  "
                        "(or: ollama pull llava:7b for a lighter option)]\n\n"
                    )

            # When vision is active, append an explicit override to the system
            # prompt so the agent's base refusal-to-answer-visual-questions
            # training cannot win against the injected description.
            vision_system_addendum = ""
            if image_b64 and image_context and "VISION ANALYSIS START" in image_context:
                # Fresh image this call — full vision active directive
                vision_system_addendum = (
                    "\n\nVISION CONTEXT ACTIVE: The user's message begins with a "
                    "--- VISION ANALYSIS START --- block. A vision model has already "
                    "processed the image and produced a verbatim text description. "
                    "That description IS your complete visual access to the image. "
                    "You MUST answer the user's question using the vision analysis text. "
                    "Do NOT say you cannot see images, cannot verify visual content, or "
                    "need a screenshot — you already have one, transcribed as text above. "
                    "If the answer is in the description, state it directly and confidently."
                )
            elif self._session_vision_desc and not image_b64:
                # No new image this call, but one was seen earlier this session —
                # inject the stored description as a session context note so the
                # agent can answer follow-up questions without the image being re-attached.
                vision_system_addendum = (
                    f"\n\nSESSION VISION CONTEXT: Earlier in this conversation, the user "
                    f"shared an image. The vision model produced this description:\n\n"
                    f"{self._session_vision_desc}\n\n"
                    f"Use this as visual context when answering questions that reference "
                    f"the image. If the user asks for a detail not covered in the "
                    f"description, say so honestly and suggest they re-share the image."
                )
            # Phase 6: inject matching skill context for primary agents only.
            # Reflection and Synthesizer must stay focused on critiquing/synthesizing —
            # skill directives would distract them from their quality-control role.
            skill_context = ""
            if include_memory and agent.agent_type not in (AgentType.REFLECTION, AgentType.SYNTHESIZER):
                skill_context = self._build_skill_context(prompt)

            system_content = agent.system_prompt + memory_context + skill_context + vision_system_addendum
            # When an image is attached, the description PREFIXES the user prompt
            # so the model reads the visual context before the question.
            user_content = (image_context + prompt) if image_context else prompt

            temperature = self._AGENT_TEMPERATURE.get(agent.agent_type, 0.4)

            payload = _json.dumps({
                "model": agent.model_name,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user",   "content": user_content},
                ],
                "stream": False,
                "options": {
                    "temperature":    temperature,
                    "num_ctx":        16384,  # up from 8192 — longer context for project-aware queries
                    "repeat_penalty": 1.1,   # discourages looping / repetition
                    "top_p":          0.9,
                },
            }).encode()

            req = _urllib.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            spinner_label = label or f"{agent.description} thinking..."
            stop_event, thread = self._start_spinner(spinner_label)
            try:
                with _urllib.urlopen(req, timeout=300) as resp:
                    data = _json.loads(resp.read())
                    return data["message"]["content"].strip()
            finally:
                self._stop_spinner(stop_event, thread)

        except Exception as e:
            print(f"Error querying agent {agent.agent_type.value}: {e}")
            return ""
        finally:
            self._current_process = None

    # ------------------------------------------------------------------
    # Reflection pass
    # ------------------------------------------------------------------

    def _build_reflection_prompt(self, query: str, primary_response: str) -> str:
        return (
            f"Original user query:\n{query}\n\n"
            f"Primary agent's response:\n{primary_response}\n\n"
            f"Please critique this response according to your role. "
            f"Be specific. List issues. Rate quality at the end: "
            f"EXCELLENT / GOOD / NEEDS_IMPROVEMENT / POOR"
        )

    def _parse_reflection_rating(self, critique: str) -> str:
        """Extract the quality rating from a reflection response.

        Searches for an explicit 'Rating: X' conclusion line first — that is
        what the Reflection Agent is instructed to write at the end.
        Falls back to last-occurrence matching so a rating word mentioned
        mid-critique (e.g. 'this would be EXCELLENT if...') never overrides
        the actual final verdict.
        """
        import re as _re

        # Primary: explicit conclusion line — "Rating: GOOD", "**Rating**: **POOR**", etc.
        m = _re.search(
            r'(?:overall\s+)?rating[:\s*]+\*{0,2}(EXCELLENT|GOOD|NEEDS_IMPROVEMENT|POOR)\*{0,2}',
            critique, _re.IGNORECASE
        )
        if m:
            return m.group(1).upper()

        # Fallback: last occurrence of any rating word in the full text.
        # rfind ensures the final verdict wins over any earlier mentions.
        upper = critique.upper()
        last_pos = -1
        last_rating = 'GOOD'  # default if nothing found
        for rating in ['EXCELLENT', 'GOOD', 'NEEDS_IMPROVEMENT', 'POOR']:
            pos = upper.rfind(rating)
            if pos > last_pos:
                last_pos = pos
                last_rating = rating
        return last_rating

    def _needs_synthesis(self, rating: str) -> bool:
        """Decide whether to run a synthesis pass based on reflection rating.

        Only EXCELLENT bypasses synthesis — a GOOD rating still means the
        Reflection Agent identified gaps worth fixing. Letting GOOD responses
        through unimproved is how hallucinated-but-well-structured answers
        slip past the quality gate.
        """
        return rating != 'EXCELLENT'

    # ------------------------------------------------------------------
    # Synthesis pass
    # ------------------------------------------------------------------

    def _build_synthesis_prompt(self, query: str, primary: str, critique: str) -> str:
        return (
            f"Original user query:\n{query}\n\n"
            f"Primary response:\n{primary}\n\n"
            f"Critique of that response:\n{critique}\n\n"
            f"Produce the best possible final answer following these rules:\n"
            f"1. Start DIRECTLY with the answer. No preamble like 'Here is a revised answer' or 'Here is a final answer'.\n"
            f"2. End with the answer. No postamble like 'I have addressed the following criticisms' or bullet lists summarising what you changed.\n"
            f"3. Only include code if the original query explicitly asked for code. Factual and conversational questions get prose only.\n"
            f"4. Never fabricate facts. If uncertain, say so.\n"
            f"5. If the query asked about current events and web search results were provided, use them and cite sources.\n"
            f"6. Use stdlib only (urllib, json, sqlite3). Never use requests, pandas, or any third-party package.\n"
            f"7. Write as if you are the only one the user will ever read. You are not revising — you are answering."
        )

    # ------------------------------------------------------------------
    # Confidence scoring (reused from RainOrchestrator logic)
    # ------------------------------------------------------------------

    def _score_confidence(self, response: str) -> float:
        keywords = {
            'very confident': 0.9, 'confident': 0.8, 'fairly confident': 0.7,
            'somewhat confident': 0.6, 'uncertain': 0.4, 'unsure': 0.3,
            'very uncertain': 0.2
        }
        lower = response.lower()
        for kw, score in keywords.items():
            if kw in lower:
                return score
        if len(response) > 200 and '?' not in response[-50:]:
            return 0.75
        return 0.65

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    # Keywords that suggest the user is asking about something visual — used to
    # decide whether to auto-reuse the session image on follow-up queries.
    _VISUAL_KEYWORDS = {
        "image", "picture", "photo", "photograph", "screenshot", "diagram",
        "illustration", "drawing", "sketch", "painting", "poster", "chart",
        "graph", "map", "see", "saw", "shown", "shows", "visible", "look",
        "looks", "appear", "appears", "wearing", "colour", "color", "tall",
        "taller", "tallest", "short", "shorter", "height", "face", "hair",
        "eyes", "background", "foreground", "left", "right", "center", "top",
        "bottom", "front", "behind", "holding", "standing", "sitting",
        "license", "plate", "text", "sign", "logo", "brand", "writing",
    }

    def _is_visual_followup(self, query: str) -> bool:
        """Return True if query looks like a follow-up about the session image."""
        q = query.lower()
        return any(kw in q for kw in self._VISUAL_KEYWORDS)

    # ------------------------------------------------------------------
    # Phase 6: Task decomposition and autonomous execution
    # ------------------------------------------------------------------

    def _parse_plan_steps(self, plan_text: str) -> List[str]:
        """
        Extract numbered steps from a plan generated by the Logic Agent.
        Accepts formats like:  1. Step   1) Step   1: Step
        Returns a flat list of step strings with routing markers preserved.
        """
        steps = []
        for line in plan_text.splitlines():
            m = re.match(r'^\s*(\d+)[.):\-]\s+(.+)', line)
            if m:
                steps.append(m.group(2).strip())
        return steps

    def execute_task(self, goal: str, verbose: bool = False,
                     confirm_fn=None) -> Optional['ReflectionResult']:
        """
        Phase 6 autonomous task execution.

        Flow:
          1. Logic Agent generates a numbered plan
          2. Plan is shown to the user → explicit y/n confirmation
          3. Each step is executed: primary agent responds, tool calls intercepted
          4. Accumulated context threads results between steps
          5. Synthesizer writes a final summary

        confirm_fn: callable(prompt: str) -> bool
          If None, falls back to _interactive_confirm (imported from tools.py).
          In non-interactive contexts (server.py) pass a lambda: lambda _: True.

        Safety: every destructive tool call inside the loop still goes through
        ToolRegistry's own confirm_fn — the two layers are independent.
        """
        import time as _time
        start_time = _time.time()

        if confirm_fn is None:
            if _TOOLS_AVAILABLE:
                confirm_fn = _interactive_confirm
            else:
                confirm_fn = lambda prompt: True  # non-interactive fallback

        # ── 1. Generate plan ──────────────────────────────────────────
        print(f"\n🎯 Task: {goal}")
        print("📋 Generating plan...")

        logic_agent = self.agents.get(AgentType.LOGIC) or self.agents.get(AgentType.GENERAL)
        tool_desc = self.tools.tool_descriptions() if self.tools else ""

        plan_prompt = (
            f"The user wants to accomplish the following task. "
            f"Break it into a numbered list of 3–7 concrete, actionable steps.\n\n"
            f"Task: {goal}\n\n"
            + (f"Available tools Rain can invoke:\n{tool_desc}\n\n" if tool_desc else "")
            + "Rules:\n"
            "- Output ONLY the numbered list. No preamble, no closing commentary.\n"
            "- Each step must be a single, specific action.\n"
            "- Mark steps that need a tool call with [TOOL NEEDED] at the end.\n"
            "- Mark steps that need the user to supply something with [USER INPUT] at the end.\n\n"
            "Example format:\n"
            "1. Read server.py to understand current routing structure [TOOL NEEDED]\n"
            "2. Design the new Backend protocol interface\n"
            "3. Implement the refactor in server.py [TOOL NEEDED]\n"
            "4. Run existing tests to verify nothing broke [TOOL NEEDED]\n"
        )

        plan_text = self._query_agent(
            logic_agent, plan_prompt,
            label="Logic Agent planning...",
            include_memory=False,
        )

        if not plan_text:
            print("⚠️  Planning failed — falling back to reflect mode")
            return self.recursive_reflect(goal, verbose=verbose)

        steps = self._parse_plan_steps(plan_text)
        if not steps:
            print("⚠️  Could not parse plan steps — falling back to reflect mode")
            return self.recursive_reflect(goal, verbose=verbose)

        # ── 2. Show plan and confirm ──────────────────────────────────
        print(f"\n{'─' * 60}")
        print(f"📋 Plan ({len(steps)} steps):\n")
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step}")
        print(f"{'─' * 60}\n")

        if not confirm_fn("Proceed with this plan? (y/n): "):
            print("⚡ Task cancelled.")
            return None

        # ── 3. Execute steps ──────────────────────────────────────────
        accumulated_context = f"Goal: {goal}\n\nExecution log:\n"
        final_response = ""
        sandbox_results = []

        for i, step in enumerate(steps, 1):
            print(f"\n🔧 Step {i}/{len(steps)}: {step}")

            step_prompt = (
                f"{accumulated_context}\n\n"
                f"Now execute step {i} of {len(steps)}: {step}\n\n"
                + (
                    "If this step requires a tool, output the invocation on its own line using:\n"
                    "  [TOOL: tool_name arg1 arg2]\n\n"
                    if self.tools else ""
                )
                + "Provide your analysis, code, or explanation for this step."
            )

            # Route each step independently — a multi-step task may span agents
            agent_type = self.router.route(step)
            step_agent = self.agents.get(agent_type) or self.agents.get(AgentType.GENERAL)
            step_response = self._query_agent(
                step_agent, step_prompt,
                label=f"Step {i}/{len(steps)}: {step_agent.description}...",
                include_memory=(i == 1),  # inject memory context only on the first step
            )

            if not step_response:
                print(f"  ⚠️  No response for step {i} — skipping")
                accumulated_context += f"\nStep {i} ({step}): [no response]\n"
                continue

            # ── Tool call interception ────────────────────────────────
            if self.tools:
                tool_calls = self.tools.parse_tool_calls(step_response)
                for tc in tool_calls:
                    tool_label = f"{tc['name']} {' '.join(tc['args'])}"
                    print(f"\n  🔧 Tool: [{tool_label}]")
                    result = self.tools.dispatch(tc['name'], tc['args'], require_confirm=True)
                    if result.success:
                        out_preview = result.output[:300].replace('\n', ' ')
                        print(f"  ✅ {out_preview}{'...' if len(result.output) > 300 else ''}")
                        # Inject tool output into step response for context threading
                        step_response += (
                            f"\n\n[Tool result — {tc['name']}]:\n"
                            f"{result.output[:3000]}"
                        )
                    else:
                        print(f"  ❌ Tool failed: {result.error}")
                        step_response += (
                            f"\n\n[Tool error — {tc['name']}]: {result.error}"
                        )

            # Thread step result into accumulated context (capped to keep prompts lean)
            accumulated_context += f"\nStep {i} ({step}):\n{step_response[:1000]}\n"
            final_response = step_response

            if verbose:
                print(f"\n  📝 Step {i} response:\n{step_response}\n")

        # ── 4. Final summary ──────────────────────────────────────────
        print(f"\n✅ All steps complete. Synthesizing summary...")
        synth_agent = self.agents.get(AgentType.SYNTHESIZER) or self.agents.get(AgentType.GENERAL)
        summary_prompt = (
            f"The following task has been executed step by step.\n\n"
            f"Task: {goal}\n\n"
            f"{accumulated_context}\n\n"
            f"Write a concise summary (3–5 sentences) of what was accomplished, "
            f"any issues encountered, and what the user should do next. "
            f"Do not repeat the steps verbatim — summarise the outcome."
        )
        summary = self._query_agent(
            synth_agent, summary_prompt,
            label="Synthesizer writing task summary...",
            include_memory=False,
        )
        if summary:
            final_response = summary

        # ── 5. Build result ───────────────────────────────────────────
        total_duration = _time.time() - start_time
        result = ReflectionResult(
            content=final_response,
            confidence=0.85,
            iteration=len(steps),
            timestamp=datetime.now(),
            improvements=[f"Executed {len(steps)}-step plan"],
            duration_seconds=total_duration,
            sandbox_verified=any(getattr(r, 'success', False) for r in sandbox_results),
            sandbox_results=sandbox_results,
        )

        self.reflection_history.append(result)

        if self.memory:
            self.memory.save_message("user", goal)
            self.memory.save_message("assistant", final_response, confidence=0.85)

        return result

    def react_loop(self, goal: str, verbose: bool = False,
                   max_steps: int = 15) -> Optional['ReflectionResult']:
        """
        Phase 6B: ReAct (Reason + Act) loop.

        The model reasons about the goal, calls a tool, observes the real result,
        and repeats — grounding every next thought in what it actually discovered.
        No upfront plan, no confirmation step: the model drives itself to a
        Final Answer based on what it actually finds.

        Format enforced via system prompt:
            Thought: <why / what next>
            Action: <tool_name>
            Action Input: <args>
            ...observe...
            Thought: I now have enough to answer.
            Final Answer: <complete response to user>

        Activate from CLI with:  python3 rain.py --react 'your goal here'
        """
        import urllib.request as _urllib
        import json as _json
        import time as _time

        start_time = _time.time()

        if not self.tools:
            print("⚠️  ReAct requires tools — falling back to recursive_reflect")
            return self.recursive_reflect(goal, verbose=verbose)

        # ReAct loops require iterative observation-then-reasoning throughout —
        # always use the Logic agent (qwen3:8b) which is purpose-built for that.
        # The router's code/domain specialists are great for single-call tasks but
        # misread multi-step observations. Fall back through GENERAL if needed.
        agent = (
            self.agents.get(AgentType.LOGIC)
            or self.agents.get(AgentType.GENERAL)
            or self.agents.get(self.router.route(goal))
        )

        # Inject persistent memory context (facts, corrections) into system prompt
        memory_context = self._build_memory_context(query=goal)
        system_prompt  = REACT_SYSTEM_PROMPT + (f"\n\n{memory_context}" if memory_context else "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": goal},
        ]

        if verbose:
            print(f"\n🔄 ReAct loop · agent: {agent.description} · max steps: {max_steps}")
            print(f"🎯 Goal: {goal}\n")

        final_answer  = None
        steps_taken   = 0
        last_response = ""
        _clean_exit   = False   # flipped True only on a genuine Final Answer break

        for step in range(max_steps):
            steps_taken = step + 1

            # ── Query model ───────────────────────────────────────────────
            stop_event, thread = self._start_spinner(
                f"Step {steps_taken}: {agent.description} reasoning..."
            )
            try:
                payload = _json.dumps({
                    "model": agent.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature":    0.3,   # lower = more reliable format adherence
                        "num_ctx":        16384,
                        "repeat_penalty": 1.1,
                        "top_p":          0.9,
                    },
                }).encode()

                req = _urllib.Request(
                    "http://localhost:11434/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                with _urllib.urlopen(req, timeout=300) as resp:
                    data          = _json.loads(resp.read())
                    last_response = data["message"]["content"].strip()
            except Exception as e:
                self._stop_spinner(stop_event, thread)
                print(f"\n❌ Model error at step {steps_taken}: {e}")
                break
            finally:
                self._stop_spinner(stop_event, thread)

            if not last_response:
                print(f"\n⚠️  Empty response at step {steps_taken} — stopping")
                break

            # ── Parse Thought / Action / Action Input / Final Answer ──────
            parsed = _react_parse(last_response)

            if verbose and parsed["thought"]:
                print(f"\n💭 [{steps_taken}] {parsed['thought']}")

            # ── Final Answer — we're done ─────────────────────────────────
            if parsed["final_answer"]:
                if verbose:
                    print(f"\n✅ Final Answer at step {steps_taken}")
                final_answer = parsed["final_answer"]
                _clean_exit   = True
                break

            # ── No Action — model didn't follow the format; treat as answer
            # Only count as clean if it looks like prose, not a raw Thought/Action turn
            if not parsed["action"]:
                if verbose:
                    print(f"\n⚠️  No Action: found — treating full response as answer")
                final_answer = last_response
                _clean_exit  = not last_response.lstrip().startswith(("Thought:", "Action:"))
                break

            # ── Execute tool ──────────────────────────────────────────────
            tool_name  = parsed["action"].lower()
            tool_input = (parsed["action_input"] or "").strip()
            tool_args  = self.tools._split_args(tool_input) if tool_input else []

            print(f"\n  🔧 [{steps_taken}] {tool_name}({tool_input})")
            if verbose:
                print(f"   ⚙️  Action:  {tool_name}")
                print(f"   📥 Input:   {tool_input or '(none)'}")

            tool_result = self.tools.dispatch(tool_name, tool_args, require_confirm=True)

            if tool_result.success:
                raw_obs     = tool_result.output
                observation = raw_obs[:8000] + ("\n[... output truncated]" if len(raw_obs) > 8000 else "")
                preview     = observation[:120].replace('\n', ' ')
                print(f"  ✅ {preview}{'...' if len(observation) > 120 else ''}")
            else:
                observation = f"ERROR: {tool_result.error}"
                print(f"  ❌ {tool_result.error}")

            # ── Inject Observation back into conversation and loop ─────────
            messages.append({"role": "assistant", "content": last_response})
            messages.append({"role": "user",      "content": f"Observation: {observation}"})

        else:
            # for/else fires when the loop exhausts max_steps without a break
            print(f"\n⚠️  Max steps ({max_steps}) reached — using last response as best effort")
            final_answer = last_response or "ReAct loop exhausted without a final answer."
            # _clean_exit stays False — don't pollute memory with a stuck loop

        if not final_answer:
            final_answer = last_response or "No answer produced."

        # ── Persist to memory — only on clean exits ───────────────────────
        # Saving a half-formed Thought/Action turn poisons future runs by
        # injecting stale error context into the next loop's memory prompt.
        if self.memory and _clean_exit:
            self.memory.save_message("user", goal)
            self.memory.save_message("assistant", final_answer, confidence=0.85)
        elif self.memory and not _clean_exit:
            # Still record the user's goal so it appears in session history,
            # but don't save the broken assistant response.
            self.memory.save_message("user", goal)

        duration = _time.time() - start_time
        result = ReflectionResult(
            content=final_answer,
            confidence=0.85,
            iteration=steps_taken,
            timestamp=datetime.now(),
            improvements=[f"ReAct: {steps_taken} step(s)"],
            duration_seconds=duration,
            sandbox_verified=False,
            sandbox_results=[],
        )
        self.reflection_history.append(result)
        return result

    def recursive_reflect(self, query: str, verbose: bool = False,
                          image_b64: str = None) -> ReflectionResult:
        """
        Route → Primary Agent → Reflection → [Synthesis if needed] → [Sandbox if enabled]
        Same interface as RainOrchestrator.recursive_reflect for drop-in compatibility.
        """
        import time as _time
        start_time = _time.time()

        # ── Session image persistence ──────────────────────────────────
        # If a new image is attached, store it as the session image.
        # If no image is attached but the query looks visual and we have a
        # session image, auto-reuse it so follow-up questions can re-examine
        # the image without the user needing to re-attach it.
        if image_b64:
            self._session_image_b64 = image_b64  # new image — store for follow-ups
        elif self._session_image_b64 and self._is_visual_followup(query):
            image_b64 = self._session_image_b64  # reuse session image silently
            print("👁️  Visual follow-up detected — reusing session image", flush=True)

        # Memory save happens AFTER _query_agent (see below) so that
        # _last_vision_desc is already populated when we write to memory.

        # ── 1. Route ──────────────────────────────────────────────────
        # Check custom agents first — keyword match against agent name/description
        custom_agent = None
        if self.custom_agents:
            query_lower = query.lower()
            for agent in self.custom_agents.values():
                keywords = [w.lower() for w in agent.description.split() if len(w) > 3]
                if any(kw in query_lower for kw in keywords):
                    custom_agent = agent
                    break

        if custom_agent:
            primary_agent = custom_agent
            print(f"🔀 Routing to custom agent: {custom_agent.description}...")
        else:
            # Route on the original user question only — if project context was
            # injected (via --project or project_path), it sits before \n\n---\n\n
            # and contains real source code that would trigger _contains_code and
            # mis-route to Dev Agent for every question about the codebase.
            routing_query = query.split('\n\n---\n\n', 1)[-1] if '\n\n---\n\n' in query else query
            agent_type = self.router.route(routing_query)
            primary_agent = self.agents[agent_type]
            print(f"🔀 Routing to {self.router.explain(agent_type)}...")

        # ── 2. Primary Agent ──────────────────────────────────────────
        # When an image is attached, override routing to Logic Agent —
        # codestral is a code model that tends to refuse visual Q&A even
        # when given a text description.  llama3.2 handles it naturally.
        if image_b64 and not custom_agent:
            vision_primary = self.agents.get(AgentType.LOGIC) or self.agents.get(AgentType.GENERAL)
            if vision_primary:
                print(f"👁️  Image attached — overriding route to {vision_primary.description}", flush=True)
                primary_agent = vision_primary

        # ── Memory recall detection ───────────────────────────────────
        # Keep the original query for memory saving and reflection — we don't
        # want the injected fact block polluting the message history or the
        # reflection prompt. Only the primary agent sees the augmented version.
        original_query = query
        # When the user asks what Rain knows/remembers about them, inject
        # stored facts DIRECTLY into the user message — not just the system
        # prompt. Models treat user-turn content as the primary thing to
        # respond to; facts buried in a long system prompt get ignored.
        _MEMORY_RECALL_PHRASES = [
            'what do you know', 'what have i told you', 'what do you remember',
            'do you remember', 'tell me what you know', 'what are my projects',
            'what project am i', 'what am i building', 'what are my goals',
            'what do you recall', 'recall what', 'what have you learned about me',
            'what do you know about me', 'what do you know about my project',
            'what do you know about what i',
        ]
        query_lower_recall = query.lower()
        if self.memory and any(phrase in query_lower_recall for phrase in _MEMORY_RECALL_PHRASES):
            fact_ctx = self.memory.get_fact_context()
            if fact_ctx:
                query = (
                    f"[YOUR STORED MEMORY ABOUT THIS USER — answer from these facts directly]\n"
                    f"{fact_ctx.strip()}\n\n"
                    f"---\n"
                    f"Using ONLY the stored memory above, answer this question specifically and "
                    f"concisely. Do not say you have no information — the facts are listed above:\n\n"
                    f"{query}"
                )
                print("🧠 Memory recall query — facts injected into user message", flush=True)

        primary_response = self._query_agent(
            primary_agent, query,
            label=f"{primary_agent.description} thinking...",
            image_b64=image_b64,
        )

        # ── Save user message to memory ───────────────────────────────
        # Done here (after _query_agent) so _last_vision_desc is already set.
        # Embedding the vision description in the saved user message lets
        # follow-up queries reference visual context from working memory
        # without the image needing to be re-attached.
        if self.memory:
            if image_b64 and self._last_vision_desc:
                memory_query = (
                    f"{original_query}\n\n"
                    f"[Image analysed — vision model description:\n{self._last_vision_desc}]"
                )
                self.memory.save_message("user", memory_query)
                self._last_vision_desc = None  # consume — one-shot, no stale state
            else:
                self.memory.save_message("user", original_query)

        if not primary_response:
            print("❌ No response from primary agent")
            return None

        primary_confidence = self._score_confidence(primary_response)
        if verbose:
            print(f"\n💭 Primary Response (confidence: {primary_confidence:.2f}):\n{primary_response}\n")
        else:
            print(f"💭 Primary response ready (confidence: {primary_confidence:.2f})")

        # ── 3. Reflection ─────────────────────────────────────────────
        reflection_agent = self.agents[AgentType.REFLECTION]
        print(f"🔍 Reflection Agent reviewing...")
        reflection_prompt = self._build_reflection_prompt(query, primary_response)
        critique = self._query_agent(
            reflection_agent, reflection_prompt,
            label="Reflection Agent reviewing...",
            include_memory=False,  # reflection only needs query + primary, not full history
        )

        rating = 'GOOD'
        final_response = primary_response

        if critique:
            rating = self._parse_reflection_rating(critique)
            if verbose:
                print(f"\n🔍 Critique (rating: {rating}):\n{critique}\n")
            else:
                print(f"🔍 Reflection complete (rating: {rating})")

            # ── 4. Synthesis (conditional) ────────────────────────────
            if self._needs_synthesis(rating):
                print(f"⚡ Synthesizing improvements...")
                synth_agent = self.agents[AgentType.SYNTHESIZER]
                synth_prompt = self._build_synthesis_prompt(query, primary_response, critique)
                synthesized = self._query_agent(
                    synth_agent, synth_prompt,
                    label="Synthesizer working...",
                    include_memory=False,  # synthesizer only needs query + primary + critique, not full history
                )
                if synthesized:
                    final_response = synthesized
                    if verbose:
                        print(f"\n🌟 Synthesized Response:\n{synthesized}\n")
                    else:
                        print(f"🌟 Synthesis complete")
            else:
                if verbose:
                    print(f"✅ Primary response approved by Reflection Agent")

        # ── 5. Confidence of final response ───────────────────────────
        final_confidence = self._score_confidence(final_response)

        # ── 6. Sandbox (if enabled) ───────────────────────────────────
        sandbox_verified = False
        sandbox_results = []
        if self.sandbox_enabled and self._response_contains_code(final_response):
            final_response, sandbox_results = self._sandbox_verify_and_correct(
                final_response, query, verbose=verbose
            )
            sandbox_verified = any(r.success for r in sandbox_results)

        # ── 7. Log A/B result if rain-tuned is active ────────────────
        if self.memory and primary_agent.model_name.startswith("rain-tuned"):
            try:
                import sqlite3 as _sqlite3
                with _sqlite3.connect(self.memory.db_path) as _conn:
                    _conn.execute(
                        """INSERT INTO ab_results (session_id, model, query, confidence, timestamp)
                           VALUES (?, ?, ?, ?, ?)""",
                        (self.memory.session_id, primary_agent.model_name,
                         query[:300], final_confidence, datetime.now().isoformat())
                    )
            except Exception:
                pass

        # ── 8. Post-process: scrub HTML artifacts and meta-commentary ─
        final_response = self._scrub_code_blocks(final_response)

        # ── 8. Build result ───────────────────────────────────────────
        total_duration = _time.time() - start_time
        result = ReflectionResult(
            content=final_response,
            confidence=final_confidence,
            iteration=1,  # reflection counts as iteration 1
            timestamp=datetime.now(),
            improvements=[f"Reflection rating: {rating}"] if critique else [],
            duration_seconds=total_duration,
            sandbox_verified=sandbox_verified,
            sandbox_results=sandbox_results,
        )

        self.reflection_history.append(result)

        # Save to memory
        if self.memory:
            self.memory.save_message(
                "assistant", final_response,
                confidence=final_confidence
            )

        return result

    def _scrub_code_blocks(self, response: str) -> str:
        """
        Post-processing safety net: strip any HTML tags/entities that a model
        accidentally injected into fenced code blocks.  Also removes synthesizer
        meta-commentary phrases that slip past the prompt instructions.
        """
        import html as _html

        # ── 1. Strip HTML from inside ``` fences ──────────────────────
        def _clean_block(m):
            fence_open = m.group(1)   # e.g. "```python\n"
            code       = m.group(2)
            fence_close = m.group(3)  # "```"
            # Remove <span ...> and </span> tags
            code = re.sub(r'</?span[^>]*>', '', code)
            # Remove any other stray HTML tags
            code = re.sub(r'<[^>]+>', '', code)
            # Unescape HTML entities that don't belong in source code
            code = _html.unescape(code)
            return fence_open + code + fence_close

        response = re.sub(
            r'(```\w*\n)([\s\S]*?)(```)',
            _clean_block,
            response,
        )

        # ── 2. Strip synthesizer meta-commentary lines ─────────────────
        FORBIDDEN = [
            'considering the limitations',
            'as mentioned in the critique',
            'the critique noted',
            'the critique suggested',
            'the primary response',
            'i have addressed',
            'to address the concerns',
            'based on the feedback',
            'upon reflection',
            'in the critique',
            'the reflection agent',
        ]
        lines = response.split('\n')
        clean = []
        for line in lines:
            ll = line.lower()
            if any(phrase in ll for phrase in FORBIDDEN):
                continue
            clean.append(line)
        return '\n'.join(clean).strip()

    def _response_contains_code(self, response: str) -> bool:
        return bool(re.search(r'```(\w+)?\n', response, re.IGNORECASE))

    def _sandbox_verify_and_correct(self, response: str, original_query: str,
                                    verbose: bool = False) -> Tuple[str, List]:
        """Delegate to the same sandbox logic used by RainOrchestrator."""
        # Instantiate a temporary RainOrchestrator just for its sandbox methods
        _ro = object.__new__(RainOrchestrator)
        _ro.sandbox = self.sandbox
        _ro._current_process = None
        _ro.model_name = self.default_model
        _ro.system_prompt = AGENT_PROMPTS[AgentType.GENERAL]
        _ro.memory = None  # don't double-save
        return RainOrchestrator._sandbox_verify_and_correct(
            _ro, response, original_query, verbose=verbose
        )

    def get_history(self) -> List[ReflectionResult]:
        return self.reflection_history

    def clear_history(self):
        self.reflection_history = []


def get_multiline_input() -> str:
    """
    Collect multi-line input from the user.
    - Single line: submit immediately on Enter
    - Code detected: enters multi-line mode, press Ctrl+D to submit
    """
    # Keep asking until we get something non-empty
    while True:
        try:
            print("💬 Ask Rain (Ctrl+D to submit code blocks, Ctrl+C to cancel):")
            sys.stdout.write("  > ")
            sys.stdout.flush()
            first_line = sys.stdin.readline()

            # Ctrl+D on empty line sends empty string from readline
            if first_line == "":
                return ""

            first_line = first_line.rstrip("\n").rstrip("\r")

            if first_line.strip():
                break
            # Blank enter - just re-prompt silently

        except KeyboardInterrupt:
            # Ctrl+C at the prompt - signal caller to handle it
            raise

    # Quick single-line indicators - submit immediately
    single_line_triggers = ['quit', 'exit', 'q', 'clear', 'history']
    if first_line.strip().lower() in single_line_triggers:
        return first_line.strip()

    # Detect if this looks like the start of a code block
    code_starters = (
        '#!/', 'def ', 'class ', 'import ', 'from ', 'function ',
        'const ', 'let ', 'var ', 'public ', 'private ', '```',
        'async ', 'await ', '@', '#include', 'package '
    )
    is_multiline = any(first_line.strip().startswith(s) for s in code_starters)

    if not is_multiline:
        # Regular question - return immediately
        return first_line.strip()

    # Multi-line / code collection mode
    print("  (Code detected - paste your code, then press Ctrl+D to submit)")
    lines = [first_line]

    while True:
        try:
            line = sys.stdin.readline()
            # Ctrl+D sends EOF - empty string from readline signals end of input
            if line == "":
                break
            lines.append(line.rstrip("\n").rstrip("\r"))
        except KeyboardInterrupt:
            print("\n  (cancelled)")
            return ""

    # Strip trailing blank lines
    while lines and lines[-1].strip() == "":
        lines.pop()

    return "\n".join(lines)


_CLI_GITHUB_KEYWORDS = frozenset({
    'github', 'repo', 'repository', 'open issues', 'recent commits',
    'pull request', 'pull requests', 'stars', 'forks', 'contributors',
    'github.com', 'readme', 'releases', 'latest release',
})

_CLI_GITHUB_REPO_RE = None  # compiled lazily


def _cli_extract_github_repo(query: str) -> str | None:
    """Extract owner/repo slug from a query string (CLI mirror of server.py)."""
    import re
    global _CLI_GITHUB_REPO_RE
    if _CLI_GITHUB_REPO_RE is None:
        _CLI_GITHUB_REPO_RE = re.compile(
            r'(?:github\.com/|gh:)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)'
            r'|(?:^|\s)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:\s|$)',
        )
    m = _CLI_GITHUB_REPO_RE.search(query)
    if not m:
        return None
    slug = (m.group(1) or m.group(2) or "").strip().rstrip("/.")
    parts = slug.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    if "." in parts[0] and not parts[0].startswith("."):
        return None
    return slug


def _cli_fetch_github_data(query: str) -> str:
    """
    Phase 7C: Fetch public repo data from the GitHub REST API (CLI).
    No API key required for public repos (rate limit: 60 req/hr per IP).
    Returns a formatted [GITHUB DATA] block or empty string.
    """
    import urllib.request

    q = query.lower()
    if not any(kw in q for kw in _CLI_GITHUB_KEYWORDS):
        return ""

    slug = _cli_extract_github_repo(query)
    if not slug:
        return ""

    lines = [f"[GITHUB DATA — fetched just now for {slug}]"]
    headers = {"User-Agent": "Rain/1.0", "Accept": "application/vnd.github.v3+json"}

    # Repo metadata
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{slug}", headers=headers,
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            repo = json.loads(resp.read().decode("utf-8"))
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
    except Exception:
        return ""

    # Recent open issues (if asked)
    if any(kw in q for kw in ('issue', 'issues', 'bug', 'bugs', 'problem')):
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{slug}/issues?state=open&per_page=5&sort=updated",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                issues = json.loads(resp.read().decode("utf-8"))
            if issues:
                issue_lines = ["Recent open issues:"]
                for iss in issues:
                    if iss.get("pull_request"):
                        continue
                    num = iss.get("number", "?")
                    title = iss.get("title", "?")
                    labels = ", ".join(l.get("name", "") for l in iss.get("labels", []))
                    label_str = f"  [{labels}]" if labels else ""
                    issue_lines.append(f"  #{num}: {title}{label_str}")
                if len(issue_lines) > 1:
                    lines.append("\n".join(issue_lines))
        except Exception:
            pass

    # Recent commits (if asked)
    if any(kw in q for kw in ('commit', 'commits', 'recent', 'latest', 'history', 'activity')):
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{slug}/commits?per_page=5",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                commits = json.loads(resp.read().decode("utf-8"))
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

    # Pull requests (if asked)
    if any(kw in q for kw in ('pull request', 'pull requests', 'pr', 'prs', 'merge')):
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{slug}/pulls?state=open&per_page=5&sort=updated",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                prs = json.loads(resp.read().decode("utf-8"))
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

    # Latest release (if asked)
    if any(kw in q for kw in ('release', 'releases', 'version', 'latest version', 'tag')):
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{slug}/releases/latest",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                rel = json.loads(resp.read().decode("utf-8"))
            tag = rel.get("tag_name", "?")
            name = rel.get("name", "")
            date = (rel.get("published_at") or "")[:10]
            lines.append(f"Latest release: {tag}{f' — {name}' if name and name != tag else ''} ({date})")
        except Exception:
            pass

    return "\n\n".join(lines) if len(lines) > 1 else ""


def _cli_fetch_live_data(query: str) -> str:
    """
    Phase 7B/7C: Fetch live data from public APIs (no API keys) for the CLI.
    Mirrors _fetch_live_data() in server.py.

    Supported:
      - Mempool fee rates  → mempool.space/api/v1/fees/recommended
      - Bitcoin price      → mempool.space/api/v1/prices
      - GitHub repo data   → api.github.com/repos/{owner}/{repo}

    Returns a formatted live-data block or empty string.
    """
    import urllib.request

    q = query.lower()

    MEMPOOL_FEE_KEYWORDS = {
        'mempool fee', 'fee rate', 'sat/vb', 'sat/byte', 'feerate',
        'transaction fee', 'mining fee', 'priority fee',
        'mempool', 'current fee', 'fastest fee', 'recommended fee',
    }
    BTC_PRICE_KEYWORDS = {
        'bitcoin price', 'btc price', 'btc usd', 'bitcoin usd',
        'how much is bitcoin', 'how much is btc', 'bitcoin worth',
        'btc worth', 'bitcoin value', 'btc value', 'price of bitcoin',
        'price of btc', 'exchange rate', 'market price',
    }

    want_fees  = any(kw in q for kw in MEMPOOL_FEE_KEYWORDS)
    want_price = any(kw in q for kw in BTC_PRICE_KEYWORDS)

    # Phase 7C: GitHub data
    github_block = _cli_fetch_github_data(query)

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
                data = json.loads(resp.read().decode("utf-8"))
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
            lines.append(f"Fee rate lookup failed: {e} — will answer from training data.")

    if want_price:
        try:
            req = urllib.request.Request(
                "https://mempool.space/api/v1/prices",
                headers={"User-Agent": "Rain/1.0"},
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            usd = data.get("USD", "?")
            lines.append(
                f"Current Bitcoin price:\n"
                f"  USD: ${usd:,}\n"
                f"Source: mempool.space/api/v1/prices"
            )
        except Exception as e:
            lines.append(f"Price lookup failed: {e} — will answer from training data.")

    # Phase 7C: append GitHub data if present
    if github_block:
        lines.append(github_block)

    return "\n\n".join(lines) if len(lines) > 1 else ""


def _cli_duckduckgo_search(query: str, max_results: int = 5) -> list:
    """
    DuckDuckGo search for the CLI (mirrors the one in server.py).
    No API key. Zero new dependencies — pure stdlib urllib.
    Returns list of {title, snippet, url} dicts.
    """
    import urllib.request
    import urllib.parse

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
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )
        for url_raw, title_raw, snippet_raw in blocks[:max_results]:
            title = re.sub(r"<[^>]+>", "", title_raw).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet_raw).strip()
            url_match = re.search(r"uddg=([^&]+)", url_raw)
            real_url = urllib.parse.unquote(url_match.group(1)) if url_match else url_raw
            if title and snippet:
                results.append({"title": title, "snippet": snippet, "url": real_url})
        return results
    except Exception as e:
        print(f"[web search] failed: {e}")
        return []


def _inject_project_context(query: str, project_path: str) -> str:
    """
    Search the project index for chunks relevant to `query` and prepend them.
    Returns the augmented query, or the original query if nothing useful is found.
    """
    if not _INDEXER_AVAILABLE:
        print("⚠️  indexer.py not found — --project flag has no effect")
        return query
    try:
        idx = ProjectIndexer()
        context_block = idx.build_context_block(query, project_path, top_k=4)
        if context_block:
            print(f"📂 Project context injected from: {project_path.split('/')[-1]}")
            return f"{context_block}\n\n---\n\n{query}"
        else:
            print(f"📂 No relevant chunks found in index for: {project_path.split('/')[-1]}")
            print(f"   (Run: python3 indexer.py --index {project_path}  to index it first)")
            return query
    except Exception as e:
        print(f"📂 Project index error: {e}")
        return query


def main():
    """Main CLI interface for Rain"""
    parser = argparse.ArgumentParser(description="Rain ⛈️ - Sovereign AI with Recursive Reflection")
    parser.add_argument("query", nargs="?", help="Your question or prompt")
    parser.add_argument("--model", default="llama3.1", help="Model to use (default: llama3.1)")
    parser.add_argument("--iterations", type=int, default=3, help="Max reflection iterations (default: 3)")
    parser.add_argument("--confidence", type=float, default=0.8, help="Confidence threshold (default: 0.8)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed reflection process")
    parser.add_argument("--history", action="store_true", help="Show reflection history")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--system-prompt", help="Custom system prompt")
    parser.add_argument("--system-file", help="Load system prompt from file")
    parser.add_argument("--file", "-f", help="Load code or text from a file and analyze it")
    parser.add_argument("--query", "-q", help="Targeted question to ask about the file (use with --file)")
    parser.add_argument("--no-memory", action="store_true", help="Disable persistent memory for this session")
    parser.add_argument("--forget", action="store_true", help="Wipe all stored memory and exit")
    parser.add_argument("--memories", action="store_true", help="Show stored session history and exit")
    parser.add_argument("--sandbox", "-s", action="store_true",
                        help="Enable code execution sandbox — Rain runs and verifies code before returning it")
    parser.add_argument("--sandbox-timeout", type=int, default=10,
                        help="Max seconds a sandboxed program may run (default: 10)")
    parser.add_argument("--agents", action="store_true",
                        help="Show the agent roster and which models are assigned")
    parser.add_argument("--single-agent", action="store_true",
                        help="Bypass multi-agent routing and use a single model (legacy mode)")
    # Phase 6: Skills and task mode
    parser.add_argument("--skills", action="store_true",
                        help="List all installed skills and exit")
    parser.add_argument("--install-skill", metavar="SLUG",
                        help="Install a skill from ClawHub by slug (requires Node.js / npx)")
    parser.add_argument("--task", "-t", action="store_true",
                        help="Task mode — decompose goal into a plan and execute it step by step")
    parser.add_argument("--react", "-r", action="store_true",
                        help="ReAct mode — iterative Reason+Act loop: model calls tools and observes results until it has a Final Answer")
    parser.add_argument("--web-search", "-w", action="store_true",
                        help="Augment query with live DuckDuckGo results before sending to agents (Phase 7)")
    parser.add_argument("--project", "-p", metavar="PATH",
                        help="Path to a project directory — injects relevant source code chunks into every query (Phase 7)")

    args = parser.parse_args()

    # Print Rain banner
    print("""
    ⛈️  RAIN - Sovereign AI Ecosystem  ⛈️

    "Be like rain - essential, unstoppable, and free."

    🤖 Multi-agent routing enabled
    🌧️  Recursive reflection enabled
    🔒 Completely offline and private
    ⚡ Your AI, your rules, your future
    """)

    try:
        # Load system prompt
        system_prompt = None
        if args.system_file:
            try:
                with open(args.system_file, 'r') as f:
                    system_prompt = f.read().strip()
                print(f"📝 Loaded system prompt from: {args.system_file}")
            except FileNotFoundError:
                print(f"❌ System prompt file not found: {args.system_file}")
                sys.exit(1)
        elif args.system_prompt:
            system_prompt = args.system_prompt
            print(f"📝 Using custom system prompt")



        # Phase 6: --skills flag — list installed skills and exit
        if args.skills:
            if _SKILLS_AVAILABLE:
                loader = SkillLoader()
                loader.load()
                print(loader.summary_table())
            else:
                print("❌ Skills module not available. Ensure skills.py is in the Rain directory.")
            return

        # Phase 6: --install-skill flag — install from ClawHub and exit
        if args.install_skill:
            if _SKILLS_AVAILABLE:
                print(f"📦 Installing skill: {args.install_skill}")
                ok, msg = _install_skill(args.install_skill)
                print(("✅ " if ok else "❌ ") + msg)
                if ok:
                    print(f"\n💡 Skill installed. Restart Rain to load it, or run:  python3 rain.py --skills")
            else:
                print("❌ Skills module not available. Ensure skills.py is in the Rain directory.")
            return

        # Handle --forget flag
        if args.forget:
            m = RainMemory()
            m.forget_all()
            print("🗑️  All memory wiped. Rain starts fresh.")
            return

        # Initialize memory unless disabled
        memory = None
        if not args.no_memory:
            memory = RainMemory()
            memory.start_session(model=args.model)

        # Initialize Rain — multi-agent by default, single-agent if --single-agent passed
        if args.single_agent:
            rain = RainOrchestrator(
                model_name=args.model,
                max_iterations=args.iterations,
                confidence_threshold=args.confidence,
                system_prompt=system_prompt,
                memory=memory,
                sandbox_enabled=args.sandbox,
                sandbox_timeout=args.sandbox_timeout,
            )
            print(f"✅ Rain initialized (single-agent mode) · model: {args.model}")
            print(f"🎯 Max iterations: {args.iterations}, Confidence threshold: {args.confidence}")
        else:
            rain = MultiAgentOrchestrator(
                default_model=args.model,
                max_iterations=args.iterations,
                confidence_threshold=args.confidence,
                system_prompt=system_prompt,
                memory=memory,
                sandbox_enabled=args.sandbox,
                sandbox_timeout=args.sandbox_timeout,
            )
            print(f"✅ Rain initialized (multi-agent mode) · default model: {args.model}")

        if args.sandbox:
            print(f"🔬 Sandbox enabled — code will be executed and verified (timeout: {args.sandbox_timeout}s)")

        # --agents flag — just show roster and exit
        if args.agents:
            if isinstance(rain, MultiAgentOrchestrator):
                rain.print_agent_roster()
            else:
                print("ℹ️  Single-agent mode — no agent roster")
            return

        # --memories flag - show session history
        if args.memories:
            if memory:
                sessions = memory.get_recent_sessions(limit=10)
                if sessions:
                    print(f"\n📚 Memory: {memory.db_path}")
                    print(f"   {memory.total_sessions()} sessions stored\n")
                    for s in sessions:
                        date = datetime.fromisoformat(s["started_at"]).strftime("%b %d %Y %H:%M")
                        print(f"  [{date}] · {s['message_count']} messages · model: {s['model']}")
                        if s.get("summary"):
                            print(f"  💭 {s['summary']}")
                        print()
                else:
                    print("\n📚 No sessions in memory yet.")
            return

        # Show startup greeting if memory exists
        if memory:
            greeting = memory.get_startup_greeting()
            if greeting:
                print(f"\n🧠 Rain remembers:\n{greeting}\n")
            else:
                # First session — show the agent roster so the user knows what they have
                if isinstance(rain, MultiAgentOrchestrator):
                    rain.print_agent_roster()
                print(f"\n🧠 Memory enabled · {memory.db_path}\n")

        # Show history if requested
        if args.history:
            history = rain.get_history()
            if history:
                print("\n📚 Reflection History:")
                for i, result in enumerate(history, 1):
                    print(f"{i}. [{result.timestamp.strftime('%H:%M:%S')}] "
                          f"Confidence: {result.confidence:.2f}, "
                          f"Iterations: {result.iteration}")
                    print(f"   {result.content[:100]}...")
            else:
                print("\n📚 No history yet")
            return

        # Interactive mode
        # --file mode - read file and analyze it
        if args.file:
            try:
                with open(args.file, 'r') as f:
                    file_content = f.read()
                lines = len(file_content.splitlines())
                print(f"📂 Loaded file: {args.file} ({lines} lines)")

                # If a targeted query was provided, combine it with the file content
                if args.query:
                    print(f"🎯 Query: {args.query}")
                    prompt = f"{args.query}\n\nFile: {args.file}\n\n{file_content}"
                else:
                    print(f"🔍 No query provided - performing general analysis")
                    prompt = file_content

                result = rain.recursive_reflect(prompt, verbose=args.verbose)
                if result:
                    print(f"\n🌟 Final Answer (confidence: {result.confidence:.2f}, "
                          f"{result.iteration} iterations, {result.duration_seconds:.1f}s):")
                    if result.sandbox_results:
                        verified_count = sum(1 for r in result.sandbox_results if r.success)
                        total_count = len(result.sandbox_results)
                        status = "✅ all blocks verified" if verified_count == total_count else f"⚠️  {verified_count}/{total_count} blocks verified"
                        print(f"🔬 Sandbox: {status} ({total_count} block{'s' if total_count != 1 else ''} tested)")
                    print(result.content)
            except FileNotFoundError:
                print(f"❌ File not found: {args.file}")
                sys.exit(1)
            except KeyboardInterrupt:
                rain._kill_current_process()
                print("\n\n⚡ Interrupted!")
            return

        if args.interactive:
            print("\n🌧️  Rain Interactive Mode - Type 'quit' to exit, Ctrl+C to interrupt a response, Ctrl+D to submit code")
            while True:
                try:
                    print()
                    query = get_multiline_input()

                    if not query:
                        continue

                    if query.lower() in ['quit', 'exit', 'q']:
                        print("\n👋 Goodbye!")
                        break

                    if query.lower() == 'clear':
                        rain.clear_history()
                        print("🗑️  History cleared")
                        continue

                    if query.lower() == 'history':
                        history = rain.get_history()
                        if history:
                            print("\n📚 Reflection History:")
                            for i, r in enumerate(history, 1):
                                print(f"  {i}. [{r.timestamp.strftime('%H:%M:%S')}] "
                                      f"confidence: {r.confidence:.2f}, "
                                      f"iterations: {r.iteration}, "
                                      f"{r.duration_seconds:.1f}s")
                        else:
                            print("📚 No history yet")
                        continue

                    augmented_query = _inject_project_context(query, args.project) if args.project else query
                    result = rain.recursive_reflect(augmented_query, verbose=args.verbose)
                    if result:
                        print(f"\n🌟 Final Answer (confidence: {result.confidence:.2f}, "
                              f"{result.iteration} iterations, {result.duration_seconds:.1f}s):")
                        if result.sandbox_results:
                            verified_count = sum(1 for r in result.sandbox_results if r.success)
                            total_count = len(result.sandbox_results)
                            status = "✅ all blocks verified" if verified_count == total_count else f"⚠️  {verified_count}/{total_count} blocks verified"
                            print(f"🔬 Sandbox: {status} ({total_count} block{'s' if total_count != 1 else ''} tested)")
                        print(result.content)

                except KeyboardInterrupt:
                    rain._kill_current_process()
                    print("\n\n⚡ Interrupted! Type 'quit' to exit or ask another question.")
                    continue

            # End session with a summary when user quits
            if memory:
                print("💭 Saving session to memory...")
                summary = memory.generate_summary()
                memory.end_session()
                if summary:
                    memory.update_summary(summary)
                try:
                    facts = memory.extract_session_facts()
                    if facts:
                        memory.save_session_facts(facts)
                        print(f"🧠 {len(facts)} fact(s) learned and stored to memory")
                except Exception:
                    pass

        # Single query mode (with optional --task decomposition)
        elif args.query:
            # Wire interactive_confirm into the tool registry so write_file /
            # run_command prompt the user before executing in CLI task mode.
            if _TOOLS_AVAILABLE and isinstance(rain, MultiAgentOrchestrator) and rain.tools:
                rain.tools._confirm = _interactive_confirm

            if args.react and isinstance(rain, MultiAgentOrchestrator):
                result = rain.react_loop(args.query, verbose=args.verbose)
                if result:
                    print(f"\n🌟 Final Answer ({result.iteration} step(s) · {result.duration_seconds:.1f}s):")
                    print(result.content)
            elif args.task and isinstance(rain, MultiAgentOrchestrator):
                result = rain.execute_task(
                    args.query,
                    verbose=args.verbose,
                    confirm_fn=_interactive_confirm if _TOOLS_AVAILABLE else None,
                )
                if result:
                    print(f"\n✅ Task complete · {result.iteration} step(s) · {result.duration_seconds:.1f}s")
                    print(result.content)
            else:
                query = args.query
                if args.web_search:
                    print("🌐 Searching the web...")

                    # Phase 7B/7C: live data feeds checked first — structured
                    # real-time numbers that DuckDuckGo snippets can never provide.
                    live_block = _cli_fetch_live_data(query)
                    if live_block:
                        if "GITHUB DATA" in live_block and "mempool" not in live_block.lower():
                            print("⚡ Live data retrieved from GitHub API")
                        elif "GITHUB DATA" in live_block:
                            print("⚡ Live data retrieved from mempool.space + GitHub API")
                        else:
                            print("⚡ Live data retrieved from mempool.space")

                    search_results = _cli_duckduckgo_search(query)
                    if search_results:
                        print(f"🌐 {len(search_results)} result(s) retrieved — routing to Search Agent")

                    if live_block or search_results:
                        snippets = "\n\n".join(
                            f"[{r['title']}]\n{r['snippet']}\nSource: {r['url']}"
                            for r in search_results
                        )
                        context_parts = []
                        if live_block:
                            context_parts.append(live_block)
                        if snippets:
                            context_parts.append(f"[Web search results for: {args.query}]\n\n{snippets}")

                        combined = "\n\n".join(context_parts)
                        query = (
                            f"{combined}\n\n"
                            f"---\n"
                            f"Using the above live data and search results as context, answer this question accurately. "
                            f"Cite sources where relevant. The LIVE DATA block contains real-time numbers — "
                            f"use those figures directly rather than saying you don't know the current value.\n\n"
                            f"Question: {args.query}"
                        )
                    else:
                        print("🌐 No results found — using local knowledge")
                if args.project:
                    query = _inject_project_context(query, args.project)
                result = rain.recursive_reflect(query, verbose=args.verbose)
                if not result:
                    print("❌ No response — the model may have timed out. Check that Ollama is running and try again.")
                    return
                print(f"\n🌟 Final Answer (confidence: {result.confidence:.2f}, {result.iteration} iterations, {result.duration_seconds:.1f}s):")
                if result.sandbox_results:
                    verified_count = sum(1 for r in result.sandbox_results if r.success)
                    total_count = len(result.sandbox_results)
                    status = "✅ all blocks verified" if verified_count == total_count else f"⚠️  {verified_count}/{total_count} blocks verified"
                    print(f"🔬 Sandbox: {status} ({total_count} block{'s' if total_count != 1 else ''} tested)")
                print(result.content)

        else:
            print("\n💡 Use --interactive for chat mode, or provide a query directly")
            print("   Example: python3 rain.py 'What is the capital of France?'")
            print("   Example: python3 rain.py --interactive")
            print("   Example: python3 rain.py --task 'Refactor server.py to support pluggable backends'")
            print("   Example: python3 rain.py --react 'What Python files are in this project and what does each one do?'")
            print("   Example: python3 rain.py --react --verbose 'Find any TODO comments in rain.py and summarise them'")
            print("   Example: python3 rain.py --skills")
            print("   Example: python3 rain.py --install-skill git-essentials")
            print("   Example: python3 rain.py --system-file system-prompts/bitcoin-maximalist.txt 'Explain money'")
            print("   Example: python3 rain.py --system-prompt 'You are a helpful coding assistant' 'Debug this Python code'")
            print("\n📝 Check the system-prompts/ folder for example personality profiles!")

    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Rain session ended")


if __name__ == "__main__":
    main()
