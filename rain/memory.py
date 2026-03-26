"""
Rain ⛈️ — Persistent Memory System

6-tier memory architecture with SQLite storage:
  Tier 1: Episodic (session summaries)
  Tier 2: Working memory (recent messages)
  Tier 2.5: Session anchor (pinned opening messages)
  Tier 3: Semantic (vector retrieval via nomic-embed-text)
  Tier 4: Corrections (plausibility-filtered)
  Tier 5: User profile + session facts
  Tier 6: Knowledge graph (external — see knowledge_graph.py)
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class RainMemory:
    """
    Persistent memory for Rain using local SQLite.
    Stores sessions and messages in ~/.rain/memory.db
    Zero external dependencies - uses Python built-in sqlite3.
    """

    def __init__(self, test_mode: bool = False):
        self.db_path = Path.home() / ".rain" / "memory.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now()
        # test_mode: feedback and gap writes are suppressed so diagnostic
        # sessions can't poison the calibration table.
        self.test_mode = test_mode
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
                    agent_type TEXT,
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
                    query_embedding BLOB,
                    agent_type TEXT,
                    confidence REAL
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

            # ── Migrate existing feedback table (add calibration columns if absent) ──
            try:
                conn.execute("ALTER TABLE feedback ADD COLUMN agent_type TEXT")
            except Exception:
                pass  # column already exists
            try:
                conn.execute("ALTER TABLE feedback ADD COLUMN confidence REAL")
            except Exception:
                pass  # column already exists
            # ── Migrate existing messages table (add agent_type column if absent) ──
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN agent_type TEXT")
            except Exception:
                pass  # column already exists
            # ── Migrate existing feedback table (add plausibility column if absent) ──
            try:
                conn.execute("ALTER TABLE feedback ADD COLUMN plausibility_score REAL")
            except Exception:
                pass  # column already exists
            try:
                conn.execute("ALTER TABLE feedback ADD COLUMN access_count INTEGER DEFAULT 0")
            except Exception:
                pass  # column already exists
            # ── Create synthesis_log table (dual-response logging — Priority 2) ──
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS synthesis_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    primary_response TEXT NOT NULL,
                    synthesized_response TEXT NOT NULL,
                    primary_confidence REAL,
                    synthesis_confidence REAL,
                    rating TEXT,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_synthesis_log_hash
                    ON synthesis_log(query_hash);
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
            "Extract key facts about the USER (not about Rain/AI) from this conversation.\n"
            "Return a JSON array of objects with exactly these fields:\n"
            '  "type": one of: technology, project, preference, goal, person\n'
            '  "key": use ONLY these canonical keys when applicable:\n'
            '    user_name, preferred_language, active_project, os_platform,\n'
            '    tech_stack, goal, organization, expertise_area, project_type\n'
            '    (use a descriptive snake_case key only if none of the above fit)\n'
            '  "value": short string (e.g. "Python", "Rain trading bot")\n\n'
            "Rules:\n"
            "- Only facts about the human user — skip facts about Rain, AI, or general knowledge.\n"
            "- Only include clearly stated facts, not guesses.\n"
            "- Skip facts where the value is a generic word like 'time', 'array', 'my_project'.\n"
            "- Maximum 8 facts. Prefer quality over quantity.\n"
            "- Return ONLY the JSON array — no preamble, no explanation, no markdown.\n\n"
            f"Conversation:\n{transcript}\n\nFacts JSON:"
        )

        try:
            payload = json.dumps({
                "model": "llama3.2",
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

    # Canonical key aliases — map noisy LLM-extracted keys to stable canonical keys
    _KEY_ALIASES = {
        "language": "preferred_language",
        "programming_language": "preferred_language",
        "coding_language": "preferred_language",
        "primary_language": "preferred_language",
        "project": "active_project",
        "project_name": "active_project",
        "current_project": "active_project",
        "main_project": "active_project",
        "name": "user_name",
        "username": "user_name",
        "platform": "os_platform",
        "operating_system": "os_platform",
        "expertise": "expertise_area",
        "experience": "expertise_area",
    }
    # Values that are clearly garbage / too generic to store
    _GARBAGE_VALUES = {
        "my_project", "project", "code", "time", "array", "stdlib",
        "unknown", "n/a", "none", "null", "undefined", "yes", "no",
        "true", "false", "1", "0", "it", "this", "that",
    }
    # Keys whose values are Rain-the-system, not user facts
    _AI_SYSTEM_KEYS = {"name", "rain", "rain_name", "ai_name", "ai_type", "ai_system_name"}

    def save_session_facts(self, facts: List[Dict]):
        """
        Persist extracted session facts and roll persistent ones into user_profile.
        Normalizes keys via _KEY_ALIASES and filters garbage before writing.
        """
        if not facts:
            return
        now = datetime.now().isoformat()
        PERSISTENT_TYPES = {"preference", "project", "technology", "person", "goal"}
        def _normalize(fact: Dict) -> Optional[Dict]:
            """Normalize key aliases and filter garbage. Returns None to skip."""
            raw_key   = fact.get("key", "").strip().lower()
            raw_value = fact.get("value", "").strip()
            if not raw_key or not raw_value:
                return None
            # Drop AI-system facts (Rain talking about itself)
            if raw_key in self._AI_SYSTEM_KEYS:
                return None
            if raw_value.lower() in {"rain", "rain ai", "sovereign ai"}:
                return None
            # Drop garbage values
            if raw_value.lower() in self._GARBAGE_VALUES or len(raw_value) <= 1:
                return None
            # Normalize key to canonical form
            key = self._KEY_ALIASES.get(raw_key, raw_key)
            return {**fact, "key": key, "value": raw_value}

        normalized = [n for f in facts if (n := _normalize(f)) is not None]
        if not normalized:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                for fact in normalized:
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
                for fact in normalized:
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

    def get_fact_context(self, query: str = None) -> str:
        """
        Build a Tier 5 context block from the user profile and recent session facts.
        Injected into every agent prompt to give Rain persistent knowledge of the user.

        Phase 11: when query is provided, session facts are relevance-filtered via
        keyword overlap so only topically relevant facts consume context budget.
        User profile rows are always included (persistent identity facts).
        Returns an empty string if nothing is stored yet.
        """
        import re as _re

        _STOP = {
            'the','a','an','is','are','was','were','be','been','have','has','had',
            'do','does','did','will','would','could','should','may','might','must',
            'can','to','of','in','for','on','with','at','by','from','up','about',
            'into','i','you','it','we','they','what','which','who','this','that',
            'and','or','but','not','so','if','then','than','there','how','when',
        }

        def _kw_score(text: str) -> float:
            if not query:
                return 1.0
            q_words = {w for w in _re.findall(r'\w+', query.lower()) if w not in _STOP and len(w) > 2}
            if not q_words:
                return 1.0
            t_words = {w for w in _re.findall(r'\w+', text.lower()) if w not in _STOP and len(w) > 2}
            return len(q_words & t_words) / len(q_words)

        parts = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # User profile — persistent facts, always injected (identity context)
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

                # Recent session facts — relevance-gated when query is provided
                fact_rows = conn.execute(
                    """SELECT sf.fact_type, sf.fact_key, sf.fact_value
                       FROM session_facts sf
                       WHERE sf.session_id != ?
                       ORDER BY sf.timestamp DESC
                       LIMIT 40""",
                    (self.session_id,),
                ).fetchall()
                if fact_rows:
                    if query:
                        # Keep facts with any keyword overlap, cap at 12
                        filtered = [
                            r for r in fact_rows
                            if _kw_score(f"{r['fact_key']} {r['fact_value']}") > 0.05
                        ][:12]
                        if not filtered:
                            filtered = list(fact_rows[:5])  # always keep 5 most recent
                    else:
                        filtered = list(fact_rows[:20])
                    lines = [
                        f"  [{r['fact_type']}] {r['fact_key']}: {r['fact_value']}"
                        for r in filtered
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

    def save_message(self, role: str, content: str, is_code: bool = False,
                     confidence: float = None, agent_type: str = None):
        """Save a single message and asynchronously embed it for semantic search.

        agent_type records which agent produced this response — used by
        implicit feedback detection to correlate follow-up signals with the
        correct agent for calibration purposes.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO messages
                       (session_id, timestamp, role, content, is_code, confidence, agent_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.session_id, datetime.now().isoformat(), role, content,
                 int(is_code), confidence, agent_type)
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

    def save_feedback(self, query: str, response: str, rating: str,
                      correction: str = None, agent_type: str = None,
                      confidence: float = None):
        """Persist a thumbs-up/down signal (and optional correction) from the user.

        agent_type and confidence are recorded for calibration — over time Rain
        learns which agents produce reliable responses and adjusts their
        confidence scores accordingly.

        In test_mode, feedback writes are suppressed so diagnostic sessions
        can't poison the calibration table.
        """
        if self.test_mode:
            print("🧪 TEST MODE — feedback not recorded", flush=True)
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO feedback
                       (session_id, query, response, rating, correction, timestamp,
                        agent_type, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.session_id, query, response, rating, correction,
                 datetime.now().isoformat(), agent_type, confidence)
            )
            feedback_id = cursor.lastrowid

        # Embed the query in background so we can do semantic retrieval later
        t = threading.Thread(
            target=self._store_feedback_embedding,
            args=(feedback_id, query),
            daemon=True,
        )
        t.start()

        # Compute correction plausibility in background (bad ratings with corrections only)
        # This checks whether Rain's response is consistent with its historically good answers
        # on the same topic — if so, the correction may be a false negative (user error).
        if rating == 'bad' and correction:
            t2 = threading.Thread(
                target=self._compute_and_store_plausibility,
                args=(feedback_id, query, response),
                daemon=True,
            )
            t2.start()

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

    def _calculate_plausibility_score(self, query: str, response: str) -> Optional[float]:
        """
        Pure computation: assess how plausible a correction is by checking whether
        Rain's response is consistent with its historically good answers on the same topic.

        Returns a float in [0, 1] if enough history exists to make a judgment, or None
        if there are no prior good-rated responses on the same topic (unknown territory).

        High score (> 0.5) → Rain has been consistent here; the correction is suspicious.
        Low score or None  → new territory; correction should be treated as authoritative.

        Called synchronously before implicit-feedback calibration updates (to gate false
        corrections) and from the background plausibility thread (to persist the score).
        """
        response_vec = self._embed(response[:1500])
        if response_vec is None:
            return None

        query_vec = self._embed(query)
        if query_vec is None:
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """SELECT query, response, query_embedding FROM feedback
                       WHERE rating = 'good'
                         AND query_embedding IS NOT NULL
                         AND response IS NOT NULL
                         AND response != ''
                       ORDER BY id DESC LIMIT 100"""
                ).fetchall()
        except Exception:
            return None

        if not rows:
            return None

        # Find good-rated responses on the same topic (similar query)
        similar_responses = []
        for fb_query, fb_response, emb_blob in rows:
            try:
                fb_query_vec = json.loads(emb_blob.decode("utf-8"))
                if self._cosine_similarity(query_vec, fb_query_vec) > 0.5:
                    similar_responses.append(fb_response)
            except Exception:
                pass

        if not similar_responses:
            # No history on this topic — plausibility is unknown
            return None

        # Compare Rain's current response against those good-rated responses.
        # High similarity = Rain has been consistent with what users previously approved.
        similarities = []
        for resp_text in similar_responses[:10]:          # cap to avoid slow loops
            resp_vec = self._embed(resp_text[:1500])
            if resp_vec:
                similarities.append(self._cosine_similarity(response_vec, resp_vec))

        if not similarities:
            return None

        return round(max(similarities), 3)   # use max: one strong match is enough

    def _compute_and_store_plausibility(self, feedback_id: int, query: str, response: str):
        """
        Assess plausibility of a user correction and persist the score to the DB.

        Delegates computation to _calculate_plausibility_score(), then writes the result
        to the feedback row so get_relevant_corrections() can annotate suspicious entries.

        Runs in a background thread at correction-ingestion time.  Score is stored once
        and read back by get_relevant_corrections() at prompt-build time.
        """
        plausibility = self._calculate_plausibility_score(query, response)
        if plausibility is None:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE feedback SET plausibility_score = ? WHERE id = ?",
                    (plausibility, feedback_id)
                )
        except Exception:
            pass

    def get_session_anchor(self, session_id: str, limit: int = 3) -> List[dict]:
        """
        Return the first `limit` messages of the given session.

        These act as an anchor — the opening messages of a session establish
        goals, constraints, and project context that must never drift out of
        working memory as the conversation grows past the recent-messages window.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT role, content, timestamp FROM messages
                       WHERE session_id = ?
                       ORDER BY id ASC LIMIT ?""",
                    (session_id, limit)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_calibration_factors(self) -> dict:
        """
        Compute per-agent confidence adjustment factors from feedback history.

        For each agent type with at least 5 ratings in the last 90 days:
          accuracy  = good_count / total_count
          factor    = 0.7 + (accuracy * 0.6)   → [0.7 … 1.3], clamped to [0.72, 1.10]

        A factor < 1.0 deflates confidence scores, making synthesis trigger more
        often for unreliable agents.  A factor > 1.0 inflates them, letting
        reliable agents skip reflection more frequently.

        Returns an empty dict when there is not yet enough data.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """SELECT agent_type,
                              SUM(CASE WHEN rating = 'good' THEN 1 ELSE 0 END) AS good_count,
                              COUNT(*) AS total_count
                       FROM feedback
                       WHERE agent_type IS NOT NULL
                         AND timestamp > datetime('now', '-90 days')
                       GROUP BY agent_type"""
                ).fetchall()

            factors = {}
            for agent_type, good_count, total_count in rows:
                if total_count < 5:
                    continue  # not enough data to calibrate yet
                accuracy = good_count / total_count
                raw = 0.7 + (accuracy * 0.6)
                factors[agent_type] = round(max(0.72, min(1.10, raw)), 3)
            return factors
        except Exception:
            return {}

    def log_synthesis(self, query_hash: str, primary_response: str,
                      synthesized_response: str, primary_confidence: float,
                      synthesis_confidence: float) -> int:
        """
        Record both the primary and synthesized response for a query.

        Returns the synthesis_log row ID.  Over time this table reveals whether
        the synthesizer is actually improving responses or just adding latency.
        Call update_synthesis_rating() when the user provides feedback so the
        rating can be attached retroactively.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """INSERT INTO synthesis_log
                           (session_id, query_hash, primary_response, synthesized_response,
                            primary_confidence, synthesis_confidence, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (self.session_id, query_hash,
                     primary_response[:3000], synthesized_response[:3000],
                     primary_confidence, synthesis_confidence,
                     datetime.now().isoformat())
                )
                return cursor.lastrowid
        except Exception:
            return -1

    def update_synthesis_rating(self, query_hash: str, rating: str):
        """
        Attach a user rating to the most recent synthesis_log entry for this query.
        Called by the feedback endpoint after the user submits thumbs-up/down so
        we can track whether synthesis improved or degraded the final response.

        Two-stage match:
          1. Primary   — exact query_hash match (precise, works when the query
                         the UI sends is identical to what was hashed at synthesis time)
          2. Fallback  — most recent unrated entry in the current session
                         (handles cases where context injection caused the hash to
                         diverge between synthesis time and feedback time)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """UPDATE synthesis_log SET rating = ?
                       WHERE query_hash = ?
                         AND id = (SELECT MAX(id) FROM synthesis_log
                                   WHERE query_hash = ?)""",
                    (rating, query_hash, query_hash)
                )
                if cursor.rowcount == 0:
                    # Hash didn't match — fall back to the most recent unrated
                    # synthesis entry for this session so ratings are never lost.
                    conn.execute(
                        """UPDATE synthesis_log SET rating = ?
                           WHERE session_id = ?
                             AND rating IS NULL
                             AND id = (SELECT MAX(id) FROM synthesis_log
                                       WHERE session_id = ? AND rating IS NULL)""",
                        (rating, self.session_id, self.session_id)
                    )
        except Exception:
            pass

    def get_synthesis_accuracy(self) -> dict:
        """
        Return synthesis effectiveness stats for the agent roster display.

        Returns a dict with:
          total                       — all synthesis runs recorded
          rated                       — runs with user feedback attached
          improved                    — runs rated 'good' by the user
          improvement_rate            — good / rated  (0.0 when no ratings yet)
          confidence_improved         — runs where synthesis_confidence > primary_confidence
          confidence_improvement_rate — confidence_improved / total
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM synthesis_log"
                ).fetchone()[0]
                rated = conn.execute(
                    "SELECT COUNT(*) FROM synthesis_log WHERE rating IS NOT NULL"
                ).fetchone()[0]
                improved = conn.execute(
                    "SELECT COUNT(*) FROM synthesis_log WHERE rating = 'good'"
                ).fetchone()[0]
                conf_improved = conn.execute(
                    """SELECT COUNT(*) FROM synthesis_log
                       WHERE synthesis_confidence > primary_confidence"""
                ).fetchone()[0]
            return {
                'total': total,
                'rated': rated,
                'improved': improved,
                'improvement_rate': round(improved / rated, 3) if rated > 0 else 0.0,
                'confidence_improved': conf_improved,
                'confidence_improvement_rate': round(conf_improved / total, 3) if total > 0 else 0.0,
            }
        except Exception:
            return {}

    def get_relevant_corrections(self, query: str, limit: int = 3) -> List[dict]:
        """
        Find past corrections that are semantically relevant to the current query.
        Only returns 'bad'-rated feedback that has a user-supplied correction.
        Uses cosine similarity on embedded queries; falls back to recency if
        embeddings are unavailable.

        Corrections with plausibility_score > 0.5 are prefixed with
        [LOW CONFIDENCE CORRECTION] — Rain gave a similar answer before and users
        liked it, so this correction may be a false negative (user error rather
        than Rain error).  The prefix signals the model to treat the correction
        with caution rather than accepting it as authoritative.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Phase 11: decay old corrections that have been seen rarely.
            # Corrections older than 90 days are only kept if accessed 3+ times
            # (meaning they fired repeatedly and are genuinely useful).
            rows = conn.execute(
                """SELECT id, query, response, correction, query_embedding,
                          plausibility_score, timestamp,
                          COALESCE(access_count, 0) AS access_count
                   FROM feedback
                   WHERE rating = 'bad' AND correction IS NOT NULL AND correction != ''
                     AND (
                       CAST(julianday('now') - julianday(COALESCE(timestamp, '2000-01-01')) AS INTEGER) <= 90
                       OR COALESCE(access_count, 0) >= 3
                     )
                   ORDER BY id DESC LIMIT 50"""
            ).fetchall()

        if not rows:
            return []

        def _annotate(correction_text: str, plausibility) -> str:
            """Prefix suspicious corrections so the model treats them with caution."""
            if plausibility is not None and plausibility > 0.5:
                return f"[LOW CONFIDENCE CORRECTION — Rain has given consistent answers on this topic before; verify before accepting] {correction_text}"
            return correction_text

        query_vec = self._embed(query)
        if query_vec is None:
            # No embedding available — return most recent corrections
            return [
                {"query": r[1], "response": r[2], "correction": _annotate(r[3], r[5])}
                for r in rows[:limit]  # r[5] = plausibility_score, unchanged
            ]

        scored = []
        for row in rows:
            _, fb_query, fb_response, fb_correction, emb_blob, fb_plausibility, _ts, _ac = row
            if emb_blob:
                try:
                    fb_vec = json.loads(emb_blob.decode("utf-8"))
                    sim = self._cosine_similarity(query_vec, fb_vec)
                    scored.append((sim, fb_query, fb_response, fb_correction, fb_plausibility))
                except Exception:
                    pass

        scored.sort(key=lambda x: x[0], reverse=True)
        # Only inject corrections with meaningful similarity (>0.5)
        results = []
        for item in scored[:limit]:
            if item[0] > 0.5:
                s, q, r, c, plausibility = item
                results.append({
                    "query": q,
                    "response": r,
                    "correction": _annotate(c, plausibility),
                    "similarity": round(s, 2),
                    "plausibility": plausibility,
                })
        return results

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
        """Get recent messages including the current session for working memory.

        Previously this excluded the current session (session_id != ?), which
        meant Rain had no in-session context: it couldn't answer 'what was my
        first question?' and prior responses in this conversation were invisible,
        causing format drift (e.g. bullet patterns reinforced from old sessions).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT m.role, m.content, m.timestamp, m.is_code, s.started_at
                   FROM messages m
                   JOIN sessions s ON m.session_id = s.id
                   ORDER BY m.timestamp DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def get_current_session_messages(self) -> List[Dict]:
        """Get messages from the current session only, in chronological order.

        Used for conversation history recall ("what was my first question?") and
        for Tier 2 working memory injection — distinct from get_recent_messages
        which spans sessions. Includes rowid for targeted deletion in pruning.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT m.rowid, m.role, m.content, m.timestamp, m.is_code, s.started_at
                   FROM messages m
                   JOIN sessions s ON m.session_id = s.id
                   WHERE m.session_id = ?
                   ORDER BY m.timestamp ASC""",
                (self.session_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def log_knowledge_gap(self, session_id: str, query: str, gap: str, confidence: float):
        """Persist a knowledge gap Rain identified in its own response.

        Called when synthesis fired or confidence was low — Rain records what it
        was uncertain about so patterns can be surfaced later via get_top_gaps().
        Suppressed in test_mode to keep diagnostic sessions from polluting the table.
        """
        if self.test_mode:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS knowledge_gaps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        query TEXT,
                        gap_description TEXT,
                        confidence REAL,
                        resolved INTEGER DEFAULT 0,
                        timestamp TEXT
                    )""",
                )
                conn.execute(
                    """INSERT INTO knowledge_gaps
                           (session_id, query, gap_description, confidence, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, query[:300], gap[:500], confidence,
                     datetime.now().isoformat())
                )
        except Exception:
            pass

    def get_top_gaps(self, limit: int = 5) -> List[Dict]:
        """Return the most common unresolved knowledge gaps across all sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT gap_description, COUNT(*) as frequency,
                              AVG(confidence) as avg_confidence
                       FROM knowledge_gaps
                       WHERE resolved = 0
                       GROUP BY gap_description
                       HAVING COUNT(*) >= 2
                       ORDER BY frequency DESC, avg_confidence ASC
                       LIMIT ?""",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def prune_session_memory(self, keep_recent: int = 20, model_query_fn=None) -> int:
        """Deliberate forgetting — compress old session messages when the session grows large.

        When a session exceeds `keep_recent * 2` messages, the oldest half is
        summarized into a single compressed entry in the session_facts table and
        deleted from the messages table.  The most recent `keep_recent` messages
        are always preserved verbatim.

        Returns the number of messages pruned (0 if nothing to do).
        """
        msgs = self.get_current_session_messages()
        prune_threshold = keep_recent * 2  # only prune if session is genuinely large
        if len(msgs) <= prune_threshold:
            return 0

        # Split: old half to compress, recent half to keep
        split = len(msgs) - keep_recent
        to_prune = msgs[:split]
        if not to_prune:
            return 0

        # Build a summary of the pruned messages
        summary_lines = []
        for msg in to_prune:
            role = "User" if msg["role"] == "user" else "Rain"
            summary_lines.append(f"{role}: {msg['content'][:300]}")
        raw = "\n".join(summary_lines)

        if model_query_fn:
            # Ask the model to compress — produces a tighter summary
            compressed = model_query_fn(
                f"Summarize the following conversation excerpt in 3-5 sentences, "
                f"capturing the key topics, decisions, and facts. Be concise:\n\n{raw}"
            )
        else:
            # Fallback: naive truncation if no model available
            compressed = f"[Compressed: {len(to_prune)} earlier messages — {raw[:400]}...]"

        # Persist the compressed summary as a session fact
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO session_facts (session_id, fact, confidence, timestamp)
                       VALUES (?, ?, ?, ?)""",
                    (self.session_id,
                     f"[Pruned memory summary] {compressed}",
                     0.9,
                     datetime.now().isoformat())
                )
                # Delete the pruned messages
                ids = [m["rowid"] for m in to_prune if "rowid" in m]
                if ids:
                    conn.executemany("DELETE FROM messages WHERE rowid = ?", [(i,) for i in ids])
        except Exception:
            pass

        return len(to_prune)

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

