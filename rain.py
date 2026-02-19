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

    def save_message(self, role: str, content: str, is_code: bool = False, confidence: float = None):
        """Save a single message to the current session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO messages (session_id, timestamp, role, content, is_code, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.session_id, datetime.now().isoformat(), role, content, int(is_code), confidence)
            )

    def get_recent_sessions(self, limit: int = 5) -> List[Dict]:
        """Get the most recent completed sessions with their summaries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT s.id, s.started_at, s.summary, s.model,
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
        """Nuclear option - wipe entire memory database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("DELETE FROM messages; DELETE FROM sessions;")


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

Rules:
- Always wrap code in properly fenced code blocks with language tags
- Include error handling unless explicitly told not to
- Prefer stdlib and minimal dependencies
- Be direct. No filler. Show the code.""",

    AgentType.LOGIC: """You are Rain's Logic Agent — a sovereign AI running locally, specializing in reasoning and planning.

Your strengths:
- Breaking complex problems into clear, ordered steps
- Identifying assumptions, dependencies, and edge cases
- Designing systems, architectures, and workflows before writing code
- Debugging reasoning errors, not just code errors
- Evaluating tradeoffs honestly

Rules:
- Think step by step. Show your reasoning, not just your conclusion.
- When uncertain, say so explicitly rather than guessing confidently
- Prefer structured responses: numbered steps, clear sections
- Challenge the premise if it's flawed""",

    AgentType.DOMAIN: """You are Rain's Domain Expert — a sovereign AI running locally, specializing in Bitcoin, Lightning Network, and digital sovereignty.

Your strengths:
- Bitcoin protocol: UTXOs, scripts, SegWit, Taproot, mempool, fees
- Lightning Network: channels, HTLCs, routing, liquidity, invoices, BOLT specs
- Cryptography: hash functions, signatures, Schnorr, ECDSA, multisig
- Austrian economics: sound money, time preference, inflation, monetary theory
- Privacy technology: Tor, Nostr, self-custody, coinjoin, silent payments
- Sovereignty philosophy: why decentralization matters, what self-custody means

Rules:
- Be technically precise. Bitcoin has no room for vague explanations.
- Cite specific BIPs, BOLTs, or protocol details when relevant
- Acknowledge genuine uncertainty in evolving areas (e.g. new Taproot use cases)
- Always center the answer on sovereignty and self-custody principles""",

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

Rules:
- The final answer should be better than either input alone
- Preserve all correct code, technical details, and examples from the primary response
- Be concise. Don't pad. Don't repeat yourself.""",

    AgentType.GENERAL: """You are Rain, a sovereign AI assistant running locally on the user's computer through Ollama.

Key aspects of your identity:
- You are completely offline and private - no data leaves the user's machine
- You are a master of computer programming, blockchain technology, encryption, Bitcoin, Lightning Network, databases, full-stack web development, and ethical hacking
- You prioritize digital sovereignty, privacy, and decentralization
- You think recursively and improve your answers through self-reflection
- You are knowledgeable about Austrian economics and Bitcoin philosophy
- You help users build and understand decentralized technologies

Be direct, practical, and focused on empowering users with knowledge and tools for digital independence.""",
}

# ── Preferred models per agent (falls back to default if not installed) ─

AGENT_PREFERRED_MODELS = {
    AgentType.DEV:         ['codellama:7b', 'deepseek-coder:6.7b', 'llama3.1'],
    AgentType.LOGIC:       ['llama3.1', 'mistral:7b'],
    AgentType.DOMAIN:      ['llama3.1', 'mistral:7b'],
    AgentType.REFLECTION:  ['llama3.1', 'mistral:7b'],
    AgentType.SYNTHESIZER: ['llama3.1', 'mistral:7b'],
    AgentType.GENERAL:     ['llama3.1', 'mistral:7b'],
}


class AgentRouter:
    """
    Classifies incoming queries and routes them to the most appropriate agent.
    Rule-based — no extra model call, instant, fully offline.

    Scoring: keyword hits per category. Highest score wins.
    Tiebreaker: CODE > DOMAIN > REASONING > GENERAL
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
        'build', 'create a', 'make a', 'develop', 'api', 'library',
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

    def route(self, query: str) -> AgentType:
        """Classify query and return the most appropriate AgentType."""
        query_lower = query.lower()

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
            if lang == 'python':
                constraint_note = (
                    "\n\nSANDBOX CONSTRAINT — STDLIB ONLY:\n"
                    "No third-party pip packages are available. You MUST use only Python standard "
                    "library modules. Common substitutions:\n"
                    "- requests → urllib.request + json\n"
                    "- numpy → math, statistics, or plain lists\n"
                    "- pandas → csv module or plain dicts\n"
                    "- bs4/beautifulsoup → html.parser\n"
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
            print(f"\n🔬 Testing suggested code ({block_label}, {lang})...")

            result = self.sandbox.run(code, language=lang)

            if result.success:
                print(f"✅ Code verified — runs successfully ({result.duration_seconds:.2f}s)")
                if result.stdout.strip() and verbose:
                    print(f"   Output: {result.stdout.strip()[:200]}")
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
        self._installed_models: List[str] = []
        self._current_process = None

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
        """Pick the best installed model for an agent, falling back to default."""
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

    def _build_memory_context(self) -> str:
        """
        Two-tier memory context — same as RainOrchestrator.
        Tier 1: session summaries (long-term). Tier 2: last 20 messages (working).
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

    def _query_agent(self, agent: Agent, prompt: str, label: str = None) -> str:
        """Send a prompt to a specific agent's model and return the response."""
        self._current_process = None
        try:
            memory_context = self._build_memory_context()
            full_prompt = (
                f"{agent.system_prompt}{memory_context}\n\n"
                f"User: {prompt}\n\nAssistant:"
            )
            self._current_process = subprocess.Popen(
                ['ollama', 'run', agent.model_name, full_prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            spinner_label = label or f"{agent.description} thinking..."
            stop_event, thread = self._start_spinner(spinner_label)
            try:
                stdout, _ = self._current_process.communicate()
            finally:
                self._stop_spinner(stop_event, thread)
            if self._current_process.returncode != 0:
                return ""
            return stdout.strip()
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
        """Extract the quality rating from a reflection response."""
        upper = critique.upper()
        for rating in ['EXCELLENT', 'GOOD', 'NEEDS_IMPROVEMENT', 'POOR']:
            if rating in upper:
                return rating
        return 'GOOD'  # default if unparseable

    def _needs_synthesis(self, rating: str) -> bool:
        """Decide whether to run a synthesis pass based on reflection rating."""
        return rating in ('NEEDS_IMPROVEMENT', 'POOR')

    # ------------------------------------------------------------------
    # Synthesis pass
    # ------------------------------------------------------------------

    def _build_synthesis_prompt(self, query: str, primary: str, critique: str) -> str:
        return (
            f"Original user query:\n{query}\n\n"
            f"Primary response:\n{primary}\n\n"
            f"Critique of that response:\n{critique}\n\n"
            f"Produce the best possible final answer, addressing every valid "
            f"criticism. Do not mention this synthesis process."
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

    def recursive_reflect(self, query: str, verbose: bool = False) -> Optional[ReflectionResult]:
        """
        Route → Primary Agent → Reflection → [Synthesis if needed] → [Sandbox if enabled]
        Same interface as RainOrchestrator.recursive_reflect for drop-in compatibility.
        """
        import time as _time
        start_time = _time.time()

        # Save to memory
        if self.memory:
            self.memory.save_message("user", query)

        # ── 1. Route ──────────────────────────────────────────────────
        agent_type = self.router.route(query)
        primary_agent = self.agents[agent_type]
        print(f"🔀 Routing to {self.router.explain(agent_type)}...")

        # ── 2. Primary Agent ──────────────────────────────────────────
        primary_response = self._query_agent(
            primary_agent, query,
            label=f"{primary_agent.description} thinking..."
        )
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
            label="Reflection Agent reviewing..."
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
                    label="Synthesizer working..."
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

        # ── 7. Build result ───────────────────────────────────────────
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

                    result = rain.recursive_reflect(query, verbose=args.verbose)
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
                recent = memory.get_recent_messages(limit=20)
                if recent:
                    history_text = "\n".join([
                        f"{'User' if m['role'] == 'user' else 'Rain'}: {m['content'][:200]}"
                        for m in recent
                    ])
                    summary_prompt = f"Summarize this conversation in 2-3 sentences, focusing on what was discussed and accomplished:\n\n{history_text}\n\nSummary:"
                    summary = rain._query_model(summary_prompt)
                    memory.end_session(summary=summary.strip() if summary else None)
                else:
                    memory.end_session()

        # Single query mode
        elif args.query:
            result = rain.recursive_reflect(args.query, verbose=args.verbose)
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
