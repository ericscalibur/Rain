#!/usr/bin/env python3
"""
Rain ⛈️ - Project Indexer (Phase 7)

Semantically indexes entire project directories so Rain can answer questions
about your actual codebase, not just your conversations.

Files are chunked into ~800-char segments, embedded using nomic-embed-text
(the same model used by RainMemory), and stored in the project_index table
in Rain's existing SQLite database at ~/.rain/memory.db.

Zero new dependencies — pure stdlib + the Ollama HTTP API already running.

Usage (standalone):
    python3 indexer.py --index /path/to/project
    python3 indexer.py --search /path/to/project "how does the payment flow work?"
    python3 indexer.py --list
    python3 indexer.py --remove /path/to/project

Usage (from Rain):
    from indexer import ProjectIndexer
    idx = ProjectIndexer()
    stats = idx.index_project("/path/to/project")
    results = idx.search_project("how does authentication work?", "/path/to/project")
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ── Constants ─────────────────────────────────────────────────────────

CHUNK_SIZE    = 900    # characters per chunk
CHUNK_OVERLAP = 120    # overlap between consecutive chunks to preserve context
MAX_FILE_SIZE = 500_000  # 500 KB — skip larger files (usually generated/binary)
EMBED_MODEL   = "nomic-embed-text"
EMBED_URL     = "http://localhost:11434/api/embed"
MIN_SIMILARITY = 0.35   # minimum cosine similarity to surface a result

# Directories that are never useful to index
# Specific filenames to exclude from indexing — these are meta-documents written
# for Claude (the AI assistant), not for Rain. Indexing them would inject
# instructions meant for a different AI into Rain's project context, confusing
# its self-model and capability understanding.
IGNORE_FILES = frozenset({
    "CLAUDE.md",           # Instructions for Claude AI assistant — not for Rain
    "SESSION_HANDOFF.md",  # Claude session carry-forward notes — not for Rain
})

IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", ".pnp",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".venv", "venv", "env", ".env",
    "dist", "build", "out", "_build", ".next", ".nuxt", ".output",
    "target",          # Rust / Maven
    "vendor",          # Go / PHP
    ".cache",
    "coverage", ".nyc_output",
    "eggs", ".eggs", "*.egg-info",
    "site-packages",
    ".idea", ".vscode", ".vs",
    "Pods",            # iOS
    ".gradle",
})

# Extensions that are certainly binary / not useful as text context
IGNORE_EXTENSIONS = frozenset({
    # compiled / binary
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll", ".exe", ".obj", ".o",
    ".class", ".jar", ".war", ".ear", ".wasm",
    # archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz",
    # media
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".avif",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".mp3", ".wav", ".ogg", ".flac",
    ".ttf", ".woff", ".woff2", ".eot",
    # documents (non-text)
    ".pdf", ".docx", ".xlsx", ".pptx", ".odt",
    # database / index files
    ".db", ".sqlite", ".sqlite3", ".mdb",
    ".idx", ".pack",
    # lock / hash files that add noise without value
    ".lock",   # package-lock.json is fine (text), Cargo.lock is fine — but .lock extension might be used for others
    ".sum",    # go.sum — large, repetitive
    # misc
    ".DS_Store", ".env",
})

# Extensions we know are definitely worth indexing
TEXT_EXTENSIONS = frozenset({
    # programming languages
    ".py", ".pyi",
    ".js", ".mjs", ".cjs",
    ".ts", ".tsx", ".jsx",
    ".rs", ".go", ".rb",
    ".java", ".kt", ".kts",
    ".swift", ".m",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
    ".cs", ".fs", ".fsx",
    ".php", ".pl", ".pm",
    ".lua", ".r", ".rmd",
    ".scala", ".groovy",
    ".ex", ".exs",      # Elixir
    ".erl", ".hrl",     # Erlang
    ".clj", ".cljs",    # Clojure
    ".hs", ".lhs",      # Haskell
    ".ml", ".mli",      # OCaml
    ".nim",
    ".zig",
    ".v",               # Vlang / Verilog
    # web / markup
    ".html", ".htm", ".xhtml",
    ".css", ".scss", ".sass", ".less",
    ".xml", ".xsd", ".xsl",
    ".vue", ".svelte",
    ".astro",
    # data / config
    ".json", ".jsonc", ".json5",
    ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".conf", ".config",
    ".env.example", ".env.sample",
    ".properties",
    ".plist",
    # shell / scripts
    ".sh", ".bash", ".zsh", ".fish", ".ksh",
    ".ps1", ".psm1",
    ".bat", ".cmd",
    # docs / text
    ".md", ".mdx", ".markdown",
    ".rst", ".txt", ".asciidoc", ".adoc",
    ".tex",
    # database
    ".sql", ".graphql", ".gql",
    # other
    ".proto", ".thrift",
    ".dockerfile",
    ".tf", ".hcl",       # Terraform
    ".bicep",
    ".nix",
    ".vim", ".vimrc",
    ".gitignore", ".gitattributes",
    ".editorconfig",
    ".prettierrc", ".eslintrc",
    "Makefile", "makefile",
    "Dockerfile",
    "Procfile",
    "Gemfile",
    "Rakefile",
    "Vagrantfile",
})


# ── ProjectIndexer ────────────────────────────────────────────────────

class ProjectIndexer:
    """
    Indexes a project directory for semantic search.

    Each text file is split into overlapping chunks, embedded with
    nomic-embed-text via the local Ollama API, and stored in the
    project_index table of Rain's SQLite memory database.

    Search returns the most semantically similar chunks ranked by
    cosine similarity — no cloud, no new pip installs.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (Path.home() / ".rain" / "memory.db")
        self._ensure_table()

    # ── Public API ────────────────────────────────────────────────────

    def index_project(
        self,
        project_path: str,
        force: bool = False,
        progress_fn=None,
    ) -> Dict:
        """
        Walk project_path and index every text file.

        Args:
            project_path: Absolute or relative path to the project root.
            force:        Re-index files that are already in the DB.
            progress_fn:  Optional callable(file_path: str) called for each
                          file as it's indexed — useful for streaming progress
                          to a UI.

        Returns:
            {
                "project_path":  str,
                "files_indexed": int,
                "files_skipped": int,
                "chunks_total":  int,
                "errors":        int,
                "duration_s":    float,
            }
        """
        start = datetime.now()
        project_path = str(Path(project_path).resolve())
        root = Path(project_path)

        if not root.exists():
            return {"error": f"Path not found: {project_path}"}
        if not root.is_dir():
            return {"error": f"Not a directory: {project_path}"}

        files_indexed = 0
        files_skipped = 0
        chunks_total  = 0
        errors        = 0

        for file_path in self._walk(root):
            file_str = str(file_path)
            try:
                if not force and self._is_indexed(project_path, file_str):
                    files_skipped += 1
                    continue

                content = self._read_file(file_path)
                if not content:
                    files_skipped += 1
                    continue

                chunks = self._chunk(content, file_path)
                if not chunks:
                    files_skipped += 1
                    continue

                self._index_file(project_path, file_str, chunks)

                if progress_fn:
                    try:
                        progress_fn(file_str)
                    except Exception:
                        pass

                files_indexed += 1
                chunks_total  += len(chunks)

            except Exception:
                errors += 1

        duration = (datetime.now() - start).total_seconds()
        return {
            "project_path":  project_path,
            "files_indexed": files_indexed,
            "files_skipped": files_skipped,
            "chunks_total":  chunks_total,
            "errors":        errors,
            "duration_s":    round(duration, 1),
        }

    def search_project(
        self,
        query: str,
        project_path: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find the most relevant file chunks for a query within a project.

        Returns a list of dicts (sorted by similarity, highest first):
            {
                "file_path":    str,   # absolute path
                "rel_path":     str,   # relative to project root
                "chunk_index":  int,
                "content":      str,
                "similarity":   float,
            }
        """
        query_vec = self._embed(query)
        if query_vec is None:
            return []

        project_path = str(Path(project_path).resolve())

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT file_path, chunk_index, content, embedding
                       FROM project_index
                       WHERE project_path = ?""",
                    (project_path,),
                ).fetchall()
        except Exception:
            return []

        root = Path(project_path)
        scored = []
        for row in rows:
            try:
                vec = json.loads(row["embedding"].decode("utf-8"))
                sim = _cosine_similarity(query_vec, vec)
                if sim >= MIN_SIMILARITY:
                    abs_path = row["file_path"]
                    try:
                        rel = str(Path(abs_path).relative_to(root))
                    except ValueError:
                        rel = abs_path
                    scored.append({
                        "file_path":   abs_path,
                        "rel_path":    rel,
                        "chunk_index": row["chunk_index"],
                        "content":     row["content"],
                        "similarity":  round(sim, 3),
                    })
            except Exception:
                continue

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def build_context_block(
        self,
        query: str,
        project_path: str,
        top_k: int = 4,
    ) -> str:
        """
        Return a formatted context string ready to inject into an agent prompt.
        Returns an empty string if no relevant results are found.
        """
        hits = self.search_project(query, project_path, top_k=top_k)
        if not hits:
            return ""

        lines = [f"[Project context from: {Path(project_path).name}/]"]
        for h in hits:
            lines.append(
                f"\n--- {h['rel_path']} "
                f"(chunk {h['chunk_index']}, {round(h['similarity'] * 100)}% match) ---"
            )
            lines.append(h["content"].strip())
        return "\n".join(lines)

    def get_project_tree(self, project_path: str, max_lines: int = 150) -> str:
        """
        Return a compact file-tree summary of the project directory.
        Uses only the filesystem — no DB required.
        Useful as a one-time orientation block for the first message in a session.
        """
        root = Path(project_path).resolve()
        if not root.exists():
            return f"Project not found: {project_path}"

        lines = [f"{root.name}/"]
        count = [0]

        def _recurse(path: Path, depth: int):
            if count[0] >= max_lines:
                return
            try:
                entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except PermissionError:
                return

            for entry in entries:
                if count[0] >= max_lines:
                    lines.append("  ... (truncated)")
                    return
                if entry.is_dir():
                    if entry.name in IGNORE_DIRS or entry.name.startswith("."):
                        continue
                    prefix = "  " * depth + "├── "
                    lines.append(f"{prefix}{entry.name}/")
                    count[0] += 1
                    _recurse(entry, depth + 1)
                else:
                    if entry.suffix.lower() in IGNORE_EXTENSIONS:
                        continue
                    prefix = "  " * depth + "│   "
                    lines.append(f"{prefix}{entry.name}")
                    count[0] += 1

        _recurse(root, 1)
        return "\n".join(lines)

    def list_indexed_projects(self) -> List[Dict]:
        """
        Return all indexed projects with file and chunk counts.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT
                           project_path,
                           COUNT(DISTINCT file_path) AS file_count,
                           COUNT(*)                  AS chunk_count,
                           MAX(indexed_at)           AS last_indexed
                       FROM project_index
                       GROUP BY project_path
                       ORDER BY last_indexed DESC"""
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def remove_project(self, project_path: str) -> int:
        """
        Remove all indexed data for a project.
        Returns the number of chunks deleted.
        """
        project_path = str(Path(project_path).resolve())
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "DELETE FROM project_index WHERE project_path = ?",
                    (project_path,),
                )
                return cur.rowcount
        except Exception:
            return 0

    def get_changed_files(self, project_path: str) -> List[str]:
        """
        Phase 7C: Compare indexed files' mtime against their indexed_at
        timestamp. Returns a list of file paths that have been modified
        since they were last indexed.

        Also detects new files that exist on disk but aren't indexed yet.
        """
        import os
        project_path = str(Path(project_path).resolve())
        root = Path(project_path)
        if not root.is_dir():
            return []

        changed = []

        # Get all indexed files and their latest indexed_at for this project
        indexed_files: Dict[str, str] = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT file_path, MAX(indexed_at) as last_indexed "
                    "FROM project_index WHERE project_path = ? "
                    "GROUP BY file_path",
                    (project_path,),
                ).fetchall()
                for row in rows:
                    indexed_files[row[0]] = row[1]
        except Exception:
            return []

        # Walk the project and check mtimes
        for file_path in self._walk(root):
            file_str = str(file_path)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            except OSError:
                continue

            if file_str in indexed_files:
                # File is indexed — check if it's been modified since
                try:
                    indexed_at = datetime.fromisoformat(indexed_files[file_str])
                    if mtime > indexed_at:
                        changed.append(file_str)
                except (ValueError, TypeError):
                    changed.append(file_str)
            else:
                # New file not yet indexed
                changed.append(file_str)

        return changed

    def reindex_file(self, project_path: str, file_path: str) -> int:
        """
        Re-index a single file (e.g. after it's been edited).
        Returns the number of chunks stored.
        """
        project_path = str(Path(project_path).resolve())
        file_path_obj = Path(file_path).resolve()

        content = self._read_file(file_path_obj)
        if not content:
            return 0

        chunks = self._chunk(content, file_path_obj)
        if not chunks:
            return 0

        self._index_file(project_path, str(file_path_obj), chunks)
        return len(chunks)

    # ── Internal helpers ──────────────────────────────────────────────

    def _ensure_table(self):
        """
        Make sure the project_index table exists.
        RainMemory._init_db() also creates it — this is a safety net for
        standalone use of ProjectIndexer without a RainMemory instance.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS project_index (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_path  TEXT    NOT NULL,
                        file_path     TEXT    NOT NULL,
                        chunk_index   INTEGER NOT NULL,
                        content       TEXT    NOT NULL,
                        embedding     BLOB    NOT NULL,
                        indexed_at    TEXT    NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_project_index_path
                        ON project_index(project_path);
                """)
        except Exception:
            pass

    def _walk(self, root: Path):
        """
        Yield text file paths under root, skipping ignored dirs and extensions.
        """
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            # Skip if any ancestor directory is in the ignore set
            rel_parts = set(path.relative_to(root).parts[:-1])
            if rel_parts & IGNORE_DIRS:
                continue
            # Also skip dotdirs (e.g. .git, .venv even if not in the set above)
            if any(part.startswith(".") for part in rel_parts):
                continue

            ext = path.suffix.lower()
            name = path.name

            # Hard-skip known binary extensions
            if ext in IGNORE_EXTENSIONS:
                continue

            # Skip filenames reserved for Claude — indexing these would inject
            # AI-assistant instructions into Rain's own project context.
            if name in IGNORE_FILES:
                continue

            # Accept known text extensions OR small files without a listed extension
            if ext not in TEXT_EXTENSIONS and name not in TEXT_EXTENSIONS:
                try:
                    if path.stat().st_size > 20_000:
                        continue  # unknown extension + large → skip
                except OSError:
                    continue

            yield path

    def _read_file(self, path: Path) -> str:
        """Read a file as text. Returns '' on any error or oversized file."""
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                return ""
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def _chunk(self, text: str, file_path: Path) -> List[str]:
        """
        Split text into overlapping chunks of ~CHUNK_SIZE characters.

        Tries to split at newlines to preserve logical boundaries.
        Prepends a short file header to each chunk so the model knows
        which file the context came from.
        """
        text = text.strip()
        if not text:
            return []

        header = f"# File: {file_path.name}\n"

        # If the whole file fits in one chunk, return it as-is
        if len(header) + len(text) <= CHUNK_SIZE:
            return [header + text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE - len(header)
            raw_chunk = text[start:end]

            # Try to end at a newline to avoid splitting mid-line
            if end < len(text):
                last_nl = raw_chunk.rfind("\n")
                if last_nl > CHUNK_SIZE // 2:
                    raw_chunk = raw_chunk[:last_nl]

            chunk = (header + raw_chunk).strip()
            if chunk and chunk != header.strip():
                chunks.append(chunk)

            # Advance, leaving overlap from the end of this chunk
            advance = len(raw_chunk) - CHUNK_OVERLAP
            if advance <= 0:
                advance = max(1, len(raw_chunk))
            start += advance

        return chunks

    def _is_indexed(self, project_path: str, file_path: str) -> bool:
        """Return True if the file already has entries in the index."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """SELECT 1 FROM project_index
                       WHERE project_path = ? AND file_path = ?
                       LIMIT 1""",
                    (project_path, file_path),
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def _index_file(self, project_path: str, file_path: str, chunks: List[str]):
        """Embed each chunk and store it in the DB, replacing any existing entries."""
        # Remove stale entries for this file first
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM project_index WHERE project_path = ? AND file_path = ?",
                    (project_path, file_path),
                )
        except Exception:
            pass

        now = datetime.now().isoformat()
        for i, chunk in enumerate(chunks):
            vec = self._embed(chunk)
            if vec is None:
                continue
            blob = json.dumps(vec).encode("utf-8")
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """INSERT INTO project_index
                               (project_path, file_path, chunk_index, content, embedding, indexed_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (project_path, file_path, i, chunk, blob, now),
                    )
            except Exception:
                pass

    def _embed(self, text: str) -> Optional[List[float]]:
        """
        Embed text using nomic-embed-text via the local Ollama HTTP API.
        Returns None if Ollama is unreachable or the model isn't installed.
        """
        import urllib.request

        try:
            payload = json.dumps({
                "model": EMBED_MODEL,
                "input": text[:2000],   # nomic supports up to 8192 tokens; 2000 chars is safe
            }).encode("utf-8")
            req = urllib.request.Request(
                EMBED_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            embeddings = data.get("embeddings", [])
            if embeddings and isinstance(embeddings[0], list):
                return embeddings[0]

            # Fallback: older Ollama versions use "embedding" (singular)
            embedding = data.get("embedding")
            if embedding and isinstance(embedding, list):
                return embedding

        except Exception:
            pass
        return None


# ── Pure-Python cosine similarity (no numpy) ──────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── NoteIndexer ───────────────────────────────────────────────────────

NOTES_DIR = Path.home() / ".rain" / "notes"
NOTES_EXTENSIONS = frozenset({".md", ".txt"})


class NoteIndexer:
    """
    Indexes personal notes from ~/.rain/notes/ for semantic search.

    Drop any .md or .txt file into ~/.rain/notes/ and Rain will chunk,
    embed, and retrieve relevant excerpts automatically.  Uses the same
    SQLite DB and nomic-embed-text pipeline as ProjectIndexer.

    Table: notes_chunks
        id          INTEGER PRIMARY KEY
        filepath    TEXT NOT NULL          -- absolute path to the note file
        chunk_text  TEXT NOT NULL
        symbol      TEXT                   -- optional heading/section label
        embedding   BLOB NOT NULL
        indexed_at  TEXT NOT NULL
    """

    def __init__(self, db_path: Optional[Path] = None, notes_dir: Optional[Path] = None):
        self.db_path   = db_path   or (Path.home() / ".rain" / "memory.db")
        self.notes_dir = notes_dir or NOTES_DIR
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    # ── Public API ────────────────────────────────────────────────────

    def index_notes(self, force: bool = False) -> Dict:
        """
        Walk notes_dir and index all .md / .txt files.

        Returns:
            {"files_indexed": int, "files_skipped": int,
             "chunks_total": int, "errors": int}
        """
        files_indexed = 0
        files_skipped = 0
        chunks_total  = 0
        errors        = 0

        for note_path in sorted(self.notes_dir.rglob("*")):
            if not note_path.is_file():
                continue
            if note_path.suffix.lower() not in NOTES_EXTENSIONS:
                continue

            filepath_str = str(note_path)
            try:
                if not force and self._is_indexed(filepath_str):
                    files_skipped += 1
                    continue

                text = note_path.read_text(encoding="utf-8", errors="replace").strip()
                if not text:
                    files_skipped += 1
                    continue

                chunks = self._chunk_note(text, note_path)
                if not chunks:
                    files_skipped += 1
                    continue

                self._index_note(filepath_str, chunks)
                files_indexed += 1
                chunks_total  += len(chunks)
            except Exception:
                errors += 1

        return {
            "files_indexed": files_indexed,
            "files_skipped": files_skipped,
            "chunks_total":  chunks_total,
            "errors":        errors,
        }

    def search_notes(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Find the most relevant note chunks for *query*.

        Returns list of dicts sorted by similarity (highest first):
            {"filepath": str, "chunk_text": str, "symbol": str,
             "similarity": float}
        """
        query_vec = self._embed(query)
        if query_vec is None:
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT filepath, chunk_text, symbol, embedding FROM notes_chunks"
                ).fetchall()
        except Exception:
            return []

        if not rows:
            return []

        try:
            import numpy as np
            parsed = []
            vecs   = []
            for row in rows:
                try:
                    vec = json.loads(row["embedding"].decode("utf-8"))
                    parsed.append(row)
                    vecs.append(vec)
                except Exception:
                    continue

            if not parsed:
                return []

            matrix = np.array(vecs, dtype=np.float32)
            q      = np.array(query_vec, dtype=np.float32)
            norms  = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-10
            sims   = (matrix @ q / norms).tolist()

            results = []
            for sim, row in zip(sims, parsed):
                if sim >= MIN_SIMILARITY:
                    results.append({
                        "filepath":   row["filepath"],
                        "chunk_text": row["chunk_text"],
                        "symbol":     row["symbol"] or "",
                        "similarity": round(sim, 3),
                    })

            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]

        except ImportError:
            # numpy unavailable — fall back to pure-Python
            scored = []
            for row in rows:
                try:
                    vec = json.loads(row["embedding"].decode("utf-8"))
                    sim = _cosine_similarity(query_vec, vec)
                    if sim >= MIN_SIMILARITY:
                        scored.append({
                            "filepath":   row["filepath"],
                            "chunk_text": row["chunk_text"],
                            "symbol":     row["symbol"] or "",
                            "similarity": round(sim, 3),
                        })
                except Exception:
                    continue
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return scored[:top_k]

    def build_context_block(self, query: str, top_k: int = 3) -> str:
        """
        Return a formatted context string ready to inject into an agent prompt.
        Returns empty string if no relevant notes found.
        """
        hits = self.search_notes(query, top_k=top_k)
        if not hits:
            return ""

        lines = ["[Notes context from ~/.rain/notes/]"]
        for h in hits:
            label = h["symbol"] or Path(h["filepath"]).name
            lines.append(
                f"\n--- {label} ({round(h['similarity'] * 100)}% match) ---"
            )
            lines.append(h["chunk_text"].strip())
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────

    def _ensure_table(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS notes_chunks (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        filepath   TEXT NOT NULL,
                        chunk_text TEXT NOT NULL,
                        symbol     TEXT,
                        embedding  BLOB NOT NULL,
                        indexed_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_notes_chunks_filepath
                        ON notes_chunks(filepath);
                """)
        except Exception:
            pass

    def _is_indexed(self, filepath: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT 1 FROM notes_chunks WHERE filepath = ? LIMIT 1",
                    (filepath,),
                ).fetchone()
                return row is not None
        except Exception:
            return False

    def _chunk_note(self, text: str, note_path: Path) -> List[tuple]:
        """
        Paragraph-chunk a note file.  Each tuple is (chunk_text, symbol)
        where symbol is the most recent markdown heading (if any).

        Falls back to fixed-size chunking for non-markdown files.
        """
        chunks: List[tuple] = []
        current_heading = ""
        current_para: List[str] = []

        def _flush():
            body = "\n".join(current_para).strip()
            if body:
                header = f"# Note: {note_path.name}"
                if current_heading:
                    header += f" — {current_heading}"
                chunks.append((header + "\n" + body, current_heading or note_path.stem))

        for line in text.splitlines():
            if line.startswith("#"):
                _flush()
                current_para = [line]
                current_heading = line.lstrip("#").strip()
            elif line.strip() == "":
                if current_para:
                    _flush()
                    current_para = []
            else:
                current_para.append(line)
                # Flush if chunk is getting large
                if sum(len(l) for l in current_para) > CHUNK_SIZE:
                    _flush()
                    current_para = []

        _flush()
        return chunks

    def _index_note(self, filepath: str, chunks: List[tuple]):
        """Delete stale entries then embed and store all chunks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM notes_chunks WHERE filepath = ?", (filepath,))
        except Exception:
            pass

        now = datetime.now().isoformat()
        for chunk_text, symbol in chunks:
            vec = self._embed(chunk_text)
            if vec is None:
                continue
            blob = json.dumps(vec).encode("utf-8")
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """INSERT INTO notes_chunks
                               (filepath, chunk_text, symbol, embedding, indexed_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (filepath, chunk_text, symbol, blob, now),
                    )
            except Exception:
                pass

    def _embed(self, text: str) -> Optional[List[float]]:
        """Delegate to the module-level ProjectIndexer embed logic."""
        import urllib.request
        try:
            payload = json.dumps({
                "model": EMBED_MODEL,
                "input": text[:2000],
            }).encode("utf-8")
            req = urllib.request.Request(
                EMBED_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            embeddings = data.get("embeddings", [])
            if embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            embedding = data.get("embedding")
            if embedding and isinstance(embedding, list):
                return embedding
        except Exception:
            pass
        return None


def search_notes(query: str, top_k: int = 3) -> List[Dict]:
    """
    Module-level convenience wrapper.  Returns top_k note chunks relevant
    to *query*, or [] if the notes dir is empty / Ollama is unavailable.
    """
    return NoteIndexer().search_notes(query, top_k=top_k)


# ── CLI (standalone use) ──────────────────────────────────────────────

def _fmt_path(p: str) -> str:
    """Shorten a path relative to home for display."""
    try:
        return "~/" + str(Path(p).relative_to(Path.home()))
    except ValueError:
        return p


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rain ⛈️  Project Indexer — semantic index of your codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 indexer.py --index ~/projects/myapp
  python3 indexer.py --index ~/projects/myapp --force
  python3 indexer.py --search ~/projects/myapp "how does the payment flow work?"
  python3 indexer.py --list
  python3 indexer.py --tree ~/projects/myapp
  python3 indexer.py --remove ~/projects/myapp
        """,
    )
    parser.add_argument("--index",  metavar="PATH", help="Index a project directory")
    parser.add_argument("--search", metavar="PATH", help="Project path to search within")
    parser.add_argument("query",    nargs="?",       help="Query string (use with --search)")
    parser.add_argument("--list",   action="store_true", help="List all indexed projects")
    parser.add_argument("--tree",   metavar="PATH", help="Print file tree of a project")
    parser.add_argument("--remove", metavar="PATH", help="Remove a project from the index")
    parser.add_argument("--force",  action="store_true",
                        help="Re-index all files even if already indexed (use with --index)")
    parser.add_argument("--top-k",  type=int, default=5,
                        help="Max results to return from --search (default: 5)")

    args = parser.parse_args()
    indexer = ProjectIndexer()

    if args.index:
        path = args.index
        print(f"⛈️  Indexing {_fmt_path(path)} ...")
        print("    (this may take a few minutes for large projects — embedding with nomic-embed-text)\n")

        def on_progress(fp):
            try:
                rel = Path(fp).relative_to(Path(path).resolve())
                print(f"  ✓ {rel}")
            except ValueError:
                print(f"  ✓ {fp}")

        stats = indexer.index_project(path, force=args.force, progress_fn=on_progress)
        if "error" in stats:
            print(f"\n❌ {stats['error']}")
            sys.exit(1)
        print(f"\n✅ Done in {stats['duration_s']}s")
        print(f"   {stats['files_indexed']} file(s) indexed · "
              f"{stats['chunks_total']} chunks · "
              f"{stats['files_skipped']} skipped · "
              f"{stats['errors']} error(s)")

    elif args.search and args.query:
        hits = indexer.search_project(args.query, args.search, top_k=args.top_k)
        if not hits:
            print(f"No results found for: \"{args.query}\"")
            print("Make sure the project is indexed first:  python3 indexer.py --index <path>")
            sys.exit(0)
        print(f"Top {len(hits)} result(s) for: \"{args.query}\"\n")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h['rel_path']}  (chunk {h['chunk_index']}, {round(h['similarity']*100)}% match)")
            print("─" * 60)
            print(h["content"][:600])
            if len(h["content"]) > 600:
                print("  ...")
            print()

    elif args.list:
        projects = indexer.list_indexed_projects()
        if not projects:
            print("No projects indexed yet.")
            print("Run:  python3 indexer.py --index <path>")
        else:
            print(f"{'PROJECT':<50} {'FILES':>6} {'CHUNKS':>7}  LAST INDEXED")
            print("─" * 90)
            for p in projects:
                short = _fmt_path(p["project_path"])
                dt = p["last_indexed"][:16].replace("T", " ") if p["last_indexed"] else "unknown"
                print(f"{short:<50} {p['file_count']:>6} {p['chunk_count']:>7}  {dt}")

    elif args.tree:
        print(indexer.get_project_tree(args.tree))

    elif args.remove:
        path = str(Path(args.remove).resolve())
        n = indexer.remove_project(path)
        if n:
            print(f"✅ Removed {n} chunk(s) for {_fmt_path(path)}")
        else:
            print(f"Nothing found for {_fmt_path(path)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
