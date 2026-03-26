#!/usr/bin/env python3
"""
Rain ⛈️ - Knowledge Graph & Deep Project Intelligence (Phase 10)

Builds a directed graph of your codebase in SQLite: nodes are files,
functions, classes, methods, imports, and concepts; edges are
relationships like "calls", "imports", "defines", "inherits", "contains".

Also integrates git history and a decision log so Rain can answer
"why was this written this way?" by pointing to the commit or
conversation where the choice was made.

Zero new dependencies — pure stdlib (ast, re, subprocess, sqlite3, json)
plus the Ollama HTTP API already running locally.

Usage (standalone):
    python3 knowledge_graph.py --build /path/to/project
    python3 knowledge_graph.py --summary /path/to/project
    python3 knowledge_graph.py --find /path/to/project function_name
    python3 knowledge_graph.py --callers /path/to/project function_name
    python3 knowledge_graph.py --history /path/to/project [file_path]
    python3 knowledge_graph.py --decisions [/path/to/project]
    python3 knowledge_graph.py --onboard /path/to/project
    python3 knowledge_graph.py --stats /path/to/project

Usage (from Rain):
    from knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    stats = kg.build_graph("/path/to/project")
    nodes = kg.find_nodes("/path/to/project", name="recursive_reflect")
    callers = kg.get_callers("/path/to/project", "recursive_reflect")
    context = kg.build_context_block("why does recursive_reflect use threading?", "/path/to/project")
"""

import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

# Max file size to attempt parsing (500KB — matches indexer.py)
MAX_PARSE_SIZE = 500_000

# Specific filenames to exclude — these are meta-documents written for Claude
# (the AI assistant), not for Rain. Including them would inject instructions
# meant for a different AI into Rain's project context and knowledge graph.
IGNORE_FILES = frozenset({
    "CLAUDE.md",           # Instructions for Claude AI assistant — not for Rain
    "SESSION_HANDOFF.md",  # Claude session carry-forward notes — not for Rain
})

# Directories to skip (mirrors indexer.py)
IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", ".pnp",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".venv", "venv", "env", ".env",
    "dist", "build", "out", "_build", ".next", ".nuxt", ".output",
    "target", "vendor", ".cache",
    "coverage", ".nyc_output",
    "eggs", ".eggs", "site-packages",
    ".idea", ".vscode", ".vs",
    "Pods", ".gradle",
})

# File extensions we can structurally parse
PYTHON_EXTENSIONS = frozenset({".py", ".pyi"})
JS_TS_EXTENSIONS = frozenset({".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"})
RUST_EXTENSIONS = frozenset({".rs"})
GO_EXTENSIONS = frozenset({".go"})

# All parseable extensions
PARSEABLE_EXTENSIONS = PYTHON_EXTENSIONS | JS_TS_EXTENSIONS | RUST_EXTENSIONS | GO_EXTENSIONS

# Extensions that are code files worth tracking as file nodes even if
# we can't do deep structural parsing on them
CODE_EXTENSIONS = PARSEABLE_EXTENSIONS | frozenset({
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".java", ".kt", ".kts", ".scala", ".groovy",
    ".rb", ".php", ".swift", ".m",
    ".lua", ".r", ".ex", ".exs", ".erl",
    ".hs", ".ml", ".clj", ".cljs",
    ".html", ".htm", ".css", ".scss", ".sass",
    ".vue", ".svelte", ".astro",
    ".sql", ".graphql", ".gql",
    ".sh", ".bash", ".zsh",
    ".md", ".mdx", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml",
    ".xml",
    ".dockerfile", ".tf", ".hcl",
    ".proto",
})


# ── KnowledgeGraph ────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Builds and queries a structural knowledge graph of one or more projects.

    The graph lives in Rain's SQLite database alongside memory and the
    project index.  Tables:
        kg_nodes          — files, functions, classes, methods, imports
        kg_edges          — calls, imports, defines, inherits, contains
        kg_decisions      — architectural decisions from conversations
        kg_project_summaries — LLM-generated project overviews
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (Path.home() / ".rain" / "memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    # ══════════════════════════════════════════════════════════════════
    #  Graph building
    # ══════════════════════════════════════════════════════════════════

    def build_graph(
        self,
        project_path: str,
        force: bool = False,
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """
        Parse every code file in the project, extract nodes and edges,
        and store them in the knowledge graph tables.

        Args:
            project_path:  Absolute or relative path to the project root.
            force:         Drop existing graph data for this project first.
            progress_fn:   Optional callback(file_path) for progress updates.

        Returns dict with stats: files_parsed, nodes, edges, duration_s, errors.
        """
        start = datetime.now()
        project_path = str(Path(project_path).resolve())
        root = Path(project_path)

        if not root.is_dir():
            return {"error": f"Not a directory: {project_path}"}

        if force:
            self._clear_project(project_path)

        files_parsed = 0
        total_nodes = 0
        total_edges = 0
        errors = 0

        # Phase 1: create file nodes and parse structure
        for file_path in self._walk(root):
            rel_path = str(file_path.relative_to(root))
            ext = file_path.suffix.lower()

            if progress_fn:
                try:
                    progress_fn(rel_path)
                except Exception:
                    pass

            try:
                # Always create a file node
                file_node_id = self._upsert_node(
                    project_path=project_path,
                    file_path=str(file_path),
                    node_type="file",
                    name=rel_path,
                    metadata_dict={"extension": ext, "size": file_path.stat().st_size},
                )

                # Structural parsing based on language
                if ext in PYTHON_EXTENSIONS:
                    n, e = self._parse_python(project_path, file_path, file_node_id)
                    total_nodes += n
                    total_edges += e
                elif ext in JS_TS_EXTENSIONS:
                    n, e = self._parse_js_ts(project_path, file_path, file_node_id)
                    total_nodes += n
                    total_edges += e
                elif ext in RUST_EXTENSIONS:
                    n, e = self._parse_rust(project_path, file_path, file_node_id)
                    total_nodes += n
                    total_edges += e
                elif ext in GO_EXTENSIONS:
                    n, e = self._parse_go(project_path, file_path, file_node_id)
                    total_nodes += n
                    total_edges += e

                total_nodes += 1  # count the file node itself
                files_parsed += 1

            except Exception:
                errors += 1

        # Phase 2: resolve cross-file edges (imports → file nodes)
        try:
            self._resolve_import_edges(project_path, root)
        except Exception:
            pass

        # Phase 3: git history integration
        try:
            if (root / ".git").is_dir():
                self._index_git_history(project_path, root)
                if progress_fn:
                    progress_fn("[git history indexed]")
        except Exception:
            pass

        duration = (datetime.now() - start).total_seconds()
        return {
            "project_path": project_path,
            "files_parsed": files_parsed,
            "nodes": total_nodes,
            "edges": total_edges,
            "errors": errors,
            "duration_s": round(duration, 1),
        }

    # ══════════════════════════════════════════════════════════════════
    #  Query API
    # ══════════════════════════════════════════════════════════════════

    def find_nodes(
        self,
        project_path: str,
        name: Optional[str] = None,
        node_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Search for nodes by name and/or type.
        Name matching is case-insensitive substring match.
        """
        project_path = str(Path(project_path).resolve())
        conditions = ["project_path = ?"]
        params: list = [project_path]

        if name:
            conditions.append("LOWER(name) LIKE ?")
            params.append(f"%{name.lower()}%")
        if node_type:
            conditions.append("node_type = ?")
            params.append(node_type)

        where = " AND ".join(conditions)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"SELECT * FROM kg_nodes WHERE {where} ORDER BY name LIMIT ?",
                    params + [limit],
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_node(self, node_id: int) -> Optional[Dict]:
        """Get a single node by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM kg_nodes WHERE id = ?", (node_id,)
                ).fetchone()
                return dict(row) if row else None
        except Exception:
            return None

    def get_edges(
        self,
        node_id: int,
        direction: str = "both",
        edge_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get edges connected to a node.
        direction: 'outgoing', 'incoming', or 'both'
        """
        results = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                type_filter = " AND edge_type = ?" if edge_type else ""
                type_param = [edge_type] if edge_type else []

                if direction in ("outgoing", "both"):
                    rows = conn.execute(
                        f"SELECT e.*, n.name as target_name, n.node_type as target_type "
                        f"FROM kg_edges e JOIN kg_nodes n ON e.target_id = n.id "
                        f"WHERE e.source_id = ?{type_filter}",
                        [node_id] + type_param,
                    ).fetchall()
                    results.extend(dict(r) for r in rows)

                if direction in ("incoming", "both"):
                    rows = conn.execute(
                        f"SELECT e.*, n.name as source_name, n.node_type as source_type "
                        f"FROM kg_edges e JOIN kg_nodes n ON e.source_id = n.id "
                        f"WHERE e.target_id = ?{type_filter}",
                        [node_id] + type_param,
                    ).fetchall()
                    results.extend(dict(r) for r in rows)
        except Exception:
            pass
        return results

    def get_callers(self, project_path: str, function_name: str) -> List[Dict]:
        """Who calls this function? Returns nodes that have a 'calls' edge to it."""
        project_path = str(Path(project_path).resolve())
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT DISTINCT src.*
                       FROM kg_edges e
                       JOIN kg_nodes tgt ON e.target_id = tgt.id
                       JOIN kg_nodes src ON e.source_id = src.id
                       WHERE tgt.project_path = ?
                         AND LOWER(tgt.name) = ?
                         AND e.edge_type = 'calls'""",
                    (project_path, function_name.lower()),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_callees(self, project_path: str, function_name: str) -> List[Dict]:
        """What does this function call? Returns nodes it has 'calls' edges to."""
        project_path = str(Path(project_path).resolve())
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT DISTINCT tgt.*
                       FROM kg_edges e
                       JOIN kg_nodes src ON e.source_id = src.id
                       JOIN kg_nodes tgt ON e.target_id = tgt.id
                       WHERE src.project_path = ?
                         AND LOWER(src.name) = ?
                         AND e.edge_type = 'calls'""",
                    (project_path, function_name.lower()),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_file_structure(self, project_path: str, file_path: str) -> Dict:
        """
        Return a summary of a file: its functions, classes, methods, imports.
        file_path can be relative or absolute.
        """
        project_path = str(Path(project_path).resolve())
        root = Path(project_path)
        abs_path = str((root / file_path).resolve()) if not os.path.isabs(file_path) else file_path

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Get file node
                file_node = conn.execute(
                    "SELECT * FROM kg_nodes WHERE project_path = ? AND file_path = ? AND node_type = 'file'",
                    (project_path, abs_path),
                ).fetchone()

                if not file_node:
                    return {"error": f"File not in graph: {file_path}"}

                # Get all child nodes
                children = conn.execute(
                    """SELECT n.* FROM kg_nodes n
                       JOIN kg_edges e ON e.target_id = n.id
                       WHERE e.source_id = ? AND e.edge_type = 'contains'
                       ORDER BY n.line_start""",
                    (file_node["id"],),
                ).fetchall()

                return {
                    "file": dict(file_node),
                    "functions": [dict(c) for c in children if c["node_type"] == "function"],
                    "classes": [dict(c) for c in children if c["node_type"] == "class"],
                    "methods": [dict(c) for c in children if c["node_type"] == "method"],
                    "imports": [dict(c) for c in children if c["node_type"] == "import"],
                }
        except Exception as e:
            return {"error": str(e)}

    def get_project_stats(self, project_path: str) -> Dict:
        """Return counts and summary stats for the knowledge graph."""
        project_path = str(Path(project_path).resolve())
        try:
            with sqlite3.connect(self.db_path) as conn:
                def count(table, extra=""):
                    sql = f"SELECT COUNT(*) FROM {table} WHERE project_path = ? {extra}"
                    return conn.execute(sql, (project_path,)).fetchone()[0]

                node_types = conn.execute(
                    "SELECT node_type, COUNT(*) as cnt FROM kg_nodes "
                    "WHERE project_path = ? GROUP BY node_type ORDER BY cnt DESC",
                    (project_path,),
                ).fetchall()

                edge_types = conn.execute(
                    "SELECT edge_type, COUNT(*) as cnt FROM kg_edges "
                    "WHERE project_path = ? GROUP BY edge_type ORDER BY cnt DESC",
                    (project_path,),
                ).fetchall()

                decision_count = conn.execute(
                    "SELECT COUNT(*) FROM kg_decisions WHERE project_path = ?",
                    (project_path,),
                ).fetchone()[0]

                return {
                    "project_path": project_path,
                    "total_nodes": count("kg_nodes"),
                    "total_edges": count("kg_edges"),
                    "decisions": decision_count,
                    "node_types": {r[0]: r[1] for r in node_types},
                    "edge_types": {r[0]: r[1] for r in edge_types},
                }
        except Exception as e:
            return {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════
    #  Git history integration
    # ══════════════════════════════════════════════════════════════════

    def get_git_history(
        self,
        project_path: str,
        file_path: Optional[str] = None,
        n: int = 20,
    ) -> List[Dict]:
        """
        Return recent git commits for the project or a specific file.
        Reads directly from git (not cached) — always fresh.
        """
        project_path = str(Path(project_path).resolve())
        root = Path(project_path)

        if not (root / ".git").is_dir():
            return []

        cmd = f"git log --format='%H|%ai|%an|%s' -n {int(n)}"
        if file_path:
            abs_path = str((root / file_path).resolve()) if not os.path.isabs(file_path) else file_path
            cmd += f" -- {abs_path}"

        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(root),
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return []

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("|", 3)
                if len(parts) == 4:
                    commits.append({
                        "sha": parts[0].strip("'"),
                        "date": parts[1].strip(),
                        "author": parts[2].strip(),
                        "message": parts[3].strip(),
                    })
            return commits
        except Exception:
            return []

    def get_file_blame_summary(
        self,
        project_path: str,
        file_path: str,
        function_name: Optional[str] = None,
    ) -> str:
        """
        Return a git blame summary for a file or a specific function within it.
        Shows who wrote each section and when.
        """
        project_path = str(Path(project_path).resolve())
        root = Path(project_path)
        abs_path = str((root / file_path).resolve()) if not os.path.isabs(file_path) else file_path

        if not (root / ".git").is_dir():
            return "Not a git repository."

        # If a function name is given, find its line range from the graph
        line_range = ""
        if function_name:
            nodes = self.find_nodes(project_path, name=function_name)
            for node in nodes:
                if node.get("file_path") == abs_path and node.get("line_start"):
                    start = node["line_start"]
                    end = node.get("line_end") or start + 50
                    line_range = f"-L {start},{end}"
                    break

        cmd = f"git blame --line-porcelain {line_range} -- {abs_path}"
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(root),
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return f"git blame failed: {result.stderr[:200]}"

            # Parse porcelain output into a summary
            authors: Dict[str, int] = {}
            current_author = None
            for line in result.stdout.split("\n"):
                if line.startswith("author "):
                    current_author = line[7:].strip()
                    authors[current_author] = authors.get(current_author, 0) + 1

            if not authors:
                return "No blame data available."

            total = sum(authors.values())
            lines = [f"Blame summary for {Path(abs_path).name}" +
                     (f" → {function_name}()" if function_name else "") + ":"]
            for author, count in sorted(authors.items(), key=lambda x: -x[1]):
                pct = round(100 * count / total)
                lines.append(f"  {author}: {count} lines ({pct}%)")
            return "\n".join(lines)
        except Exception as e:
            return f"git blame error: {e}"

    def get_commit_for_function(
        self,
        project_path: str,
        file_path: str,
        function_name: str,
    ) -> Optional[Dict]:
        """
        Find the commit that introduced a function (first appearance in git log).
        Answers the question: "When was this written and by whom?"
        """
        project_path = str(Path(project_path).resolve())
        root = Path(project_path)
        abs_path = str((root / file_path).resolve()) if not os.path.isabs(file_path) else file_path

        if not (root / ".git").is_dir():
            return None

        # Use git log -S to find the commit that introduced the function name
        cmd = (
            f"git log --format='%H|%ai|%an|%s' -S 'def {function_name}' "
            f"--diff-filter=A -- {abs_path}"
        )
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(root),
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0 or not result.stdout.strip():
                # Fallback: find first commit mentioning this function
                cmd2 = (
                    f"git log --format='%H|%ai|%an|%s' -S '{function_name}' "
                    f"--reverse -n 1 -- {abs_path}"
                )
                result = subprocess.run(
                    cmd2, shell=True, cwd=str(root),
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    return None

            line = result.stdout.strip().split("\n")[-1]  # last (earliest) match
            parts = line.split("|", 3)
            if len(parts) == 4:
                return {
                    "sha": parts[0].strip("'"),
                    "date": parts[1].strip(),
                    "author": parts[2].strip(),
                    "message": parts[3].strip(),
                }
        except Exception:
            pass
        return None

    # ══════════════════════════════════════════════════════════════════
    #  Decision log
    # ══════════════════════════════════════════════════════════════════

    def log_decision(
        self,
        title: str,
        description: str,
        project_path: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[str] = None,
        alternatives: Optional[str] = None,
        rationale: Optional[str] = None,
        tags: Optional[str] = None,
        commit_sha: Optional[str] = None,
    ) -> int:
        """
        Manually log an architectural decision.
        Returns the decision ID.
        """
        if project_path:
            project_path = str(Path(project_path).resolve())
        now = datetime.now().isoformat()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """INSERT INTO kg_decisions
                       (project_path, session_id, title, description, context,
                        alternatives, rationale, tags, timestamp, commit_sha)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (project_path, session_id, title, description, context,
                     alternatives, rationale, tags, now, commit_sha),
                )
                return cur.lastrowid
        except Exception:
            return -1

    def extract_decisions_from_transcript(
        self,
        transcript: str,
        project_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Use the local LLM to extract architectural decisions from a
        conversation transcript.  Returns a list of decision dicts and
        also persists them to the kg_decisions table.
        """
        import urllib.request

        prompt = (
            "Extract architectural decisions from this conversation transcript. "
            "A decision is a deliberate choice about technology, design, implementation, "
            "or architecture. NOT every statement is a decision — only extract clear choices.\n\n"
            "Return a JSON array of objects. Each object must have:\n"
            '  "title": short decision title (e.g. "Use FastAPI over Flask")\n'
            '  "description": one-sentence description of what was decided\n'
            '  "rationale": why this choice was made (if stated)\n'
            '  "alternatives": what else was considered (if mentioned)\n'
            '  "tags": comma-separated keywords\n\n'
            "Rules:\n"
            "- Only include clear, deliberate decisions — not preferences or opinions.\n"
            "- Maximum 5 decisions. Skip trivial ones.\n"
            "- Return ONLY the JSON array — no preamble, no markdown.\n"
            "- If no decisions were made, return []\n\n"
            f"Transcript:\n{transcript[:3000]}\n\nDecisions JSON:"
        )

        try:
            payload = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw = data.get("response", "").strip()

            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []

            decisions = json.loads(raw[start:end])
            if not isinstance(decisions, list):
                return []

            saved = []
            for d in decisions:
                if not isinstance(d, dict) or not d.get("title"):
                    continue
                dec_id = self.log_decision(
                    title=d.get("title", ""),
                    description=d.get("description", ""),
                    project_path=project_path,
                    session_id=session_id,
                    rationale=d.get("rationale"),
                    alternatives=d.get("alternatives"),
                    tags=d.get("tags"),
                )
                d["id"] = dec_id
                saved.append(d)
            return saved
        except Exception:
            return []

    def list_decisions(
        self,
        project_path: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """List stored decisions, optionally filtered by project."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if project_path:
                    project_path = str(Path(project_path).resolve())
                    rows = conn.execute(
                        "SELECT * FROM kg_decisions WHERE project_path = ? "
                        "ORDER BY timestamp DESC LIMIT ?",
                        (project_path, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM kg_decisions ORDER BY timestamp DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def search_decisions(self, query: str, project_path: Optional[str] = None) -> List[Dict]:
        """Keyword search across decision titles, descriptions, and rationale."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                q = f"%{query.lower()}%"
                if project_path:
                    project_path = str(Path(project_path).resolve())
                    rows = conn.execute(
                        """SELECT * FROM kg_decisions
                           WHERE project_path = ?
                             AND (LOWER(title) LIKE ? OR LOWER(description) LIKE ?
                                  OR LOWER(rationale) LIKE ? OR LOWER(tags) LIKE ?)
                           ORDER BY timestamp DESC LIMIT 20""",
                        (project_path, q, q, q, q),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM kg_decisions
                           WHERE LOWER(title) LIKE ? OR LOWER(description) LIKE ?
                                 OR LOWER(rationale) LIKE ? OR LOWER(tags) LIKE ?
                           ORDER BY timestamp DESC LIMIT 20""",
                        (q, q, q, q),
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    # ══════════════════════════════════════════════════════════════════
    #  Project onboarding & summaries
    # ══════════════════════════════════════════════════════════════════

    def onboard_project(
        self,
        project_path: str,
        force: bool = False,
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """
        Full project onboarding: build graph + generate LLM summary.

        This is the "drop a new project path and Rain understands it" feature.
        Returns the graph stats plus a human-readable project summary.
        """
        project_path = str(Path(project_path).resolve())

        # Step 1: build the graph
        if progress_fn:
            progress_fn("Building knowledge graph...")
        stats = self.build_graph(project_path, force=force, progress_fn=progress_fn)
        if "error" in stats:
            return stats

        # Step 2: generate LLM summary
        if progress_fn:
            progress_fn("Generating project summary...")

        graph_stats = self.get_project_stats(project_path)
        summary_text = self._generate_project_summary(project_path, graph_stats)

        # Step 3: persist summary
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO kg_project_summaries
                       (project_path, summary, file_count, function_count,
                        class_count, languages, key_files, generated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(project_path) DO UPDATE SET
                           summary = excluded.summary,
                           file_count = excluded.file_count,
                           function_count = excluded.function_count,
                           class_count = excluded.class_count,
                           languages = excluded.languages,
                           key_files = excluded.key_files,
                           generated_at = excluded.generated_at""",
                    (
                        project_path,
                        summary_text,
                        graph_stats.get("node_types", {}).get("file", 0),
                        graph_stats.get("node_types", {}).get("function", 0),
                        graph_stats.get("node_types", {}).get("class", 0),
                        json.dumps(self._detect_languages(project_path)),
                        json.dumps(self._detect_key_files(project_path)),
                        datetime.now().isoformat(),
                    ),
                )
        except Exception:
            pass

        stats["summary"] = summary_text
        return stats

    def get_project_summary(self, project_path: str) -> Optional[str]:
        """Return the stored LLM-generated project summary, or None."""
        project_path = str(Path(project_path).resolve())
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT summary FROM kg_project_summaries WHERE project_path = ?",
                    (project_path,),
                ).fetchone()
                return row[0] if row else None
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    #  Context block builder (for agent prompt injection)
    # ══════════════════════════════════════════════════════════════════

    def build_context_block(self, query: str, project_path: str) -> str:
        """
        Build a context block from the knowledge graph to inject into
        an agent prompt.  Looks for:
          1. Function/class names mentioned in the query
          2. Relevant decisions
          3. Git history for mentioned files/functions
          4. File structure for mentioned files

        Returns a formatted string ready to prepend, or "".
        """
        project_path = str(Path(project_path).resolve())
        parts: List[str] = []
        q_lower = query.lower()

        # ── Extract names that might be identifiers ───────────────────
        # Look for snake_case, CamelCase, or dotted names
        ident_re = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\b')
        candidates = set(ident_re.findall(query))
        # Filter out common English words
        _STOP = {"the","is","are","was","were","how","why","what","when","where","does",
                 "do","this","that","it","in","on","for","to","from","with","a","an",
                 "and","or","but","not","use","uses","used","using","like","about",
                 "can","could","would","should","will","have","has","had","been",
                 "which","there","their","my","me","we","you","your","our","its",
                 "all","any","each","some","no","yes","true","false","none","if",
                 "else","then","than","of","at","by","as","so","be","get","set"}
        identifiers = [c for c in candidates if c.lower() not in _STOP and len(c) > 2]

        # ── Look up identifiers in the graph ──────────────────────────
        found_nodes: List[Dict] = []
        for ident in identifiers[:10]:  # cap to avoid huge queries
            nodes = self.find_nodes(project_path, name=ident, limit=5)
            for n in nodes:
                if n not in found_nodes:
                    found_nodes.append(n)

        if found_nodes:
            node_lines = ["Relevant code elements from the knowledge graph:"]
            for n in found_nodes[:15]:
                ntype = n.get("node_type", "?")
                name = n.get("name", "?")
                sig = n.get("signature", "")
                loc = ""
                if n.get("line_start"):
                    fp = n.get("file_path", "")
                    try:
                        rel = str(Path(fp).relative_to(project_path))
                    except ValueError:
                        rel = fp
                    loc = f" ({rel}:{n['line_start']})"
                sig_str = f"  {sig}" if sig else ""
                doc = ""
                if n.get("docstring"):
                    doc = f"\n    doc: {n['docstring'][:120]}"
                node_lines.append(f"  [{ntype}] {name}{loc}{sig_str}{doc}")

                # Add caller/callee info for functions
                if ntype in ("function", "method"):
                    callers = self.get_callers(project_path, name)
                    if callers:
                        caller_names = [c.get("name", "?") for c in callers[:5]]
                        node_lines.append(f"    called by: {', '.join(caller_names)}")
                    callees = self.get_callees(project_path, name)
                    if callees:
                        callee_names = [c.get("name", "?") for c in callees[:5]]
                        node_lines.append(f"    calls: {', '.join(callee_names)}")

            parts.append("\n".join(node_lines))

        # ── Decisions related to the query ────────────────────────────
        # Search for decisions matching any keyword in the query
        keywords = [w for w in q_lower.split() if w not in _STOP and len(w) > 2]
        matched_decisions: List[Dict] = []
        for kw in keywords[:5]:
            decs = self.search_decisions(kw, project_path)
            for d in decs:
                if d not in matched_decisions:
                    matched_decisions.append(d)

        if matched_decisions:
            dec_lines = ["Relevant architectural decisions:"]
            for d in matched_decisions[:5]:
                dec_lines.append(f"  • {d.get('title', '?')}")
                if d.get("description"):
                    dec_lines.append(f"    {d['description']}")
                if d.get("rationale"):
                    dec_lines.append(f"    Rationale: {d['rationale']}")
                if d.get("timestamp"):
                    dec_lines.append(f"    Decided: {d['timestamp'][:10]}")
            parts.append("\n".join(dec_lines))

        # ── Git history for mentioned functions ───────────────────────
        is_why_query = any(w in q_lower for w in ("why", "when", "who wrote", "history", "blame", "commit", "introduced"))
        if is_why_query and found_nodes:
            for n in found_nodes[:3]:
                if n.get("node_type") in ("function", "method", "class"):
                    commit = self.get_commit_for_function(
                        project_path,
                        n.get("file_path", ""),
                        n.get("name", ""),
                    )
                    if commit:
                        parts.append(
                            f"Git history for {n['name']}:\n"
                            f"  Introduced by: {commit['author']} on {commit['date'][:10]}\n"
                            f"  Commit: {commit['sha'][:7]} — {commit['message']}"
                        )

        if not parts:
            return ""

        header = f"[Knowledge graph context for: {Path(project_path).name}/]"
        return header + "\n\n" + "\n\n".join(parts)

    # ══════════════════════════════════════════════════════════════════
    #  Cross-project intelligence
    # ══════════════════════════════════════════════════════════════════

    def find_similar_patterns(self, query: str, exclude_project: Optional[str] = None) -> List[Dict]:
        """
        Search across ALL indexed projects for functions, classes, or decisions
        that match the query.  Useful for: "You solved a similar problem in
        Disrupt — want to apply that approach here?"
        """
        exclude = str(Path(exclude_project).resolve()) if exclude_project else None
        results = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                q = f"%{query.lower()}%"
                # Search nodes
                rows = conn.execute(
                    """SELECT * FROM kg_nodes
                       WHERE (LOWER(name) LIKE ? OR LOWER(docstring) LIKE ?)
                       ORDER BY name LIMIT 20""",
                    (q, q),
                ).fetchall()
                for r in rows:
                    if exclude and r["project_path"] == exclude:
                        continue
                    results.append({
                        "type": "node",
                        "project": Path(r["project_path"]).name,
                        "project_path": r["project_path"],
                        **dict(r),
                    })

                # Search decisions
                rows = conn.execute(
                    """SELECT * FROM kg_decisions
                       WHERE LOWER(title) LIKE ? OR LOWER(description) LIKE ?
                             OR LOWER(rationale) LIKE ?
                       ORDER BY timestamp DESC LIMIT 10""",
                    (q, q, q),
                ).fetchall()
                for r in rows:
                    pp = r["project_path"] or ""
                    if exclude and pp == exclude:
                        continue
                    results.append({
                        "type": "decision",
                        "project": Path(pp).name if pp else "global",
                        "project_path": pp,
                        **dict(r),
                    })
        except Exception:
            pass
        return results[:20]

    # ══════════════════════════════════════════════════════════════════
    #  Internal: Language-specific parsers
    # ══════════════════════════════════════════════════════════════════

    def _parse_python(self, project_path: str, file_path: Path, file_node_id: int) -> Tuple[int, int]:
        """Parse a Python file using ast. Returns (nodes_created, edges_created)."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            if len(source) > MAX_PARSE_SIZE:
                return (0, 0)
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            return (0, 0)

        nodes_created = 0
        edges_created = 0
        file_str = str(file_path)

        for node in ast.walk(tree):
            # ── Imports ───────────────────────────────────────────
            if isinstance(node, ast.Import):
                for alias in node.names:
                    nid = self._upsert_node(
                        project_path, file_str, "import", alias.name,
                        line_start=node.lineno,
                    )
                    self._upsert_edge(project_path, file_node_id, nid, "contains")
                    nodes_created += 1
                    edges_created += 1

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    nid = self._upsert_node(
                        project_path, file_str, "import", full_name,
                        line_start=node.lineno,
                    )
                    self._upsert_edge(project_path, file_node_id, nid, "contains")
                    nodes_created += 1
                    edges_created += 1

            # ── Functions ─────────────────────────────────────────
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Skip methods — they'll be handled under their class
                # Check if parent is a ClassDef by looking at _parent annotation
                # (we set this below)
                parent_cls = getattr(node, '_parent_class', None)
                ntype = "method" if parent_cls else "function"

                # Build signature
                args = []
                for arg in node.args.args:
                    ann = ""
                    if arg.annotation:
                        try:
                            ann = f": {ast.unparse(arg.annotation)}"
                        except Exception:
                            pass
                    args.append(f"{arg.arg}{ann}")
                sig = f"def {node.name}({', '.join(args)})"

                # Return annotation
                if node.returns:
                    try:
                        sig += f" -> {ast.unparse(node.returns)}"
                    except Exception:
                        pass

                # Docstring
                docstring = ast.get_docstring(node) or ""
                if len(docstring) > 300:
                    docstring = docstring[:300] + "..."

                # Decorators
                decorators = []
                for dec in node.decorator_list:
                    try:
                        decorators.append(f"@{ast.unparse(dec)}")
                    except Exception:
                        pass

                end_line = getattr(node, 'end_lineno', node.lineno + 1)

                meta = {}
                if decorators:
                    meta["decorators"] = decorators
                if parent_cls:
                    meta["class"] = parent_cls

                nid = self._upsert_node(
                    project_path, file_str, ntype, node.name,
                    signature=sig, line_start=node.lineno, line_end=end_line,
                    docstring=docstring, metadata_dict=meta,
                )

                # Edge: file contains function/method
                self._upsert_edge(project_path, file_node_id, nid, "contains")
                nodes_created += 1
                edges_created += 1

                # Find function calls within this function body
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        call_name = self._extract_call_name(child)
                        if call_name:
                            # Create a placeholder target node (will be resolved later)
                            target_id = self._upsert_node(
                                project_path, file_str, "function", call_name,
                            )
                            self._upsert_edge(project_path, nid, target_id, "calls")
                            edges_created += 1

            # ── Classes ───────────────────────────────────────────
            elif isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node) or ""
                if len(docstring) > 300:
                    docstring = docstring[:300] + "..."

                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except Exception:
                        pass

                end_line = getattr(node, 'end_lineno', node.lineno + 1)

                meta = {}
                if bases:
                    meta["bases"] = bases

                cls_id = self._upsert_node(
                    project_path, file_str, "class", node.name,
                    signature=f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
                    line_start=node.lineno, line_end=end_line,
                    docstring=docstring, metadata_dict=meta,
                )

                # Edge: file contains class
                self._upsert_edge(project_path, file_node_id, cls_id, "contains")
                nodes_created += 1
                edges_created += 1

                # Inheritance edges
                for base_name in bases:
                    base_id = self._upsert_node(
                        project_path, file_str, "class", base_name,
                    )
                    self._upsert_edge(project_path, cls_id, base_id, "inherits")
                    edges_created += 1

                # Annotate methods with their parent class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        item._parent_class = node.name

        return (nodes_created, edges_created)

    def _parse_js_ts(self, project_path: str, file_path: Path, file_node_id: int) -> Tuple[int, int]:
        """Parse a JS/TS file using regex. Returns (nodes_created, edges_created)."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            if len(source) > MAX_PARSE_SIZE:
                return (0, 0)
        except Exception:
            return (0, 0)

        nodes_created = 0
        edges_created = 0
        file_str = str(file_path)

        # ── Imports ───────────────────────────────────────────────
        for m in re.finditer(r'import\s+(?:(?:\{[^}]+\}|\*\s+as\s+\w+|\w+)\s+from\s+)?["\']([^"\']+)["\']', source):
            nid = self._upsert_node(project_path, file_str, "import", m.group(1), line_start=source[:m.start()].count('\n') + 1)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1
            edges_created += 1

        # ── require() ────────────────────────────────────────────
        for m in re.finditer(r'require\(["\']([^"\']+)["\']\)', source):
            nid = self._upsert_node(project_path, file_str, "import", m.group(1), line_start=source[:m.start()].count('\n') + 1)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1
            edges_created += 1

        # ── Functions ────────────────────────────────────────────
        # Named function declarations
        for m in re.finditer(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)', source):
            name = m.group(1)
            params = m.group(2).strip()
            line = source[:m.start()].count('\n') + 1
            sig = f"function {name}({params})"
            nid = self._upsert_node(project_path, file_str, "function", name, signature=sig, line_start=line)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1
            edges_created += 1

        # Arrow functions assigned to const/let/var
        for m in re.finditer(r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>', source):
            name = m.group(1)
            line = source[:m.start()].count('\n') + 1
            nid = self._upsert_node(project_path, file_str, "function", name, signature=f"const {name} = (...) =>", line_start=line)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1
            edges_created += 1

        # ── Classes ──────────────────────────────────────────────
        for m in re.finditer(r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?', source):
            name = m.group(1)
            base = m.group(2)
            line = source[:m.start()].count('\n') + 1
            sig = f"class {name}" + (f" extends {base}" if base else "")
            meta = {"bases": [base]} if base else {}
            cls_id = self._upsert_node(project_path, file_str, "class", name, signature=sig, line_start=line, metadata_dict=meta)
            self._upsert_edge(project_path, file_node_id, cls_id, "contains")
            nodes_created += 1
            edges_created += 1
            if base:
                base_id = self._upsert_node(project_path, file_str, "class", base)
                self._upsert_edge(project_path, cls_id, base_id, "inherits")
                edges_created += 1

        return (nodes_created, edges_created)

    def _parse_rust(self, project_path: str, file_path: Path, file_node_id: int) -> Tuple[int, int]:
        """Parse a Rust file using regex. Returns (nodes_created, edges_created)."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            if len(source) > MAX_PARSE_SIZE:
                return (0, 0)
        except Exception:
            return (0, 0)

        nodes_created = 0
        edges_created = 0
        file_str = str(file_path)

        # use statements
        for m in re.finditer(r'use\s+([\w:]+(?:::\{[^}]+\})?);', source):
            nid = self._upsert_node(project_path, file_str, "import", m.group(1), line_start=source[:m.start()].count('\n') + 1)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1; edges_created += 1

        # fn declarations
        for m in re.finditer(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)', source):
            name = m.group(1)
            params = m.group(2).strip()[:100]
            line = source[:m.start()].count('\n') + 1
            nid = self._upsert_node(project_path, file_str, "function", name, signature=f"fn {name}({params})", line_start=line)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1; edges_created += 1

        # struct declarations
        for m in re.finditer(r'(?:pub\s+)?struct\s+(\w+)', source):
            name = m.group(1)
            line = source[:m.start()].count('\n') + 1
            nid = self._upsert_node(project_path, file_str, "class", name, signature=f"struct {name}", line_start=line)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1; edges_created += 1

        # impl blocks
        for m in re.finditer(r'impl(?:\s+<[^>]*>)?\s+(\w+)', source):
            name = m.group(1)
            line = source[:m.start()].count('\n') + 1
            nid = self._upsert_node(project_path, file_str, "class", name, line_start=line)
            # Don't double-count the edge if struct already exists
            nodes_created += 1

        return (nodes_created, edges_created)

    def _parse_go(self, project_path: str, file_path: Path, file_node_id: int) -> Tuple[int, int]:
        """Parse a Go file using regex. Returns (nodes_created, edges_created)."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            if len(source) > MAX_PARSE_SIZE:
                return (0, 0)
        except Exception:
            return (0, 0)

        nodes_created = 0
        edges_created = 0
        file_str = str(file_path)

        # import statements
        for m in re.finditer(r'import\s+"([^"]+)"', source):
            nid = self._upsert_node(project_path, file_str, "import", m.group(1), line_start=source[:m.start()].count('\n') + 1)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1; edges_created += 1

        # func declarations
        for m in re.finditer(r'func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s+)?(\w+)\s*\(([^)]*)\)', source):
            name = m.group(1)
            params = m.group(2).strip()[:100]
            line = source[:m.start()].count('\n') + 1
            nid = self._upsert_node(project_path, file_str, "function", name, signature=f"func {name}({params})", line_start=line)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1; edges_created += 1

        # type struct declarations
        for m in re.finditer(r'type\s+(\w+)\s+struct', source):
            name = m.group(1)
            line = source[:m.start()].count('\n') + 1
            nid = self._upsert_node(project_path, file_str, "class", name, signature=f"type {name} struct", line_start=line)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1; edges_created += 1

        # type interface declarations
        for m in re.finditer(r'type\s+(\w+)\s+interface', source):
            name = m.group(1)
            line = source[:m.start()].count('\n') + 1
            nid = self._upsert_node(project_path, file_str, "class", name, signature=f"type {name} interface", line_start=line)
            self._upsert_edge(project_path, file_node_id, nid, "contains")
            nodes_created += 1; edges_created += 1

        return (nodes_created, edges_created)

    # ══════════════════════════════════════════════════════════════════
    #  Internal: helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_call_name(call_node: ast.Call) -> Optional[str]:
        """Extract a readable name from an ast.Call node."""
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            # e.g. self.method() → "method", obj.func() → "func"
            return func.attr
        return None

    def _upsert_node(
        self,
        project_path: str,
        file_path: str,
        node_type: str,
        name: str,
        signature: Optional[str] = None,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
        docstring: Optional[str] = None,
        metadata_dict: Optional[Dict] = None,
    ) -> int:
        """Insert or update a node. Returns the node ID."""
        now = datetime.now().isoformat()
        metadata = json.dumps(metadata_dict) if metadata_dict else None

        with sqlite3.connect(self.db_path) as conn:
            # Check if node already exists (same project, file, type, name, line)
            existing = conn.execute(
                """SELECT id FROM kg_nodes
                   WHERE project_path = ? AND file_path = ?
                     AND node_type = ? AND name = ?
                     AND (line_start = ? OR (line_start IS NULL AND ? IS NULL))""",
                (project_path, file_path, node_type, name, line_start, line_start),
            ).fetchone()

            if existing:
                # Update if we have better data
                if signature or docstring:
                    conn.execute(
                        """UPDATE kg_nodes SET
                               signature = COALESCE(?, signature),
                               line_end = COALESCE(?, line_end),
                               docstring = COALESCE(?, docstring),
                               metadata = COALESCE(?, metadata),
                               indexed_at = ?
                           WHERE id = ?""",
                        (signature, line_end, docstring, metadata, now, existing[0]),
                    )
                return existing[0]
            else:
                cur = conn.execute(
                    """INSERT INTO kg_nodes
                       (project_path, file_path, node_type, name, signature,
                        line_start, line_end, docstring, metadata, indexed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (project_path, file_path, node_type, name, signature,
                     line_start, line_end, docstring, metadata, now),
                )
                return cur.lastrowid

    def _upsert_edge(
        self,
        project_path: str,
        source_id: int,
        target_id: int,
        edge_type: str,
        metadata_dict: Optional[Dict] = None,
    ):
        """Insert an edge if it doesn't already exist."""
        metadata = json.dumps(metadata_dict) if metadata_dict else None
        try:
            with sqlite3.connect(self.db_path) as conn:
                existing = conn.execute(
                    """SELECT id FROM kg_edges
                       WHERE project_path = ? AND source_id = ?
                         AND target_id = ? AND edge_type = ?""",
                    (project_path, source_id, target_id, edge_type),
                ).fetchone()
                if not existing:
                    conn.execute(
                        """INSERT INTO kg_edges
                           (project_path, source_id, target_id, edge_type, metadata)
                           VALUES (?, ?, ?, ?, ?)""",
                        (project_path, source_id, target_id, edge_type, metadata),
                    )
        except Exception:
            pass

    def _resolve_import_edges(self, project_path: str, root: Path):
        """
        After all files are parsed, resolve import nodes to actual file nodes.
        E.g. 'from rain import RainMemory' → edge from import node to rain.py file node.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Get all import nodes for this project
                imports = conn.execute(
                    "SELECT id, name FROM kg_nodes WHERE project_path = ? AND node_type = 'import'",
                    (project_path,),
                ).fetchall()

                # Get all file nodes
                files = conn.execute(
                    "SELECT id, name, file_path FROM kg_nodes WHERE project_path = ? AND node_type = 'file'",
                    (project_path,),
                ).fetchall()

                # Build a lookup: stem → file node id
                stem_to_file: Dict[str, int] = {}
                for f in files:
                    stem = Path(f["file_path"]).stem
                    stem_to_file[stem] = f["id"]
                    # Also store the relative path without extension
                    rel = f["name"].replace("/", ".").replace("\\", ".")
                    if rel.endswith(".py"):
                        stem_to_file[rel[:-3]] = f["id"]

                for imp in imports:
                    import_name = imp["name"]
                    # Try to match: 'rain' → rain.py, 'indexer' → indexer.py
                    parts = import_name.split(".")
                    for i in range(len(parts), 0, -1):
                        candidate = ".".join(parts[:i])
                        if candidate in stem_to_file:
                            self._upsert_edge(
                                project_path, imp["id"], stem_to_file[candidate], "references",
                            )
                            break
                        # Also try just the last part
                        if parts[-1] in stem_to_file:
                            self._upsert_edge(
                                project_path, imp["id"], stem_to_file[parts[-1]], "references",
                            )
                            break
        except Exception:
            pass

    def _index_git_history(self, project_path: str, root: Path):
        """
        Store git commit metadata as node metadata on file nodes.
        This enriches the graph so 'why' questions can be answered.
        """
        try:
            # Get last commit for each file in the project
            result = subprocess.run(
                "git log --format='%H|%ai|%an|%s' -n 50",
                shell=True, cwd=str(root),
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("|", 3)
                if len(parts) == 4:
                    commits.append({
                        "sha": parts[0].strip("'"),
                        "date": parts[1].strip(),
                        "author": parts[2].strip(),
                        "message": parts[3].strip(),
                    })

            # Store as project-level metadata (we don't create nodes for commits
            # to keep the graph focused on code structure)
            if commits:
                with sqlite3.connect(self.db_path) as conn:
                    # Store on the first file node as a project-wide annotation
                    # (lightweight approach — no new tables needed)
                    file_nodes = conn.execute(
                        "SELECT id FROM kg_nodes WHERE project_path = ? AND node_type = 'file' LIMIT 1",
                        (project_path,),
                    ).fetchall()
                    if file_nodes:
                        # We just confirm git is indexed — the actual history
                        # is queried live via get_git_history() for freshness
                        pass
        except Exception:
            pass

    def _generate_project_summary(self, project_path: str, stats: Dict) -> str:
        """
        Use the local LLM to generate a human-readable project summary
        from the knowledge graph stats.
        """
        import urllib.request

        node_types = stats.get("node_types", {})
        edge_types = stats.get("edge_types", {})

        # Gather key file names
        key_files = self._detect_key_files(project_path)
        languages = self._detect_languages(project_path)

        # Build a structured description for the LLM
        description = (
            f"Project: {Path(project_path).name}\n"
            f"Files: {node_types.get('file', 0)}\n"
            f"Functions: {node_types.get('function', 0)}\n"
            f"Classes: {node_types.get('class', 0)}\n"
            f"Methods: {node_types.get('method', 0)}\n"
            f"Imports: {node_types.get('import', 0)}\n"
            f"Call relationships: {edge_types.get('calls', 0)}\n"
            f"Inheritance relationships: {edge_types.get('inherits', 0)}\n"
            f"Languages: {', '.join(languages) if languages else 'unknown'}\n"
            f"Key files: {', '.join(key_files[:10]) if key_files else 'none detected'}\n"
        )

        prompt = (
            "Based on the following project statistics from a code analysis, "
            "write a concise 3-5 sentence summary of what this project likely does, "
            "its main components, and its structure. Be specific about the numbers.\n\n"
            f"{description}\n\n"
            "Project summary:"
        )

        try:
            payload = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()[:1000]
        except Exception:
            # Fallback: generate a simple stats-based summary
            return (
                f"{Path(project_path).name}: "
                f"{node_types.get('file', 0)} files, "
                f"{node_types.get('function', 0)} functions, "
                f"{node_types.get('class', 0)} classes. "
                f"Languages: {', '.join(languages) if languages else 'unknown'}."
            )

    def _detect_languages(self, project_path: str) -> List[str]:
        """Detect programming languages used in the project from file extensions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT metadata FROM kg_nodes WHERE project_path = ? AND node_type = 'file'",
                    (project_path,),
                ).fetchall()

            ext_counts: Dict[str, int] = {}
            for row in rows:
                try:
                    meta = json.loads(row[0]) if row[0] else {}
                    ext = meta.get("extension", "")
                    if ext:
                        ext_counts[ext] = ext_counts.get(ext, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass

            ext_to_lang = {
                ".py": "Python", ".pyi": "Python",
                ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
                ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript",
                ".rs": "Rust", ".go": "Go",
                ".rb": "Ruby", ".java": "Java", ".kt": "Kotlin",
                ".c": "C", ".cpp": "C++", ".h": "C/C++",
                ".swift": "Swift", ".cs": "C#",
                ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
                ".sql": "SQL", ".sh": "Shell",
                ".md": "Markdown", ".json": "JSON", ".yaml": "YAML",
            }

            languages = []
            for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
                lang = ext_to_lang.get(ext, ext)
                if lang not in languages:
                    languages.append(lang)
            return languages[:8]
        except Exception:
            return []

    def _detect_key_files(self, project_path: str) -> List[str]:
        """Detect key files by number of contained nodes (most functions/classes)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """SELECT n.name, COUNT(e.id) as child_count
                       FROM kg_nodes n
                       JOIN kg_edges e ON e.source_id = n.id AND e.edge_type = 'contains'
                       WHERE n.project_path = ? AND n.node_type = 'file'
                       GROUP BY n.id
                       ORDER BY child_count DESC
                       LIMIT 10""",
                    (project_path,),
                ).fetchall()
                return [r[0] for r in rows]
        except Exception:
            return []

    def _walk(self, root: Path):
        """Yield code file paths under root, skipping ignored dirs."""
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel_parts = set(path.relative_to(root).parts[:-1])
            if rel_parts & IGNORE_DIRS:
                continue
            if any(part.startswith(".") for part in rel_parts):
                continue
            # Skip filenames reserved for Claude — indexing these would inject
            # AI-assistant instructions into Rain's own knowledge graph.
            if path.name in IGNORE_FILES:
                continue
            ext = path.suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > MAX_PARSE_SIZE:
                    continue
            except OSError:
                continue
            yield path

    def _clear_project(self, project_path: str):
        """Remove all graph data for a project."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM kg_edges WHERE project_path = ?", (project_path,))
                conn.execute("DELETE FROM kg_nodes WHERE project_path = ?", (project_path,))
        except Exception:
            pass

    def _ensure_tables(self):
        """Create knowledge graph tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS kg_nodes (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_path  TEXT    NOT NULL,
                        file_path     TEXT    NOT NULL,
                        node_type     TEXT    NOT NULL,
                        name          TEXT    NOT NULL,
                        signature     TEXT,
                        line_start    INTEGER,
                        line_end      INTEGER,
                        docstring     TEXT,
                        metadata      TEXT,
                        indexed_at    TEXT    NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_kg_nodes_project
                        ON kg_nodes(project_path);
                    CREATE INDEX IF NOT EXISTS idx_kg_nodes_name
                        ON kg_nodes(project_path, name);
                    CREATE INDEX IF NOT EXISTS idx_kg_nodes_type
                        ON kg_nodes(project_path, node_type);

                    CREATE TABLE IF NOT EXISTS kg_edges (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_path  TEXT    NOT NULL,
                        source_id     INTEGER NOT NULL,
                        target_id     INTEGER NOT NULL,
                        edge_type     TEXT    NOT NULL,
                        metadata      TEXT,
                        FOREIGN KEY (source_id) REFERENCES kg_nodes(id),
                        FOREIGN KEY (target_id) REFERENCES kg_nodes(id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_kg_edges_source
                        ON kg_edges(source_id);
                    CREATE INDEX IF NOT EXISTS idx_kg_edges_target
                        ON kg_edges(target_id);
                    CREATE INDEX IF NOT EXISTS idx_kg_edges_project
                        ON kg_edges(project_path);

                    CREATE TABLE IF NOT EXISTS kg_decisions (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_path  TEXT,
                        session_id    TEXT,
                        title         TEXT    NOT NULL,
                        description   TEXT    NOT NULL DEFAULT '',
                        context       TEXT,
                        alternatives  TEXT,
                        rationale     TEXT,
                        tags          TEXT,
                        timestamp     TEXT    NOT NULL,
                        commit_sha    TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_kg_decisions_project
                        ON kg_decisions(project_path);

                    CREATE TABLE IF NOT EXISTS kg_project_summaries (
                        project_path  TEXT PRIMARY KEY,
                        summary       TEXT    NOT NULL,
                        file_count    INTEGER,
                        function_count INTEGER,
                        class_count   INTEGER,
                        languages     TEXT,
                        key_files     TEXT,
                        generated_at  TEXT    NOT NULL
                    );
                """)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rain ⛈️ - Knowledge Graph & Deep Project Intelligence (Phase 10)",
    )
    parser.add_argument("--build", metavar="PATH", help="Build knowledge graph for a project directory")
    parser.add_argument("--force", action="store_true", help="Clear existing graph data before building")
    parser.add_argument("--summary", metavar="PATH", help="Show stored project summary")
    parser.add_argument("--onboard", metavar="PATH", help="Full onboarding: build graph + generate summary")
    parser.add_argument("--stats", metavar="PATH", help="Show knowledge graph statistics")
    parser.add_argument("--find", nargs=2, metavar=("PATH", "NAME"), help="Find nodes by name in a project")
    parser.add_argument("--callers", nargs=2, metavar=("PATH", "NAME"), help="Find callers of a function")
    parser.add_argument("--callees", nargs=2, metavar=("PATH", "NAME"), help="Find what a function calls")
    parser.add_argument("--file-structure", nargs=2, metavar=("PATH", "FILE"), help="Show structure of a file")
    parser.add_argument("--history", metavar="PATH", help="Show git history for a project")
    parser.add_argument("--history-file", metavar="FILE", help="Show git history for a specific file (use with --history)")
    parser.add_argument("--blame", nargs=2, metavar=("PATH", "FILE"), help="Show blame summary for a file")
    parser.add_argument("--blame-function", metavar="NAME", help="Scope blame to a function (use with --blame)")
    parser.add_argument("--decisions", nargs="?", const="__all__", metavar="PATH", help="List decisions (optionally for a project)")
    parser.add_argument("--log-decision", nargs=2, metavar=("TITLE", "DESCRIPTION"), help="Log a new decision")
    parser.add_argument("--decision-project", metavar="PATH", help="Project path for --log-decision")
    parser.add_argument("--context", nargs=2, metavar=("PATH", "QUERY"), help="Build context block for a query")
    parser.add_argument("--cross-project", metavar="QUERY", help="Search across all projects for patterns")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    kg = KnowledgeGraph()

    def on_progress(msg: str):
        print(f"  📂 {msg}")

    if args.build:
        print(f"⛈️  Building knowledge graph for: {args.build}")
        stats = kg.build_graph(args.build, force=args.force, progress_fn=on_progress)
        if "error" in stats:
            print(f"❌ {stats['error']}")
        else:
            print(f"\n✅ Graph built in {stats['duration_s']}s:")
            print(f"   Files parsed:  {stats['files_parsed']}")
            print(f"   Nodes created: {stats['nodes']}")
            print(f"   Edges created: {stats['edges']}")
            if stats['errors']:
                print(f"   Errors:        {stats['errors']}")

    elif args.onboard:
        print(f"⛈️  Onboarding project: {args.onboard}")
        result = kg.onboard_project(args.onboard, force=args.force, progress_fn=on_progress)
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"\n✅ Project onboarded in {result['duration_s']}s:")
            print(f"   Files: {result['files_parsed']}  Nodes: {result['nodes']}  Edges: {result['edges']}")
            if result.get("summary"):
                print(f"\n📋 Summary:\n{result['summary']}")

    elif args.stats:
        stats = kg.get_project_stats(args.stats)
        if "error" in stats:
            print(f"❌ {stats['error']}")
        else:
            print(f"⛈️  Knowledge Graph — {Path(args.stats).name}")
            print(f"   Total nodes: {stats['total_nodes']}")
            print(f"   Total edges: {stats['total_edges']}")
            print(f"   Decisions:   {stats['decisions']}")
            if stats.get("node_types"):
                print("   Node types:")
                for ntype, count in stats["node_types"].items():
                    print(f"     {ntype}: {count}")
            if stats.get("edge_types"):
                print("   Edge types:")
                for etype, count in stats["edge_types"].items():
                    print(f"     {etype}: {count}")

    elif args.summary:
        summary = kg.get_project_summary(args.summary)
        if summary:
            print(f"📋 Project summary — {Path(args.summary).name}:\n{summary}")
        else:
            print("No summary stored. Run --onboard first.")

    elif args.find:
        path, name = args.find
        nodes = kg.find_nodes(path, name=name)
        if nodes:
            print(f"Found {len(nodes)} node(s) matching '{name}':")
            for n in nodes:
                loc = f":{n['line_start']}" if n.get("line_start") else ""
                sig = f"  {n['signature']}" if n.get("signature") else ""
                print(f"  [{n['node_type']}] {n['name']}{loc}{sig}")
        else:
            print(f"No nodes found matching '{name}'")

    elif args.callers:
        path, name = args.callers
        callers = kg.get_callers(path, name)
        if callers:
            print(f"Functions that call '{name}':")
            for c in callers:
                loc = f":{c['line_start']}" if c.get("line_start") else ""
                print(f"  [{c['node_type']}] {c['name']}{loc}")
        else:
            print(f"No callers found for '{name}'")

    elif args.callees:
        path, name = args.callees
        callees = kg.get_callees(path, name)
        if callees:
            print(f"Functions called by '{name}':")
            for c in callees:
                print(f"  [{c['node_type']}] {c['name']}")
        else:
            print(f"No callees found for '{name}'")

    elif args.file_structure:
        path, file_path = args.file_structure
        structure = kg.get_file_structure(path, file_path)
        if "error" in structure:
            print(f"❌ {structure['error']}")
        else:
            print(f"📄 {file_path}:")
            for section, items in [("Functions", structure.get("functions", [])),
                                   ("Classes", structure.get("classes", [])),
                                   ("Methods", structure.get("methods", [])),
                                   ("Imports", structure.get("imports", []))]:
                if items:
                    print(f"  {section} ({len(items)}):")
                    for item in items:
                        sig = item.get("signature", item.get("name", "?"))
                        line = f":{item['line_start']}" if item.get("line_start") else ""
                        print(f"    {sig}{line}")

    elif args.history:
        commits = kg.get_git_history(args.history, file_path=args.history_file, n=20)
        if commits:
            label = args.history_file or Path(args.history).name
            print(f"Git history — {label}:")
            for c in commits:
                print(f"  {c['sha'][:7]} {c['date'][:10]} ({c['author']}): {c['message']}")
        else:
            print("No git history found.")

    elif args.blame:
        path, file_path = args.blame
        result = kg.get_file_blame_summary(path, file_path, function_name=args.blame_function)
        print(result)

    elif args.decisions is not None:
        project_path = args.decisions if args.decisions != "__all__" else None
        decisions = kg.list_decisions(project_path=project_path)
        if decisions:
            print(f"Architectural decisions ({len(decisions)}):")
            for d in decisions:
                proj = Path(d["project_path"]).name if d.get("project_path") else "global"
                print(f"  [{d['timestamp'][:10]}] [{proj}] {d['title']}")
                if d.get("description"):
                    print(f"    {d['description']}")
                if d.get("rationale"):
                    print(f"    Rationale: {d['rationale']}")
        else:
            print("No decisions logged yet.")

    elif args.log_decision:
        title, description = args.log_decision
        dec_id = kg.log_decision(
            title=title,
            description=description,
            project_path=args.decision_project,
        )
        print(f"✅ Decision logged (ID: {dec_id}): {title}")

    elif args.context:
        path, query = args.context
        block = kg.build_context_block(query, path)
        if block:
            print(block)
        else:
            print("No relevant graph context found. Build the graph first with --build.")

    elif args.cross_project:
        results = kg.find_similar_patterns(args.cross_project)
        if results:
            print(f"Cross-project matches for '{args.cross_project}':")
            for r in results:
                rtype = r.get("type", "?")
                proj = r.get("project", "?")
                name = r.get("name") or r.get("title", "?")
                print(f"  [{rtype}] [{proj}] {name}")
        else:
            print("No cross-project matches found.")


if __name__ == "__main__":
    main()
