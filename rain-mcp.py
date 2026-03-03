#!/usr/bin/env python3
"""
Rain ⛈️ — MCP Server (Model Context Protocol)

Gives ZED, Claude Desktop, and any MCP-compatible editor access to:
  - Rain's semantic project index (search your codebase by meaning)
  - Rain's persistent user memory (what Rain knows about you)
  - Project indexing (trigger re-indexing from inside your editor)

Transport: stdio — ZED spawns this as a subprocess and communicates
over stdin/stdout using newline-delimited JSON-RPC 2.0.

Zero new dependencies — pure stdlib + Rain's existing modules.

ZED configuration (add to settings.json):
  "context_servers": {
    "rain": {
      "command": {
        "path": "/Users/ericscalibur/Documents/Rain/.venv/bin/python3",
        "args": ["/Users/ericscalibur/Documents/Rain/rain-mcp.py"]
      }
    }
  }

Usage:
  ZED agent mode will automatically call search_project when you ask
  questions about your codebase. Memory facts are always available
  via the rain://memory/profile resource.
"""

import json
import sys
import os
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# Add Rain's directory to sys.path so we can import rain.py and indexer.py
_RAIN_DIR = str(Path(__file__).parent.resolve())
if _RAIN_DIR not in sys.path:
    sys.path.insert(0, _RAIN_DIR)

# Lazy imports — failures are surfaced as tool errors, not startup crashes
try:
    from indexer import ProjectIndexer
    _INDEXER_OK = True
except ImportError:
    _INDEXER_OK = False

try:
    from rain import RainMemory
    _MEMORY_OK = True
except ImportError:
    _MEMORY_OK = False


# ── JSON-RPC transport ───────────────────────────────────────────────────────

def _send(obj: dict):
    """Write a JSON-RPC message to stdout and flush immediately."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _respond(req_id, result: dict):
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _error(req_id, code: int, message: str):
    _send({"jsonrpc": "2.0", "id": req_id,
           "error": {"code": code, "message": message}})


def _log(msg: str):
    """Write a debug message to stderr — never pollutes the JSON-RPC stream."""
    sys.stderr.write(f"[rain-mcp] {msg}\n")
    sys.stderr.flush()


# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the full contents of a specific file by path. "
            "Use this when the user asks to see a specific file (e.g. 'can you see README.md', "
            "'show me ROADMAP.md', 'read rain.py'). "
            "Do NOT use search_project for direct file requests — semantic search will find "
            "the wrong file when an exact path or filename is specified."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or project-relative path to the file (e.g. /Users/you/project/README.md or README.md)"
                },
                "project_path": {
                    "type": "string",
                    "description": "Optional: absolute path to the project root, used to resolve relative paths"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to return (default: 200). Use for large files to avoid flooding context.",
                    "default": 200
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "search_project",
        "description": (
            "Search Rain's semantic project index for source code relevant to a query. "
            "Returns the most relevant file chunks ranked by meaning, not just keyword match. "
            "Use this when you need to find how something is implemented, where a function "
            "lives, or what code exists around a concept. The project must be indexed first "
            "using index_project."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for — describe the concept, function, or behaviour"
                },
                "project_path": {
                    "type": "string",
                    "description": "Absolute path to the project directory (e.g. /Users/you/myproject)"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return (default: 5, max: 10)",
                    "default": 5
                }
            },
            "required": ["query", "project_path"]
        }
    },
    {
        "name": "get_user_memory",
        "description": (
            "Get what Rain knows about the user — persistent facts accumulated across all "
            "sessions, including current project, tech stack, goals, preferences, and past "
            "decisions. Use this to orient answers to the user's actual context without "
            "asking them to repeat themselves."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "index_project",
        "description": (
            "Index a project directory so Rain can semantically search its source code. "
            "Walks the project tree, chunks every text file, embeds with nomic-embed-text "
            "via Ollama, and stores in Rain's local SQLite database. "
            "Run once per project, then re-run when files change significantly. "
            "Already-indexed files are skipped unless force=true."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Absolute path to the project directory to index"
                },
                "force": {
                    "type": "boolean",
                    "description": "Re-index files that are already indexed (default: false)",
                    "default": False
                }
            },
            "required": ["project_path"]
        }
    },
    {
        "name": "list_indexed_projects",
        "description": (
            "List all projects currently indexed in Rain's semantic memory, "
            "with file counts, chunk counts, and last-indexed timestamps."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "list_directory",
        "description": (
            "Lists files and directories in a given path. "
            "Use this to explore the project structure before reading or editing files. "
            "Prefer grep or find_path when searching for symbols — use list_directory "
            "only when you need to see what exists in a specific folder. "
            "Defaults to the Rain project directory if no path is given."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "The directory to list. Accepts absolute paths or paths relative "
                        "to the Rain project root. Defaults to the Rain root if omitted."
                    )
                }
            },
            "required": []
        }
    },
    {
        "name": "write_file",
        "description": (
            "Creates a new file or overwrites an existing file with the given content. "
            "A .rain-backup copy of the original is saved automatically before any overwrite. "
            "Every write is logged to ~/.rain/audit.log. "
            "Use this to edit Rain's source code, config files, or any project file. "
            "Prefer targeted edits — read the file first so you only change what is needed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute or project-relative path to write. "
                        "Relative paths resolve against the Rain project root."
                    )
                },
                "content": {
                    "type": "string",
                    "description": "The full text content to write to the file."
                },
                "project_path": {
                    "type": "string",
                    "description": "Optional: project root used to resolve relative paths."
                },
                "backup": {
                    "type": "boolean",
                    "description": "Save a .rain-backup of the original before overwriting (default: true).",
                    "default": True
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "grep",
        "description": (
            "Searches the contents of files in the project with a regular expression. "
            "Prefer this tool to find_path when searching for symbols or code patterns, "
            "because you won't need to guess what path the symbol lives in. "
            "Supports full regex syntax (e.g. 'log.*Error', 'def\\\\s+\\\\w+', etc.). "
            "Pass an include_pattern to narrow the search to specific file types. "
            "Never use this tool to search for paths — use find_path for that. "
            "Results are paginated with up to 30 matches shown."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "regex": {
                    "type": "string",
                    "description": "A regex pattern to search for inside file contents."
                },
                "project_path": {
                    "type": "string",
                    "description": "Absolute path to the directory to search (defaults to Rain root)."
                },
                "include_pattern": {
                    "type": "string",
                    "description": (
                        "Optional glob to filter which files are searched, e.g. '**/*.py' "
                        "or '**/*.json'. If omitted, all text files are searched."
                    )
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the regex is case-sensitive (default: false).",
                    "default": False
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return (default: 30, max: 50).",
                    "default": 30
                }
            },
            "required": ["regex"]
        }
    },
    {
        "name": "find_path",
        "description": (
            "Fast file path pattern matching using glob syntax. "
            "Supports patterns like '**/*.py' or 'src/**/*.ts'. "
            "Returns matching file paths sorted alphabetically. "
            "Prefer grep when searching for symbols — use find_path when you need "
            "to locate files by name pattern. "
            "Results are capped at 50 matches."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "glob": {
                    "type": "string",
                    "description": "The glob pattern to match against every path in the project, e.g. '**/*.py'."
                },
                "project_path": {
                    "type": "string",
                    "description": "Absolute path to the project directory to search (defaults to Rain root)."
                }
            },
            "required": ["glob"]
        }
    }
]


# ── Resource definitions ─────────────────────────────────────────────────────

def _memory_resource():
    return {
        "uri": "rain://memory/profile",
        "name": "Rain — User Memory",
        "description": "What Rain knows about the user: current project, tech stack, goals, and preferences accumulated across all sessions.",
        "mimeType": "text/plain"
    }


# ── Tool execution ───────────────────────────────────────────────────────────

def _call_search_project(args: dict) -> str:
    if not _INDEXER_OK:
        return "❌ indexer.py not found — cannot search project index."
    query = args.get("query", "").strip()
    project_path = args.get("project_path", "").strip()
    top_k = min(int(args.get("top_k", 5)), 10)

    if not query:
        return "❌ query is required."
    if not project_path:
        return "❌ project_path is required."
    if not Path(project_path).exists():
        return f"❌ Path not found: {project_path}"

    try:
        idx = ProjectIndexer()
        context = idx.build_context_block(query, project_path, top_k=top_k)
        if context:
            return context
        return (
            f"No relevant chunks found for '{query}' in {project_path}.\n"
            f"The project may not be indexed yet — use index_project to index it first."
        )
    except Exception as e:
        return f"❌ search_project error: {type(e).__name__}: {e}"


def _call_get_user_memory(_args: dict) -> str:
    if not _MEMORY_OK:
        return "❌ rain.py not found — cannot access user memory."
    try:
        mem = RainMemory()
        ctx = mem.get_fact_context()
        if ctx and ctx.strip():
            return ctx.strip()
        return (
            "No user memory stored yet.\n"
            "Start a conversation with Rain (via rain-web or the CLI) and your "
            "project details will be remembered automatically for future sessions."
        )
    except Exception as e:
        return f"❌ get_user_memory error: {type(e).__name__}: {e}"


def _call_index_project(args: dict) -> str:
    if not _INDEXER_OK:
        return "❌ indexer.py not found — cannot index project."
    project_path = args.get("project_path", "").strip()
    force = bool(args.get("force", False))

    if not project_path:
        return "❌ project_path is required."
    if not Path(project_path).exists():
        return f"❌ Path not found: {project_path}"

    try:
        idx = ProjectIndexer()
        _log(f"Indexing {project_path} (force={force})...")
        stats = idx.index_project(project_path, force=force)
        if "error" in stats:
            return f"❌ {stats['error']}"
        return (
            f"✅ Indexed {stats['files_indexed']} files, "
            f"{stats['chunks_total']} chunks from {project_path} "
            f"in {stats['duration_s']}s\n"
            f"   ({stats['files_skipped']} files skipped — already indexed, "
            f"{stats['errors']} errors)"
        )
    except Exception as e:
        return f"❌ index_project error: {type(e).__name__}: {e}"


def _call_list_directory(args: dict) -> str:
    path_str = (args.get("path") or "").strip()

    if not path_str:
        p = Path(_RAIN_DIR)
    else:
        p = Path(path_str)
        if not p.is_absolute():
            p = Path(_RAIN_DIR) / path_str

    if not p.exists():
        return f"❌ Path not found: {path_str or _RAIN_DIR}"
    if not p.is_file() and not p.is_dir():
        return f"❌ Not a valid path: {p}"
    if p.is_file():
        return f"❌ {p} is a file, not a directory — use read_file to read it."

    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        dirs  = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        lines = [f"# {p}/\n"]
        for d in dirs:
            lines.append(f"  📁 {d.name}/")
        for f in files:
            try:
                size = f.stat().st_size
                lines.append(f"  📄 {f.name}  ({size:,} bytes)")
            except OSError:
                lines.append(f"  📄 {f.name}")
        lines.append(f"\n{len(dirs)} director{'ies' if len(dirs) != 1 else 'y'}, {len(files)} file{'s' if len(files) != 1 else ''}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ list_directory error: {type(e).__name__}: {e}"


def _call_write_file(args: dict) -> str:
    import fnmatch as _fnmatch
    from datetime import datetime as _dt

    path_str     = (args.get("path") or "").strip()
    content      = args.get("content", "")
    project_path = (args.get("project_path") or "").strip()
    do_backup    = bool(args.get("backup", True))

    if not path_str:
        return "❌ path is required."

    p = Path(path_str)
    if not p.is_absolute():
        root = Path(project_path) if project_path else Path(_RAIN_DIR)
        p = root / path_str

    try:
        p.parent.mkdir(parents=True, exist_ok=True)

        backup_note = ""
        if p.exists() and do_backup:
            backup_path = p.with_suffix(p.suffix + ".rain-backup")
            backup_path.write_bytes(p.read_bytes())
            backup_note = f" (backup → {backup_path.name})"

        p.write_text(content, encoding="utf-8")

        # Audit log
        audit = Path.home() / ".rain" / "audit.log"
        audit.parent.mkdir(parents=True, exist_ok=True)
        with open(audit, "a", encoding="utf-8") as fh:
            fh.write(f"[{_dt.now().isoformat()}] write_file: {p}\n")

        return f"✅ Written {len(content):,} chars ({len(content.splitlines())} lines) to {p}{backup_note}"
    except Exception as e:
        return f"❌ write_file error: {type(e).__name__}: {e}"


def _call_grep(args: dict) -> str:
    import re as _re
    import fnmatch as _fnmatch

    regex           = (args.get("regex") or "").strip()
    project_path    = (args.get("project_path") or "").strip() or _RAIN_DIR
    include_pattern = (args.get("include_pattern") or "").strip()
    case_sensitive  = bool(args.get("case_sensitive", False))
    max_results     = min(int(args.get("max_results", 30)), 50)

    if not regex:
        return "❌ regex is required."

    root = Path(project_path)
    if not root.exists():
        return f"❌ Path not found: {project_path}"

    try:
        flags   = 0 if case_sensitive else _re.IGNORECASE
        pattern = _re.compile(regex, flags)
    except _re.error as e:
        return f"❌ Invalid regex '{regex}': {e}"

    # File extensions we'll skip (binary / generated)
    _SKIP_EXTS = {
        ".pyc", ".pyo", ".so", ".dylib", ".db", ".sqlite", ".sqlite3",
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
        ".zip", ".tar", ".gz", ".bz2", ".whl", ".egg",
        ".bin", ".pkl", ".pth", ".pt", ".npz",
    }
    # Directory names we'll skip
    _SKIP_DIRS = {
        "__pycache__", ".git", "node_modules", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", "dist", "build", ".aider.tags.cache.v4",
    }

    results = []

    # Collect candidate files
    # Use rglob("*") + fnmatch filter so patterns like "*.py" and "**/*.py"
    # both work correctly. (rglob with lstrip("*/") breaks "*.py" → ".py".)
    if include_pattern:
        # Match the pattern against the filename portion only
        bare = include_pattern.split("/")[-1]  # e.g. "**/*.py" → "*.py"
        candidates = (f for f in root.rglob("*") if _fnmatch.fnmatch(f.name, bare))
    else:
        candidates = root.rglob("*")

    for file_path in candidates:
        if not file_path.is_file():
            continue
        # Skip ignored dirs
        if any(part in _SKIP_DIRS for part in file_path.parts):
            continue
        if file_path.suffix.lower() in _SKIP_EXTS:
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                try:
                    rel = file_path.relative_to(root)
                except ValueError:
                    rel = file_path
                results.append(f"{rel}:{i}: {line.rstrip()}")
                if len(results) >= max_results:
                    break

        if len(results) >= max_results:
            break

    if not results:
        return f"No matches found for '{regex}' in {root.name}/"

    header = f"# grep '{regex}' in {root.name}/  ({len(results)} match{'es' if len(results) != 1 else ''})\n"
    return header + "\n".join(results)


def _call_find_path(args: dict) -> str:
    glob_pattern = (args.get("glob") or "").strip()
    project_path = (args.get("project_path") or "").strip() or _RAIN_DIR

    if not glob_pattern:
        return "❌ glob is required."

    root = Path(project_path)
    if not root.exists():
        return f"❌ Path not found: {project_path}"

    _SKIP_DIRS = {
        "__pycache__", ".git", "node_modules", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", "dist", "build", ".aider.tags.cache.v4",
    }

    try:
        all_matches = []
        for p in root.rglob(glob_pattern):
            # Skip hidden / build dirs
            if any(part in _SKIP_DIRS or part.startswith(".") for part in p.relative_to(root).parts[:-1]):
                continue
            try:
                all_matches.append(str(p.relative_to(root)))
            except ValueError:
                all_matches.append(str(p))

        all_matches.sort()

        if not all_matches:
            return f"No files matching '{glob_pattern}' in {root.name}/"

        capped = all_matches[:50]
        lines  = [f"# Files matching '{glob_pattern}' in {root.name}/  ({len(all_matches)} total)\n"]
        lines += [f"  {m}" for m in capped]
        if len(all_matches) > 50:
            lines.append(f"\n… ({len(all_matches) - 50} more results not shown)")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ find_path error: {type(e).__name__}: {e}"


def _call_list_indexed_projects(_args: dict) -> str:
    if not _INDEXER_OK:
        return "❌ indexer.py not found."
    try:
        idx = ProjectIndexer()
        projects = idx.list_indexed_projects()
        if not projects:
            return (
                "No projects indexed yet.\n"
                "Use index_project to index a project directory."
            )
        lines = ["Indexed projects:\n"]
        for p in projects:
            last = p.get("last_indexed", "")[:10] if p.get("last_indexed") else "unknown"
            lines.append(
                f"  • {p['project_path']}\n"
                f"    {p['file_count']} files · {p['chunk_count']} chunks · last indexed {last}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ list_indexed_projects error: {type(e).__name__}: {e}"


def _call_read_file(args: dict) -> str:
    path_str = args.get("path", "").strip()
    project_path = args.get("project_path", "").strip()
    max_lines = int(args.get("max_lines", 200))

    if not path_str:
        return "❌ path is required."

    p = Path(path_str)

    # If relative, try to resolve against project_path or Rain's own directory
    if not p.is_absolute():
        if project_path:
            p = Path(project_path) / path_str
        else:
            # Strip leading project-name prefix if present.
            # e.g. "Rain/README.md" when _RAIN_DIR ends in "Rain" becomes
            # Path(_RAIN_DIR) / "README.md", not .../Rain/Rain/README.md
            rain_dir_name = Path(_RAIN_DIR).name
            parts = p.parts
            if parts and parts[0].lower() == rain_dir_name.lower() and len(parts) > 1:
                p = Path(_RAIN_DIR) / Path(*parts[1:])
            else:
                p = Path(_RAIN_DIR) / path_str

    if not p.exists():
        # Last resort: search for the filename anywhere inside Rain's directory
        matches = list(Path(_RAIN_DIR).rglob(p.name))
        if matches:
            # Sort by depth so shallower files win.
            # README.md (1 part from root) beats rain-vscode/README.md (2 parts).
            rain_root = Path(_RAIN_DIR)
            matches.sort(key=lambda m: len(m.relative_to(rain_root).parts))
            p = matches[0]
        else:
            return f"❌ File not found: {path_str}"

    if not p.is_file():
        return f"❌ Not a file: {p}"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        total = len(lines)
        if total > max_lines:
            truncated = lines[:max_lines]
            return (
                f"# {p}  ({total} lines total — showing first {max_lines})\n\n"
                + "\n".join(truncated)
                + f"\n\n… ({total - max_lines} lines truncated — request with higher max_lines to see more)"
            )
        return f"# {p}\n\n{content}"
    except Exception as e:
        return f"❌ Could not read {p}: {type(e).__name__}: {e}"


_TOOL_HANDLERS = {
    "read_file":             _call_read_file,
    "search_project":        _call_search_project,
    "get_user_memory":       _call_get_user_memory,
    "index_project":         _call_index_project,
    "list_indexed_projects": _call_list_indexed_projects,
    "list_directory":        _call_list_directory,
    "write_file":            _call_write_file,
    "grep":                  _call_grep,
    "find_path":             _call_find_path,
}


def _dispatch_tool(name: str, arguments: dict) -> list:
    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        return [{"type": "text", "text": f"❌ Unknown tool: {name}"}]
    text = handler(arguments)
    return [{"type": "text", "text": text}]


# ── Resource reading ─────────────────────────────────────────────────────────

def _read_resource(uri: str) -> str:
    if uri == "rain://memory/profile":
        return _call_get_user_memory({})
    return f"❌ Unknown resource URI: {uri}"


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    _log("Rain MCP server starting — waiting for messages on stdin")

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # Parse incoming JSON-RPC message
        try:
            msg = json.loads(raw_line)
        except json.JSONDecodeError as e:
            _log(f"JSON parse error: {e} — line: {raw_line[:120]}")
            continue

        method  = msg.get("method", "")
        req_id  = msg.get("id")        # None for notifications
        params  = msg.get("params") or {}

        _log(f"← {method} (id={req_id})")

        # ── Lifecycle ────────────────────────────────────────────────────────

        if method == "initialize":
            _respond(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools":     {},
                    "resources": {"subscribe": False, "listChanged": False},
                },
                "serverInfo": {
                    "name":    "rain",
                    "version": "1.0.0",
                }
            })

        elif method == "notifications/initialized":
            pass  # notification — no response expected

        elif method == "ping":
            _respond(req_id, {})

        # ── Tools ────────────────────────────────────────────────────────────

        elif method == "tools/list":
            _respond(req_id, {"tools": TOOLS})

        elif method == "tools/call":
            name      = params.get("name", "")
            arguments = params.get("arguments") or {}
            _log(f"  tool: {name} args={list(arguments.keys())}")
            content = _dispatch_tool(name, arguments)
            _respond(req_id, {"content": content, "isError": False})

        # ── Resources ────────────────────────────────────────────────────────

        elif method == "resources/list":
            _respond(req_id, {"resources": [_memory_resource()]})

        elif method == "resources/read":
            uri  = params.get("uri", "")
            text = _read_resource(uri)
            _respond(req_id, {
                "contents": [{
                    "uri":      uri,
                    "mimeType": "text/plain",
                    "text":     text,
                }]
            })

        elif method == "resources/templates/list":
            _respond(req_id, {"resourceTemplates": []})

        # ── Prompts (not implemented — respond gracefully) ────────────────────

        elif method == "prompts/list":
            _respond(req_id, {"prompts": []})

        # ── Unknown method ───────────────────────────────────────────────────

        else:
            if req_id is not None:
                # Only send error for requests (not notifications)
                _error(req_id, -32601, f"Method not found: {method}")
            else:
                _log(f"Ignored notification: {method}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("Shutting down")
        sys.exit(0)
    except Exception as e:
        _log(f"Fatal: {type(e).__name__}: {e}")
        sys.exit(1)
