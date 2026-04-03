#!/usr/bin/env python3
"""
Rain ⛈️ - Tool Registry (Phase 6)

Gives Rain's agents the ability to act — read files, write files, run commands,
and interact with git. Every destructive operation requires explicit confirmation
and is logged to ~/.rain/audit.log.

Principles:
  - Never execute destructive operations without confirmation
  - Always back up files before overwriting
  - Every action is logged with timestamp, tool, args, and result
  - All paths are resolved and validated before use
  - Tools work with stdlib only — no new dependencies

Available tools:
  read_file(path)                  → file contents
  write_file(path, content)        → write with backup, requires confirm
  list_dir(path)                   → directory listing
  run_command(cmd, cwd, confirm)   → shell command with optional confirm gate
  git_status(repo)                 → git status output
  git_log(repo, n)                 → last N commits
  git_diff(repo)                   → unstaged diff
  git_commit(repo, message)        → stage all + commit, requires confirm

Usage:
    from tools import ToolRegistry
    tools = ToolRegistry()
    result = tools.read_file("rain.py")
    if result.success:
        print(result.output)
"""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional


# ── Result type ───────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Outcome of a single tool invocation."""
    success: bool
    output: str
    error: Optional[str] = None
    tool_name: str = ""
    args: str = ""

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"[{self.tool_name} failed] {self.error or 'Unknown error'}"

    @classmethod
    def ok(cls, output: str, tool_name: str = "", args: str = "") -> 'ToolResult':
        return cls(success=True, output=output, tool_name=tool_name, args=args)

    @classmethod
    def fail(cls, error: str, tool_name: str = "", args: str = "") -> 'ToolResult':
        return cls(success=False, output="", error=error, tool_name=tool_name, args=args)


# ── Tool registry ─────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry of all tools Rain's agents can invoke.

    confirm_fn: optional callable(prompt: str) -> bool
      Called before any destructive operation. If None, all confirmations
      auto-approve (useful for tests / non-interactive pipelines).
      In interactive CLI mode, pass a function that prompts the user.
    """

    AUDIT_LOG = Path.home() / '.rain' / 'audit.log'
    MAX_READ_BYTES = 512_000      # 512 KB read cap — don't swallow huge files
    MAX_OUTPUT_CHARS = 8_000      # truncate long outputs before injecting into prompts

    def __init__(self, confirm_fn: Optional[Callable[[str], bool]] = None):
        self._confirm = confirm_fn or self._auto_confirm
        self.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ── File tools ────────────────────────────────────────────────────

    def read_file(self, path: str,
                  start_line: int = None, end_line: int = None) -> ToolResult:
        """
        Read a file and return its contents.

        start_line / end_line: optional 1-based line numbers (inclusive).
        When provided, only that slice is returned — useful for large files
        where grep_files has already located the relevant section.

        Caps output at MAX_READ_BYTES to protect agent context windows.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return ToolResult.fail(f"Path not found or not accessible: {path}", "read_file", path)

        if resolved.is_dir():
            return ToolResult.fail(f"{path} is a directory, not a file. Use list_dir.", "read_file", path)

        try:
            with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)

            if start_line is not None or end_line is not None:
                # 1-based, inclusive — clamp to valid range
                lo = max(1, start_line or 1)
                hi = min(total_lines, end_line or total_lines)
                slice_lines = all_lines[lo - 1:hi]
                content = "".join(slice_lines)
                header = f"# {resolved}\n# lines {lo}–{hi} of {total_lines}\n\n"
                output = header + content[:self.MAX_READ_BYTES]
                if len(content) > self.MAX_READ_BYTES:
                    output += f"\n[... truncated at {self.MAX_READ_BYTES:,} chars]"
                self._audit("read_file", f"{path}:{lo}-{hi}", "ok")
            else:
                content = "".join(all_lines)
                truncated = len(content) > self.MAX_READ_BYTES
                shown = content[:self.MAX_READ_BYTES]
                suffix = (
                    f"\n\n[... TRUNCATED — file has {total_lines} lines but only the first "
                    f"{self.MAX_READ_BYTES:,} chars are shown. "
                    f"To read a specific section: use grep_files to find the relevant line numbers, "
                    f"then call read_file {path} <start_line> <end_line>. "
                    f"Example: read_file {path} 200 260]"
                ) if truncated else ""
                output = f"# {resolved}\n# {total_lines} lines{' (truncated — see instructions below)' if truncated else ''}\n\n{shown}{suffix}"
                self._audit("read_file", path, "ok")

            return ToolResult.ok(output, "read_file", path)

        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}", "read_file", path)
        except Exception as e:
            return ToolResult.fail(str(e), "read_file", path)

    def write_file(self, path: str, content: str, require_confirm: bool = True) -> ToolResult:
        """
        Write content to a file.
        - Creates parent directories if needed.
        - If the file already exists, backs it up first as <file>.rain-backup.
        - Requires confirmation before writing (unless require_confirm=False).
        """
        resolved = Path(path).expanduser().resolve()

        exists = resolved.exists()
        action = "overwrite" if exists else "create"

        if require_confirm:
            preview_lines = content.splitlines()[:10]
            preview = "\n".join(preview_lines)
            if len(content.splitlines()) > 10:
                preview += f"\n... ({len(content.splitlines())} total lines)"

            prompt = (
                f"📝 write_file: {action} {resolved}\n"
                f"{'─' * 60}\n"
                f"{preview}\n"
                f"{'─' * 60}\n"
                f"Proceed? (y/n): "
            )
            if not self._confirm(prompt):
                self._audit("write_file", path, "cancelled by user")
                return ToolResult.fail("Write cancelled by user.", "write_file", path)

        try:
            # Backup existing file
            backup_path = None
            if exists:
                backup_path = resolved.with_suffix(resolved.suffix + '.rain-backup')
                shutil.copy2(resolved, backup_path)

            # Create parent dirs
            resolved.parent.mkdir(parents=True, exist_ok=True)

            # Write
            with open(resolved, 'w', encoding='utf-8') as f:
                f.write(content)

            detail = f"written ({len(content):,} chars)"
            if backup_path:
                detail += f", backup: {backup_path.name}"

            self._audit("write_file", path, detail)
            return ToolResult.ok(
                f"✅ Written: {resolved}\n   {len(content):,} chars · {content.count(chr(10)) + 1} lines"
                + (f"\n   Backup: {backup_path}" if backup_path else ""),
                "write_file", path
            )

        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}", "write_file", path)
        except Exception as e:
            return ToolResult.fail(str(e), "write_file", path)

    def list_dir(self, path: str = ".") -> ToolResult:
        """
        List files and subdirectories at path.
        Shows type (📁/📄), size, and modification time.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return ToolResult.fail(f"Directory not found: {path}", "list_dir", path)

        if not resolved.is_dir():
            return ToolResult.fail(f"{path} is a file, not a directory. Use read_file.", "list_dir", path)

        try:
            entries = sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            lines = [f"📁 {resolved}/\n"]
            for entry in entries:
                try:
                    stat = entry.stat()
                    if entry.is_dir():
                        lines.append(f"  📁 {entry.name}/")
                    else:
                        size = self._human_size(stat.st_size)
                        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%b %d %H:%M")
                        lines.append(f"  📄 {entry.name:<40} {size:>8}  {mtime}")
                except OSError:
                    lines.append(f"  ❓ {entry.name}  (stat failed)")

            lines.append(f"\n  {len(entries)} item(s)")
            self._audit("list_dir", path, f"{len(entries)} items")
            return ToolResult.ok("\n".join(lines), "list_dir", path)

        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}", "list_dir", path)
        except Exception as e:
            return ToolResult.fail(str(e), "list_dir", path)

    # ── Search tool ───────────────────────────────────────────────────

    def grep_files(self, pattern: str, path: str = ".", include: str = "*") -> ToolResult:
        """
        Search file contents recursively for a regex pattern.
        Returns matching lines with filename and line number — like grep -rn.

        pattern: Python regex string (case-insensitive by default)
        path:    directory or file to search (default: current directory)
        include: glob pattern to filter filenames (default: all files)

        Skips hidden dirs, __pycache__, node_modules, .venv, .git automatically.
        Output capped at MAX_OUTPUT_CHARS to protect agent context windows.
        """
        import fnmatch as _fnmatch

        search_path = self._resolve_path(path)
        if search_path is None:
            return ToolResult.fail(f"Path not found: {path}", "grep_files", pattern)

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult.fail(f"Invalid regex pattern '{pattern}': {e}", "grep_files", pattern)

        SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv',
                     '.mypy_cache', '.pytest_cache', 'dist', 'build', '.ruff_cache'}

        try:
            files_to_search: List[Path] = []
            if search_path.is_file():
                files_to_search = [search_path]
            else:
                for root, dirs, files in os.walk(search_path):
                    # Prune noise dirs in-place so os.walk skips their subtrees
                    dirs[:] = [d for d in dirs
                                if d not in SKIP_DIRS and not d.startswith('.')]
                    for fname in sorted(files):
                        if _fnmatch.fnmatch(fname, include):
                            files_to_search.append(Path(root) / fname)

            matches: List[str] = []
            for fpath in sorted(files_to_search):
                try:
                    text = fpath.read_text(encoding='utf-8', errors='replace')
                    for lineno, line in enumerate(text.splitlines(), 1):
                        if compiled.search(line):
                            try:
                                rel = fpath.relative_to(search_path) if search_path.is_dir() else fpath
                            except ValueError:
                                rel = fpath
                            matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                except (PermissionError, IsADirectoryError, OSError):
                    continue

            if not matches:
                self._audit("grep_files", f"/{pattern}/ in {path}", "0 matches")
                return ToolResult.ok(
                    f"No matches found for pattern: {pattern}",
                    "grep_files", pattern
                )

            output = "\n".join(matches)
            truncated = len(output) > self.MAX_OUTPUT_CHARS
            if truncated:
                output = output[:self.MAX_OUTPUT_CHARS]
                output += f"\n[... truncated — {len(matches)} total match(es), showing first {self.MAX_OUTPUT_CHARS} chars]"

            self._audit("grep_files", f"/{pattern}/ in {path}", f"{len(matches)} match(es)")
            return ToolResult.ok(
                f"{len(matches)} match(es):\n\n{output}",
                "grep_files", pattern
            )

        except Exception as e:
            return ToolResult.fail(str(e), "grep_files", pattern)

    # ── Shell tool ────────────────────────────────────────────────────

    def run_command(
        self,
        cmd: str,
        cwd: str = ".",
        require_confirm: bool = True,
        timeout: int = 30,
    ) -> ToolResult:
        """
        Run a shell command.
        - Always shows the full command before executing.
        - Requires confirmation by default (require_confirm=True).
        - Hard timeout to prevent hangs.
        - Captures stdout + stderr.
        """
        cwd_path = Path(cwd).expanduser().resolve()

        if require_confirm:
            prompt = (
                f"🔧 run_command\n"
                f"   Command: {cmd}\n"
                f"   Working dir: {cwd_path}\n"
                f"Proceed? (y/n): "
            )
            if not self._confirm(prompt):
                self._audit("run_command", cmd, "cancelled by user")
                return ToolResult.fail("Command cancelled by user.", "run_command", cmd)

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(cwd_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            combined = (result.stdout or "") + (result.stderr or "")
            combined = combined[:self.MAX_OUTPUT_CHARS]
            if len(result.stdout or "") + len(result.stderr or "") > self.MAX_OUTPUT_CHARS:
                combined += f"\n[... output truncated at {self.MAX_OUTPUT_CHARS} chars]"

            status = f"exit {result.returncode}"
            self._audit("run_command", cmd, status)

            if result.returncode == 0:
                return ToolResult.ok(combined or "(no output)", "run_command", cmd)
            else:
                return ToolResult.fail(
                    combined or f"Command exited with code {result.returncode}",
                    "run_command", cmd
                )

        except subprocess.TimeoutExpired:
            self._audit("run_command", cmd, f"timeout after {timeout}s")
            return ToolResult.fail(f"Command timed out after {timeout}s.", "run_command", cmd)
        except FileNotFoundError:
            return ToolResult.fail(f"Working directory not found: {cwd}", "run_command", cmd)
        except Exception as e:
            return ToolResult.fail(str(e), "run_command", cmd)

    # ── Git tools ─────────────────────────────────────────────────────

    def git_status(self, repo: str = ".") -> ToolResult:
        """Run git status in the given repo directory."""
        return self._git_read("status --short --branch", repo, "git_status")

    def git_log(self, repo: str = ".", n: int = 10) -> ToolResult:
        """Show the last N commits with one-line format."""
        return self._git_read(
            f"log --oneline --graph --decorate -n {int(n)}",
            repo, "git_log"
        )

    def git_diff(self, repo: str = ".", staged: bool = False) -> ToolResult:
        """
        Show unstaged diff (default) or staged diff (staged=True).
        Caps output to avoid flooding agent context.
        """
        flag = "--cached" if staged else ""
        return self._git_read(f"diff {flag} --stat && git diff {flag}", repo, "git_diff")

    def git_commit(self, message: str, repo: str = ".", require_confirm: bool = True) -> ToolResult:
        """
        Stage all changes and create a commit.
        Requires confirmation — this is a write operation.
        """
        repo_path = Path(repo).expanduser().resolve()

        # Show what will be committed
        status_result = self._git_read("status --short", repo, "git_status")
        status_preview = status_result.output if status_result.success else "(could not read status)"

        if require_confirm:
            prompt = (
                f"📦 git_commit\n"
                f"   Repo: {repo_path}\n"
                f"   Message: {message}\n"
                f"   Changes to commit:\n"
                f"{self._indent(status_preview, '     ')}\n"
                f"Proceed? (y/n): "
            )
            if not self._confirm(prompt):
                self._audit("git_commit", f"{repo}: {message}", "cancelled by user")
                return ToolResult.fail("Commit cancelled by user.", "git_commit", message)

        # Stage all
        add_result = self._git_write("add -A", repo_path)
        if not add_result.success:
            return ToolResult.fail(f"git add failed: {add_result.error}", "git_commit", message)

        # Commit
        commit_result = self._git_write(f'commit -m "{message}"', repo_path)
        self._audit("git_commit", f"{repo}: {message}", "ok" if commit_result.success else "failed")
        return commit_result

    # ── Tool inventory ────────────────────────────────────────────────

    def tool_descriptions(self) -> str:
        """
        Return a compact tool reference for injection into agent prompts.
        This is how the model knows what tools are available.
        """
        return """
Available tools (Rain can invoke these during task execution):

  read_file(path, [start_line], [end_line])
    → Read a file's contents. Supports any text file up to 512 KB.
      Optional start_line and end_line (1-based, inclusive) read only that slice.
      Use grep_files first to find the relevant line numbers, then read just that section.
      Example: read_file ROADMAP.md 439 480
      Example: read_file rain.py 2400 2500

  write_file(path, content)
    → Create or overwrite a file. Backs up existing file automatically.
      Always asks for confirmation before writing.

  list_dir(path)
    → List files and subdirectories at path.

  grep_files(pattern, path=".", include="*")
    → Search file contents recursively for a regex pattern.
      Returns matching lines with filename and line number.
      Use this instead of read_file when searching for specific text across files.
      Example: grep_files("TODO|FIXME", ".")
      Example: grep_files("def react", "rain.py")
      Example: grep_files("import", ".", "*.py")

  run_command(cmd, cwd=".")
    → Execute a shell command. Shows command before running.
      Always asks for confirmation. Hard 30s timeout.

  git_status(repo=".")
    → Show current git status (modified, staged, untracked files).

  git_log(repo=".", n=10)
    → Show last N commits with graph decoration.

  git_diff(repo=".", staged=False)
    → Show unstaged (or staged) diff with file stats.

  git_commit(message, repo=".")
    → Stage all changes and create a commit. Requires confirmation.

To invoke a tool, use this exact format in your response:
  [TOOL: tool_name arg1 arg2]

Example:
  [TOOL: read_file src/main.py]
  [TOOL: grep_files "TODO|FIXME" .]
  [TOOL: run_command pytest tests/]
  [TOOL: git_status .]
""".strip()

    def parse_tool_calls(self, text: str) -> List[dict]:
        """
        Parse [TOOL: name arg1 arg2 ...] invocations from model output.
        Returns list of {"name": str, "args": list[str]} dicts.
        DOTALL enabled so multiline [TOOL: write_file ...] blocks parse correctly.
        """
        pattern = r'\[TOOL:\s*(\w+)(.*?)\]'
        calls = []
        for m in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            name = m.group(1).strip()
            args_raw = m.group(2).strip()
            # write_file: first whitespace-delimited token is path, rest is content.
            # _split_args would shred multiline content into useless tokens.
            if name.lower() == 'write_file':
                parts = args_raw.split(None, 1)
                args = parts if parts else []
            else:
                args = self._split_args(args_raw)
            calls.append({"name": name, "args": args, "raw": m.group(0)})
        return calls

    def dispatch(self, name: str, args: List[str], require_confirm: bool = True) -> ToolResult:
        """
        Dispatch a tool call by name with positional args.
        Used by the task execution loop to run model-requested tools.
        """
        name = name.lower()

        try:
            if name == "read_file":
                path = args[0] if args else "."
                start = int(args[1]) if len(args) > 1 else None
                end   = int(args[2]) if len(args) > 2 else None
                return self.read_file(path, start_line=start, end_line=end)

            elif name == "write_file":
                if len(args) < 2:
                    return ToolResult.fail("write_file requires path and content args.", "write_file")
                return self.write_file(args[0], args[1], require_confirm=require_confirm)

            elif name == "list_dir":
                path = args[0] if args else "."
                return self.list_dir(path)

            elif name == "run_command":
                if not args:
                    return ToolResult.fail("run_command requires a command string.", "run_command")
                cmd = args[0]
                cwd = args[1] if len(args) > 1 else "."
                return self.run_command(cmd, cwd=cwd, require_confirm=require_confirm)

            elif name == "git_status":
                repo = args[0] if args else "."
                return self.git_status(repo)

            elif name == "git_log":
                repo = args[0] if args else "."
                n = int(args[1]) if len(args) > 1 else 10
                return self.git_log(repo, n)

            elif name == "git_diff":
                repo = args[0] if args else "."
                staged = len(args) > 1 and args[1].lower() in ("staged", "true", "cached")
                return self.git_diff(repo, staged=staged)

            elif name == "git_commit":
                if not args:
                    return ToolResult.fail("git_commit requires a commit message.", "git_commit")
                message = args[0]
                repo = args[1] if len(args) > 1 else "."
                return self.git_commit(message, repo=repo, require_confirm=require_confirm)

            elif name == "grep_files":
                if not args:
                    return ToolResult.fail("grep_files requires a pattern argument.", "grep_files")
                pattern = args[0]
                path    = args[1] if len(args) > 1 else "."
                include = args[2] if len(args) > 2 else "*"
                return self.grep_files(pattern, path, include)

            else:
                return ToolResult.fail(
                    f"Unknown tool: '{name}'. Available: read_file, write_file, list_dir, "
                    f"grep_files, run_command, git_status, git_log, git_diff, git_commit",
                    name
                )

        except IndexError as e:
            return ToolResult.fail(f"Missing argument for {name}: {e}", name)
        except Exception as e:
            return ToolResult.fail(f"Tool error in {name}: {e}", name)

    # ── Audit log ─────────────────────────────────────────────────────

    def get_audit_log(self, last_n: int = 50) -> str:
        """Return the last N lines of the audit log as a string."""
        if not self.AUDIT_LOG.exists():
            return "(no audit log yet)"
        try:
            lines = self.AUDIT_LOG.read_text(encoding='utf-8').splitlines()
            return "\n".join(lines[-last_n:])
        except Exception:
            return "(could not read audit log)"

    # ── Private helpers ───────────────────────────────────────────────

    def _audit(self, tool: str, args: str, result: str):
        """Append a line to the audit log."""
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {tool:<15} {args[:80]:<80}  →  {result}\n"
            with open(self.AUDIT_LOG, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception:
            pass  # Audit failure must never crash Rain

    def _git_read(self, git_args: str, repo: str, label: str) -> ToolResult:
        """Run a read-only git command."""
        repo_path = Path(repo).expanduser().resolve()
        try:
            result = subprocess.run(
                f"git {git_args}",
                shell=True,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=15,
            )
            output = (result.stdout or "") + (result.stderr or "")
            output = output[:self.MAX_OUTPUT_CHARS]
            self._audit(label, repo, "ok" if result.returncode == 0 else f"exit {result.returncode}")
            if result.returncode == 0:
                return ToolResult.ok(output or "(no output)", label, repo)
            else:
                return ToolResult.fail(output or f"git exited with {result.returncode}", label, repo)
        except subprocess.TimeoutExpired:
            return ToolResult.fail("git command timed out.", label, repo)
        except Exception as e:
            return ToolResult.fail(str(e), label, repo)

    def _git_write(self, git_args: str, repo_path: Path) -> ToolResult:
        """Run a write git command (no separate confirm — caller handles that)."""
        try:
            result = subprocess.run(
                f"git {git_args}",
                shell=True,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode == 0:
                return ToolResult.ok(output or "(ok)", "git", git_args)
            else:
                return ToolResult.fail(output or f"exit {result.returncode}", "git", git_args)
        except Exception as e:
            return ToolResult.fail(str(e), "git", git_args)

    @staticmethod
    def _resolve_path(path: str) -> Optional[Path]:
        """Resolve and validate a path. Returns None if it doesn't exist."""
        try:
            p = Path(path).expanduser().resolve()
            if p.exists():
                return p
            return None
        except Exception:
            return None

    @staticmethod
    def _auto_confirm(prompt: str) -> bool:
        """Default confirm_fn — auto-approves (for non-interactive use)."""
        return True

    @staticmethod
    def _human_size(n: int) -> str:
        for unit in ('B', 'KB', 'MB', 'GB'):
            if n < 1024:
                return f"{n:.0f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

    @staticmethod
    def _indent(text: str, prefix: str) -> str:
        return "\n".join(prefix + line for line in text.splitlines())

    @staticmethod
    def _split_args(raw: str) -> List[str]:
        """
        Split a raw argument string respecting both single- and double-quoted tokens.
        Each quote type toggles its own flag; the other quote type is preserved as a
        literal character inside. Quote characters themselves are stripped from output.

        Examples:
          'foo "bar baz" qux'          -> ['foo', 'bar baz', 'qux']
          '\'"temperature"|temp\' .'   -> ['"temperature"|temp', '.']
          '"it\'s fine" arg2'          -> ["it's fine", 'arg2']
        """
        args = []
        current = []
        in_double = False
        in_single = False
        for ch in raw:
            if ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "'" and not in_double:
                in_single = not in_single
            elif ch == ' ' and not in_double and not in_single:
                if current:
                    args.append(''.join(current))
                    current = []
            else:
                current.append(ch)
        if current:
            args.append(''.join(current))
        return [a for a in args if a]


# ── Interactive confirm helper (used by CLI) ──────────────────────────

def interactive_confirm(prompt: str) -> bool:
    """
    Standard confirm_fn for interactive CLI mode.
    Shows the prompt and reads y/n from stdin.
    """
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        answer = sys.stdin.readline().strip().lower()
        return answer in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        print()
        return False


# ── CLI (for testing) ─────────────────────────────────────────────────

if __name__ == '__main__':
    tools = ToolRegistry(confirm_fn=interactive_confirm)

    if len(sys.argv) < 2:
        print("Usage: python3 tools.py <tool> [args...]")
        print("Tools: read_file, list_dir, run_command, git_status, git_log, git_diff, audit")
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "audit":
        print(tools.get_audit_log())
    else:
        result = tools.dispatch(cmd, args, require_confirm=True)
        if result.success:
            print(result.output)
        else:
            print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(0 if result.success else 1)
