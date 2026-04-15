#!/usr/bin/env python3
"""
Rain ⛈️ - Sovereign AI Orchestrator (CLI Entry Point)

The brain of the Rain ecosystem that manages recursive reflection
and multi-agent AI interactions completely offline.

"Be like rain - essential, unstoppable, and free."

All core logic lives in the rain/ package:
  rain.memory       — RainMemory (6-tier persistent memory)
  rain.agents       — AgentType, Agent, prompts, model configuration
  rain.router       — AgentRouter (keyword-based query routing)
  rain.sandbox      — CodeSandbox, SandboxResult, ReflectionResult
  rain.orchestrator — MultiAgentOrchestrator (the core pipeline)
"""

import json
import re
import sys
import argparse
import signal
from pathlib import Path
from datetime import datetime

# ── Core imports from the rain package ────────────────────────────────
from rain import (
    MultiAgentOrchestrator,
    RainMemory,
    AgentType,
    auto_pick_default_model,
    _SKILLS_AVAILABLE,
    _TOOLS_AVAILABLE,
)

# ── Optional external modules (graceful degradation) ──────────────────
try:
    from skills import SkillLoader, install_skill as _install_skill
except ImportError:
    pass  # _SKILLS_AVAILABLE from rain package handles the flag

try:
    from tools import ToolRegistry, interactive_confirm as _interactive_confirm
except ImportError:
    pass  # _TOOLS_AVAILABLE from rain package handles the flag

try:
    from indexer import ProjectIndexer
    _INDEXER_AVAILABLE = True
except ImportError:
    _INDEXER_AVAILABLE = False


# ── CLI helper functions ──────────────────────────────────────────────

def get_multiline_input() -> str:
    """
    Collect multi-line input from the user.
    - Single line: submit immediately on Enter
    - Code detected: enters multi-line mode, press Ctrl+D to submit
    """
    while True:
        try:
            print("\U0001f4ac Ask Rain (Ctrl+D to submit code blocks, Ctrl+C to cancel):")
            sys.stdout.write("  > ")
            sys.stdout.flush()
            first_line = sys.stdin.readline()

            if first_line == "":
                return ""

            first_line = first_line.rstrip("\n").rstrip("\r")

            if first_line.strip():
                break

        except KeyboardInterrupt:
            raise

    single_line_triggers = ['quit', 'exit', 'q', 'clear', 'history']
    if first_line.strip().lower() in single_line_triggers:
        return first_line.strip()

    code_starters = (
        '#!/', 'def ', 'class ', 'import ', 'from ', 'function ',
        'const ', 'let ', 'var ', 'public ', 'private ', '```',
        'async ', 'await ', '@', '#include', 'package '
    )
    is_multiline = any(first_line.strip().startswith(s) for s in code_starters)

    if not is_multiline:
        return first_line.strip()

    print("  (Code detected - paste your code, then press Ctrl+D to submit)")
    lines = [first_line]

    while True:
        try:
            line = sys.stdin.readline()
            if line == "":
                break
            lines.append(line.rstrip("\n").rstrip("\r"))
        except KeyboardInterrupt:
            print("\n  (cancelled)")
            return ""

    while lines and lines[-1].strip() == "":
        lines.pop()

    return "\n".join(lines)


# ── GitHub + Live Data CLI helpers ────────────────────────────────────

_CLI_GITHUB_KEYWORDS = frozenset({
    'github', 'repo', 'repository', 'open issues', 'recent commits',
    'pull request', 'pull requests', 'stars', 'forks', 'contributors',
    'github.com', 'readme', 'releases', 'latest release',
})

_CLI_GITHUB_REPO_RE = None  # compiled lazily


def _cli_extract_github_repo(query: str) -> str | None:
    """Extract owner/repo slug from a query string (CLI mirror of server.py)."""
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

    lines = [f"[GITHUB DATA \u2014 fetched just now for {slug}]"]
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
            f"  Stars: {repo.get('stargazers_count', '?'):,}  \u00b7  Forks: {repo.get('forks_count', '?'):,}\n"
            f"  Open issues: {repo.get('open_issues_count', '?'):,}\n"
            f"  Default branch: {repo.get('default_branch', '?')}\n"
            f"  Created: {repo.get('created_at', '?')[:10]}  \u00b7  Updated: {repo.get('updated_at', '?')[:10]}\n"
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
            suffix = f" \u2014 {name}" if name and name != tag else ""
            lines.append(f"Latest release: {tag}{suffix} ({date})")
        except Exception:
            pass

    return "\n\n".join(lines) if len(lines) > 1 else ""


def _cli_fetch_live_data(query: str) -> str:
    """
    Phase 7B/7C: Fetch live data from public APIs (no API keys) for the CLI.
    Mirrors _fetch_live_data() in server.py.
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

    github_block = _cli_fetch_github_data(query)

    if not (want_fees or want_price) and not github_block:
        return ""

    sources = []
    if want_fees or want_price:
        sources.append("mempool.space")
    if github_block:
        sources.append("GitHub API")
    lines = [f"[LIVE DATA \u2014 fetched just now from {' + '.join(sources)}]"]

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
            lines.append(f"Fee rate lookup failed: {e} \u2014 will answer from training data.")

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
            lines.append(f"Price lookup failed: {e} \u2014 will answer from training data.")

    if github_block:
        lines.append(github_block)

    return "\n\n".join(lines) if len(lines) > 1 else ""


def _cli_duckduckgo_search(query: str, max_results: int = 5) -> list:
    """
    DuckDuckGo search for the CLI (mirrors the one in server.py).
    No API key. Zero new dependencies \u2014 pure stdlib urllib.
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


def _handle_todo_mutation(query: str):
    """
    Intercept to-do add/remove mutations before the LLM runs.
    Uses numbered list format (1. Item) with auto-renumber on remove.

    Handles:
    - Ordinal references: "task 1", "task number three", "#2"
    - "remove all except X" / "keep only X"
    - Compound requests: remove + add in one message
    - "let's add X" / "also add X" (no explicit to-do keyword needed)
    - File-path queries: "where is the file", "save to disk"

    Returns (handled: bool, response: str).
    Short-circuits the LLM — no hallucination possible.
    """
    import os as _os, re as _re, difflib as _difflib

    q = query.lower().strip()

    # Locate Erics-to-do.md from the repo root (needed early for path queries)
    root = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(6):
        if _os.path.exists(_os.path.join(root, '.git')):
            break
        root = _os.path.dirname(root)
    todo_path = _os.path.join(root, 'Erics-to-do.md')

    # ── File-path / "save to disk" read queries ──────────────────────────
    is_path_q = bool(
        _re.search(r'\b(file\s*path|save.*disk|verify.*file|access.*file|explicit\w*\s+save)\b', q)
        and _re.search(r'\b(to[\s-]?do|todo|list|it)\b', q)
    )
    if is_path_q:
        if _os.path.isfile(todo_path):
            with open(todo_path, 'r') as f:
                contents = f.read().strip()
            return True, f"The to-do list is saved at:\n\n`{todo_path}`\n\nCurrent contents:\n\n{contents}"
        else:
            return True, f"The to-do list will be saved at:\n\n`{todo_path}`\n\nThe file does not exist yet — tell me what to put on the list and I'll create it."

    # is_remove catches both explicit removes and "mark done / complete" phrasing
    is_remove = bool(_re.search(
        r'\b(remove|removed|delete|cross.?off|done with|completed?|finished?|check.?off|mark.*done|are\s+done|can\s+be\s+removed|the\s+rest)\b', q))
    # Detect "remove all except X" / "keep only X" — implies is_remove
    remove_except_m = _re.search(
        r'\b(?:remove|delete)\s+(?:all|everything)(?:\s+\w+){0,4}\s+except\s+(?:for\s+)?["\']?(.+?)["\']?\s*(?:[,.]|$)',
        query, _re.IGNORECASE,
    ) or _re.search(
        r'\bkeep\s+only\s+["\']?(.+?)["\']?\s*(?:[,.]|$)',
        query, _re.IGNORECASE,
    )
    if remove_except_m:
        is_remove = True
    # "keep first item, remove the rest" — keep item 1, remove everything else
    keep_first_rest_m = (
        _re.search(r"\bhaven.t\s+done.{0,50}\bfirst\b|\bfirst\s+item.{0,40}\bkeep\b|\bkeep\s+that\b", q, _re.IGNORECASE)
        and _re.search(r'\bthe\s+rest\b', q, _re.IGNORECASE)
    )
    if keep_first_rest_m:
        is_remove = True
    # "to the list", "to my list", "new task/tasks", "new item", "let's add X", or compound remove+add
    is_add = bool(
        _re.search(r'\b(add|new item|put)\b', q)
        and (
            _re.search(r'\b(to[- ]?do|to\s+(?:the|my)\s+list|tasks?|new\s+item)\b', q)
            or is_remove
            or _re.search(r"(?:let'?s\s+add|also\s+add|and\s+add|please\s+add)", q)
        )
    )

    if not (is_remove or is_add):
        return False, ""

    if not _os.path.exists(todo_path):
        if is_add and not is_remove:
            # Create the file so the add can proceed
            open(todo_path, 'w').close()
            lines: list = []
        else:
            return False, ""
    else:
        with open(todo_path, 'r') as f:
            lines = f.readlines()

    # Numbered list items: "1. Item text"
    item_texts = [_re.sub(r'^\d+\.\s+', '', l).strip()
                  for l in lines if _re.match(r'^\d+\.\s+', l.strip())]

    _ORDINALS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        **{str(n): n for n in range(1, 11)},
    }

    def _resolve_positions(q_text):
        positions = set()
        for m in _re.finditer(
            r'\btask\s+(?:number\s+)?(?:#)?(\d+|\w+)\b'
            r'|\bnumber\s+(\d+|\w+)\b'
            r'|\b#(\d+)\b',
            q_text, _re.IGNORECASE,
        ):
            raw = next(g for g in m.groups() if g is not None).lower()
            if raw in _ORDINALS:
                positions.add(_ORDINALS[raw])
        return sorted(positions)

    def _renumber(lines_list):
        """Re-number all numbered list items sequentially starting at 1."""
        n, result = 1, []
        for l in lines_list:
            if _re.match(r'^\d+\.\s+', l.strip()):
                result.append(_re.sub(r'^\d+\.', f'{n}.', l.strip()) + '\n')
                n += 1
            else:
                result.append(l)
        return result

    action_descs = []

    # ── "Remove all except X" / "keep only X" ────────────────────────────
    if remove_except_m and is_remove and item_texts:
        keep_text = remove_except_m.group(1).strip().strip("'\".,;")
        keep_lower = keep_text.lower()
        fuzzy_keep = _difflib.get_close_matches(keep_lower, [t.lower() for t in item_texts], n=1, cutoff=0.3)
        if fuzzy_keep:
            kept_item = item_texts[[t.lower() for t in item_texts].index(fuzzy_keep[0])]
        else:
            keep_words = [w for w in keep_lower.split() if len(w) > 3]
            kept_item = next((t for t in item_texts if any(w in t.lower() for w in keep_words)), None)
        if kept_item:
            removed_items = [t for t in item_texts if t != kept_item]
            lines = _renumber([l for l in lines if not any(t in l for t in removed_items)])
            item_texts = [kept_item]
            if removed_items:
                action_descs.append(f"Removed {len(removed_items)} task(s), kept \"{kept_item}\"")
            is_remove = False  # handled; skip normal remove logic

    # ── "Keep first item, remove the rest" ───────────────────────────────
    if keep_first_rest_m and is_remove and item_texts and len(item_texts) > 1:
        kept_item = item_texts[0]
        removed_items = item_texts[1:]
        lines = _renumber([l for l in lines if not any(t in l for t in removed_items)])
        item_texts = [kept_item]
        action_descs.append(f"Kept \"{kept_item}\", removed {len(removed_items)} completed task(s)")
        is_remove = False  # handled

    # ── Remove ────────────────────────────────────────────────────────────
    if is_remove and item_texts:
        positions = _resolve_positions(q)
        matched = []
        if positions:
            for pos in positions:
                if 1 <= pos <= len(item_texts):
                    matched.append(item_texts[pos - 1])
        else:
            q_clean = _re.sub(
                r'\b(remove|delete|cross off|done with|completed?|finished?|check off|'
                r'mark(?:\s+as)?\s+done|from|off|my|the|to[\s-]?do|list|task(?:s)?|number|it)\b',
                '', q, flags=_re.IGNORECASE,
            ).strip()
            matches = _difflib.get_close_matches(q_clean, item_texts, n=1, cutoff=0.25)
            if not matches:
                matches = [t for t in item_texts
                           if any(w in t.lower() for w in q_clean.split() if len(w) > 3)][:1]
            matched = matches

        if matched:
            lines = _renumber([l for l in lines if not any(t in l for t in matched)])
            for t in matched:
                action_descs.append(f"Removed \"{t}\"")

    # ── Add ───────────────────────────────────────────────────────────────
    if is_add:
        item = None
        m = _re.search(
            r'(?:to\s+(?:the|my)\s+list|new\s+task|to[\s-]?do)\s*:\s*(.+?)(?:\s*$|\s*\.)',
            query, _re.IGNORECASE,
        )
        if m:
            item = m.group(1).strip()
        if not item:
            m = _re.search(r'\badd\s+(.+?)\s+to\b', query, _re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if not _re.match(r'^(?:a\s+)?(?:new\s+)?(?:task|item)$', candidate, _re.IGNORECASE):
                    item = candidate
        if not item:
            # "let's add X", "also add X", bare "add 'X'" at end of message
            m = _re.search(
                r"(?:let'?s\s+|also\s+|and\s+|please\s+)?add\s+['\"]?(.+?)['\"]?\s*[,.]?\s*$",
                query, _re.IGNORECASE,
            )
            if m:
                candidate = m.group(1).strip().strip("'\"")
                if candidate and not _re.match(r'^(?:a\s+)?(?:new\s+)?(?:task|item)$', candidate, _re.IGNORECASE):
                    item = candidate
        if item:
            item = item.rstrip('.,;!').capitalize()
            num = sum(1 for l in lines if _re.match(r'^\d+\.\s+', l.strip())) + 1
            lines.append(f'{num}. {item}\n')
            action_descs.append(f"Added \"{item}\"")

    if not action_descs:
        return False, ""

    with open(todo_path, 'w') as f:
        f.writelines(lines)

    updated = ''.join(lines).strip()
    summary = ' · '.join(action_descs)
    return True, f"✅ {summary}.\n\nUpdated to-do list:\n\n{updated}"


def _inject_file_context(query: str) -> str:
    """
    Detect queries about to-do lists or named markdown files and inject the
    actual file contents so the model reads real data instead of hallucinating.
    """
    import os as _os
    import re as _re

    todo_patterns = [
        r"\bto[\s-]?do\b",
        r"\btodo\b",
        r"erics[\s-]to[\s-]do",
        r"eric'?s to.?do",
    ]
    md_file_pattern = r"([\w\-]+\.md)\b"

    lower = query.lower()
    candidate_paths: list[str] = []

    if any(_re.search(p, lower) for p in todo_patterns):
        candidate_paths.append("Erics-to-do.md")

    for match in _re.finditer(md_file_pattern, query, _re.IGNORECASE):
        fname = match.group(1)
        if fname not in candidate_paths:
            candidate_paths.append(fname)

    if not candidate_paths:
        return query

    # Resolve the Rain repo root. When running from a worktree the __file__
    # path includes /.claude/worktrees/<name>/ — walk up to the git root.
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    repo_root = script_dir
    for _ in range(5):
        if _os.path.isdir(_os.path.join(repo_root, ".git")):
            break
        parent = _os.path.dirname(repo_root)
        if parent == repo_root:
            break
        repo_root = parent

    search_dirs = [
        repo_root,
        script_dir,
        _os.path.join(repo_root, "data"),
        _os.path.join(repo_root, "memory"),
        _os.path.join(repo_root, "context"),
        _os.path.expanduser("~/.rain"),
    ]

    injected_blocks: list[str] = []
    for fname in candidate_paths:
        for directory in search_dirs:
            fpath = _os.path.join(directory, fname)
            if _os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        contents = fh.read().strip()
                    injected_blocks.append(
                        f"[File: {fname}]\n{contents}\n[End of {fname}]"
                    )
                    print(f"\U0001f4cb Injected file context from: {fname}")
                except Exception:
                    pass
                break

    if injected_blocks:
        block = "\n\n".join(injected_blocks)
        return f"{block}\n\n---\n\n{query}"
    return query


def _inject_project_context(query: str, project_path: str) -> str:
    """
    Search the project index for chunks relevant to `query` and prepend them.
    Returns the augmented query, or the original query if nothing useful is found.
    """
    if not _INDEXER_AVAILABLE:
        print("\u26a0\ufe0f  indexer.py not found \u2014 --project flag has no effect")
        return query
    try:
        idx = ProjectIndexer()
        context_block = idx.build_context_block(query, project_path, top_k=4)
        if context_block:
            print(f"\U0001f4c2 Project context injected from: {project_path.split('/')[-1]}")
            return f"{context_block}\n\n---\n\n{query}"
        else:
            print(f"\U0001f4c2 No relevant chunks found in index for: {project_path.split('/')[-1]}")
            print(f"   (Run: python3 indexer.py --index {project_path}  to index it first)")
            return query
    except Exception as e:
        print(f"\U0001f4c2 Project index error: {e}")
        return query


# ── Main CLI ──────────────────────────────────────────────────────────

def main():
    """Main CLI interface for Rain"""
    parser = argparse.ArgumentParser(description="Rain \u26c8\ufe0f - Sovereign AI with Recursive Reflection")
    parser.add_argument("query", nargs="?", help="Your question or prompt")
    parser.add_argument("--model", default=None, help="Model to use (default: auto-detect best available)")
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
    parser.add_argument("--test-mode", action="store_true",
                        help="Test mode \u2014 feedback and gap writes suppressed; calibration read-only. "
                             "Use when running diagnostic prompts to prevent poisoning the calibration table.")
    parser.add_argument("--forget", action="store_true", help="Wipe all stored memory and exit")
    parser.add_argument("--memories", action="store_true", help="Show stored session history and exit")
    parser.add_argument("--sandbox", "-s", action="store_true",
                        help="Enable code execution sandbox \u2014 Rain runs and verifies code before returning it")
    parser.add_argument("--sandbox-timeout", type=int, default=10,
                        help="Max seconds a sandboxed program may run (default: 10)")
    parser.add_argument("--agents", action="store_true",
                        help="Show the agent roster and which models are assigned")
    # Phase 6: Skills and task mode
    parser.add_argument("--skills", action="store_true",
                        help="List all installed skills and exit")
    parser.add_argument("--install-skill", metavar="SLUG",
                        help="Install a skill from ClawHub by slug (requires Node.js / npx)")
    parser.add_argument("--task", "-t", action="store_true",
                        help="Task mode \u2014 decompose goal into a plan and execute it step by step")
    parser.add_argument("--react", "-r", action="store_true",
                        help="ReAct mode \u2014 iterative Reason+Act loop: model calls tools and observes results until it has a Final Answer")
    parser.add_argument("--web-search", "-w", action="store_true",
                        help="Augment query with live DuckDuckGo results before sending to agents (Phase 7)")
    parser.add_argument("--project", "-p", metavar="PATH",
                        help="Path to a project directory \u2014 injects relevant source code chunks into every query (Phase 7)")
    parser.add_argument("--meta", action="store_true",
                        help="Phase 11: run metacognition agent \u2014 generate a self-assessment report and exit")
    parser.add_argument("--quiet", "-Q", action="store_true",
                        help="Suppress startup banner and all informational prints \u2014 output the response only")

    args = parser.parse_args()

    # Print Rain banner
    if not args.quiet:
        print("""
    \u26c8\ufe0f  RAIN - Sovereign AI Ecosystem  \u26c8\ufe0f

    "Be like rain - essential, unstoppable, and free."

    \U0001f916 Multi-agent routing enabled
    \U0001f327\ufe0f  Recursive reflection enabled
    \U0001f512 Completely offline and private
    \u26a1 Your AI, your rules, your future
    """)

    try:
        # Load system prompt
        system_prompt = None
        if args.system_file:
            try:
                with open(args.system_file, 'r') as f:
                    system_prompt = f.read().strip()
                print(f"\U0001f4dd Loaded system prompt from: {args.system_file}")
            except FileNotFoundError:
                print(f"\u274c System prompt file not found: {args.system_file}")
                sys.exit(1)
        elif args.system_prompt:
            system_prompt = args.system_prompt
            print(f"\U0001f4dd Using custom system prompt")

        # Phase 6: --skills flag \u2014 list installed skills and exit
        if args.skills:
            if _SKILLS_AVAILABLE:
                loader = SkillLoader()
                loader.load()
                print(loader.summary_table())
            else:
                print("\u274c Skills module not available. Ensure skills.py is in the Rain directory.")
            return

        # Phase 6: --install-skill flag \u2014 install from ClawHub and exit
        if args.install_skill:
            if _SKILLS_AVAILABLE:
                print(f"\U0001f4e6 Installing skill: {args.install_skill}")
                ok, msg = _install_skill(args.install_skill)
                print(("\u2705 " if ok else "\u274c ") + msg)
                if ok:
                    print(f"\n\U0001f4a1 Skill installed. Restart Rain to load it, or run:  python3 rain.py --skills")
            else:
                print("\u274c Skills module not available. Ensure skills.py is in the Rain directory.")
            return

        # Handle --forget flag
        if args.forget:
            m = RainMemory()
            m.forget_all()
            print("\U0001f5d1\ufe0f  All memory wiped. Rain starts fresh.")
            return

        # Resolve model \u2014 auto-detect if not explicitly provided
        if args.model:
            resolved_model = args.model
        else:
            resolved_model = auto_pick_default_model()
            if not args.quiet:
                print(f"\U0001f50d Auto-detected model: {resolved_model}")

        # Initialize memory unless disabled
        memory = None
        if not args.no_memory:
            memory = RainMemory(test_mode=getattr(args, 'test_mode', False))
            if memory.test_mode and not args.quiet:
                print("\U0001f9ea TEST MODE \u2014 feedback disabled, calibration read-only", flush=True)
            memory.start_session(model=resolved_model)

        # Load RAIN.md self-knowledge document (same as server.py does at startup)
        _rain_md = ""
        _rain_md_path = Path(__file__).parent / "RAIN.md"
        try:
            _rain_md = _rain_md_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

        # Initialize Rain \u2014 multi-agent (always)
        rain = MultiAgentOrchestrator(
            default_model=resolved_model,
            max_iterations=args.iterations,
            confidence_threshold=args.confidence,
            system_prompt=system_prompt,
            memory=memory,
            sandbox_enabled=args.sandbox,
            sandbox_timeout=args.sandbox_timeout,
            quiet=args.quiet,
            rain_md=_rain_md,
        )
        if not args.quiet:
            print(f"\u2705 Rain initialized (multi-agent mode) \u00b7 default model: {resolved_model}")
        # Propagate --project path onto the orchestrator so _build_memory_context
        # can proactively query the knowledge graph for structural context.
        if args.project:
            rain.project_path = args.project

        if args.sandbox and not args.quiet:
            print(f"\U0001f52c Sandbox enabled \u2014 code will be executed and verified (timeout: {args.sandbox_timeout}s)")

        # --agents flag \u2014 just show roster and exit
        if args.agents:
            rain.print_agent_roster()
            return

        # --meta flag \u2014 Phase 11 metacognition: generate self-assessment report and exit
        if args.meta:
            if not memory:
                print("\u274c Memory required for self-assessment. Remove --no-memory flag.")
                return
            print("\n\U0001f9e0 Running self-assessment...\n")
            def _query_model(prompt: str) -> str:
                import urllib.request as _ur, json as _j
                payload = _j.dumps({"model": "llama3.2", "prompt": prompt, "stream": False}).encode()
                req = _ur.Request("http://localhost:11434/api/generate", data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")
                with _ur.urlopen(req, timeout=60) as r:
                    return _j.loads(r.read())["response"].strip()
            report = memory.generate_meta_report(_query_model)
            print(report)
            print()
            return

        # --memories flag - show session history + knowledge gaps
        if args.memories:
            if memory:
                sessions = memory.get_recent_sessions(limit=10)
                if sessions:
                    print(f"\n\U0001f4da Memory: {memory.db_path}")
                    print(f"   {memory.total_sessions()} sessions stored\n")
                    for s in sessions:
                        date = datetime.fromisoformat(s["started_at"]).strftime("%b %d %Y %H:%M")
                        print(f"  [{date}] \u00b7 {s['message_count']} messages \u00b7 model: {s['model']}")
                        if s.get("summary"):
                            print(f"  \U0001f4ad {s['summary']}")
                        print()
                else:
                    print("\n\U0001f4da No sessions in memory yet.")

                # Phase 11: show knowledge gaps
                try:
                    import sqlite3 as _sq
                    with _sq.connect(memory.db_path) as _conn:
                        gaps = _conn.execute(
                            """SELECT query, confidence, rating, timestamp FROM knowledge_gaps
                               WHERE resolved = 0 ORDER BY timestamp DESC LIMIT 10"""
                        ).fetchall()
                    if gaps:
                        print("\n\U0001f9e0 Knowledge gaps (topics Rain struggled with):")
                        for g in gaps:
                            date = datetime.fromisoformat(g[3]).strftime("%b %d")
                            print(f"  [{date}] conf {g[1]:.0%} · {g[2]} · {g[0][:80]}")
                        print()
                except Exception:
                    pass
            return

        # Show startup greeting if memory exists
        if memory and not args.quiet:
            greeting = memory.get_startup_greeting()
            if greeting:
                print(f"\n\U0001f9e0 Rain remembers:\n{greeting}\n")
            else:
                rain.print_agent_roster()
                print(f"\n\U0001f9e0 Memory enabled \u00b7 {memory.db_path}\n")

        # Show history if requested
        if args.history:
            history = rain.get_history()
            if history:
                print("\n\U0001f4da Reflection History:")
                for i, result in enumerate(history, 1):
                    print(f"{i}. [{result.timestamp.strftime('%H:%M:%S')}] "
                          f"Confidence: {result.confidence:.2f}, "
                          f"Iterations: {result.iteration}")
                    print(f"   {result.content[:100]}...")
            else:
                print("\n\U0001f4da No history yet")
            return

        # --file mode - read file and analyze it
        if args.file:
            try:
                with open(args.file, 'r') as f:
                    file_content = f.read()
                lines = len(file_content.splitlines())
                print(f"\U0001f4c2 Loaded file: {args.file} ({lines} lines)")

                if args.query:
                    print(f"\U0001f3af Query: {args.query}")
                    prompt = f"{args.query}\n\nFile: {args.file}\n\n{file_content}"
                else:
                    print(f"\U0001f50d No query provided - performing general analysis")
                    prompt = file_content

                result = rain.recursive_reflect(prompt, verbose=args.verbose)
                if result:
                    print(f"\n\U0001f31f Final Answer (confidence: {result.confidence:.2f}, "
                          f"{result.iteration} iterations, {result.duration_seconds:.1f}s):")
                    if result.sandbox_results:
                        verified_count = sum(1 for r in result.sandbox_results if r.success)
                        total_count = len(result.sandbox_results)
                        status = "\u2705 all blocks verified" if verified_count == total_count else f"\u26a0\ufe0f  {verified_count}/{total_count} blocks verified"
                        print(f"\U0001f52c Sandbox: {status} ({total_count} block{'s' if total_count != 1 else ''} tested)")
                    print(result.content)
            except FileNotFoundError:
                print(f"\u274c File not found: {args.file}")
                sys.exit(1)
            except KeyboardInterrupt:
                rain._kill_current_process()
                print("\n\n\u26a1 Interrupted!")
            return

        # Interactive mode
        if args.interactive:
            print("\n\U0001f327\ufe0f  Rain Interactive Mode - Type 'quit' to exit, Ctrl+C to interrupt a response, Ctrl+D to submit code")
            while True:
                try:
                    print()
                    query = get_multiline_input()

                    if not query:
                        continue

                    if query.lower() in ['quit', 'exit', 'q']:
                        print("\n\U0001f44b Goodbye!")
                        break

                    if query.lower() == 'clear':
                        rain.clear_history()
                        print("\U0001f5d1\ufe0f  History cleared")
                        continue

                    if query.lower() == 'history':
                        history = rain.get_history()
                        if history:
                            print("\n\U0001f4da Reflection History:")
                            for i, r in enumerate(history, 1):
                                print(f"  {i}. [{r.timestamp.strftime('%H:%M:%S')}] "
                                      f"confidence: {r.confidence:.2f}, "
                                      f"iterations: {r.iteration}, "
                                      f"{r.duration_seconds:.1f}s")
                        else:
                            print("\U0001f4da No history yet")
                        continue

                    _todo_handled, _todo_resp = _handle_todo_mutation(query)
                    if _todo_handled:
                        print(_todo_resp)
                        continue

                    augmented_query = _inject_file_context(query)
                    augmented_query = _inject_project_context(augmented_query, args.project) if args.project else augmented_query
                    result = rain.recursive_reflect(augmented_query, verbose=args.verbose)
                    if result:
                        print(f"\n\U0001f31f Final Answer (confidence: {result.confidence:.2f}, "
                              f"{result.iteration} iterations, {result.duration_seconds:.1f}s):")
                        if result.sandbox_results:
                            verified_count = sum(1 for r in result.sandbox_results if r.success)
                            total_count = len(result.sandbox_results)
                            status = "\u2705 all blocks verified" if verified_count == total_count else f"\u26a0\ufe0f  {verified_count}/{total_count} blocks verified"
                            print(f"\U0001f52c Sandbox: {status} ({total_count} block{'s' if total_count != 1 else ''} tested)")
                        print(result.content)

                except KeyboardInterrupt:
                    rain._kill_current_process()
                    print("\n\n\u26a1 Interrupted! Type 'quit' to exit or ask another question.")
                    continue

            # End session with a summary when user quits
            if memory:
                print("\U0001f4ad Saving session to memory...")
                summary = memory.generate_summary()
                memory.end_session()
                if summary:
                    memory.update_summary(summary)
                try:
                    facts = memory.extract_session_facts()
                    if facts:
                        memory.save_session_facts(facts)
                        print(f"\U0001f9e0 {len(facts)} fact(s) learned and stored to memory")
                except Exception:
                    pass

        # Single query mode (with optional --task decomposition)
        elif args.query:
            # Wire interactive_confirm into the tool registry so write_file /
            # run_command prompt the user before executing in CLI task mode.
            if _TOOLS_AVAILABLE and rain.tools:
                rain.tools._confirm = _interactive_confirm

            # Auto-detect mode when neither --react nor --task is explicit
            _use_react = args.react or (
                not args.task
                and rain._auto_mode(args.query) == 'react'
            )
            if _use_react:
                if not args.react and not args.quiet:
                    print(f"\u26a1 Auto-selected ReAct mode ({rain.router.explain_mode('react')})")
            if _use_react:
                result = rain.react_loop(args.query, verbose=args.verbose)
                if result:
                    if not args.quiet:
                        print(f"\n\U0001f31f Final Answer ({result.iteration} step(s) \u00b7 {result.duration_seconds:.1f}s):")
                    print(result.content)
            elif args.task:
                result = rain.execute_task(
                    args.query,
                    verbose=args.verbose,
                    confirm_fn=_interactive_confirm if _TOOLS_AVAILABLE else None,
                )
                if result:
                    if not args.quiet:
                        print(f"\n\u2705 Task complete \u00b7 {result.iteration} step(s) \u00b7 {result.duration_seconds:.1f}s")
                    print(result.content)
            else:
                query = args.query

                _todo_handled, _todo_resp = _handle_todo_mutation(query)
                if _todo_handled:
                    print(_todo_resp)
                    return

                if args.web_search:
                    if not args.quiet:
                        print("\U0001f310 Searching the web...")

                    live_block = _cli_fetch_live_data(query)
                    if live_block and not args.quiet:
                        if "GITHUB DATA" in live_block and "mempool" not in live_block.lower():
                            print("\u26a1 Live data retrieved from GitHub API")
                        elif "GITHUB DATA" in live_block:
                            print("\u26a1 Live data retrieved from mempool.space + GitHub API")
                        else:
                            print("\u26a1 Live data retrieved from mempool.space")

                    search_results = _cli_duckduckgo_search(query)
                    if search_results and not args.quiet:
                        print(f"\U0001f310 {len(search_results)} result(s) retrieved \u2014 routing to Search Agent")

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
                            f"Cite sources where relevant. The LIVE DATA block contains real-time numbers \u2014 "
                            f"use those figures directly rather than saying you don't know the current value.\n\n"
                            f"Question: {args.query}"
                        )
                    else:
                        if not args.quiet:
                            print("\U0001f310 No results found \u2014 using local knowledge")
                query = _inject_file_context(query)
                if args.project:
                    query = _inject_project_context(query, args.project)

                # Auto-detect react vs reflect
                _mode = (
                    rain._auto_mode(query)
                    if not args.react and not args.task
                    else ('react' if args.react else 'reflect')
                )
                if _mode == 'react':
                    if not args.quiet:
                        print(f"\u26a1 Auto-selected ReAct mode ({rain.router.explain_mode('react')})")
                    result = rain.react_loop(query, verbose=args.verbose)
                else:
                    result = rain.recursive_reflect(query, verbose=args.verbose)
                if not result:
                    print("\u274c No response \u2014 the model may have timed out. Check that Ollama is running and try again.")
                    return
                if not args.quiet:
                    print(f"\n\U0001f31f Final Answer (confidence: {result.confidence:.2f}, {result.iteration} iterations, {result.duration_seconds:.1f}s):")
                    if result.sandbox_results:
                        verified_count = sum(1 for r in result.sandbox_results if r.success)
                        total_count = len(result.sandbox_results)
                        status = "\u2705 all blocks verified" if verified_count == total_count else f"\u26a0\ufe0f  {verified_count}/{total_count} blocks verified"
                        print(f"\U0001f52c Sandbox: {status} ({total_count} block{'s' if total_count != 1 else ''} tested)")
                print(result.content)

        else:
            print("\n\U0001f4a1 Use --interactive for chat mode, or provide a query directly")
            print("   Example: python3 rain.py 'What is the capital of France?'")
            print("   Example: python3 rain.py --interactive")
            print("   Example: python3 rain.py --task 'Refactor server.py to support pluggable backends'")
            print("   Example: python3 rain.py --react 'What Python files are in this project and what does each one do?'")
            print("   Example: python3 rain.py --react --verbose 'Find any TODO comments in rain.py and summarise them'")
            print("   Example: python3 rain.py --skills")
            print("   Example: python3 rain.py --install-skill git-essentials")
            print("   Example: python3 rain.py --system-file system-prompts/bitcoin-maximalist.txt 'Explain money'")
            print("   Example: python3 rain.py --system-prompt 'You are a helpful coding assistant' 'Debug this Python code'")
            print("\n\U0001f4dd Check the system-prompts/ folder for example personality profiles!")

    except RuntimeError as e:
        print(f"\u274c Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\U0001f44b Rain session ended")


if __name__ == "__main__":
    main()
