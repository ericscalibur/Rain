#!/usr/bin/env python3
"""
Rain ⛈️ - Phase 10 Knowledge Graph Test Suite

Six focused tests that exercise the core knowledge graph features
against Rain's own codebase. Run from the Rain directory:

    python3 test_knowledge_graph.py

Each test prints a clear ✅ / ❌ with details so you can see exactly
what passed and what needs attention.

Requirements:
    - Rain's codebase at the current working directory
    - ~/.rain/memory.db writable (tests use a temporary copy)
    - git available on PATH (for history tests)
    - No Ollama needed (LLM-dependent features are skipped gracefully)
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_graph import KnowledgeGraph

# ── Helpers ────────────────────────────────────────────────────────────

PROJECT_PATH = str(Path(__file__).parent.resolve())

_pass_count = 0
_fail_count = 0


def _header(name: str):
    print(f"\n{'─' * 60}")
    print(f"  TEST: {name}")
    print(f"{'─' * 60}")


def _pass(msg: str):
    global _pass_count
    _pass_count += 1
    print(f"  ✅ {msg}")


def _fail(msg: str):
    global _fail_count
    _fail_count += 1
    print(f"  ❌ {msg}")


def _check(condition: bool, pass_msg: str, fail_msg: str):
    if condition:
        _pass(pass_msg)
    else:
        _fail(fail_msg)


def _make_kg() -> KnowledgeGraph:
    """
    Create a KnowledgeGraph backed by a temporary database so tests
    don't pollute the real ~/.rain/memory.db.
    """
    tmp_dir = tempfile.mkdtemp(prefix="rain_kg_test_")
    db_path = Path(tmp_dir) / "test_memory.db"
    return KnowledgeGraph(db_path=db_path), tmp_dir


# ══════════════════════════════════════════════════════════════════════
#  Test 1: Graph Build — parse Rain's codebase and verify node/edge counts
# ══════════════════════════════════════════════════════════════════════

def test_graph_build():
    _header("1 · Graph Build (parse Rain's own codebase)")

    kg, tmp_dir = _make_kg()
    try:
        stats = kg.build_graph(PROJECT_PATH, force=True)

        _check(
            "error" not in stats,
            f"Graph built successfully in {stats.get('duration_s', '?')}s",
            f"Graph build failed: {stats.get('error', 'unknown')}",
        )

        files_parsed = stats.get("files_parsed", 0)
        _check(
            files_parsed >= 8,
            f"Parsed {files_parsed} files (expected ≥ 8 — rain.py, server.py, indexer.py, etc.)",
            f"Only parsed {files_parsed} files (expected ≥ 8)",
        )

        total_nodes = stats.get("nodes", 0)
        _check(
            total_nodes >= 100,
            f"Created {total_nodes} nodes (expected ≥ 100)",
            f"Only {total_nodes} nodes created (expected ≥ 100)",
        )

        total_edges = stats.get("edges", 0)
        _check(
            total_edges >= 200,
            f"Created {total_edges} edges (expected ≥ 200)",
            f"Only {total_edges} edges created (expected ≥ 200)",
        )

        errors = stats.get("errors", 0)
        _check(
            errors == 0,
            "Zero parse errors",
            f"{errors} parse error(s) — check file compatibility",
        )

        # Verify stats endpoint works
        graph_stats = kg.get_project_stats(PROJECT_PATH)
        node_types = graph_stats.get("node_types", {})

        _check(
            "function" in node_types and node_types["function"] > 50,
            f"Node types look right: {json.dumps(node_types, indent=None)}",
            f"Unexpected node types: {json.dumps(node_types, indent=None)}",
        )

        edge_types = graph_stats.get("edge_types", {})
        _check(
            "calls" in edge_types and "contains" in edge_types,
            f"Edge types look right: {json.dumps(edge_types, indent=None)}",
            f"Missing expected edge types: {json.dumps(edge_types, indent=None)}",
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
#  Test 2: Python AST Parsing — verify function signatures, classes,
#           methods, docstrings extracted from rain.py
# ══════════════════════════════════════════════════════════════════════

def test_python_ast_parsing():
    _header("2 · Python AST Parsing (functions, classes, signatures)")

    kg, tmp_dir = _make_kg()
    try:
        kg.build_graph(PROJECT_PATH, force=True)

        # Check that RainMemory class was found
        rain_memory_nodes = kg.find_nodes(PROJECT_PATH, name="RainMemory", node_type="class")
        _check(
            len(rain_memory_nodes) >= 1,
            f"Found RainMemory class ({len(rain_memory_nodes)} node(s))",
            "RainMemory class NOT found in graph",
        )

        # Check that MultiAgentOrchestrator class was found
        orchestrator_nodes = kg.find_nodes(PROJECT_PATH, name="MultiAgentOrchestrator", node_type="class")
        _check(
            len(orchestrator_nodes) >= 1,
            f"Found MultiAgentOrchestrator class ({len(orchestrator_nodes)} node(s))",
            "MultiAgentOrchestrator class NOT found in graph",
        )

        # Check a known function with a specific signature
        reflect_nodes = kg.find_nodes(PROJECT_PATH, name="recursive_reflect")
        has_signature = any(
            n.get("signature") and "recursive_reflect" in n.get("signature", "")
            for n in reflect_nodes
        )
        _check(
            has_signature,
            f"recursive_reflect found with signature ({len(reflect_nodes)} node(s))",
            "recursive_reflect missing or has no signature",
        )

        # Check docstrings are captured
        has_docstring = any(
            n.get("docstring") and len(n["docstring"]) > 10
            for n in reflect_nodes
        )
        _check(
            has_docstring,
            "recursive_reflect has a captured docstring",
            "recursive_reflect docstring NOT captured",
        )

        # Check that _init_db (a method inside RainMemory) was found
        init_db_nodes = kg.find_nodes(PROJECT_PATH, name="_init_db")
        _check(
            len(init_db_nodes) >= 1,
            f"Found _init_db method ({len(init_db_nodes)} node(s))",
            "_init_db method NOT found — method parsing may be broken",
        )

        # Check line numbers are present
        has_lines = any(
            n.get("line_start") and n["line_start"] > 0
            for n in reflect_nodes
        )
        _check(
            has_lines,
            "Line numbers present on function nodes",
            "Line numbers missing from function nodes",
        )

        # Check imports were captured
        import_nodes = kg.find_nodes(PROJECT_PATH, node_type="import")
        _check(
            len(import_nodes) >= 20,
            f"Found {len(import_nodes)} import nodes (expected ≥ 20)",
            f"Only {len(import_nodes)} import nodes (expected ≥ 20)",
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
#  Test 3: Callers & Callees — trace call relationships
# ══════════════════════════════════════════════════════════════════════

def test_callers_and_callees():
    _header("3 · Callers & Callees (call graph traversal)")

    kg, tmp_dir = _make_kg()
    try:
        kg.build_graph(PROJECT_PATH, force=True)

        # recursive_reflect should be called by multiple functions
        callers = kg.get_callers(PROJECT_PATH, "recursive_reflect")
        caller_names = [c.get("name", "") for c in callers]

        _check(
            len(callers) >= 3,
            f"recursive_reflect has {len(callers)} callers: {', '.join(caller_names[:8])}",
            f"recursive_reflect only has {len(callers)} caller(s) (expected ≥ 3): {caller_names}",
        )

        # Check specific known callers
        expected_callers = {"main", "_stream_chat", "run_pipeline"}
        found_expected = expected_callers & set(caller_names)
        _check(
            len(found_expected) >= 2,
            f"Found expected callers: {found_expected}",
            f"Missing expected callers — found {set(caller_names)} but wanted at least 2 of {expected_callers}",
        )

        # recursive_reflect should call several things itself
        callees = kg.get_callees(PROJECT_PATH, "recursive_reflect")
        callee_names = [c.get("name", "") for c in callees]

        _check(
            len(callees) >= 3,
            f"recursive_reflect calls {len(callees)} functions: {', '.join(callee_names[:10])}",
            f"recursive_reflect only calls {len(callees)} function(s) (expected ≥ 3)",
        )

        # build_graph should be callable and have callers/callees too
        bg_callers = kg.get_callers(PROJECT_PATH, "build_graph")
        _check(
            len(bg_callers) >= 1,
            f"build_graph has {len(bg_callers)} caller(s)",
            "build_graph has 0 callers — caller detection may be broken for knowledge_graph.py",
        )

        # Check that get_edges works for a specific node
        nodes = kg.find_nodes(PROJECT_PATH, name="recursive_reflect", node_type="method")
        if nodes:
            edges = kg.get_edges(nodes[0]["id"], direction="outgoing", edge_type="calls")
            _check(
                len(edges) >= 1,
                f"get_edges() returned {len(edges)} outgoing 'calls' edges for recursive_reflect",
                "get_edges() returned 0 outgoing 'calls' edges",
            )
        else:
            _fail("Could not find recursive_reflect method node to test get_edges()")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
#  Test 4: Git History — verify commit data and blame
# ══════════════════════════════════════════════════════════════════════

def test_git_history():
    _header("4 · Git History (commits, blame, introducing commit)")

    kg, tmp_dir = _make_kg()
    try:
        kg.build_graph(PROJECT_PATH, force=True)

        # Project-wide git history
        commits = kg.get_git_history(PROJECT_PATH, n=10)
        _check(
            len(commits) >= 1,
            f"Got {len(commits)} commit(s) from project history",
            "No git history returned — is this a git repo?",
        )

        if commits:
            c = commits[0]
            has_fields = all(k in c for k in ("sha", "date", "author", "message"))
            _check(
                has_fields,
                f"Commit has all fields: {c['sha'][:7]} by {c['author']} — {c['message'][:50]}",
                f"Commit missing fields: {list(c.keys())}",
            )

        # Per-file history for rain.py
        rain_commits = kg.get_git_history(PROJECT_PATH, file_path="rain.py", n=5)
        _check(
            len(rain_commits) >= 1,
            f"Got {len(rain_commits)} commit(s) for rain.py",
            "No git history for rain.py",
        )

        # Blame summary for server.py
        blame = kg.get_file_blame_summary(PROJECT_PATH, "server.py")
        _check(
            "Blame summary" in blame or "%" in blame,
            f"Blame summary returned: {blame[:80]}...",
            f"Blame summary unexpected: {blame[:120]}",
        )

        # Find introducing commit for a known function
        commit = kg.get_commit_for_function(PROJECT_PATH, "rain.py", "recursive_reflect")
        if commit:
            _check(
                "sha" in commit and "author" in commit,
                f"recursive_reflect introduced by {commit['author']} on {commit['date'][:10]} "
                f"({commit['sha'][:7]}): {commit['message'][:50]}",
                f"Introducing commit has unexpected shape: {commit}",
            )
        else:
            # This can happen if git log -S doesn't find it (e.g. the function
            # was renamed or the git history is shallow)
            _pass("get_commit_for_function returned None — acceptable if history is shallow")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
#  Test 5: Decision Log — manual logging, listing, searching
# ══════════════════════════════════════════════════════════════════════

def test_decision_log():
    _header("5 · Decision Log (log, list, search)")

    kg, tmp_dir = _make_kg()
    try:
        # Log a few test decisions
        id1 = kg.log_decision(
            title="Use FastAPI over Flask",
            description="FastAPI chosen for async support, SSE streaming, and automatic OpenAPI docs.",
            project_path=PROJECT_PATH,
            rationale="Flask requires additional libraries for async; FastAPI handles it natively.",
            alternatives="Flask, Starlette, aiohttp",
            tags="web,backend,streaming",
        )
        _check(
            id1 > 0,
            f"Decision 1 logged (ID: {id1}): Use FastAPI over Flask",
            f"Decision 1 failed to log (returned {id1})",
        )

        id2 = kg.log_decision(
            title="SQLite for all storage",
            description="Single SQLite database for memory, vectors, feedback, project index, and knowledge graph.",
            project_path=PROJECT_PATH,
            rationale="Zero external dependencies, fully portable, single-file backup.",
            alternatives="PostgreSQL, TinyDB, separate DBs per feature",
            tags="database,architecture,sovereignty",
        )
        _check(
            id2 > 0,
            f"Decision 2 logged (ID: {id2}): SQLite for all storage",
            f"Decision 2 failed to log (returned {id2})",
        )

        id3 = kg.log_decision(
            title="Server-Sent Events over WebSockets",
            description="SSE chosen for streaming responses from the multi-agent pipeline.",
            project_path=PROJECT_PATH,
            rationale="SSE works natively with uvicorn, no buffering issues, simpler client code.",
            alternatives="WebSockets, long-polling",
            tags="streaming,web,architecture",
        )

        # List all decisions
        all_decisions = kg.list_decisions(project_path=PROJECT_PATH)
        _check(
            len(all_decisions) >= 3,
            f"Listed {len(all_decisions)} decisions for project",
            f"Expected ≥ 3 decisions, got {len(all_decisions)}",
        )

        # Search by keyword
        sqlite_results = kg.search_decisions("SQLite", project_path=PROJECT_PATH)
        _check(
            len(sqlite_results) >= 1,
            f"Search for 'SQLite' returned {len(sqlite_results)} result(s): "
            f"{sqlite_results[0]['title'] if sqlite_results else '?'}",
            "Search for 'SQLite' returned 0 results",
        )

        streaming_results = kg.search_decisions("streaming")
        _check(
            len(streaming_results) >= 1,
            f"Search for 'streaming' returned {len(streaming_results)} result(s)",
            "Search for 'streaming' returned 0 results",
        )

        # Verify decision fields are complete
        d = all_decisions[0]
        required_fields = {"id", "title", "description", "timestamp"}
        has_all = required_fields.issubset(set(d.keys()))
        _check(
            has_all,
            f"Decision has all required fields: {sorted(d.keys())}",
            f"Decision missing fields — has: {sorted(d.keys())}, needs: {required_fields}",
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
#  Test 6: Context Block Builder — the piece that feeds agents
# ══════════════════════════════════════════════════════════════════════

def test_context_block_builder():
    _header("6 · Context Block Builder (agent prompt injection)")

    kg, tmp_dir = _make_kg()
    try:
        kg.build_graph(PROJECT_PATH, force=True)

        # Log a test decision so context builder can find it
        kg.log_decision(
            title="Use nomic-embed-text for embeddings",
            description="Chosen for local embedding via Ollama, zero pip deps.",
            project_path=PROJECT_PATH,
            rationale="Runs locally, good quality, already available via Ollama.",
            tags="embeddings,memory,semantic",
        )

        # Query mentioning a known function
        block1 = kg.build_context_block(
            "how does recursive_reflect work?",
            PROJECT_PATH,
        )
        _check(
            "recursive_reflect" in block1 and "Knowledge graph" in block1,
            f"Context block for 'recursive_reflect' is {len(block1)} chars with graph header",
            f"Context block missing expected content (len={len(block1)})",
        )

        _check(
            "called by" in block1.lower() or "calls" in block1.lower(),
            "Context block includes caller/callee information",
            "Context block missing caller/callee info",
        )

        # Query mentioning a class
        block2 = kg.build_context_block(
            "what is the RainMemory class?",
            PROJECT_PATH,
        )
        _check(
            "RainMemory" in block2,
            f"Context block for 'RainMemory' found the class ({len(block2)} chars)",
            "Context block did NOT find RainMemory",
        )

        # "Why" query should trigger git history context
        block3 = kg.build_context_block(
            "why was recursive_reflect introduced?",
            PROJECT_PATH,
        )
        _check(
            "git" in block3.lower() or "introduced" in block3.lower() or "commit" in block3.lower(),
            "Context block for 'why' query includes git history",
            f"Context block for 'why' query missing git history (len={len(block3)})",
        )

        # Query mentioning a decision keyword
        block4 = kg.build_context_block(
            "why did we choose nomic-embed-text for embeddings?",
            PROJECT_PATH,
        )
        _check(
            "nomic" in block4.lower() or "decision" in block4.lower() or "embedding" in block4.lower(),
            f"Context block for decision query includes relevant content ({len(block4)} chars)",
            "Context block for decision query missing decision data",
        )

        # Query with no matching identifiers should return empty or minimal
        block_empty = kg.build_context_block(
            "what is the weather today?",
            PROJECT_PATH,
        )
        _check(
            len(block_empty) < 50,
            f"Irrelevant query returns minimal context ({len(block_empty)} chars) — good, no noise",
            f"Irrelevant query returned {len(block_empty)} chars of context — may inject noise",
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Rain ⛈️  — Phase 10 Knowledge Graph Test Suite")
    print("=" * 60)
    print(f"  Project: {PROJECT_PATH}")
    print(f"  Using temp databases (production memory.db is untouched)")

    test_graph_build()
    test_python_ast_parsing()
    test_callers_and_callees()
    test_git_history()
    test_decision_log()
    test_context_block_builder()

    print(f"\n{'=' * 60}")
    total = _pass_count + _fail_count
    if _fail_count == 0:
        print(f"  🎉 ALL {total} CHECKS PASSED")
    else:
        print(f"  ✅ {_pass_count}/{total} passed  ·  ❌ {_fail_count}/{total} failed")
    print(f"{'=' * 60}\n")

    sys.exit(0 if _fail_count == 0 else 1)


if __name__ == "__main__":
    main()
