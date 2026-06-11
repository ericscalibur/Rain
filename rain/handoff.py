"""
Auto-generated session handoff — runtime state from live data, not human memory.

Every fact in the generated block is read from the running system at generation
time: git for code state, the orchestrator for the resolved roster, memory.db
for performance numbers. Nothing is written from recollection, so nothing in
the block can go stale the way hand-written handoff notes do.

The block is written between BEGIN/END markers inside SESSION_HANDOFF.md.
Everything outside the markers — the human narrative — is never touched.
"""

import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

BEGIN_MARKER = "<!-- BEGIN AUTO-GENERATED RUNTIME STATE — edit nothing here, run `python3 rain.py --handoff` to refresh -->"
END_MARKER = "<!-- END AUTO-GENERATED RUNTIME STATE -->"


def _git(args, cwd):
    try:
        out = subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _pct(value):
    return f"{round(value * 100)}%" if value is not None else "n/a"


def generate_runtime_state(orchestrator, memory) -> str:
    """Build the runtime-state markdown block from live system data."""
    repo = Path(__file__).resolve().parent.parent
    lines = []
    lines.append(BEGIN_MARKER)
    lines.append("")
    lines.append("## Runtime State (auto-generated)")
    lines.append("")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by `--handoff`. "
                 f"Trust this section over hand-written notes elsewhere in this file._")
    lines.append("")

    # ── Git state ──────────────────────────────────────────────────────
    branch = _git(["branch", "--show-current"], repo)
    commits = _git(["log", "--format=- `%h` %ad %s", "--date=short", "-8"], repo)
    dirty = _git(["status", "--porcelain"], repo)
    lines.append(f"**Branch:** `{branch or 'unknown'}`"
                 + (" (uncommitted changes present)" if dirty else ""))
    lines.append("")
    if commits:
        lines.append("**Recent commits:**")
        lines.append(commits)
        lines.append("")

    # ── Resolved agent roster ──────────────────────────────────────────
    lines.append("**Live agent roster** (resolved by `_best_model_for`, not the docs):")
    lines.append("")
    lines.append("| Agent | Model |")
    lines.append("|---|---|")
    for agent_type, agent in orchestrator.agents.items():
        lines.append(f"| {agent_type.value} | `{agent.model_name}` |")
    lines.append("")
    installed = sorted(orchestrator._installed_models)
    lines.append(f"**Installed Ollama models ({len(installed)}):** "
                 + ", ".join(f"`{m}`" for m in installed))
    lines.append("")

    # ── Performance from memory.db ─────────────────────────────────────
    if memory:
        stats = memory.get_performance_stats()
        overall = stats.get("overall", {})
        lines.append("**Feedback & performance:**")
        lines.append(f"- {overall.get('total_feedback', 0)} total ratings across "
                     f"{overall.get('sessions', 0)} sessions · overall accuracy "
                     f"{_pct(overall.get('accuracy'))} · open gaps: {stats.get('open_gaps', 0)}")
        for row in stats.get("by_agent_30d", []):
            lines.append(f"- last 30d · {row['agent']}: {row['total']} ratings, "
                         f"accuracy {_pct(row['accuracy'])}, avg conf {row['avg_conf']}")

        synth = memory.get_synthesis_accuracy()
        if synth.get("total"):
            lines.append(f"- synthesis: {synth['total']} runs, {synth['rated']} rated, "
                         f"improvement rate {_pct(synth.get('improvement_rate'))}, "
                         f"confidence gain rate {_pct(synth.get('confidence_improvement_rate'))}")

        # rain-tuned usage (ab_results is written on every rain-tuned response)
        try:
            with sqlite3.connect(memory.db_path) as conn:
                row = conn.execute(
                    """SELECT COUNT(*), ROUND(AVG(confidence), 2),
                              MIN(substr(timestamp,1,10)), MAX(substr(timestamp,1,10))
                       FROM ab_results WHERE model LIKE 'rain-tuned%'"""
                ).fetchone()
            if row and row[0]:
                lines.append(f"- rain-tuned: {row[0]} logged responses "
                             f"({row[2]} → {row[3]}), avg confidence {row[1]}")
        except Exception:
            pass
        lines.append("")

        gaps = memory.get_top_gaps(limit=5)
        if gaps:
            lines.append("**Open knowledge gaps (most recent first):**")
            for g in gaps:
                desc = (g.get("gap_description") or "").replace("\n", " ")[:140]
                lines.append(f"- [{round((g.get('confidence') or 0) * 100)}%] {desc}")
            lines.append("")

    lines.append(END_MARKER)
    return "\n".join(lines)


def update_handoff_file(block: str, path: Path = None) -> Path:
    """Insert or replace the auto-generated block in SESSION_HANDOFF.md.

    Content outside the BEGIN/END markers is preserved byte-for-byte. If the
    markers aren't present yet, the block is appended to the end of the file
    (or a new file is created).
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "SESSION_HANDOFF.md"

    if path.exists():
        text = path.read_text()
        if BEGIN_MARKER in text and END_MARKER in text:
            head, rest = text.split(BEGIN_MARKER, 1)
            _, tail = rest.split(END_MARKER, 1)
            new_text = head + block + tail
        else:
            new_text = text.rstrip("\n") + "\n\n---\n\n" + block + "\n"
    else:
        new_text = "# Rain ⛈️ — Session Handoff\n\n" + block + "\n"

    path.write_text(new_text)
    return path
