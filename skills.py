#!/usr/bin/env python3
"""
Rain ⛈️ - Skills Runtime (Phase 6)

Implements the OpenClaw/ClawBot skill format for Rain's extensibility layer.
Skills are plain SKILL.md files with YAML frontmatter. They live in:
  - ~/.rain/skills/          (global, user-installed)
  - <project>/skills/        (workspace-local, takes priority)

Zero new dependencies — frontmatter parsed with stdlib re.

Skill format (YAML frontmatter):
  ---
  name: Git Essentials
  slug: git-essentials
  description: Helps with common git operations
  tags: [git, vcs, developer-tools]
  primaryEnv: GIT_AUTHOR_NAME
  ---

  # Git Essentials
  (body content — instructions, context, examples)

Usage:
    from skills import SkillLoader
    loader = SkillLoader(project_path="/path/to/project")
    loader.load()
    matches = loader.find_matching_skills("how do I stage and commit my changes?")
    for skill in matches:
        print(skill.name, skill.description)
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Data model ────────────────────────────────────────────────────────

@dataclass
class SkillMeta:
    """Parsed metadata and content for a single skill."""
    name: str
    slug: str
    description: str
    tags: List[str]
    primary_env: List[str]          # required env vars (may be empty)
    file_path: Path
    content: str                    # full SKILL.md body (after frontmatter)
    raw: str                        # full file content including frontmatter

    @property
    def env_satisfied(self) -> bool:
        """True if all required environment variables are present."""
        return all(os.environ.get(var) for var in self.primary_env)

    @property
    def source_label(self) -> str:
        """Human-readable source location."""
        home = Path.home()
        try:
            rel = self.file_path.relative_to(home)
            return f"~/{rel}"
        except ValueError:
            return str(self.file_path)

    def as_context_block(self) -> str:
        """
        Format skill as a directive context block for injection into an agent prompt.
        Keeps it tight — name, description, env status, then the body.
        """
        env_note = ""
        if self.primary_env:
            missing = [v for v in self.primary_env if not os.environ.get(v)]
            if missing:
                env_note = f"\n⚠️  Required env vars not set: {', '.join(missing)}"
            else:
                env_note = f"\n✅ Required env vars present: {', '.join(self.primary_env)}"

        return (
            f"[SKILL: {self.name}]{env_note}\n"
            f"Description: {self.description}\n"
            f"---\n"
            f"{self.content.strip()}"
        )


# ── YAML frontmatter parser ───────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Extract YAML frontmatter from a SKILL.md file.

    Returns (metadata_dict, body_text).
    Handles the subset of YAML used in OpenClaw skills:
      - Simple strings:  key: value
      - Inline lists:    tags: [item1, item2, item3]
      - Block lists:     tags:\n  - item1\n  - item2
      - Optional quotes: key: "value" or key: value

    Falls back to empty dict on parse failure — never raises.
    """
    # Must start with ---
    if not text.startswith('---'):
        return {}, text

    # Find the closing ---
    end = text.find('\n---', 3)
    if end == -1:
        return {}, text

    frontmatter_block = text[3:end].strip()
    body = text[end + 4:].lstrip('\n')

    meta: dict = {}

    lines = frontmatter_block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip blank lines and comments
        if not line.strip() or line.strip().startswith('#'):
            i += 1
            continue

        # key: value
        m = re.match(r'^(\w[\w\-]*)\s*:\s*(.*)', line)
        if not m:
            i += 1
            continue

        key = m.group(1).lower().replace('-', '_')
        raw_val = m.group(2).strip()

        if not raw_val:
            # Block list: next lines are "  - item"
            items = []
            i += 1
            while i < len(lines) and re.match(r'^\s+-\s+', lines[i]):
                items.append(lines[i].strip().lstrip('- ').strip())
                i += 1
            meta[key] = items
            continue

        # Inline list: [item1, item2]
        if raw_val.startswith('[') and raw_val.endswith(']'):
            inner = raw_val[1:-1]
            items = [s.strip().strip('"\'') for s in inner.split(',') if s.strip()]
            meta[key] = items
            i += 1
            continue

        # Plain string (strip optional quotes)
        meta[key] = raw_val.strip('"\'')
        i += 1

    return meta, body


# ── Skill loader ──────────────────────────────────────────────────────

class SkillLoader:
    """
    Discovers and indexes skills from the filesystem.

    Search order (workspace takes priority over global):
      1. <project_path>/skills/   (if project_path provided)
      2. ~/.rain/skills/

    Skills in workspace dirs shadow global skills with the same slug.
    """

    GLOBAL_SKILLS_DIR = Path.home() / '.rain' / 'skills'

    def __init__(self, project_path: Optional[str] = None):
        self._project_path = Path(project_path) if project_path else None
        self._skills: List[SkillMeta] = []
        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────

    def load(self) -> 'SkillLoader':
        """
        Scan skill directories and build the index.
        Safe to call multiple times — reloads on each call.
        """
        found: dict[str, SkillMeta] = {}  # slug -> skill (later entries win)

        # Global skills first (lower priority)
        for skill in self._scan_dir(self.GLOBAL_SKILLS_DIR):
            found[skill.slug] = skill

        # Workspace skills override global (higher priority)
        if self._project_path:
            workspace_skills_dir = self._project_path / 'skills'
            for skill in self._scan_dir(workspace_skills_dir):
                found[skill.slug] = skill

        self._skills = list(found.values())
        self._loaded = True
        return self

    @property
    def skills(self) -> List[SkillMeta]:
        """All loaded skills. Call load() first."""
        if not self._loaded:
            self.load()
        return self._skills

    @property
    def count(self) -> int:
        return len(self.skills)

    def find_matching_skills(self, query: str, top_k: int = 3) -> List[SkillMeta]:
        """
        Score each skill against the query and return the top matches.

        Scoring:
          +3  for each tag word that appears in the query
          +2  for each word in the skill name that appears in the query
          +1  for each word in the description that appears in the query
          Skills with score 0 are excluded.
          Skills with unsatisfied required env vars are deprioritised (-1).
        """
        if not self.skills:
            return []

        query_lower = query.lower()
        scored: list[tuple[int, SkillMeta]] = []

        for skill in self.skills:
            score = 0

            # Tags (highest weight)
            for tag in skill.tags:
                if tag.lower() in query_lower or query_lower in tag.lower():
                    score += 3
                else:
                    # partial match: individual words in multi-word tags
                    for word in tag.replace('-', ' ').split():
                        if word in query_lower:
                            score += 1

            # Name words
            for word in skill.name.lower().replace('-', ' ').split():
                if len(word) > 2 and word in query_lower:
                    score += 2

            # Slug words
            for word in skill.slug.replace('-', ' ').split():
                if len(word) > 2 and word in query_lower:
                    score += 2

            # Description words (lower weight, filter short words)
            desc_words = set(re.findall(r'\b\w{4,}\b', skill.description.lower()))
            query_words = set(re.findall(r'\b\w{4,}\b', query_lower))
            overlap = desc_words & query_words
            score += len(overlap)

            # Penalise skills with missing env vars (still show, just lower priority)
            if not skill.env_satisfied and skill.primary_env:
                score -= 1

            if score > 0:
                scored.append((score, skill))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored[:top_k]]

    def get_skill(self, slug: str) -> Optional[SkillMeta]:
        """Look up a skill by exact slug."""
        for skill in self.skills:
            if skill.slug == slug:
                return skill
        return None

    def summary_table(self) -> str:
        """
        Returns a formatted table of all installed skills suitable for
        printing to the terminal (used by --skills flag).
        """
        if not self.skills:
            return (
                "No skills installed.\n\n"
                "Install a skill with:  python3 rain.py --install-skill <slug>\n"
                "Browse skills at:      https://clawhub.ai\n"
                f"Skills directory:      {self.GLOBAL_SKILLS_DIR}"
            )

        lines = [
            f"{'NAME':<28} {'SLUG':<28} {'TAGS':<35} {'ENV':<6}",
            "-" * 100,
        ]
        for skill in sorted(self.skills, key=lambda s: s.name.lower()):
            env_marker = "✅" if skill.env_satisfied else ("⚠️ " if skill.primary_env else "  ")
            tags_str = ", ".join(skill.tags[:3])
            if len(skill.tags) > 3:
                tags_str += f" +{len(skill.tags) - 3}"
            lines.append(
                f"{skill.name[:27]:<28} {skill.slug[:27]:<28} {tags_str[:34]:<35} {env_marker}"
            )

        lines.append("")
        lines.append(f"  {len(self.skills)} skill(s) installed  ·  {self.GLOBAL_SKILLS_DIR}")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────

    def _scan_dir(self, directory: Path) -> List[SkillMeta]:
        """
        Recursively scan a directory for SKILL.md files.
        Each skill lives in its own subdirectory:
          skills/
            git-essentials/
              SKILL.md
            search-web/
              SKILL.md
        """
        skills = []
        if not directory.exists():
            return skills

        # Find all SKILL.md files (case-insensitive filename match)
        for path in directory.rglob('*'):
            if path.is_file() and path.name.lower() == 'skill.md':
                skill = self._parse_skill_file(path)
                if skill:
                    skills.append(skill)

        return skills

    def _parse_skill_file(self, path: Path) -> Optional[SkillMeta]:
        """Parse a single SKILL.md file. Returns None on failure."""
        try:
            raw = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return None

        meta, body = _parse_frontmatter(raw)

        # Derive slug from directory name if not in frontmatter
        slug = meta.get('slug') or path.parent.name or path.stem
        name = meta.get('name') or slug.replace('-', ' ').title()
        description = meta.get('description') or ''
        tags_raw = meta.get('tags', [])
        if isinstance(tags_raw, str):
            # comma-separated string fallback
            tags_raw = [t.strip() for t in tags_raw.split(',')]
        tags = [str(t) for t in tags_raw]

        # primaryEnv can be a string (single var) or list
        env_raw = meta.get('primary_env') or meta.get('primaryenv') or []
        if isinstance(env_raw, str):
            primary_env = [env_raw] if env_raw else []
        else:
            primary_env = [str(v) for v in env_raw if v]

        # Body: if empty, fall back to the whole file as context
        if not body.strip():
            body = raw

        return SkillMeta(
            name=name,
            slug=slug,
            description=description,
            tags=tags,
            primary_env=primary_env,
            file_path=path,
            content=body,
            raw=raw,
        )


# ── Install helper ────────────────────────────────────────────────────

def install_skill(slug: str, global_dir: Path = None) -> tuple[bool, str]:
    """
    Install a skill from ClawHub via `npx clawhub@latest install <slug>`.

    The ClawHub CLI places the skill in the current directory's `skills/`
    folder by default. We redirect it to ~/.rain/skills/ using the --dir flag.

    Returns (success: bool, message: str).
    """
    import subprocess

    target_dir = global_dir or SkillLoader.GLOBAL_SKILLS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ['npx', 'clawhub@latest', 'install', slug, '--dir', str(target_dir)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True, result.stdout.strip() or f"Skill '{slug}' installed to {target_dir}"
        else:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "No output from clawhub"
            return False, f"clawhub install failed: {detail}"
    except FileNotFoundError:
        return False, (
            "npx not found. Install Node.js to use the ClawHub registry.\n"
            "Alternatively, manually create a skill directory:\n"
            f"  mkdir -p {target_dir}/{slug}\n"
            f"  # Add a SKILL.md with YAML frontmatter to {target_dir}/{slug}/SKILL.md"
        )
    except subprocess.TimeoutExpired:
        return False, "clawhub install timed out after 60 seconds."
    except Exception as e:
        return False, f"Unexpected error: {e}"


# ── CLI (for testing / standalone use) ───────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        loader = SkillLoader()
        loader.load()
        print(loader.summary_table())
    elif len(sys.argv) > 2 and sys.argv[1] == '--match':
        query = ' '.join(sys.argv[2:])
        loader = SkillLoader()
        loader.load()
        matches = loader.find_matching_skills(query)
        if matches:
            print(f"Top matches for: \"{query}\"\n")
            for s in matches:
                print(f"  [{s.slug}] {s.name} — {s.description}")
        else:
            print(f"No matching skills for: \"{query}\"")
    elif len(sys.argv) > 2 and sys.argv[1] == '--install':
        slug = sys.argv[2]
        ok, msg = install_skill(slug)
        print(("✅ " if ok else "❌ ") + msg)
    else:
        print("Usage:")
        print("  python3 skills.py --list")
        print("  python3 skills.py --match <query>")
        print("  python3 skills.py --install <slug>")
