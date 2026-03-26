"""
Sandboxed code execution module for Rain.

Provides CodeSandbox for safely running Python and JavaScript code
in isolated temporary directories with configurable timeouts.
Includes SandboxResult and ReflectionResult dataclasses for result handling.
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


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
