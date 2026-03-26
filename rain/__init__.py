"""
Rain ⛈️ — Sovereign Local AI Ecosystem

A multi-agent AI system built to run entirely on local hardware via Ollama.
No cloud. No API keys. No telemetry.

Package structure:
  rain.memory       — RainMemory (6-tier persistent memory)
  rain.agents       — AgentType, Agent, prompts, model configuration
  rain.router       — AgentRouter (keyword-based query routing)
  rain.sandbox      — CodeSandbox, SandboxResult, ReflectionResult
  rain.orchestrator — MultiAgentOrchestrator (the core pipeline)
"""

# ── Public API ────────────────────────────────────────────────────────
# These re-exports maintain backward compatibility so that:
#   from rain import MultiAgentOrchestrator, RainMemory, AgentType, ...
# continues to work for server.py, rain-mcp.py, and any external code.

from .memory import RainMemory
from .agents import (
    AgentType,
    Agent,
    AGENT_PROMPTS,
    AGENT_PREFERRED_MODELS,
    VISION_PREFERRED_MODELS,
    auto_pick_default_model,
    _IMPLICIT_NEG_SIGNALS,
    _IMPLICIT_POS_SIGNALS,
    _LOGIC_FAST_PREFERRED,
    _LOGIC_COMPLEX_MARKERS,
    _DEFAULT_MODEL_PRIORITY,
    _EMBED_MODEL_FRAGMENTS,
)
from .router import AgentRouter
from .sandbox import ReflectionResult, SandboxResult, CodeSandbox
from .orchestrator import (
    MultiAgentOrchestrator,
    REACT_SYSTEM_PROMPT,
    _react_parse,
    _SKILLS_AVAILABLE,
    _TOOLS_AVAILABLE,
)

__all__ = [
    # Core classes
    'MultiAgentOrchestrator',
    'RainMemory',
    'AgentType',
    'Agent',
    'AgentRouter',
    'CodeSandbox',
    'ReflectionResult',
    'SandboxResult',
    # Configuration
    'AGENT_PROMPTS',
    'AGENT_PREFERRED_MODELS',
    'VISION_PREFERRED_MODELS',
    'REACT_SYSTEM_PROMPT',
    # Functions
    'auto_pick_default_model',
    '_react_parse',
    # Flags
    '_SKILLS_AVAILABLE',
    '_TOOLS_AVAILABLE',
]
