#!/usr/bin/env python3
"""
recommend_models.py — Hardware scanner & Ollama model recommender
Analyzes your machine's CPU, RAM, GPU, and unified memory, then
recommends the best Ollama models to pull for your setup.

Usage:
    python3 recommend_models.py
    python3 recommend_models.py --json        # machine-readable output
    python3 recommend_models.py --pull        # auto-pull all recommended models
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── ANSI colors (disabled automatically if not a TTY) ────────────────────────
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def bold(t):   return _c("1", t)
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def red(t):    return _c("31", t)
def dim(t):    return _c("2", t)


# ── Hardware snapshot ─────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    os_name: str = ""
    os_version: str = ""
    cpu_brand: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    ram_gb: float = 0.0
    # Apple Silicon unified memory is shared CPU+GPU — the whole pool is available to the model
    is_apple_silicon: bool = False
    apple_chip: str = ""           # e.g. "M1 Pro", "M3 Max"
    apple_chip_gen: int = 0        # 1, 2, 3, 4 …
    apple_chip_tier: str = ""      # "", "Pro", "Max", "Ultra"
    unified_memory_gb: float = 0.0
    # Discrete / dedicated GPU
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    has_cuda: bool = False
    has_metal: bool = False        # always True on Apple Silicon
    # Effective pool available to Ollama
    effective_pool_gb: float = 0.0
    notes: List[str] = field(default_factory=list)


def _run(cmd: List[str], timeout: int = 5) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def _parse_gb(text: str) -> float:
    """Parse strings like '16 GB', '32768 MB', '16384' (bytes) → float GB."""
    text = text.strip()
    m = re.search(r"([\d,.]+)\s*(TB|GB|MB|KB)?", text, re.IGNORECASE)
    if not m:
        return 0.0
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").upper()
    if unit == "TB":  return val * 1024
    if unit == "GB":  return val
    if unit == "MB":  return val / 1024
    if unit == "KB":  return val / (1024 * 1024)
    # raw bytes
    if val > 1e9:     return val / (1024 ** 3)
    if val > 1e6:     return val / (1024 ** 2)
    return val


def _scan_macos(profile: HardwareProfile):
    profile.has_metal = True

    # system_profiler for hardware overview
    hw = _run(["system_profiler", "SPHardwareDataType"])

    # Chip
    m = re.search(r"Chip:\s*(.+)", hw)
    if m:
        chip_full = m.group(1).strip()
        profile.apple_chip = chip_full
        profile.is_apple_silicon = "Apple" in chip_full

        if profile.is_apple_silicon:
            # Generation
            gen_m = re.search(r"M(\d+)", chip_full)
            if gen_m:
                profile.apple_chip_gen = int(gen_m.group(1))
            # Tier
            for tier in ("Ultra", "Max", "Pro"):
                if tier in chip_full:
                    profile.apple_chip_tier = tier
                    break
            else:
                profile.apple_chip_tier = "base"

    # CPU brand fallback
    cpu_brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    profile.cpu_brand = cpu_brand or profile.apple_chip

    # Cores
    physical = _run(["sysctl", "-n", "hw.physicalcpu"])
    logical  = _run(["sysctl", "-n", "hw.logicalcpu"])
    profile.cpu_cores_physical = int(physical) if physical.isdigit() else 0
    profile.cpu_cores_logical  = int(logical)  if logical.isdigit()  else 0

    # RAM / unified memory
    mem_m = re.search(r"Memory:\s*([\d.]+\s*\w+)", hw)
    if mem_m:
        profile.ram_gb = _parse_gb(mem_m.group(1))
        if profile.is_apple_silicon:
            profile.unified_memory_gb = profile.ram_gb

    # Effective pool
    if profile.is_apple_silicon:
        # Ollama uses ~80% of unified memory by default (leaves headroom for OS)
        profile.effective_pool_gb = profile.unified_memory_gb * 0.80
        profile.notes.append(
            "Apple Silicon unified memory: CPU & GPU share the same pool. "
            "Ollama uses ~80% of total RAM for the model context window."
        )
    else:
        profile.effective_pool_gb = profile.ram_gb * 0.75

    # Discrete GPU (non-Apple-Silicon Macs, or eGPU)
    gpu_out = _run(["system_profiler", "SPDisplaysDataType"])
    vram_m = re.search(r"VRAM.*?:\s*([\d.]+\s*\w+)", gpu_out, re.IGNORECASE)
    if vram_m:
        profile.gpu_vram_gb = _parse_gb(vram_m.group(1))
    gpu_name_m = re.search(r"Chipset Model:\s*(.+)", gpu_out)
    if gpu_name_m:
        profile.gpu_name = gpu_name_m.group(1).strip()


def _scan_linux(profile: HardwareProfile):
    # CPU
    cpuinfo = ""
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
    except Exception:
        pass
    m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    if m:
        profile.cpu_brand = m.group(1).strip()
    physical = len(set(re.findall(r"physical id\s*:\s*(\d+)", cpuinfo)))
    logical  = cpuinfo.count("processor\t:")
    profile.cpu_cores_physical = max(physical, 1)
    profile.cpu_cores_logical  = logical or 1

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(re.search(r"\d+", line).group())
                    profile.ram_gb = kb / (1024 ** 2)
                    break
    except Exception:
        pass
    profile.effective_pool_gb = profile.ram_gb * 0.75

    # NVIDIA GPU via nvidia-smi
    nvsmi = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if nvsmi:
        parts = nvsmi.split(",")
        if len(parts) >= 2:
            profile.gpu_name    = parts[0].strip()
            profile.gpu_vram_gb = float(parts[1].strip()) / 1024  # MiB → GiB
            profile.has_cuda    = True
            profile.effective_pool_gb = profile.gpu_vram_gb * 0.90
            profile.notes.append("NVIDIA GPU detected — Ollama will offload layers to VRAM.")
    else:
        # AMD ROCm
        rocm = _run(["rocm-smi", "--showmeminfo", "vram"])
        if rocm:
            profile.gpu_name = "AMD GPU (ROCm)"
            profile.notes.append("AMD ROCm GPU detected. Ollama ROCm build required.")


def _scan_windows(profile: HardwareProfile):
    # CPU via WMIC
    cpu = _run(["wmic", "cpu", "get", "name,NumberOfCores,NumberOfLogicalProcessors", "/format:csv"])
    lines = [l for l in cpu.splitlines() if l.strip() and not l.startswith("Node")]
    if lines:
        parts = lines[0].split(",")
        if len(parts) >= 4:
            profile.cpu_brand          = parts[3].strip()
            profile.cpu_cores_physical = int(parts[1]) if parts[1].isdigit() else 0
            profile.cpu_cores_logical  = int(parts[2]) if parts[2].isdigit() else 0

    # RAM via WMIC
    ram_out = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/format:csv"])
    for line in ram_out.splitlines():
        m = re.search(r"(\d{9,})", line)
        if m:
            profile.ram_gb = int(m.group(1)) / (1024 ** 3)
            break
    profile.effective_pool_gb = profile.ram_gb * 0.75

    # NVIDIA GPU via nvidia-smi
    nvsmi = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if nvsmi:
        parts = nvsmi.split(",")
        if len(parts) >= 2:
            profile.gpu_name    = parts[0].strip()
            profile.gpu_vram_gb = float(parts[1].strip()) / 1024
            profile.has_cuda    = True
            profile.effective_pool_gb = profile.gpu_vram_gb * 0.90
            profile.notes.append("NVIDIA GPU detected — Ollama will offload layers to VRAM.")


def scan_hardware() -> HardwareProfile:
    profile = HardwareProfile()
    profile.os_name    = platform.system()
    profile.os_version = platform.version()

    if profile.os_name == "Darwin":
        _scan_macos(profile)
    elif profile.os_name == "Linux":
        _scan_linux(profile)
    elif profile.os_name == "Windows":
        _scan_windows(profile)
    else:
        profile.notes.append(f"Unknown OS: {profile.os_name} — defaulting to conservative recommendations.")

    return profile


# ── Model catalog ─────────────────────────────────────────────────────────────

@dataclass
class ModelRec:
    name: str                  # ollama pull name
    display: str               # human-readable label
    size_gb: float             # approximate VRAM/RAM footprint at Q4
    category: str              # "general", "code", "vision", "embedding", "reasoning"
    description: str
    pull_cmd: str = ""
    min_pool_gb: float = 0.0   # minimum effective pool needed
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.pull_cmd = f"ollama pull {self.name}"
        if not self.min_pool_gb:
            self.min_pool_gb = self.size_gb * 1.2  # 20% headroom


# Catalog — sizes are Q4_K_M footprints (what Ollama uses by default)
MODEL_CATALOG: List[ModelRec] = [
    # ── Tiny (< 2 GB) ─────────────────────────────────────────────────────
    ModelRec("qwen2.5:0.5b",         "Qwen 2.5 0.5B",       0.4,  "general",
             "Tiny but surprisingly capable for simple Q&A and autocomplete.",
             tags=["tiny", "fast"]),
    ModelRec("qwen2.5:1.5b",         "Qwen 2.5 1.5B",       1.0,  "general",
             "Good for constrained devices. Fast, low memory.",
             tags=["tiny", "fast"]),
    ModelRec("phi3:mini",            "Phi-3 Mini (3.8B)",    2.2,  "general",
             "Microsoft's punchy small model. Strong reasoning for its size.",
             tags=["small", "reasoning"]),
    ModelRec("llama3.2:3b",          "Llama 3.2 3B",        2.0,  "general",
             "Meta's compact model. Fast responses, good for chat.",
             tags=["small", "fast"]),
    ModelRec("nomic-embed-text",     "Nomic Embed Text",     0.3,  "embedding",
             "Semantic embeddings for RAG, memory search, and similarity. Required for local vector search.",
             tags=["embedding", "required"]),

    # ── Small (2–5 GB) ────────────────────────────────────────────────────
    ModelRec("llama3.2:latest",      "Llama 3.2 (latest)",  2.0,  "general",
             "Meta's fast, capable 3B model. Great reflection/summarization agent.",
             tags=["small", "fast"]),
    ModelRec("gemma2:2b",            "Gemma 2 2B",          1.6,  "general",
             "Google's efficient 2B model. Punches above its weight.",
             tags=["small", "fast"]),
    ModelRec("mistral:7b",           "Mistral 7B",          4.1,  "general",
             "Solid all-rounder. Strong instruction following.",
             tags=["medium", "general"]),
    ModelRec("qwen2.5-coder:7b",     "Qwen 2.5 Coder 7B",  4.7,  "code",
             "Best-in-class open code model at 7B. Ideal for code generation and debugging.",
             tags=["code", "medium"]),

    # ── Medium (5–10 GB) ──────────────────────────────────────────────────
    ModelRec("llama3.1:8b",          "Llama 3.1 8B",        4.9,  "general",
             "Meta's flagship 8B. Excellent general reasoning.",
             tags=["medium", "general"]),
    ModelRec("qwen3:8b",             "Qwen 3 8B",           5.2,  "general",
             "Alibaba's sharp 8B. Strong at rewriting, summarization, and synthesis.",
             tags=["medium", "synthesis"]),
    ModelRec("qwen3.5:9b",           "Qwen 3.5 9B",         6.6,  "reasoning",
             "Best mid-size reasoning model. Strong at logic, domain knowledge, and analysis.",
             tags=["medium", "reasoning"]),
    ModelRec("gemma2:9b",            "Gemma 2 9B",          5.4,  "general",
             "Google's capable 9B. Strong at structured outputs.",
             tags=["medium", "general"]),
    ModelRec("deepseek-r1:8b",       "DeepSeek R1 8B",      5.0,  "reasoning",
             "Distilled reasoning model. Shows its chain-of-thought.",
             tags=["medium", "reasoning"]),

    # ── Large (10–20 GB) ──────────────────────────────────────────────────
    ModelRec("qwen2.5-coder:14b",    "Qwen 2.5 Coder 14B",  8.9, "code",
             "Serious code model. Handles large files and complex refactors.",
             tags=["large", "code"]),
    ModelRec("llama3.1:latest",      "Llama 3.1 (latest)",  4.9, "general",
             "Meta's latest 8B — or larger if you have the RAM.",
             tags=["large", "general"]),
    ModelRec("mistral-nemo:12b",     "Mistral Nemo 12B",    7.1, "general",
             "Mistral's extended context (128K) model.",
             tags=["large", "context"]),
    ModelRec("codestral:latest",     "Codestral",          12.0, "code",
             "Mistral's dedicated code model. Excellent at fill-in-the-middle and large file edits.",
             tags=["large", "code"]),
    ModelRec("llama3.2-vision:11b",  "Llama 3.2 Vision 11B", 7.9, "vision",
             "Multimodal — understands images and text together. Use for vision tasks.",
             tags=["large", "vision"]),
    ModelRec("qwen2.5:14b",          "Qwen 2.5 14B",        8.9, "general",
             "Strong multilingual reasoning. Good LOGIC agent alternative.",
             tags=["large", "reasoning"]),

    # ── Extra Large (20+ GB) ──────────────────────────────────────────────
    ModelRec("llama3.3:70b",         "Llama 3.3 70B",      43.0, "general",
             "Meta's flagship 70B. Near-GPT-4 quality. Needs 48+ GB effective pool.",
             tags=["xlarge", "flagship"]),
    ModelRec("qwen2.5:72b",          "Qwen 2.5 72B",       45.0, "general",
             "Alibaba's 72B. Multilingual powerhouse.",
             tags=["xlarge", "flagship"]),
    ModelRec("deepseek-r1:32b",      "DeepSeek R1 32B",    19.0, "reasoning",
             "Full reasoning model with visible chain-of-thought. Needs 24+ GB.",
             tags=["xlarge", "reasoning"]),
    ModelRec("qwen2.5-coder:32b",    "Qwen 2.5 Coder 32B", 19.0, "code",
             "Best open code model available. Handles repo-scale tasks.",
             tags=["xlarge", "code"]),
    ModelRec("deepseek-r1:70b",      "DeepSeek R1 70B",    43.0, "reasoning",
             "Full 70B reasoning model. Exceptional but needs 48+ GB.",
             tags=["xlarge", "reasoning"]),
]


# ── Recommendation engine ─────────────────────────────────────────────────────

TIER_LABELS = {
    "must":      ("✅", "Essential — pull these first"),
    "strong":    ("⭐", "Strongly recommended for your specs"),
    "optional":  ("💡", "Optional — nice to have if you have headroom"),
    "upgrade":   ("🚀", "Upgrade path — for when you have more RAM/VRAM"),
    "skip":      ("❌", "Skip — too large for your hardware"),
}

@dataclass
class Recommendation:
    model: ModelRec
    tier: str       # must / strong / optional / upgrade / skip
    reason: str


def recommend(profile: HardwareProfile) -> List[Recommendation]:
    recs = []
    pool = profile.effective_pool_gb

    def add(name: str, tier: str, reason: str):
        m = next((m for m in MODEL_CATALOG if m.name == name), None)
        if m:
            recs.append(Recommendation(m, tier, reason))

    # nomic-embed-text is the standard local embedding model
    add("nomic-embed-text", "must",
        "Standard local embedding model. Required for RAG, semantic search, and vector memory.")

    # ── Apple Silicon unified memory tiers ────────────────────────────────
    if profile.is_apple_silicon:
        umem  = profile.unified_memory_gb

        if umem >= 128:
            add("llama3.3:70b",        "must",    "128 GB unified — 70B runs comfortably at full speed.")
            add("qwen2.5-coder:32b",   "must",    "Best open-source code model. Handles repo-scale tasks.")
            add("deepseek-r1:70b",     "strong",  "70B reasoning model with visible chain-of-thought.")
            add("qwen2.5:72b",         "strong",  "Multilingual 72B — strong across languages and domains.")
            add("deepseek-r1:32b",     "strong",  "32B reasoning — fast at this pool size.")
            add("codestral:latest",    "strong",  "Mistral's dedicated code model. Excellent fill-in-the-middle.")
            add("llama3.2-vision:11b", "optional","Multimodal — understands images and text together.")
        elif umem >= 64:
            add("qwen2.5-coder:32b",   "must",    "Best open-source code model. Fits well at 64 GB.")
            add("deepseek-r1:32b",     "must",    "32B reasoning model with chain-of-thought.")
            add("llama3.3:70b",        "optional","70B fits but is tight — slower on base M-chips.")
            add("codestral:latest",    "strong",  "Mistral's dedicated code model. Excellent fill-in-the-middle.")
            add("llama3.2-vision:11b", "strong",  "Multimodal — understands images and text together.")
            add("qwen3:8b",            "optional","Fast 8B for quick tasks when the big model feels like overkill.")
        elif umem >= 32:
            add("qwen2.5-coder:14b",   "must",    "14B code model — great quality, fits well at 32 GB.")
            add("qwen3.5:9b",          "must",    "Best mid-size reasoning model. Strong logic and analysis.")
            add("llama3.2-vision:11b", "strong",  "Multimodal — understands images and text together.")
            add("codestral:latest",    "strong",  "Mistral's dedicated code model. Excellent fill-in-the-middle.")
            add("deepseek-r1:8b",      "optional","8B reasoning model with visible chain-of-thought.")
            add("qwen3:8b",            "optional","Fast 8B for quick tasks.")
            add("deepseek-r1:32b",     "upgrade", "32B reasoning — tight at 32 GB but possible.")
        elif umem >= 16:
            add("qwen3:8b",            "must",    "Best all-around 8B model for 16 GB. Strong reasoning and writing.")
            add("qwen2.5-coder:7b",    "strong",  "Best 7B code model. Great for generation, debugging, and refactoring.")
            add("qwen3.5:9b",          "strong",  "9B reasoning model — fits at 16 GB, stronger than qwen3:8b on logic.")
            add("llama3.2:latest",     "optional","Compact 3B — fast for simple tasks when speed matters.")
            add("deepseek-r1:8b",      "optional","8B reasoning model with visible chain-of-thought.")
            add("llama3.2-vision:11b", "upgrade", "11B multimodal — tight at 16 GB, upgrade RAM for best results.")
            add("codestral:latest",    "upgrade", "12B — too large to run comfortably alongside other models.")
        elif umem >= 8:
            add("llama3.2:latest",     "must",    "Compact 3B — fast and fits comfortably in 8 GB.")
            add("qwen2.5-coder:7b",    "strong",  "7B code model — fits but is snug. Best code option at this size.")
            add("phi3:mini",           "strong",  "3.8B — strong reasoning for its size, fast on 8 GB.")
            add("qwen3:8b",            "optional","8B — fits if it's the only model loaded.")
            add("qwen2.5:1.5b",        "optional","1.5B — instant responses, minimal memory.")
            add("qwen3.5:9b",          "upgrade", "9B — just over budget. Upgrade RAM for this one.")
        else:
            add("llama3.2:3b",         "must",    "3B is the best fit under 8 GB unified memory.")
            add("phi3:mini",           "strong",  "3.8B — strong reasoning per GB.")
            add("qwen2.5:1.5b",        "optional","1.5B for fast, low-memory completions.")
            add("qwen2.5:0.5b",        "optional","Tiny fallback. Useful only for the simplest tasks.")

    # ── NVIDIA CUDA ───────────────────────────────────────────────────────
    elif profile.has_cuda and profile.gpu_vram_gb > 0:
        vram = profile.gpu_vram_gb

        if vram >= 80:
            add("llama3.3:70b",        "must",    f"{vram:.0f} GB VRAM — 70B runs at full precision.")
            add("qwen2.5-coder:32b",   "strong",  "Best open-source code model.")
            add("deepseek-r1:70b",     "strong",  "70B reasoning model with visible chain-of-thought.")
        elif vram >= 48:
            add("deepseek-r1:32b",     "must",    f"{vram:.0f} GB VRAM — 32B reasoning fits well.")
            add("qwen2.5-coder:32b",   "must",    "Best open-source code model at this tier.")
            add("llama3.3:70b",        "optional","70B — tight but fits with Q4 quantization.")
        elif vram >= 24:
            add("qwen2.5-coder:14b",   "must",    f"{vram:.0f} GB VRAM — 14B code model runs well.")
            add("qwen3.5:9b",          "must",    "Best mid-size reasoning model. Fits comfortably.")
            add("deepseek-r1:32b",     "strong",  "32B reasoning at Q4 — fits at this VRAM.")
            add("codestral:latest",    "strong",  "Mistral's dedicated code model. Excellent fill-in-the-middle.")
            add("llama3.2-vision:11b", "optional","Multimodal — understands images and text together.")
        elif vram >= 16:
            add("qwen3.5:9b",          "must",    f"{vram:.0f} GB VRAM — strong reasoning model, fits well.")
            add("qwen2.5-coder:7b",    "must",    "Best 7B code model for generation and debugging.")
            add("qwen2.5-coder:14b",   "strong",  "14B code — fits at 16 GB VRAM for higher quality.")
            add("deepseek-r1:8b",      "optional","8B reasoning model with visible chain-of-thought.")
            add("llama3.2-vision:11b", "optional","Multimodal — fits at 16 GB.")
        elif vram >= 10:
            add("qwen2.5-coder:7b",    "must",    f"{vram:.0f} GB VRAM — 7B code model fits well.")
            add("qwen3.5:9b",          "must",    "9B reasoning — fits snugly at this VRAM.")
            add("qwen3:8b",            "strong",  "8B general model — fits if not running alongside the 9B.")
            add("mistral:7b",          "optional","Strong instruction-following 7B alternative.")
        elif vram >= 6:
            add("qwen2.5-coder:7b",    "must",    f"{vram:.0f} GB VRAM — 7B at Q4 fits.")
            add("phi3:mini",           "strong",  "3.8B — fast and capable for this VRAM budget.")
            add("llama3.2:latest",     "optional","Compact 3B for quick tasks.")
            add("mistral:7b",          "optional","7B — tight at 6 GB.")
        else:
            add("llama3.2:3b",         "must",    f"{vram:.0f} GB VRAM — 3B is the safe choice.")
            add("phi3:mini",           "strong",  "3.8B — may fit depending on quantization.")
            add("qwen2.5:1.5b",        "optional","1.5B as a fast fallback.")

    # ── CPU-only (no GPU / no Apple Silicon) ─────────────────────────────
    else:
        ram = profile.ram_gb
        if ram >= 128:
            add("qwen2.5-coder:14b",   "must",    f"{ram:.0f} GB RAM — 14B runs at reasonable CPU speed.")
            add("deepseek-r1:32b",     "must",    "32B reasoning — usable on high-RAM CPU, just slow.")
            add("qwen3.5:9b",          "strong",  "9B reasoning model — good balance of quality and CPU speed.")
            add("llama3.3:70b",        "optional","70B — possible but very slow on CPU.")
        elif ram >= 64:
            add("qwen2.5-coder:14b",   "must",    f"{ram:.0f} GB RAM — 14B fits and runs at usable speed.")
            add("qwen3.5:9b",          "must",    "9B reasoning — good CPU performance.")
            add("qwen3:8b",            "strong",  "8B general model — faster than the 9B on CPU.")
            add("deepseek-r1:32b",     "optional","32B — slow on CPU but possible at 64 GB.")
        elif ram >= 32:
            add("qwen3.5:9b",          "must",    f"{ram:.0f} GB RAM — 9B is the sweet spot for CPU.")
            add("qwen2.5-coder:7b",    "must",    "7B code model — good CPU speed.")
            add("qwen3:8b",            "optional","8B alternative — similar speed, different strengths.")
            add("mistral:7b",          "optional","7B with strong instruction following.")
        elif ram >= 16:
            add("phi3:mini",           "must",    f"{ram:.0f} GB RAM — 3.8B is the best quality for CPU-only.")
            add("qwen2.5-coder:7b",    "strong",  "7B code — fits but is slow on CPU.")
            add("llama3.2:latest",     "strong",  "Compact 3B — fast on CPU.")
            add("qwen3.5:9b",          "upgrade", "9B — needs more RAM for comfortable CPU use.")
        else:
            add("phi3:mini",           "must",    f"{ram:.0f} GB RAM — 3.8B is the ceiling for CPU-only.")
            add("llama3.2:3b",         "strong",  "3B fast model.")
            add("qwen2.5:1.5b",        "optional","1.5B for instant responses.")
            add("qwen2.5:0.5b",        "optional","Tiny but functional.")

    return recs


# ── Ollama integration ────────────────────────────────────────────────────────

def _normalize_model_name(name: str) -> str:
    """Strip :latest tag so 'nomic-embed-text:latest' == 'nomic-embed-text'."""
    if name.endswith(":latest"):
        return name[: -len(":latest")]
    return name


def get_installed_models() -> List[str]:
    """Return list of already-pulled model names (normalized, no :latest suffix)."""
    if not shutil.which("ollama"):
        return []
    out = _run(["ollama", "list"])
    models = []
    for line in out.splitlines()[1:]:  # skip header
        parts = line.split()
        if parts:
            models.append(_normalize_model_name(parts[0]))
    return models


def ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def pull_model(name: str) -> bool:
    print(f"  {cyan('ollama pull')} {name} ...")
    result = subprocess.run(["ollama", "pull", name])
    return result.returncode == 0


# ── Display ───────────────────────────────────────────────────────────────────

def print_profile(profile: HardwareProfile):
    print(bold("\n━━━  Hardware Profile  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
    print(f"  OS:      {profile.os_name} {profile.os_version[:40]}")
    print(f"  CPU:     {profile.cpu_brand or 'Unknown'}")
    print(f"  Cores:   {profile.cpu_cores_physical}P / {profile.cpu_cores_logical}L")

    if profile.is_apple_silicon:
        print(f"  Chip:    {bold(profile.apple_chip)}")
        print(f"  Memory:  {bold(f'{profile.unified_memory_gb:.0f} GB')} unified  "
              f"{dim(f'(effective pool ~{profile.effective_pool_gb:.0f} GB)')}")
        print(f"  Backend: Metal {green('✓')}")
    else:
        print(f"  RAM:     {profile.ram_gb:.1f} GB  "
              f"{dim(f'(effective pool ~{profile.effective_pool_gb:.0f} GB)')}")
        if profile.gpu_name:
            print(f"  GPU:     {profile.gpu_name}")
            if profile.gpu_vram_gb:
                print(f"  VRAM:    {bold(f'{profile.gpu_vram_gb:.1f} GB')}")
            if profile.has_cuda:
                print(f"  Backend: CUDA {green('✓')}")

    for note in profile.notes:
        print(f"\n  {dim('ℹ')}  {dim(note)}")


def print_recommendations(recs: List[Recommendation], installed: List[str]):
    # Group by tier
    order = ["must", "strong", "optional", "upgrade"]
    groups = {t: [] for t in order}
    for r in recs:
        if r.tier in groups:
            groups[r.tier].append(r)

    print(bold("\n━━━  Model Recommendations  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))

    for tier in order:
        tier_recs = groups[tier]
        if not tier_recs:
            continue
        icon, label = TIER_LABELS[tier]
        print(f"\n  {bold(icon + '  ' + label)}")
        for rec in tier_recs:
            already = _normalize_model_name(rec.model.name) in installed
            status  = green(" ✓ installed") if already else ""
            size    = dim(f"~{rec.model.size_gb:.1f} GB")
            cat     = dim(f"[{rec.model.category}]")
            print(f"    {bold(rec.model.name):<36} {size}  {cat}{status}")
            print(f"      {rec.reason}")
            if not already:
                print(f"      {dim(rec.model.pull_cmd)}")


def print_ollama_status(installed: List[str]):
    print(bold("\n━━━  Ollama Status  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
    if not shutil.which("ollama"):
        print(f"  {red('✗')} Ollama not found in PATH.  Install: {cyan('https://ollama.com')}")
        return
    if not ollama_running():
        print(f"  {yellow('⚠')} Ollama installed but not running.  Start: {cyan('ollama serve')}")
    else:
        print(f"  {green('✓')} Ollama is running at localhost:11434")

    if installed:
        print(f"\n  Already installed ({len(installed)} models):")
        for m in installed:
            print(f"    {green('·')} {m}")
    else:
        print(f"  {dim('No models installed yet.')}")


def print_quick_start(recs: List[Recommendation], installed: List[str]):
    to_pull = [r for r in recs if r.tier in ("must", "strong") and _normalize_model_name(r.model.name) not in installed]
    if not to_pull:
        print(bold("\n━━━  You're all set!  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
        print(f"  {green('✓')} All recommended models are already installed.")
        return

    print(bold("\n━━━  Quick Start  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
    print(f"  Pull all essential + strongly recommended models:\n")
    for r in to_pull:
        print(f"    {cyan(r.model.pull_cmd)}")
    print(f"\n  Or run this script with {bold('--pull')} to pull them automatically.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan hardware and recommend Ollama models."
    )
    parser.add_argument("--json",  action="store_true", help="Output JSON instead of formatted text")
    parser.add_argument("--pull",  action="store_true", help="Auto-pull all must/strong recommendations")
    parser.add_argument("--all",   action="store_true", help="Include optional and upgrade tiers in --pull")
    args = parser.parse_args()

    profile   = scan_hardware()
    recs      = recommend(profile)
    installed = get_installed_models()

    if args.json:
        out = {
            "hardware": {
                "os":                   profile.os_name,
                "cpu":                  profile.cpu_brand,
                "cores_physical":       profile.cpu_cores_physical,
                "cores_logical":        profile.cpu_cores_logical,
                "ram_gb":               profile.ram_gb,
                "is_apple_silicon":     profile.is_apple_silicon,
                "apple_chip":           profile.apple_chip,
                "unified_memory_gb":    profile.unified_memory_gb,
                "effective_pool_gb":    profile.effective_pool_gb,
                "gpu":                  profile.gpu_name,
                "gpu_vram_gb":          profile.gpu_vram_gb,
                "has_cuda":             profile.has_cuda,
            },
            "installed": installed,
            "recommendations": [
                {
                    "model":       r.model.name,
                    "tier":        r.tier,
                    "size_gb":     r.model.size_gb,
                    "category":    r.model.category,
                    "reason":      r.reason,
                    "pull_cmd":    r.model.pull_cmd,
                    "installed":   _normalize_model_name(r.model.name) in installed,
                }
                for r in recs
            ],
        }
        print(json.dumps(out, indent=2))
        return

    print(bold(cyan("\n⛅  Ollama Model Recommender")))
    print(dim("   Scanning hardware...\n"))
    print_profile(profile)
    print_ollama_status(installed)
    print_recommendations(recs, installed)
    print_quick_start(recs, installed)
    print()

    if args.pull:
        pull_tiers = {"must", "strong"}
        if args.all:
            pull_tiers.update({"optional"})
        to_pull = [r for r in recs
                   if r.tier in pull_tiers and r.model.name not in installed]
        if not to_pull:
            print(green("  ✓  Nothing new to pull."))
            return
        if not shutil.which("ollama"):
            print(red("  ✗  Ollama not installed — can't pull."))
            return
        print(bold(f"\n  Pulling {len(to_pull)} model(s)...\n"))
        failed = []
        for r in to_pull:
            ok = pull_model(r.model.name)
            if not ok:
                failed.append(r.model.name)
        if failed:
            print(red(f"\n  ✗  Failed to pull: {', '.join(failed)}"))
        else:
            print(green(f"\n  ✓  All models pulled successfully."))


if __name__ == "__main__":
    main()
