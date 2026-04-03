#!/usr/bin/env python3
"""
Scans your hardware and recommends the best local model for Claude Code / Rain.
Usage: python3 recommend_model.py
"""

import subprocess
import shutil
import json


def get_hardware():
    info = {"ram_gb": 0, "chip": "unknown", "cores": 0, "disk_free_gb": 0}

    # RAM and chip (macOS)
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True
        )
        for line in out.splitlines():
            if "Memory:" in line:
                parts = line.strip().split()
                info["ram_gb"] = int(parts[1])
            elif "Chip:" in line:
                info["chip"] = line.split("Chip:")[-1].strip()
            elif "Total Number of Cores:" in line:
                info["cores"] = int(line.split(":")[1].strip().split()[0])
    except Exception:
        # Linux fallback
        try:
            out = subprocess.check_output(["free", "-g"], text=True)
            info["ram_gb"] = int(out.splitlines()[1].split()[1])
            info["chip"] = "x86/unknown"
        except Exception:
            pass

    # Free disk space
    try:
        out = subprocess.check_output(["df", "-BG", "/"], text=True)
        info["disk_free_gb"] = int(out.splitlines()[1].split()[3].replace("G", ""))
    except Exception:
        try:
            import shutil as sh
            info["disk_free_gb"] = sh.disk_usage("/").free // (1024**3)
        except Exception:
            pass

    return info


def get_installed_ollama_models():
    if not shutil.which("ollama"):
        return []
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        models = []
        for line in out.splitlines()[1:]:  # skip header
            if line.strip():
                models.append(line.split()[0])
        return models
    except Exception:
        return []


def recommend(ram_gb, chip):
    is_apple_silicon = "apple" in chip.lower() or any(
        x in chip.lower() for x in ["m1", "m2", "m3", "m4"]
    )

    # Apple Silicon unified memory is more efficient than discrete GPU VRAM
    # Rule of thumb: model params in GB ≈ param_count * 2 bytes (Q4 quant ~= param_count * 0.5 GB)
    # Safe usable RAM for model = total * 0.75 (leave headroom for OS + context)
    usable = ram_gb * 0.75

    tiers = [
        {
            "min_ram": 64,
            "model": "qwen2.5:72b",
            "size_gb": 45,
            "quality": "Near-frontier",
            "notes": "Best local reasoning available. Approaches GPT-4 class on many tasks.",
        },
        {
            "min_ram": 32,
            "model": "qwen2.5:32b",
            "size_gb": 20,
            "quality": "Excellent",
            "notes": "Strong reasoning, low hallucination. Best bang for buck at this tier.",
        },
        {
            "min_ram": 24,
            "model": "mistral-small:22b",
            "size_gb": 14,
            "quality": "Very good",
            "notes": "Great at instruction following. Good alternative to qwen2.5:32b.",
        },
        {
            "min_ram": 16,
            "model": "qwen2.5:14b",
            "size_gb": 9,
            "quality": "Good",
            "notes": "Solid upgrade over 9B. Noticeably better logic and less hallucination.",
        },
        {
            "min_ram": 8,
            "model": "qwen2.5-coder:7b",
            "size_gb": 5,
            "quality": "Decent",
            "notes": "Best 7B for coding tasks. Weaker on open-ended reasoning.",
        },
        {
            "min_ram": 4,
            "model": "llama3.2:3b",
            "size_gb": 2,
            "quality": "Basic",
            "notes": "Fast, lightweight. Fine for simple tasks, struggles with complex reasoning.",
        },
    ]

    best = None
    for tier in tiers:
        if ram_gb >= tier["min_ram"]:
            best = tier
            break

    return best, tiers


def main():
    print("\n🔍 Scanning hardware...\n")
    hw = get_hardware()
    models = get_installed_ollama_models()

    print(f"  Chip:        {hw['chip']}")
    print(f"  RAM:         {hw['ram_gb']} GB")
    print(f"  CPU Cores:   {hw['cores']}")
    print(f"  Free Disk:   {hw['disk_free_gb']} GB")
    print(f"  Ollama:      {'installed ✓' if shutil.which('ollama') else 'not found ✗'}")

    if models:
        print(f"\n  Installed models:")
        for m in models:
            print(f"    • {m}")

    print()
    best, tiers = recommend(hw["ram_gb"], hw["chip"])

    if not best:
        print("⚠️  Less than 4GB RAM — local models not recommended.\n")
        return

    print("─" * 55)
    print(f"  ✅ RECOMMENDED:  {best['model']}")
    print(f"  Quality:         {best['quality']}")
    print(f"  Download size:   ~{best['size_gb']} GB")
    print(f"  Notes:           {best['notes']}")
    print("─" * 55)

    # Check if already installed
    installed_names = [m.split(":")[0] for m in models]
    rec_name = best["model"].split(":")[0]
    if rec_name in installed_names:
        print(f"\n  ✓ Already installed! Run with: ollama run {best['model']}")
    else:
        if hw["disk_free_gb"] < best["size_gb"] + 5:
            print(f"\n  ⚠️  Low disk space — need ~{best['size_gb'] + 5}GB free, have {hw['disk_free_gb']}GB")
        else:
            print(f"\n  Install with:  ollama pull {best['model']}")
            print(f"  Use in Rain:   update LOGIC/DOMAIN agent to '{best['model']}'")

    # Show full tier table
    print("\n  Full tier table for your system:\n")
    print(f"  {'Model':<25} {'RAM needed':<12} {'Quality':<15} {'Size'}")
    print(f"  {'─'*25} {'─'*12} {'─'*15} {'─'*8}")
    for tier in tiers:
        fits = "✓" if hw["ram_gb"] >= tier["min_ram"] else "✗"
        arrow = " ◀ you" if tier == best else ""
        print(f"  {fits} {tier['model']:<23} {str(tier['min_ram'])+'GB':<12} {tier['quality']:<15} ~{tier['size_gb']}GB{arrow}")

    print()


if __name__ == "__main__":
    main()
