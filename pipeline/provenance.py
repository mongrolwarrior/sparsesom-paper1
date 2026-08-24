"""Provenance: capture submodule SHAs, GPU info, driver, CUDA version, timestamps."""

import json
import subprocess
import time
from pathlib import Path


EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "external"

SUBMODULES = [
    "SparseBinarySOM",
    "StandardSparseSOM",
    "SparseBinEval",
    "MedSOM",
]


def submodule_shas() -> dict[str, str]:
    shas = {}
    for name in SUBMODULES:
        p = EXTERNAL_DIR / name
        if not (p / ".git").exists():
            shas[name] = "not-a-git-repo"
            continue
        try:
            r = subprocess.run(
                ["git", "-C", str(p), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            shas[name] = r.stdout.strip()
        except subprocess.CalledProcessError:
            shas[name] = "unknown"
    return shas


def gpu_info() -> dict:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        parts = r.stdout.strip().split(", ")
        return {
            "gpu_name": parts[0],
            "driver_version": parts[1] if len(parts) > 1 else "unknown",
            "vram_mb": int(parts[2]) if len(parts) > 2 else 0,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"gpu_name": "unknown", "driver_version": "unknown", "vram_mb": 0}


def cuda_version() -> str:
    try:
        r = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True,
        )
        for line in r.stdout.splitlines():
            if "release" in line.lower():
                return line.strip()
        return r.stdout.strip().splitlines()[-1]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_provenance(config_hash: str) -> dict:
    return {
        "config_hash": config_hash,
        "submodule_shas": submodule_shas(),
        "gpu": gpu_info(),
        "cuda": cuda_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def save_provenance(prov: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prov, f, indent=2)
