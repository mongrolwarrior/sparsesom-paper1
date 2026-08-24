"""Run 10 — Roofline: whole-BMU-phase DRAM traffic + bandwidth (§5.7).

RTX 4090, out-of-CI (requires NVIDIA Nsight Compute, `ncu`). Captures every
kernel in the BMU phase — norms + csrmm + argmax for the cuSPARSE baselines,
the fused kernel for sbsom — at edge 128 (exact) and edge 256 (sampled via
NCU_CAP, per-tile DRAM being uniform), then aggregates to
roofline_phase_summary.csv and roofline_phase_breakdown.csv. Reuses the
tested standalone scripts rather than reimplementing the ncu invocation.

Skips gracefully (no failure) if ncu is absent, since the rest of the
reproduction does not depend on it.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(data_dir: str, results_dir: str, **kwargs) -> Path:
    data, out = Path(data_dir), Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = str(data / "corpus.train.sbcsr")
    scripts = Path(__file__).resolve().parent.parent.parent / "scripts"
    summary = out / "roofline_phase_summary.csv"

    if not shutil.which(os.environ.get("NCU", "ncu")):
        print("  roofline: Nsight Compute (ncu) not found — SKIPPED (out-of-CI; "
              "§5.7 requires ncu). Rest of the reproduction is unaffected.")
        return summary

    env = {**os.environ, "OUTDIR": str(out)}
    print("  roofline: edge 128 (exact, all impls) ...")
    subprocess.run(["bash", str(scripts / "ncu_full_epoch.sh"), "128", corpus],
                   check=True, cwd=scripts.parent, env=env)
    print("  roofline: edge 256 (sampled, NCU_CAP=60) ...")
    subprocess.run(["bash", str(scripts / "ncu_full_epoch.sh"), "256", corpus],
                   check=True, cwd=scripts.parent, env={**env, "NCU_CAP": "60"})
    subprocess.run([sys.executable, str(scripts / "aggregate_phase.py"), str(out)],
                   check=True)
    print(f"\nRoofline: {summary}")
    return summary
