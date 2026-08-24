"""Run 11 — somoclu CPU baseline for the efficiency comparison (§5.5).

Times somoclu's sparse CPU kernel (the best available CPU SOM library) on the
full corpus at edges 32/64/128/256 and appends somoclu rows to
efficiency_sweep.parquet, so §5.5 can report the CPU-vs-GPU ratio across map
sizes. CPU/OpenMP only — no GPU. Must run after ex08 (which creates the
efficiency parquet the rows are appended to).

Builds the somoclu CPU binary from the submodule if it is not already present
(autotools may be absent; the build script compiles it directly with g++).
"""

import os
import subprocess
import sys
from pathlib import Path


def _ensure_binary(repo_root: Path) -> str:
    binary = os.environ.get(
        "SOMOCLU", str(repo_root / "external/somoclu/somoclu_cpu"))
    if not Path(binary).exists():
        build = repo_root / "scripts/somoclu_build/build_cpu.sh"
        print("  somoclu: binary missing — building CPU kernel ...")
        subprocess.run(["bash", str(build)], check=True)
    return binary


def run(data_dir: str, results_dir: str, edges=None, **kwargs) -> Path:
    data, out = Path(data_dir), Path(results_dir)
    corpus = str(data / "corpus.train.sbcsr")
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts = repo_root / "scripts"

    eff = out / "efficiency_sweep.parquet"
    if not eff.exists():
        print("  somoclu: efficiency_sweep.parquet not found — run ex08 first; "
              "SKIPPED.")
        return eff

    binary = _ensure_binary(repo_root)
    edge_arg = ",".join(str(e) for e in (edges or [32, 64, 128, 256]))
    print(f"  somoclu: edges {edge_arg} (CPU, full corpus) ...")
    subprocess.run(
        [sys.executable, str(scripts / "run_somoclu_anchor.py"),
         str(out), corpus, "--edges", edge_arg],
        check=True, env={**os.environ, "SOMOCLU": binary})
    return eff
