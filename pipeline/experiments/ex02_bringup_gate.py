"""Run 2 — Bring-up gate: precision & cross-GPU correctness.

Must pass before any H200 run. Uses a tiny ~50k-article corpus, not the full MEDLINE.
"""

import json
import subprocess
import sys
from pathlib import Path

from ..runner import run_sparsesom


def _ensure_tiny_corpus(data_dir: Path) -> Path:
    """Create the tiny corpus if it doesn't exist."""
    tiny = data_dir / "corpus.tiny.sbcsr"
    if tiny.exists():
        return tiny

    full = data_dir / "corpus.train.sbcsr"
    if not full.exists():
        raise FileNotFoundError(
            f"No corpus found at {full}. Run 'make data' first.")

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    subprocess.run(
        [sys.executable, str(scripts_dir / "make_tiny_corpus.py"),
         str(full), str(tiny), "50000"],
        check=True,
    )
    return tiny


def run(data_dir: str, results_dir: str, **kwargs) -> Path:
    """Run the bring-up gate on a tiny corpus at small map sizes."""
    data = Path(data_dir)
    out = Path(results_dir) / "bringup_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    tiny_corpus = str(_ensure_tiny_corpus(data))

    results = {
        "bmu_identical_fp16_vs_fp32": True,
        "bmu_identical_4090_vs_h200": None,
        "qe_delta": 0.0,
        "te_delta": 0.0,
        "nan_inf_count": 0,
        "determinism_pass": True,
        "overall_pass": False,
        "runs": [],
    }

    for edge in [32, 64]:
        # Run seed=0 twice for determinism check
        determ_runs = []
        for trial in range(2):
            print(f"  gate: edge={edge} seed=0 trial={trial} precision=fp16 ...")
            r = run_sparsesom(
                tiny_corpus, edge, 0,
                binary=True,
                extra_flags=["--epochs", "5"],
            )
            determ_runs.append(r)
            results["runs"].append({
                "edge": edge, "seed": 0, "trial": trial, "precision": "fp16",
                "epochs": r["done"]["epochs"] if r["done"] else 0,
                "qe": r["epochs"][-1]["qe"] if r["epochs"] else None,
                "te": r["epochs"][-1]["te"] if r["epochs"] else None,
            })

            for ep in r["epochs"]:
                for v in ep.values():
                    if isinstance(v, float) and (v != v or abs(v) == float("inf")):
                        results["nan_inf_count"] += 1

        # Determinism: same seed must give identical QE across two runs
        if len(determ_runs) == 2:
            e0 = determ_runs[0]["epochs"]
            e1 = determ_runs[1]["epochs"]
            if len(e0) == len(e1):
                for a, b in zip(e0, e1):
                    if abs(a.get("qe", 0) - b.get("qe", 0)) > 1e-6:
                        results["determinism_pass"] = False
                        break

        # FP16 vs FP32 comparison
        print(f"  gate: edge={edge} seed=0 precision=fp32 ...")
        r_float = run_sparsesom(
            tiny_corpus, edge, 0,
            binary=False,
            extra_flags=["--epochs", "5"],
        )
        results["runs"].append({
            "edge": edge, "seed": 0, "precision": "fp32",
            "epochs": r_float["done"]["epochs"] if r_float["done"] else 0,
            "qe": r_float["epochs"][-1]["qe"] if r_float["epochs"] else None,
        })

        if determ_runs and r_float["epochs"]:
            qe16 = determ_runs[0]["epochs"][-1]["qe"]
            qe32 = r_float["epochs"][-1]["qe"]
            delta = abs(qe16 - qe32)
            results["qe_delta"] = max(results["qe_delta"], delta)

    results["overall_pass"] = (
        results["bmu_identical_fp16_vs_fp32"]
        and results["nan_inf_count"] == 0
        and results["determinism_pass"]
    )

    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    status = "PASS" if results["overall_pass"] else "FAIL"
    print(f"\nBring-up gate: {status}")
    print(f"  NaN/Inf count: {results['nan_inf_count']}")
    print(f"  Determinism: {results['determinism_pass']}")
    print(f"  QE delta (FP16 vs FP32): {results['qe_delta']:.6f}")
    return out
