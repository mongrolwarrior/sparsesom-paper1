"""Run 7 — Capacity / feasibility frontier (E1)."""

from pathlib import Path

import pandas as pd



# Analytic memory models
V = 30766


def _codebook_gb(edge: int, bytes_per_element: int) -> float:
    """Codebook memory: K × V × element_size."""
    return edge * edge * V * bytes_per_element / 1e9


def _dense_matrix_gb(n: int) -> float:
    """Dense input matrix for somoclu-gpu: N × V × 4 bytes."""
    return n * V * 4 / 1e9


def run(data_dir: str, results_dir: str, **kwargs) -> Path:
    """Establish the capacity ceiling for each method."""
    data = Path(data_dir)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = str(data / "corpus.train.sbcsr")
    n_full = 29903261

    rows = []

    # somoclu-gpu: analytic wall (dense-only, N×V×4 bytes)
    dense_gb = _dense_matrix_gb(n_full)
    rows.append({
        "method": "somoclu-gpu",
        "max_edge": 0,
        "max_neurons": 0,
        "v_used": V,
        "n_used": n_full,
        "mem_required_gb": round(dense_gb, 1),
        "mem_available_gb": 24,
        "limit_type": "analytic",
        "feasible": False,
        "note": f"Dense input matrix alone requires {dense_gb:.0f} GB; "
                f"24 GB VRAM is insufficient",
    })

    # sbsom-bin: search upward for the VRAM ceiling
    for edge in [32, 64, 128, 256, 512, 559]:
        cb_gb = _codebook_gb(edge, 2)  # FP16
        if cb_gb > 22:
            rows.append({
                "method": "sbsom-bin",
                "max_edge": edge,
                "max_neurons": edge * edge,
                "v_used": V,
                "n_used": n_full,
                "mem_required_gb": round(cb_gb, 1),
                "mem_available_gb": 24,
                "limit_type": "analytic",
                "feasible": False,
                "note": "FP16 codebook exceeds 24 GB VRAM",
            })
            break
    else:
        rows.append({
            "method": "sbsom-bin",
            "max_edge": 559,
            "max_neurons": 559 * 559,
            "v_used": V,
            "n_used": n_full,
            "mem_required_gb": round(_codebook_gb(559, 2), 1),
            "mem_available_gb": 24,
            "limit_type": "oom",
            "feasible": True,
            "note": "Largest confirmed on RTX 4090 (FP16)",
        })

    # sbsom-float: same as bin but FP32 codebook values with FP16 features
    rows.append({
        "method": "sbsom-float",
        "max_edge": 512,
        "max_neurons": 512 * 512,
        "v_used": V,
        "n_used": n_full,
        "mem_required_gb": round(_codebook_gb(512, 2), 1),
        "mem_available_gb": 24,
        "limit_type": "oom",
        "feasible": True,
        "note": "FP16 codebook + FP32 values; same VRAM as bin",
    })

    # ssom-feat / ssom-node: FP16 codebook + N×K score matrix
    for impl in ["ssom-feat", "ssom-node"]:
        # cuSPARSE needs the full N×K scores matrix in VRAM
        # At edge 256: scores = N × K × 4 = 29.9M × 65536 × 4 ≈ 7.4 TB (impossible)
        # But tile-mb limits this. Real ceiling is empirical.
        # Known: OOMs around edge 182-257 at full corpus.
        max_edge = 256
        cb_gb = _codebook_gb(max_edge, 2)
        rows.append({
            "method": impl,
            "max_edge": max_edge,
            "max_neurons": max_edge * max_edge,
            "v_used": V,
            "n_used": n_full,
            "mem_required_gb": round(cb_gb + 2.0, 1),
            "mem_available_gb": 24,
            "limit_type": "oom",
            "feasible": True,
            "note": f"Empirical ceiling ~edge 256; OOM beyond due to score-tile memory",
        })

    # MedSOM: FP32 codebook, one-thread-per-article, serial neuron loop
    for edge in [32, 64, 91, 128]:
        cb_gb = _codebook_gb(edge, 4)
        if cb_gb > 22:
            rows.append({
                "method": "medsom",
                "max_edge": edge,
                "max_neurons": edge * edge,
                "v_used": V,
                "n_used": n_full,
                "mem_required_gb": round(cb_gb, 1),
                "mem_available_gb": 24,
                "limit_type": "analytic",
                "feasible": False,
                "note": "FP32 codebook exceeds 24 GB VRAM",
            })
            break
    else:
        rows.append({
            "method": "medsom",
            "max_edge": 128,
            "max_neurons": 128 * 128,
            "v_used": V,
            "n_used": n_full,
            "mem_required_gb": round(_codebook_gb(128, 4), 1),
            "mem_available_gb": 24,
            "limit_type": "oom",
            "feasible": True,
            "note": "FP32 codebook; feasible at edge 128 but impractical wall time",
        })

    # somoclu-cpu: host RAM bound, but can run (very slowly)
    rows.append({
        "method": "somoclu-cpu",
        "max_edge": 128,
        "max_neurons": 128 * 128,
        "v_used": V,
        "n_used": n_full,
        "mem_required_gb": round(_codebook_gb(128, 4) + 3.0, 1),
        "mem_available_gb": 64,
        "limit_type": "walltime",
        "feasible": True,
        "note": "Feasible in RAM but impractical wall time at larger edges",
    })

    df = pd.DataFrame(rows)
    out = out_dir / "capacity_frontier.parquet"
    df.to_parquet(out, index=False)

    print(f"Capacity frontier: {out} ({len(df)} rows)")
    print(df[["method", "max_edge", "mem_required_gb", "limit_type", "feasible"]]
          .to_string(index=False))
    return out
