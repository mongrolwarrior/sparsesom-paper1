"""Run 8 — Full-corpus efficiency sweep (supersedes Table 7 / Fig 5, + S1)."""

from pathlib import Path

import pandas as pd

from ..runner import (
    run_sparsesom, run_standardsparsesom, run_medsom,
)


EDGES = [32, 64, 128, 256, 512]
SEEDS = [0, 1, 2, 3, 4]


def run(data_dir: str, results_dir: str, edges=None, seeds=None, **kwargs) -> Path:
    """Run the efficiency sweep: all implementations up the edge-doubling ladder."""
    data = Path(data_dir)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = str(data / "corpus.train.sbcsr")
    pca_file = str(data / "corpus.sompca")

    edges = edges or EDGES
    seeds = seeds or SEEDS
    fixed_epochs = 20

    rows = []

    seedable_impls = [
        ("sbsom-bin", True),
        ("sbsom-float", False),
        ("ssom-feat", None),
        ("ssom-node", None),
    ]

    for impl_name, binary in seedable_impls:
        dropped = False
        for edge in edges:
            if dropped:
                break

            for seed in seeds:
                print(f"  efficiency: edge={edge} impl={impl_name} seed={seed} ...")

                try:
                    result = _run_seedable(
                        impl_name, binary, corpus, pca_file,
                        edge, seed, fixed_epochs,
                    )
                except RuntimeError as e:
                    if "OOM" in str(e) or "out of memory" in str(e).lower():
                        result = {"oom": True}
                    else:
                        raise

                row = _build_row(impl_name, edge, seed, fixed_epochs, result)
                rows.append(row)

                if result.get("oom"):
                    dropped = True
                    break

    medsom_scratch = str(out_dir / "medsom_scratch")
    Path(medsom_scratch).mkdir(exist_ok=True)
    medsom_dropped = False
    for edge in edges:
        if medsom_dropped:
            break

        print(f"  efficiency: edge={edge} impl=medsom seed=42 (hardcoded) ...")

        try:
            result = run_medsom(
                corpus, edge, epochs=fixed_epochs,
                scratch_dir=medsom_scratch,
            )
        except RuntimeError as e:
            if "OOM" in str(e) or "out of memory" in str(e).lower():
                result = {"oom": True}
            else:
                raise

        row = {
            "edge": edge,
            "impl": "medsom",
            "precision": "fp32",
            "seed": 42,
            "epochs": fixed_epochs,
            "oom": result.get("oom", False),
        }

        if result.get("oom"):
            row["dropout_edge"] = edge
            medsom_dropped = True
        else:
            ep_data = result.get("epochs", [])
            total_time = sum(e.get("epoch_time_s", 0) for e in ep_data)
            row.update({
                "bmu_s_per_ep": None,
                "upd_s_per_ep": None,
                "total_wall_s": total_time,
                "dropout_edge": None,
            })

        rows.append(row)

    df = pd.DataFrame(rows)
    out = out_dir / "efficiency_sweep.parquet"
    df.to_parquet(out, index=False)

    print(f"\nEfficiency sweep: {out} ({len(df)} rows)")
    return out


def _run_seedable(impl_name: str, binary: bool | None, corpus: str,
                  pca_file: str, edge: int, seed: int, epochs: int) -> dict:
    if impl_name.startswith("sbsom"):
        return run_sparsesom(
            corpus, edge, seed,
            binary=(binary is True),
            pca_init=pca_file,
            extra_flags=["--epochs", str(epochs)],
        )
    elif impl_name in ("ssom-feat", "ssom-node"):
        layout = "feature" if impl_name == "ssom-feat" else "node"
        return run_standardsparsesom(
            corpus, edge, seed,
            layout=layout,
            neighbourhood="box",
            stop="fixed",
            precision="fp16",
            extra_flags=["--epochs", str(epochs)],
        )
    else:
        raise ValueError(f"Unknown impl: {impl_name}")


def _build_row(impl_name, edge, seed, fixed_epochs, result):
    row = {
        "edge": edge,
        "impl": impl_name,
        "precision": "fp16",
        "seed": seed,
        "epochs": fixed_epochs,
        "oom": result.get("oom", False),
    }
    if result.get("oom"):
        row["dropout_edge"] = edge
    else:
        done = result.get("done", {}) or {}
        epochs = done.get("epochs", fixed_epochs)
        row.update({
            "bmu_s_per_ep": done.get("bmu_s", 0) / max(epochs, 1),
            "upd_s_per_ep": done.get("update_s", 0) / max(epochs, 1),
            "total_wall_s": done.get("wall_s", 0),
            "dropout_edge": None,
        })
    return row
