"""Run 5 — Master size sweep, power law / no elbow (Table 8, Fig 6, Fig 4)."""

from pathlib import Path

import pandas as pd

from ..runner import run_sparsesom


EDGES = [32, 64, 128, 256, 512]
SEEDS = [0, 1, 2, 3, 4]


def run(data_dir: str, results_dir: str, edges=None, seeds=None, **kwargs) -> Path:
    """Run the edge-doubling size sweep."""
    data = Path(data_dir)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = str(data / "corpus.train.sbcsr")
    eval_corpus = str(data / "corpus.heldout.sbcsr")
    pca_file = str(data / "corpus.sompca")

    sweep_rows = []
    epoch_rows = []

    for edge in (edges or EDGES):
        for seed in (seeds or SEEDS):
            print(f"  size_sweep: edge={edge} seed={seed} ...")

            r = run_sparsesom(
                corpus, edge, seed,
                binary=True,
                eval_corpus=eval_corpus,
                pca_init=pca_file,
            )

            done = r["done"] or {}
            metrics = r["metrics"] or {}

            sweep_rows.append({
                "edge": edge,
                "neurons": edge * edge,
                "seed": seed,
                "epochs": done.get("epochs", 0),
                "wall_s": done.get("wall_s", 0),
                "qe_heldout": metrics.get("qe_cosine", None),
                "qe_eucl": metrics.get("qe_euclidean", None),
                "te": metrics.get("topographic_error", None),
                "dead_frac": metrics.get("dead_fraction", None),
                "converged": done.get("converged", False),
            })

            bmu_per_epoch = done.get("bmu_s", 0) / max(done.get("epochs", 1), 1)
            update_per_epoch = done.get("update_s", 0) / max(done.get("epochs", 1), 1)

            for ep in r["epochs"]:
                epoch_rows.append({
                    "edge": edge,
                    "seed": seed,
                    "epoch": ep["epoch"],
                    "sigma": ep["sigma"],
                    "qe_heldout": ep["qe"],
                    "bmu_s": bmu_per_epoch,
                    "update_s": update_per_epoch,
                })

    df_sweep = pd.DataFrame(sweep_rows)
    df_epochs = pd.DataFrame(epoch_rows)

    out_sweep = out_dir / "size_sweep.parquet"
    out_epochs = out_dir / "size_sweep_epochs.parquet"

    df_sweep.to_parquet(out_sweep, index=False)
    df_epochs.to_parquet(out_epochs, index=False)

    print(f"\nSize sweep: {out_sweep} ({len(df_sweep)} rows)")
    print(f"Epoch data: {out_epochs} ({len(df_epochs)} rows)")
    _print_summary(df_sweep)
    return out_sweep


def _print_summary(df: pd.DataFrame):
    summary = df.groupby("edge").agg(
        qe_mean=("qe_heldout", "mean"),
        wall_mean=("wall_s", "mean"),
        epochs_mean=("epochs", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))
