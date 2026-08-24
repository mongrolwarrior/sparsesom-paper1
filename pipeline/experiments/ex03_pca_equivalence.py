"""Run 3 — PCA-init vs random-init equivalence (Table 5, P1 supplementary).

Gates the PCA default for runs 5-8.
"""

from pathlib import Path

import pandas as pd

from ..runner import run_sparsesom


def run(data_dir: str, results_dir: str, **kwargs) -> Path:
    """Run PCA vs random init at edges 32, 64, 128 with 5 seeds."""
    data = Path(data_dir)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = str(data / "corpus.train.sbcsr")
    eval_corpus = str(data / "corpus.heldout.sbcsr")
    pca_file = str(data / "corpus.sompca")

    edges = [32, 64, 128]
    seeds = [0, 1, 2, 3, 4]
    inits = ["pca", "random"]

    equiv_rows = []
    train_rows = []

    for edge in edges:
        for seed in seeds:
            for init in inits:
                print(f"  edge={edge} seed={seed} init={init} ...")

                pca_path = pca_file if init == "pca" else None

                r = run_sparsesom(
                    corpus, edge, seed,
                    binary=True,
                    eval_corpus=eval_corpus,
                    pca_init=pca_path,
                )

                done = r["done"] or {}
                metrics = r["metrics"] or {}

                train_rows.append({
                    "edge": edge,
                    "seed": seed,
                    "init": init,
                    "epochs": done.get("epochs", 0),
                    "wall_s": done.get("wall_s", 0),
                    "qe_eucl": metrics.get("qe_euclidean", None),
                    "te": metrics.get("topographic_error", None),
                    "dead_frac": metrics.get("dead_fraction", None),
                    "converged": done.get("converged", False),
                })

                for ep in r["epochs"]:
                    equiv_rows.append({
                        "init": init,
                        "edge": edge,
                        "seed": seed,
                        "epoch": ep["epoch"],
                        "qe_heldout": ep["qe"],
                        "te": ep.get("te", None),
                        "dead_frac": ep.get("dead_frac", None),
                    })

    df_equiv = pd.DataFrame(equiv_rows)
    df_train = pd.DataFrame(train_rows)

    out_equiv = out_dir / "pca_equivalence.parquet"
    out_train = out_dir / "fullcorpus_train.parquet"

    df_equiv.to_parquet(out_equiv, index=False)
    df_train.to_parquet(out_train, index=False)

    print(f"\nPCA equivalence data: {out_equiv} ({len(df_equiv)} rows)")
    print(f"Full-corpus training: {out_train} ({len(df_train)} rows)")

    _print_summary(df_train)
    return out_equiv


def _print_summary(df: pd.DataFrame):
    """Print a quick comparison table."""
    summary = df.groupby(["edge", "init"]).agg(
        epochs_mean=("epochs", "mean"),
        wall_mean=("wall_s", "mean"),
        converged=("converged", "all"),
    ).reset_index()
    print("\nSummary:")
    print(summary.to_string(index=False))
