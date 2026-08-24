"""PCA equivalence supplementary table and convergence curve figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.stats import t_interval, tost_equivalence


def generate(results_dir: Path, output_dir: Path):
    p_equiv = results_dir / "pca_equivalence.parquet"
    p_train = results_dir / "fullcorpus_train.parquet"

    if not p_equiv.exists():
        raise FileNotFoundError(p_equiv)
    if not p_train.exists():
        raise FileNotFoundError(p_train)

    df_equiv = pd.read_parquet(p_equiv)
    df_train = pd.read_parquet(p_train)

    # Convergence curves: QE vs epoch, one curve per init, at each edge
    edges = sorted(df_equiv["edge"].unique())

    fig, axes = plt.subplots(1, len(edges), figsize=(5 * len(edges), 5), sharey=True)
    if len(edges) == 1:
        axes = [axes]

    for ax, edge in zip(axes, edges):
        for init, color in [("pca", "#2ca02c"), ("random", "#d62728")]:
            sub = df_equiv[(df_equiv["edge"] == edge) & (df_equiv["init"] == init)]
            if sub.empty:
                continue

            # Mean QE per epoch across seeds
            ep_agg = sub.groupby("epoch")["qe_heldout"].agg(["mean", "std", "count"])
            ep_agg = ep_agg.reset_index()

            ax.plot(ep_agg["epoch"], ep_agg["mean"], color=color, label=init, linewidth=2)
            if "std" in ep_agg.columns and ep_agg["count"].iloc[0] > 1:
                ax.fill_between(
                    ep_agg["epoch"],
                    ep_agg["mean"] - 1.96 * ep_agg["std"] / np.sqrt(ep_agg["count"]),
                    ep_agg["mean"] + 1.96 * ep_agg["std"] / np.sqrt(ep_agg["count"]),
                    alpha=0.2, color=color,
                )

        ax.set_xlabel("Epoch")
        ax.set_title(f"Edge {edge}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Held-out QE")
    fig.suptitle("PCA vs Random Init — Convergence Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_equivalence.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pca_equivalence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Summary table
    summary_rows = []
    for edge in sorted(df_train["edge"].unique()):
        for init in ["pca", "random"]:
            sub = df_train[(df_train["edge"] == edge) & (df_train["init"] == init)]
            if sub.empty:
                continue
            epochs_ci = t_interval(sub["epochs"].values)
            wall_ci = t_interval(sub["wall_s"].values)
            summary_rows.append({
                "edge": edge,
                "init": init,
                "epochs": f"{epochs_ci['mean']:.1f} ± {epochs_ci['half_width']:.1f}",
                "wall_s": f"{wall_ci['mean']:.1f} ± {wall_ci['half_width']:.1f}",
                "converged": sub["converged"].all(),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "pca_equivalence_table.csv", index=False)
