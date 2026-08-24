"""Fig 4 — BMU dominance: BMU share (%) vs neuron count K on log x-axis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.stats import t_interval


def generate(results_dir: Path, output_dir: Path):
    p = results_dir / "size_sweep_epochs.parquet"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)

    # Per (edge, seed): sum bmu_s and update_s across epochs
    agg = df.groupby(["edge", "seed"]).agg(
        total_bmu=("bmu_s", "sum"),
        total_update=("update_s", "sum"),
    ).reset_index()
    agg["bmu_share"] = agg["total_bmu"] / (agg["total_bmu"] + agg["total_update"])

    # Aggregate across seeds
    edges = sorted(agg["edge"].unique())
    K_vals = [e * e for e in edges]
    means = []
    ci_lo = []
    ci_hi = []

    for edge in edges:
        vals = agg[agg["edge"] == edge]["bmu_share"].values * 100
        ci = t_interval(vals)
        means.append(ci["mean"])
        ci_lo.append(ci["ci_lo"])
        ci_hi.append(ci["ci_hi"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(K_vals, ci_lo, ci_hi, alpha=0.2, color="#2ca02c")
    ax.plot(K_vals, means, "o-", color="#2ca02c", linewidth=2, markersize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Neuron count K")
    ax.set_ylabel("BMU share of wall time (%)")
    ax.set_title("BMU Dominance vs Map Size")
    ax.set_ylim(80, 101)
    ax.grid(True, alpha=0.3)

    # Annotate endpoints
    if means:
        ax.annotate(f"{means[0]:.1f}%", (K_vals[0], means[0]),
                     textcoords="offset points", xytext=(10, -15))
        ax.annotate(f"{means[-1]:.1f}%", (K_vals[-1], means[-1]),
                     textcoords="offset points", xytext=(-40, 10))

    fig.tight_layout()
    fig.savefig(output_dir / "fig4_bmu_dominance.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig4_bmu_dominance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
