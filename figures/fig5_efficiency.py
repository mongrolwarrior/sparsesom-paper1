"""Fig 5 — Efficiency sweep: wall time (log y) vs K (log x), one line per impl."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.stats import t_interval


IMPL_STYLE = {
    "sbsom-bin":  {"color": "#2ca02c", "marker": "o", "label": "sbsom-bin"},
    "sbsom-float": {"color": "#98df8a", "marker": "D", "label": "sbsom-float"},
    "ssom-feat":  {"color": "#1f77b4", "marker": "s", "label": "ssom-feat (cuSPARSE)"},
    "ssom-node":  {"color": "#aec7e8", "marker": "^", "label": "ssom-node (cuSPARSE)"},
    "medsom":     {"color": "#d62728", "marker": "v", "label": "MedSOM"},
    "somoclu-cpu": {"color": "#7f7f7f", "marker": "x", "label": "somoclu (CPU)"},
}


def generate(results_dir: Path, output_dir: Path):
    p = results_dir / "efficiency_sweep.parquet"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)
    df = df[~df["oom"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    for impl in df["impl"].unique():
        idf = df[df["impl"] == impl]
        style = IMPL_STYLE.get(impl, {"color": "gray", "marker": ".", "label": impl})

        # Aggregate across seeds
        edges = sorted(idf["edge"].unique())
        K_vals = []
        means = []
        ci_los = []
        ci_his = []

        for edge in edges:
            vals = idf[idf["edge"] == edge]["total_wall_s"].dropna().values
            if len(vals) == 0:
                continue
            ci = t_interval(vals)
            K_vals.append(edge * edge)
            means.append(ci["mean"])
            ci_los.append(ci["ci_lo"])
            ci_his.append(ci["ci_hi"])

        if not K_vals:
            continue

        ax.fill_between(K_vals, ci_los, ci_his, alpha=0.15, color=style["color"])
        ax.plot(K_vals, means, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=7)

        # Mark OOM dropout
        oom_rows = df[(df["impl"] == impl) & (df["oom"])]
        if len(oom_rows) > 0:
            oom_edge = oom_rows["edge"].min()
            ax.axvline(oom_edge * oom_edge, color=style["color"],
                       linestyle=":", alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Neuron count K")
    ax.set_ylabel("Total wall time (s, log scale)")
    ax.set_title("Full-Corpus Efficiency Sweep (matched update, fixed epochs)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_dir / "fig5_efficiency.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig5_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
