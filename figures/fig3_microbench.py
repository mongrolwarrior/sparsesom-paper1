"""Fig 3 — BMU kernel microbenchmark: time (log y) vs problem size, one line per kernel."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.stats import t_interval, paired_log_ratio_t


def generate(results_dir: Path, output_dir: Path):
    p = results_dir / "microbench.parquet"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)

    fig, ax = plt.subplots(figsize=(8, 5))

    kernels = ["node-major", "feature-major", "spmm-tile"]
    colors = {"node-major": "#d62728", "feature-major": "#ff7f0e", "spmm-tile": "#2ca02c"}
    markers = {"node-major": "s", "feature-major": "^", "spmm-tile": "o"}

    passes = sorted(df["pass"].unique())
    x_positions = {p: i for i, p in enumerate(passes)}

    for kernel in kernels:
        kdf = df[df["kernel"] == kernel]
        means = []
        errors = []
        xs = []
        for p_name in passes:
            pdata = kdf[kdf["pass"] == p_name]["time_ms"]
            if len(pdata) == 0:
                continue
            ci = t_interval(pdata.values)
            means.append(ci["mean"])
            errors.append(ci["half_width"])
            xs.append(x_positions[p_name])

        ax.errorbar(xs, means, yerr=errors,
                     label=kernel, marker=markers[kernel],
                     color=colors[kernel], capsize=4, linewidth=2, markersize=8)

    ax.set_yscale("log")
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(passes)
    ax.set_xlabel("Problem size")
    ax.set_ylabel("BMU time per epoch (ms, log scale)")
    ax.set_title("BMU Kernel Microbenchmark (RTX 4090)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "fig3_microbench.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig3_microbench.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
