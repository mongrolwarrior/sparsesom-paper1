"""Fig 7 — Roofline: (a) achieved FLOP/s vs AI; (b) DRAM per epoch bar chart."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# RTX 4090 hardware constants
PEAK_BW_GBS = 1008
PEAK_FLOPS_TFLOPS = 82.6
RIDGE_POINT = PEAK_FLOPS_TFLOPS * 1000 / PEAK_BW_GBS  # ~81.9 FLOP/byte

IMPL_COLORS = {
    "sbsom-bin":  "#2ca02c",
    "sbsom-float": "#98df8a",
    "ssom-feat":  "#1f77b4",
    "ssom-node":  "#aec7e8",
    "medsom":     "#d62728",
}


def generate(results_dir: Path, output_dir: Path):
    p = results_dir / "roofline_dram.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Roofline plot
    # Hardware ceilings
    ai_range = np.logspace(-2, 3, 500)
    bw_line = PEAK_BW_GBS * ai_range           # GFLOP/s from bandwidth
    compute_line = np.full_like(ai_range, PEAK_FLOPS_TFLOPS * 1000)
    roof = np.minimum(bw_line, compute_line)

    ax1.plot(ai_range, roof, "k-", linewidth=2, alpha=0.3, label="Hardware ceiling")
    ax1.axvline(RIDGE_POINT, color="gray", linestyle=":", alpha=0.5)
    ax1.text(RIDGE_POINT * 1.2, PEAK_FLOPS_TFLOPS * 800, "ridge", fontsize=8,
             color="gray")

    for _, row in df.iterrows():
        impl = row["impl"]
        total_bytes = row["dram_bytes_read"] + row["dram_bytes_write"]
        if total_bytes == 0:
            continue

        ai = row.get("flops", 0) / total_bytes if total_bytes > 0 else 0
        gflops = row.get("flops", 0) / row["duration_s"] / 1e9 if row["duration_s"] > 0 else 0

        color = IMPL_COLORS.get(impl, "gray")
        ax1.plot(ai, gflops, "o", color=color, markersize=10, label=impl)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax1.set_ylabel("Performance (GFLOP/s)")
    ax1.set_title("(a) Roofline — all implementations bandwidth-bound")
    ax1.set_xlim(0.01, 500)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")

    # (b) DRAM per epoch bar chart
    impls = df["impl"].unique()
    dram_gb = []
    colors = []
    labels = []

    for impl in impls:
        row = df[df["impl"] == impl].iloc[0]
        total_bytes = (row["dram_bytes_read"] + row["dram_bytes_write"])
        scale = row.get("scale_factor", 1.0)
        gb = total_bytes * scale / 1e9
        dram_gb.append(gb)
        colors.append(IMPL_COLORS.get(impl, "gray"))
        labels.append(impl)

    y_pos = range(len(impls))
    ax2.barh(y_pos, dram_gb, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xscale("log")
    ax2.set_xlabel("DRAM per epoch (GB, log scale)")
    ax2.set_title("(b) DRAM volume — compression is the story")

    for i, v in enumerate(dram_gb):
        ax2.text(v * 1.1, i, f"{v:.0f} GB", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "fig7_roofline.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig7_roofline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
