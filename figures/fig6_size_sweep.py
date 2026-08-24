"""Fig 6 — Size sweep: (a) held-out QE vs K log-log with power-law fit; (b) residuals."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from pipeline.stats import t_interval


def generate(results_dir: Path, output_dir: Path):
    p = results_dir / "size_sweep.parquet"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)

    # Aggregate QE across seeds per edge
    edges = sorted(df["edge"].unique())
    K_vals = np.array([e * e for e in edges])
    log_K = np.log10(K_vals)

    qe_means = []
    qe_ci_lo = []
    qe_ci_hi = []
    gpu_labels = []

    for edge in edges:
        sub = df[df["edge"] == edge]
        vals = sub["qe_heldout"].dropna().values
        ci = t_interval(vals)
        qe_means.append(ci["mean"])
        qe_ci_lo.append(ci["ci_lo"])
        qe_ci_hi.append(ci["ci_hi"])
        gpu_labels.append(sub["gpu"].iloc[0] if "gpu" in sub.columns else "rtx4090")

    qe_means = np.array(qe_means)
    log_qe = np.log10(qe_means)

    # Power-law fit: log10(QE) = a + β·log10(K)
    slope, intercept, r, p_val, se = sp_stats.linregress(log_K, log_qe)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) QE vs K with fit
    for i, (K, qe, lo, hi, gpu) in enumerate(
            zip(K_vals, qe_means, qe_ci_lo, qe_ci_hi, gpu_labels)):
        color = "#2ca02c" if "4090" in str(gpu) else "#d62728"
        marker = "o" if "4090" in str(gpu) else "D"
        label = None
        if i == 0:
            label = "RTX 4090 (5 seeds)"
        elif "h200" in str(gpu).lower() and not any(
                "h200" in str(g).lower() for g in gpu_labels[:i]):
            label = "H200 (single seed)"
        ax1.errorbar(K, qe, yerr=[[qe - lo], [hi - qe]],
                      fmt=marker, color=color, capsize=4, markersize=8, label=label)

    # Fit line
    K_fit = np.logspace(np.log10(K_vals.min()) - 0.1, np.log10(K_vals.max()) + 0.1, 100)
    qe_fit = 10 ** (intercept + slope * np.log10(K_fit))
    ax1.plot(K_fit, qe_fit, "k--", alpha=0.5,
             label=f"β = {slope:.4f} ± {se:.4f}, R² = {r**2:.4f}")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Neuron count K")
    ax1.set_ylabel("Held-out cosine QE")
    ax1.set_title("(a) Size sweep — no elbow")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    # (b) Residuals
    residuals = log_qe - (intercept + slope * log_K)
    for i, (lk, res, gpu) in enumerate(zip(log_K, residuals, gpu_labels)):
        color = "#2ca02c" if "4090" in str(gpu) else "#d62728"
        marker = "o" if "4090" in str(gpu) else "D"
        ax2.plot(lk, res, marker, color=color, markersize=8)

    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("log₁₀(K)")
    ax2.set_ylabel("Residual (log₁₀ QE)")
    ax2.set_title("(b) Residuals")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "fig6_size_sweep.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig6_size_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
