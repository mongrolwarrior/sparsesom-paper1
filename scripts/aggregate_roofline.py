#!/usr/bin/env python3
"""Aggregate raw ncu_<impl>_<edge>.csv profiles into roofline_dram.csv.

ncu_profile.sh captures one row per (metric, kernel launch). This picks
the BMU kernel per impl/edge (the launch group that moves the most DRAM
overall — BMU dominates every implementation), sums its DRAM traffic
across captured launches, and records the mean achieved DRAM-bandwidth
utilisation. Writes one row per impl/edge to roofline_dram.csv.

Caveat: ncu was capped at -c 500 launches (uncapped replay of MedSOM's
per-neuron kernels would take days). For impls whose BMU is one big tiled
launch per epoch, the capture is complete; where more launches exist per
epoch than were captured, bytes are per-captured-launch, not per-epoch —
the launches_captured column records this so per-epoch totals can be
reconstructed downstream. Bandwidth utilisation (%) is launch-count
independent and always valid.

Usage: aggregate_roofline.py RESULTS_DIR
"""

import glob
import os
import re
import sys

import pandas as pd

DRAM_R = "dram__bytes_read.sum"
DRAM_W = "dram__bytes_write.sum"
SM_PCT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"       # compute occupancy
DRAM_BW = "dram__throughput.avg.pct_of_peak_sustained_elapsed"    # REAL bandwidth %
DUR = "gpu__time_duration.sum"                                    # per-kernel exec time

# substrings that identify each impl's BMU kernel; None -> pick dominant
BMU_HINT = {
    "sbsom-bin": "bmu", "sbsom-float": "bmu",
    "ssom-feat": None, "ssom-node": None, "medsom": None,
}

# BMU kernel launches per epoch, per impl/edge. sbsom runs one persistent
# kernel over the whole epoch (grid = n_samples/TA); cuSPARSE ssom tiles
# the SpMM into n_samples/(512MiB / neurons·2B) launches (from the run
# logs' "tiling BMU: N tiles" lines). MedSOM launches per-neuron kernels.
LAUNCHES_PER_EPOCH = {
    ("sbsom-bin", 128): 1, ("sbsom-bin", 256): 1,
    ("sbsom-float", 128): 1, ("sbsom-float", 256): 1,
    ("ssom-feat", 128): 1643, ("ssom-feat", 256): 6571,
    ("ssom-node", 128): 1643, ("ssom-node", 256): 6571,
}


def load(path):
    df = pd.read_csv(path, skiprows=2)
    df.columns = [c.strip() for c in df.columns]
    # metric values are strings and may carry thousands separators
    df["val"] = pd.to_numeric(
        df["Metric Value"].astype(str).str.replace(",", ""), errors="coerce")
    return df


def per_launch_table(df):
    """One row per launch: kernel short name + the three metrics."""
    df = df.copy()
    df["k"] = df["Kernel Name"].str.replace(r"[<(].*", "", regex=True)
    piv = df.pivot_table(index=["ID", "k"], columns="Metric Name",
                         values="val", aggfunc="first").reset_index()
    return piv


def main():
    results = sys.argv[1]
    rows = []
    for path in sorted(glob.glob(os.path.join(results, "ncu_*.csv"))):
        m = re.search(r"ncu_(.+)_(\d+)\.csv$", os.path.basename(path))
        if not m:
            continue
        impl, edge = m.group(1), int(m.group(2))
        piv = per_launch_table(load(path))
        piv["dram"] = piv.get(DRAM_R, 0).fillna(0) + piv.get(DRAM_W, 0).fillna(0)

        # choose the BMU kernel group
        by_k = piv.groupby("k").agg(dram=("dram", "sum"),
                                    n=("ID", "nunique")).sort_values(
                                        "dram", ascending=False)
        hint = BMU_HINT.get(impl)
        pick = None
        if hint:
            cand = [k for k in by_k.index if hint in k.lower()]
            if cand:
                pick = by_k.loc[cand].sort_values("dram",
                                                  ascending=False).index[0]
        if pick is None:
            pick = by_k.index[0]        # dominant DRAM mover = the BMU kernel

        g = piv[piv["k"] == pick]
        n_cap = int(g["ID"].nunique())
        read = float(g.get(DRAM_R, 0).sum())
        write = float(g.get(DRAM_W, 0).sum())
        lpe = LAUNCHES_PER_EPOCH.get((impl, edge))
        full = (lpe is not None and n_cap >= lpe)   # captured a whole epoch
        # per-epoch DRAM: exact if we captured every launch, else scaled
        per_ep = (read + write) if full else (
            (read + write) / n_cap * lpe if (lpe and n_cap) else None)
        # summed kernel exec time over the epoch (for the launch-structure gap
        # test) — only meaningful when the whole epoch was captured
        dur_sum = None
        if DUR in g.columns:
            d = g[DUR].sum(min_count=1)
            dur_sum = float(d) / 1e9 if pd.notna(d) else None   # ns -> s
        rows.append({
            "impl": impl, "edge": edge,
            "bmu_kernel": pick.strip()[:60],
            "launches_captured": n_cap,
            "launches_per_epoch": lpe,
            "capture": "full_epoch_exact" if full else "scaled_estimate",
            "dram_bytes_read": read,
            "dram_bytes_write": write,
            "dram_per_epoch_tb": per_ep / 1e12 if per_ep else None,
            "dram_bw_pct_measured": (float(g[DRAM_BW].mean())
                                     if DRAM_BW in g.columns else None),
            "sm_throughput_pct": (float(g[SM_PCT].mean())
                                  if SM_PCT in g.columns else None),
            "kernel_time_s_sum": dur_sum if full else None,
        })

    df = pd.DataFrame(rows).sort_values(["edge", "impl"])
    out = os.path.join(results, "roofline_dram.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    print(df[["edge", "impl", "launches_captured", "capture",
              "dram_per_epoch_tb", "dram_bw_pct_measured",
              "sm_throughput_pct", "kernel_time_s_sum"]].to_string(index=False))


if __name__ == "__main__":
    main()
