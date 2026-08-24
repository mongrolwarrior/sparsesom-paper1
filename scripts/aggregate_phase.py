#!/usr/bin/env python3
"""Aggregate full-BMU-phase ncu captures (ncu_full_*.csv) into manuscript tables.

Produces two CSVs:
  roofline_phase_breakdown.csv — per (impl, edge, kernel): launches, DRAM
      read/write, mean measured DRAM throughput %, SM occupancy %, summed
      kernel exec time. Makes the cuSPARSE score-block round trip explicit
      (csrmm writes it, argmax_kernel reads it back).
  roofline_phase_summary.csv — per (impl, edge): phase-total DRAM (sum over
      all BMU-phase kernels), summed kernel exec time, the unprofiled BMU
      wall (from efficiency_sweep), the gap (wall - summed kernel time), and
      phase sustained bandwidth (phase DRAM / unprofiled wall, % of peak).

The gap and sustained bandwidth localise the §5.7 mechanism:
  gap ~ 0 with low sustained BW  -> cost is inside the kernels (intrinsic
       csrmm access pattern), launch structure excluded.
  gap large                      -> time between launches (sync/serialisation).

Usage: aggregate_phase.py RESULTS_DIR
"""

import glob
import os
import re
import sys

import pandas as pd

DRAM_R = "dram__bytes_read.sum"
DRAM_W = "dram__bytes_write.sum"
DRAM_BW = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
SM_PCT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DUR = "gpu__time_duration.sum"
PEAK_BW_BPS = 1008e9   # RTX 4090 peak DRAM bandwidth
N_SAMPLES = 26_912_934
SCORES_TILE_BYTES = 512 * 1024 * 1024   # cuSPARSE score-tile budget (observed)
PER_TILE = ("csrmm", "argmax", "rebase")  # launched once per sample-tile


def tiles_per_epoch(edge):
    import math
    spt = SCORES_TILE_BYTES // (edge * edge * 2)   # samples per tile (fp16)
    return max(1, math.ceil(N_SAMPLES / spt))


def is_per_tile(kernel):
    return any(t in kernel.lower() for t in PER_TILE)


def kernel_role(k):
    """Human-readable role so the score-block round trip is legible."""
    kl = k.lower()
    if "csrmm" in kl:
        return "scores: codebook read + SCORE-BLOCK WRITE"
    if "argmax" in kl:
        return "argmin: SCORE-BLOCK READ-BACK"
    if "norm" in kl:
        return "codebook norm (read)"
    if "bmu_spmm" in kl:
        return "FUSED score+argmin (no score block)"
    if "rebase" in kl:
        return "row-pointer setup (negligible)"
    return "other"


def load(path):
    df = pd.read_csv(path, skiprows=2)
    df.columns = [c.strip() for c in df.columns]
    df["val"] = pd.to_numeric(
        df["Metric Value"].astype(str).str.replace(",", ""), errors="coerce")
    df["k"] = (df["Kernel Name"].str.replace(r"[<(].*", "", regex=True)
               .str.replace(r".*::", "", regex=True).str[:34])
    return df.pivot_table(index=["ID", "k"], columns="Metric Name",
                          values="val", aggfunc="first").reset_index()


def unprofiled_bmu_wall(results):
    """Per-epoch BMU wall (s) from the efficiency sweep, per (impl, edge)."""
    p = os.path.join(results, "efficiency_sweep.parquet")
    if not os.path.exists(p):
        return {}
    df = pd.read_parquet(p)
    df = df[~df.oom]
    out = {}
    for (impl, edge), g in df.groupby(["impl", "edge"]):
        v = g["bmu_s_per_ep"].dropna()
        if len(v):
            out[(impl, edge)] = float(v.mean())
    return out


def main():
    results = sys.argv[1]
    walls = unprofiled_bmu_wall(results)
    brk, summ = [], []

    for path in sorted(glob.glob(os.path.join(results, "ncu_full_*.csv"))):
        m = re.search(r"ncu_full_(.+)_(\d+)\.csv$", os.path.basename(path))
        if not m:
            continue
        impl, edge = m.group(1), int(m.group(2))
        piv = load(path)

        g = piv.groupby("k").agg(
            launches=("ID", "nunique"),
            dram_read=(DRAM_R, "sum"),
            dram_write=(DRAM_W, "sum"),
            dram_bw_pct=(DRAM_BW, "mean"),
            sm_pct=(SM_PCT, "mean"),
            kernel_time_s=(DUR, lambda s: s.sum() / 1e9),  # ns -> s
        ).reset_index()

        # scale each per-tile kernel by ITS OWN captured launch count up to the
        # full tile count. ncu's -c cap can truncate mid-tile-iteration, so
        # csrmm and argmax are captured a different number of times (e.g. 20 vs
        # 19) and must not share one factor. Once-per-epoch kernels (norms,
        # fused bmu) are captured whole -> scale 1.
        tpe = tiles_per_epoch(edge)

        def _kscale(row):
            if is_per_tile(row.k) and row.launches and row.launches < tpe:
                return tpe / row.launches
            return 1.0
        g["scale"] = g.apply(_kscale, axis=1)
        for col in ("dram_read", "dram_write", "kernel_time_s"):
            g[col] = g[col] * g["scale"]

        for _, r in g.iterrows():
            brk.append({"impl": impl, "edge": edge, "kernel": r.k,
                        "role": kernel_role(r.k),
                        "launches_captured": int(r.launches),
                        "scaled_x": round(r.scale, 1),
                        "dram_read_tb": r.dram_read / 1e12,
                        "dram_write_tb": r.dram_write / 1e12,
                        "dram_bw_pct_of_peak": round(r.dram_bw_pct, 1),
                        "sm_occupancy_pct_NOT_bandwidth": round(r.sm_pct, 1),
                        "kernel_time_s": round(r.kernel_time_s, 3)})

        # itemise the dense score-block round trip — the whole mechanism.
        # cuSPARSE csrmm's writes ARE the score-block write; argmax_kernel's
        # reads ARE the read-back. sbsom's fused kernel does neither (0).
        def kb(substr, col):
            sel = g[g.k.str.contains(substr, case=False)]
            return float(sel[col].sum())
        sb_write = kb("csrmm", "dram_write")
        sb_read = kb("argmax", "dram_read")

        phase_dram = (g.dram_read.sum() + g.dram_write.sum())
        ktime = g.kernel_time_s.sum()
        # representative sampling factor (csrmm's) for the capture label
        cs = g[g.k.str.contains("csrmm", case=False)]["scale"]
        rep = float(cs.max()) if len(cs) else 1.0
        capture = "exact" if rep <= 1.0 else f"sampled_x{rep:.0f}"
        wall = walls.get((impl, edge))
        sust_bw = (phase_dram / wall / PEAK_BW_BPS * 100) if wall else None
        summ.append({
            "impl": impl, "edge": edge, "capture": capture,
            "phase_dram_tb": round(phase_dram / 1e12, 3),
            "score_block_write_tb": round(sb_write / 1e12, 3),
            "score_block_readback_tb": round(sb_read / 1e12, 3),
            "score_roundtrip_tb": round((sb_write + sb_read) / 1e12, 3),
            "kernel_time_sum_s": round(ktime, 3),
            "bmu_wall_unprofiled_s": round(wall, 3) if wall else None,
            "gap_s": round(wall - ktime, 3) if wall else None,
            "gap_frac": round((wall - ktime) / wall, 3) if wall else None,
            "sustained_dram_bw_pct": round(sust_bw, 1) if sust_bw else None,
        })

    bdf = pd.DataFrame(brk).sort_values(["edge", "impl", "kernel"])
    sdf = pd.DataFrame(summ).sort_values(["edge", "impl"])
    bdf.to_csv(os.path.join(results, "roofline_phase_breakdown.csv"), index=False)
    sdf.to_csv(os.path.join(results, "roofline_phase_summary.csv"), index=False)
    print("=== per-kernel breakdown (score-block write/read-back itemised) ===")
    print(bdf.to_string(index=False))
    print("\n=== phase summary ===")
    print(sdf.to_string(index=False))
    print("\n=== score-block round trip (the mechanism, itemised) ===")
    print(sdf[["impl", "edge", "score_block_write_tb",
               "score_block_readback_tb", "score_roundtrip_tb",
               "phase_dram_tb"]].to_string(index=False))


if __name__ == "__main__":
    main()
