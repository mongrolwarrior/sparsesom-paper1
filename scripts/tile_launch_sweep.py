#!/usr/bin/env python3
"""Mechanism test for §5.7, with a PRE-REGISTERED discriminator.

Question: cuSPARSE ssom-feat is not DRAM-bound (measured ~15% of peak at
edge 256 vs our ~45%). Why is it slow — per-launch sync/serialisation
between its thousands of tiles (A), or an intrinsic csrmm access pattern
that just streams slowly regardless of tiling (B)?

A naive launch-count fit is confounded: enlarging --tile-mb both cuts
launches AND cuts codebook re-read traffic (fewer sample-blocks -> fewer
full-codebook passes), so launches and DRAM move together. Instead we
measure DRAM at each tile size and report SUSTAINED BANDWIDTH
(phase DRAM / unprofiled BMU wall, % of peak).

PRE-REGISTERED (state the outcome against this, do not fit post-hoc):
  * If sustained bandwidth CLIMBS from ~15% toward sbsom's ~45% as tiles
    grow and launches fall (time dropping faster than traffic) -> mechanism
    (A), launch structure. REPORTABLE.
  * If sustained bandwidth stays near ~15% regardless of tile size -> the
    bottleneck is intrinsic to csrmm's access pattern; launch structure is
    EXCLUDED. EQUALLY REPORTABLE.

Per tile size: unprofiled 2-epoch run for BMU wall; a 1-epoch ncu capture
(BMU-phase kernels) for phase DRAM bytes and in-kernel DRAM throughput.
Larger tiles have fewer launches, so their ncu captures are cheap; the
default/small tiles have many launches and are the slow ones.

Writes tile_sweep.csv: tile_mb, launches_per_epoch, bmu_s_per_ep,
phase_dram_tb, in_kernel_dram_bw_pct, sustained_bw_pct.

Usage: tile_sweep.py RESULTS_DIR CORPUS --edge 256 [--tiles-mb 512,1024,...]
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.runner import run_standardsparsesom  # noqa: E402
from aggregate_phase import load  # reuse the ncu CSV parser  # noqa: E402

N_SAMPLES = 26_912_934
PEAK_BW_BPS = 1008e9
DRAM_R = "dram__bytes_read.sum"
DRAM_W = "dram__bytes_write.sum"
DRAM_BW = "dram__throughput.avg.pct_of_peak_sustained_elapsed"

NCU = os.environ.get("NCU", "ncu")
STANDARDSPARSESOM = os.environ.get("STANDARDSPARSESOM", "standardsparsesom")
METRICS = f"{DRAM_R},{DRAM_W},{DRAM_BW}"


def ncu_phase_bytes(corpus, edge, tile_mb, launches, scratch, cap=600):
    """1-epoch ncu capture of the BMU phase; return (phase_dram_bytes, bw%).

    Caps profiled launches (per-tile DRAM is uniform) and scales the per-tile
    kernels to the full launch count, so a 6571-tile config doesn't take days.
    """
    csv = str(scratch / f"tile_{tile_mb:.0f}.csv")
    cmd = [NCU, "--metrics", METRICS,
           "-k", "regex:norms_kernel|csrmm|argmax_kernel|rebase_rowptr",
           "-c", str(cap), "--csv", "--log-file", csv,
           STANDARDSPARSESOM, corpus, "--map", str(edge), "--layout", "feature",
           "--precision", "fp16", "--stop", "fixed", "--epochs", "1", "--seed", "0",
           "--tile-mb", str(tile_mb)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    piv = load(csv)
    piv["k"] = piv["k"] if "k" in piv.columns else ""
    per_tile = piv["k"].str.contains("csrmm|argmax|rebase", case=False)
    cap_tiles = int(piv[piv["k"].str.contains("csrmm", case=False)]["ID"].nunique() or 1)
    scale = launches / cap_tiles if cap_tiles < launches else 1.0
    read = piv.get(DRAM_R, 0).fillna(0)
    write = piv.get(DRAM_W, 0).fillna(0)
    factor = per_tile.map({True: scale, False: 1.0})
    dram = float((read * factor).sum() + (write * factor).sum())
    bw = piv[DRAM_BW].mean() if DRAM_BW in piv.columns else float("nan")
    return dram, float(bw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("corpus")
    ap.add_argument("--edge", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--tiles-mb", default="512,1024,2048,4096,8192")
    args = ap.parse_args()

    results = Path(args.results_dir)
    scratch = results / "tile_sweep_scratch"
    scratch.mkdir(exist_ok=True)
    neurons = args.edge * args.edge
    rows = []

    for tile_mb in [float(t) for t in args.tiles_mb.split(",")]:
        spt = int(tile_mb * 1024 * 1024 // (neurons * 2))
        if spt < 1:
            continue
        launches = math.ceil(N_SAMPLES / spt)
        print(f"  tile_mb={tile_mb:.0f}: {launches} launches/epoch ...", flush=True)

        # unprofiled wall
        r = run_standardsparsesom(
            args.corpus, args.edge, 0, layout="feature", neighbourhood="box",
            stop="fixed", precision="fp16",
            extra_flags=["--epochs", str(args.epochs), "--tile-mb", str(tile_mb)])
        if r.get("oom"):
            print(f"    OOM at tile_mb={tile_mb}")
            continue
        done = r["done"] or {}
        ep = max(done.get("epochs", args.epochs), 1)
        wall = done.get("bmu_s", 0) / ep

        # measured phase DRAM + in-kernel bandwidth (capped + scaled).
        # cap=60 -> ~20 uniform tiles; ncu overhead at edge 256 is ~14s/launch,
        # so a larger cap makes each tile-size point take hours.
        dram, bw = ncu_phase_bytes(args.corpus, args.edge, tile_mb, launches,
                                   scratch, cap=60)
        sust = dram / wall / PEAK_BW_BPS * 100 if wall else None
        rows.append({"tile_mb": tile_mb, "launches_per_epoch": launches,
                     "bmu_s_per_ep": round(wall, 2),
                     "phase_dram_tb": round(dram / 1e12, 3),
                     "in_kernel_dram_bw_pct": round(bw, 1),
                     "sustained_bw_pct": round(sust, 1) if sust else None})

    df = pd.DataFrame(rows)
    out = results / "tile_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(df.to_string(index=False))

    if len(df) >= 2:
        df = df.sort_values("launches_per_epoch")   # many -> few launches
        lo_bw = df.iloc[-1].sustained_bw_pct   # fewest launches (biggest tile)
        hi_bw = df.iloc[0].sustained_bw_pct    # most launches (smallest tile)
        print(f"\nPRE-REGISTERED DISCRIMINATOR:")
        print(f"  sustained BW: {hi_bw}% ({df.iloc[0].launches_per_epoch:.0f} "
              f"launches) -> {lo_bw}% ({df.iloc[-1].launches_per_epoch:.0f} launches)")
        if lo_bw is not None and hi_bw is not None:
            if lo_bw - hi_bw >= 10:
                print("  -> CLIMBS toward ~45% as tiles grow: mechanism (A) "
                      "launch structure / sync-serialisation.")
            elif abs(lo_bw - hi_bw) < 5:
                print("  -> FLAT (~constant): mechanism (B) intrinsic csrmm "
                      "access pattern; launch structure EXCLUDED.")
            else:
                print("  -> partial climb; report the trend, interpretation to Desktop.")


if __name__ == "__main__":
    main()
