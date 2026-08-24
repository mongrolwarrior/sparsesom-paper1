#!/usr/bin/env python3
"""Anchor the somoclu (best CPU library) point of the efficiency sweep.

E2's original spec was somoclu -> MedSOM -> sbsom, but somoclu was never
run, leaving the "three orders ahead of the best CPU library" claim
without new timing data. This runs somoclu's sparse CPU kernel on the
full corpus at one edge, measures per-epoch wall time, and appends a row
to efficiency_sweep.parquet (impl=somoclu, precision=fp32-cpu).

somoclu is CPU/OpenMP only, so this can run alongside GPU work. Wall
time is reported per-epoch and scaled to the sweep's fixed 20-epoch
basis (measured epochs, not extrapolated training to convergence).

Usage:
    run_somoclu_anchor.py RESULTS_DIR CORPUS.sbcsr --edge 128 [--epochs 3]
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SOMOCLU = os.environ.get(
    "SOMOCLU",
    str(Path(__file__).resolve().parent.parent
        / "external/somoclu/somoclu_cpu"))
SWEEP_BASIS_EPOCHS = 20


def ensure_libsvm(corpus: Path, scratch: Path) -> Path:
    libsvm = scratch / (corpus.stem + ".libsvm")
    if libsvm.exists() and libsvm.stat().st_size > 0:
        print(f"reusing {libsvm} ({libsvm.stat().st_size/1e9:.1f} GB)")
        return libsvm
    conv = Path(__file__).resolve().parent / "sbcsr_to_libsvm.py"
    print(f"converting {corpus} -> {libsvm} ...", flush=True)
    subprocess.run([sys.executable, str(conv), str(corpus), str(libsvm)],
                   check=True)
    return libsvm


def run_somoclu(libsvm: Path, edge: int, epochs: int, scratch: Path) -> dict:
    out_prefix = str(scratch / f"somoclu_{edge}")
    log_path = scratch / f"somoclu_{edge}.log"
    cmd = [SOMOCLU, "-k", "2", "-x", str(edge), "-y", str(edge),
           "-e", str(epochs), "-v", "1", str(libsvm), out_prefix]
    print("  " + " ".join(cmd), flush=True)

    # somoclu prints epoch times and the progress bar to stderr, so merge
    # both streams; persist the full output so a parse miss is recoverable.
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        log.write(proc.stdout or "")
    wall = time.monotonic() - t0

    out = proc.stdout or ""
    if proc.returncode != 0:
        low = out.lower()
        if ("bad_alloc" in low or "out of memory" in low
                or proc.returncode in (-9, 137)):   # OOM-killer
            return {"oom": True}
        raise RuntimeError(f"somoclu failed ({proc.returncode}); see "
                           f"{log_path}:\n{out[-800:]}")

    ep_times = [float(m) for m in
                re.findall(r"Time for epoch \d+:\s*([\d.eE+-]+)", out)]
    if not ep_times:
        raise RuntimeError(f"somoclu exited 0 but printed no epoch times; "
                           f"see {log_path}")
    # clean up codebook/bmu dumps (edge 128: ~2 GB .wts)
    for suf in (".wts", ".bm", ".umx"):
        Path(out_prefix + suf).unlink(missing_ok=True)
    return {"oom": False, "epoch_times": ep_times, "wall_total": wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("corpus")
    ap.add_argument("--edge", type=int, default=None,
                    help="single edge; or use --edges for several")
    ap.add_argument("--edges", default=None,
                    help="comma-separated edges to run sequentially")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    results = Path(args.results_dir)
    scratch = results / "somoclu_scratch"
    scratch.mkdir(exist_ok=True)
    out = results / "efficiency_sweep.parquet"

    if args.edges:
        edges = [int(e) for e in args.edges.split(",")]
    elif args.edge is not None:
        edges = [args.edge]
    else:
        ap.error("give --edge or --edges")

    libsvm = ensure_libsvm(Path(args.corpus), scratch)

    for edge in edges:
        r = run_somoclu(libsvm, edge, args.epochs, scratch)
        row = {"edge": edge, "impl": "somoclu", "precision": "fp32-cpu",
               "seed": 0, "epochs": SWEEP_BASIS_EPOCHS,
               "oom": r.get("oom", False)}

        if r["oom"]:
            row.update({"bmu_s_per_ep": None, "upd_s_per_ep": None,
                        "total_wall_s": None, "dropout_edge": edge})
            print(f"somoclu OOM at edge {edge}")
        else:
            ep = r["epoch_times"]
            # first epoch includes allocation/warm-up; use steady-state mean
            steady = ep[1:] if len(ep) > 1 else ep
            per_ep = sum(steady) / len(steady)
            row.update({"bmu_s_per_ep": None,   # somoclu doesn't split BMU/update
                        "upd_s_per_ep": None,
                        "total_wall_s": per_ep * SWEEP_BASIS_EPOCHS,
                        "dropout_edge": None,
                        "somoclu_s_per_ep_measured": per_ep,
                        "somoclu_epochs_measured": len(ep)})
            print(f"somoclu edge {edge}: {per_ep:.1f}s/epoch (mean of "
                  f"{len(steady)} steady epochs), "
                  f"20-epoch basis = {per_ep*SWEEP_BASIS_EPOCHS:.0f}s")

        # drop only the same (impl, edge) row so other somoclu sizes persist
        df = pd.read_parquet(out)
        df = df[~((df["impl"] == "somoclu") & (df["edge"] == edge))]
        pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_parquet(
            out, index=False)
        print(f"appended somoclu edge {edge} -> {out}")


if __name__ == "__main__":
    main()
