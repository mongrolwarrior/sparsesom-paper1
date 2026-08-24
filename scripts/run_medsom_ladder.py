#!/usr/bin/env python3
"""Run the MedSOM rungs of the efficiency sweep and append to its parquet.

Run 8's seedable grid completed but the MedSOM stage crashed on a CLI
mismatch before efficiency_sweep.parquet gained any medsom rows.  This
script runs just the MedSOM ladder (fixed epochs, stop on OOM) and
appends the rows, so the 60 GPU-hours of seedable results are not
re-run.

Usage:
    run_medsom_ladder.py RESULTS_DIR CORPUS [--edges 32,64,...] [--epochs 20]
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.runner import run_medsom  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("corpus")
    ap.add_argument("--edges", default="32,64,128,256,512")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    results = Path(args.results_dir)
    out = results / "efficiency_sweep.parquet"
    df = pd.read_parquet(out)
    df = df[df["impl"] != "medsom"]          # idempotent re-runs

    scratch = results / "medsom_scratch"
    scratch.mkdir(exist_ok=True)

    rows = []
    for edge in [int(e) for e in args.edges.split(",")]:
        print(f"  efficiency: edge={edge} impl=medsom seed=42 (hardcoded) ...",
              flush=True)
        try:
            result = run_medsom(args.corpus, edge, epochs=args.epochs,
                                scratch_dir=str(scratch))
        except RuntimeError as e:
            if "OOM" in str(e) or "out of memory" in str(e).lower():
                result = {"oom": True}
            else:
                raise
        finally:
            for f in scratch.glob("codebook_*.dat"):
                f.unlink()

        row = {
            "edge": edge,
            "impl": "medsom",
            "precision": "fp32",
            "seed": 42,
            "epochs": args.epochs,
            "oom": result.get("oom", False),
        }
        if result.get("oom"):
            row.update({"bmu_s_per_ep": None, "upd_s_per_ep": None,
                        "total_wall_s": None, "dropout_edge": edge})
            rows.append(row)
            print(f"  medsom OOM at edge {edge}; ladder ends")
            break

        ep_data = result.get("epochs", [])
        total = sum(e.get("epoch_time_s", 0) for e in ep_data)
        row.update({"bmu_s_per_ep": None, "upd_s_per_ep": None,
                    "total_wall_s": total, "dropout_edge": None})
        rows.append(row)
        print(f"  medsom edge {edge}: {len(ep_data)} epochs, {total:.0f}s total "
              f"({total/max(len(ep_data),1):.0f}s/epoch)")

        # persist incrementally so a crash on a later rung loses nothing
        pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_parquet(
            out, index=False)

    final = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    final.to_parquet(out, index=False)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"\n{out}: {len(final)} rows ({len(rows)} medsom)")


if __name__ == "__main__":
    main()
