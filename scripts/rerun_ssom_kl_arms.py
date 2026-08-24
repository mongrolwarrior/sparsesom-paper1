#!/usr/bin/env python3
"""Re-run the run-6 cuSPARSE box+kl arms with a real epoch budget.

The original run 6 left StandardSparseSOM at its default --epochs 10, so
with --stop kl the progress-driven σ schedule never reached refinement
and quality saturated (flat QE ≈0.60 at every edge). This re-runs only
the ssom-feat / ssom-node box+kl cells with --epochs 200 so the KL
plateau stop can fire, evaluates held-out metrics, and replaces those
rows in impl_compare.parquet (originals preserved in
impl_compare_ssom_kl_10ep.parquet).

Usage: rerun_ssom_kl_arms.py RESULTS_DIR DATA_DIR [--edges 32,64,128,256]
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.runner import run_standardsparsesom, run_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("data_dir")
    ap.add_argument("--edges", default="32,64,128,256")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    results = Path(args.results_dir)
    data = Path(args.data_dir)
    corpus = str(data / "corpus.train.sbcsr")
    eval_corpus = str(data / "corpus.heldout.sbcsr")

    out = results / "impl_compare.parquet"
    df = pd.read_parquet(out)

    backup = results / "impl_compare_ssom_kl_10ep.parquet"
    if not backup.exists():
        old = df[(df.impl.isin(["ssom-feat", "ssom-node"]))
                 & (df.update_variant == "box+kl")]
        old.to_parquet(backup, index=False)
        print(f"Backed up {len(old)} 10-epoch rows -> {backup}")

    rows = []
    for edge in [int(e) for e in args.edges.split(",")]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            for impl in ["ssom-feat", "ssom-node"]:
                layout = "feature" if impl == "ssom-feat" else "node"
                print(f"  rerun: edge={edge} seed={seed} impl={impl} "
                      f"update=box+kl epochs<=200 ...", flush=True)

                with tempfile.NamedTemporaryFile(suffix=".somw",
                                                 delete=False) as tmp:
                    weights_path = tmp.name

                r = run_standardsparsesom(
                    corpus, edge, seed,
                    layout=layout, neighbourhood="box", stop="kl",
                    precision="fp16",
                    extra_flags=["--epochs", "200",
                                 "--save-weights", weights_path],
                )

                row = {"edge": edge, "impl": impl, "update_variant": "box+kl",
                       "precision": "fp16", "seed": seed,
                       "oom": bool(r.get("oom", False))}

                if not r.get("oom"):
                    done = r.get("done", {}) or {}
                    epochs = done.get("epochs", 0)
                    try:
                        metrics = run_metrics(eval_corpus, weights_path,
                                              codebook_format="somw",
                                              rows=edge, cols=edge)
                    except Exception as e:
                        print(f"    WARNING: metrics failed: {e}")
                        metrics = {}
                    finally:
                        Path(weights_path).unlink(missing_ok=True)

                    row.update({
                        "wall_s": done.get("wall_s", 0),
                        "bmu_s": done.get("bmu_s", 0),
                        "bmu_s_per_ep": done.get("bmu_s", 0) / max(epochs, 1),
                        "upd_s_per_ep": done.get("update_s", 0) / max(epochs, 1),
                        "epochs": epochs,
                        "qe": metrics.get("qe_cosine", None),
                        "te": metrics.get("topographic_error", None),
                        "dead_frac": metrics.get("dead_fraction", None),
                    })
                else:
                    Path(weights_path).unlink(missing_ok=True)

                rows.append(row)

                # replace-and-persist incrementally: all 10-epoch ssom box+kl
                # rows are invalid (and backed up), so drop them wholesale
                keep = df[~((df.impl.isin(["ssom-feat", "ssom-node"]))
                            & (df.update_variant == "box+kl"))]
                pd.concat([keep, pd.DataFrame(rows)],
                          ignore_index=True).to_parquet(out, index=False)

    print(f"\nReplaced {len(rows)} rows in {out}")


if __name__ == "__main__":
    main()
