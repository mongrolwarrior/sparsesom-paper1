#!/usr/bin/env python3
"""Promote σ₀-sweep arm A runs (PCA @ 0.5·E, seed 0) into size-sweep format.

The σ₀ experiment (run 9) trained PCA-init maps at σ₀ = 0.5·E on edges
32-512 with the identical command line the size sweep (run 5) would use
(same corpus split, PCA init, deterministic σ schedule, KL stop).  Those
runs are therefore valid seed-0 size-sweep points; this script extracts
them from the captured stdout log and writes them in run-5 schema so
they can be merged with the remaining seeds instead of re-running.

Usage:
    promote_sigma0_to_size_sweep.py LOGFILE RESULTS_DIR
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import parsers  # noqa: E402

LABEL_RE = re.compile(
    r"^  (?P<arm>[A-Z]'?): edge=(?P<edge>\d+) init=(?P<init>\w+) "
    r"σ₀=(?P<frac>[\d.]+)·E=[\d.]+ seed=(?P<seed>\d+)")


def split_runs(log_text: str):
    """Yield (label_dict, stdout_segment) per run in the log."""
    lines = log_text.splitlines()
    current = None
    seg = []
    for line in lines:
        m = LABEL_RE.match(line)
        if m:
            if current:
                yield current, "\n".join(seg)
            current = m.groupdict()
            seg = []
        elif current is not None:
            seg.append(line)
    if current:
        yield current, "\n".join(seg)


def main():
    log_path, results_dir = Path(sys.argv[1]), Path(sys.argv[2])
    log_text = log_path.read_text(errors="replace")

    sweep_rows = []
    epoch_rows = []

    for label, seg in split_runs(log_text):
        if label["arm"] != "A":
            continue
        edge = int(label["edge"])
        seed = int(label["seed"])

        r = parsers.parse_sparsesom_stdout(seg)
        done = r["done"] or {}
        metrics = r["metrics"] or {}
        if not done:
            print(f"WARNING: no done-line for arm A edge={edge}; skipped")
            continue

        sweep_rows.append({
            "edge": edge,
            "neurons": edge * edge,
            "seed": seed,
            "epochs": done.get("epochs", 0),
            "wall_s": done.get("wall_s", 0),
            "qe_heldout": metrics.get("qe_cosine", None),
            "qe_eucl": metrics.get("qe_euclidean", None),
            "te": metrics.get("topographic_error", None),
            "dead_frac": metrics.get("dead_fraction", None),
            "converged": done.get("converged", False),
        })

        n_ep = max(done.get("epochs", 1), 1)
        bmu_per_epoch = done.get("bmu_s", 0) / n_ep
        update_per_epoch = done.get("update_s", 0) / n_ep
        for ep in r["epochs"]:
            epoch_rows.append({
                "edge": edge,
                "seed": seed,
                "epoch": ep["epoch"],
                "sigma": ep["sigma"],
                "qe_heldout": ep["qe"],
                "bmu_s": bmu_per_epoch,
                "update_s": update_per_epoch,
            })

    df_sweep = pd.DataFrame(sweep_rows).sort_values("edge")
    df_epochs = pd.DataFrame(epoch_rows).sort_values(["edge", "epoch"])

    out_sweep = results_dir / "size_sweep_seed0_promoted.parquet"
    out_epochs = results_dir / "size_sweep_epochs_seed0_promoted.parquet"
    df_sweep.to_parquet(out_sweep, index=False)
    df_epochs.to_parquet(out_epochs, index=False)

    print(f"Promoted {len(df_sweep)} arm-A runs -> {out_sweep}")
    print(f"Per-epoch rows: {len(df_epochs)} -> {out_epochs}")
    print(df_sweep.to_string(index=False))


if __name__ == "__main__":
    main()
