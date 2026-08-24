"""Run 9 — PCA σ₀ optimisation: does PCA earn its place under det-exp?

Tests whether PCA init with a reduced σ₀ saves epochs vs random init
at σ₀ = 0.5·E, without degrading quality. Side-experiment; does not
gate the main size sweep.

Arms:
  A  PCA    σ₀ = 0.50·E   1 seed (deterministic)
  B  PCA    σ₀ = 0.25·E   1 seed
  C  PCA    σ₀ = 0.125·E  1 seed
  R  random σ₀ = 0.50·E   5 seeds (ensemble)
  R' random σ₀ = 0.25·E   3 seeds (control)
"""

from pathlib import Path

import pandas as pd

from ..runner import run_sparsesom


EDGES = [32, 64, 128, 256, 512]

ARMS = [
    {"name": "A",  "init": "pca",    "sigma0_frac": 0.50,  "seeds": [0]},
    {"name": "B",  "init": "pca",    "sigma0_frac": 0.25,  "seeds": [0]},
    {"name": "C",  "init": "pca",    "sigma0_frac": 0.125, "seeds": [0]},
    {"name": "R",  "init": "random", "sigma0_frac": 0.50,  "seeds": [0, 1, 2, 3, 4]},
    {"name": "R'", "init": "random", "sigma0_frac": 0.25,  "seeds": [0, 1, 2]},
]


def run(data_dir: str, results_dir: str, edges=None, **kwargs) -> Path:
    data = Path(data_dir)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = str(data / "corpus.train.sbcsr")
    eval_corpus = str(data / "corpus.heldout.sbcsr")
    pca_file = str(data / "corpus.sompca")

    edges = edges or EDGES
    summary_rows = []
    epoch_rows = []

    for arm in ARMS:
        for edge in edges:
            sigma0 = arm["sigma0_frac"] * edge
            for seed in arm["seeds"]:
                label = (f"  {arm['name']}: edge={edge} init={arm['init']} "
                         f"σ₀={arm['sigma0_frac']}·E={sigma0:.1f} seed={seed}")
                print(label)

                pca_path = pca_file if arm["init"] == "pca" else None

                r = run_sparsesom(
                    corpus, edge, seed,
                    binary=True,
                    eval_corpus=eval_corpus,
                    pca_init=pca_path,
                    sigma_init=sigma0,
                )

                done = r.get("done") or {}
                metrics = r.get("metrics") or {}

                summary_rows.append({
                    "arm": arm["name"],
                    "edge": edge,
                    "init": arm["init"],
                    "sigma0_frac": arm["sigma0_frac"],
                    "seed": seed,
                    "epochs": done.get("epochs", 0),
                    "wall_s": done.get("wall_s", 0),
                    "qe_heldout": metrics.get("qe_cosine", None),
                    "qe_eucl": metrics.get("qe_euclidean", None),
                    "te": metrics.get("topographic_error", None),
                    "dead_frac": metrics.get("dead_fraction", None),
                    "converged": done.get("converged", False),
                })

                for ep in r.get("epochs", []):
                    epoch_rows.append({
                        "arm": arm["name"],
                        "edge": edge,
                        "init": arm["init"],
                        "sigma0_frac": arm["sigma0_frac"],
                        "seed": seed,
                        "epoch": ep["epoch"],
                        "sigma": ep.get("sigma", None),
                        "qe": ep.get("qe", None),
                        "te": ep.get("te", None),
                        "dead_frac": ep.get("dead_frac", None),
                    })

    df = pd.DataFrame(summary_rows)
    df_ep = pd.DataFrame(epoch_rows)

    out_summary = out_dir / "sigma0_sweep.parquet"
    out_epochs = out_dir / "sigma0_sweep_epochs.parquet"
    df.to_parquet(out_summary, index=False)
    df_ep.to_parquet(out_epochs, index=False)

    print(f"\nσ₀ sweep: {out_summary} ({len(df)} rows)")
    print(f"Per-epoch: {out_epochs} ({len(df_ep)} rows)")

    _print_decision_table(df)
    return out_summary


def _print_decision_table(df: pd.DataFrame):
    """Print the Q1/Q2 decision table."""
    print("\n" + "=" * 70)
    print("Q1: Smallest safe σ₀ per edge (PCA arms)")
    print("=" * 70)

    baseline = df[df["arm"] == "A"].set_index("edge")

    for arm_name in ["B", "C"]:
        arm = df[df["arm"] == arm_name].set_index("edge")
        print(f"\n  Arm {arm_name} (σ₀={arm['sigma0_frac'].iloc[0]}·E) vs A (0.5·E):")
        for edge in sorted(arm.index):
            if edge not in baseline.index:
                continue
            b = baseline.loc[edge]
            a = arm.loc[edge]
            dqe = (a["qe_heldout"] - b["qe_heldout"]) / b["qe_heldout"] * 100
            dte = a["te"] - b["te"]
            ddead = a["dead_frac"] - b["dead_frac"]
            epoch_save = b["epochs"] - a["epochs"]
            safe = (abs(dqe) <= 0.5 and dte <= 0.005 and ddead <= 0.02)
            tag = "SAFE" if safe else "UNSAFE"
            print(f"    edge={edge:4d}  ΔQE={dqe:+.2f}%  ΔTE={dte:+.4f}  "
                  f"Δdead={ddead:+.3f}  epochs saved={epoch_save:+d}  [{tag}]")

    print("\n" + "=" * 70)
    print("Q2: PCA@best-σ₀ vs random@0.5·E")
    print("=" * 70)

    random_ref = df[df["arm"] == "R"].groupby("edge").agg(
        epochs_mean=("epochs", "mean"),
        qe_mean=("qe_heldout", "mean"),
        te_mean=("te", "mean"),
    )

    for edge in sorted(baseline.index):
        if edge not in random_ref.index:
            continue
        b = baseline.loc[edge]
        rr = random_ref.loc[edge]
        pca_ep = b["epochs"]
        rand_ep = rr["epochs_mean"]
        saving_pct = (rand_ep - pca_ep) / rand_ep * 100
        print(f"  edge={edge:4d}  PCA(0.5E)={pca_ep:.0f}ep  "
              f"random(0.5E)={rand_ep:.1f}ep  saving={saving_pct:+.1f}%")
