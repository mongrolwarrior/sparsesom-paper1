"""Run 6 — Cross-implementation & update-rule comparison (Tables 3, 4, 6).

The cuSPARSE baselines (ssom-feat, ssom-node) run at FP16 precision to match
SparseBinarySOM's __half codebook, ensuring an apples-to-apples bandwidth comparison.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..runner import run_sparsesom, run_standardsparsesom, run_metrics


def run(data_dir: str, results_dir: str, edges=None, seeds=None, **kwargs) -> Path:
    """Run sbsom vs ssom-feat vs ssom-node with update variants."""
    data = Path(data_dir)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = str(data / "corpus.train.sbcsr")
    eval_corpus = str(data / "corpus.heldout.sbcsr")
    pca_file = str(data / "corpus.sompca")

    default_edges = [32, 64, 128, 256]
    edges = edges or default_edges
    seeds = seeds or [0, 1, 2, 3, 4]

    impls = ["sbsom-bin", "ssom-feat", "ssom-node"]
    update_variants = ["gaussian", "box", "box+kl"]

    rows = []

    for edge in edges:
        for seed in seeds:
            for impl in impls:
                for variant in update_variants:
                    print(f"  impl_compare: edge={edge} seed={seed} "
                          f"impl={impl} update={variant} ...")

                    try:
                        result, metrics = _run_one(
                            impl, variant, corpus, eval_corpus, pca_file,
                            edge, seed,
                        )
                    except RuntimeError as e:
                        if "OOM" in str(e) or "out of memory" in str(e).lower():
                            result = {"oom": True}
                            metrics = {}
                        else:
                            raise

                    precision = "fp16" if impl == "sbsom-bin" else "fp16"
                    row = {
                        "edge": edge,
                        "impl": impl,
                        "update_variant": variant,
                        "precision": precision,
                        "seed": seed,
                        "oom": result.get("oom", False),
                    }

                    if not result.get("oom"):
                        done = result.get("done", {}) or {}
                        epochs = done.get("epochs", 0)
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

                    rows.append(row)

    df = pd.DataFrame(rows)
    out = out_dir / "impl_compare.parquet"
    df.to_parquet(out, index=False)

    print(f"\nImpl compare: {out} ({len(df)} rows)")

    _print_layout_ratio(df)
    return out


def _print_layout_ratio(df: pd.DataFrame):
    """Print node-vs-feature BMU ratio with CI for the layout claim."""
    ok = df[(~df["oom"]) & (df["update_variant"] == "box+kl")]
    feat = ok[ok["impl"] == "ssom-feat"]
    node = ok[ok["impl"] == "ssom-node"]

    if feat.empty or node.empty:
        return

    print("\n" + "=" * 70)
    print("Layout claim: node-major / feature-major BMU ratio (matched precision)")
    print("=" * 70)

    for edge in sorted(feat["edge"].unique()):
        f_seeds = feat[feat["edge"] == edge].set_index("seed")["bmu_s_per_ep"]
        n_seeds = node[node["edge"] == edge].set_index("seed")["bmu_s_per_ep"]
        common = f_seeds.index.intersection(n_seeds.index)
        if len(common) < 2:
            continue
        log_ratios = np.log(n_seeds[common].values / f_seeds[common].values)
        n = len(log_ratios)
        mean_lr = log_ratios.mean()
        se = log_ratios.std(ddof=1) / np.sqrt(n)
        from scipy.stats import t as t_dist
        t_crit = t_dist.ppf(0.975, df=n - 1)
        ci_lo = np.exp(mean_lr - t_crit * se)
        ci_hi = np.exp(mean_lr + t_crit * se)
        ratio = np.exp(mean_lr)
        print(f"  edge={edge:4d}  ratio={ratio:.2f}x  "
              f"95% CI=[{ci_lo:.2f}, {ci_hi:.2f}]  (n={n} seeds)")


def _run_one(impl: str, variant: str, corpus: str, eval_corpus: str,
             pca_file: str, edge: int, seed: int) -> tuple[dict, dict]:
    """Dispatch to the appropriate binary, return (run_result, heldout_metrics)."""
    neighbourhood = "box" if "box" in variant else "gaussian"
    stop = "kl" if "+kl" in variant else "fixed"

    if impl == "sbsom-bin":
        r = run_sparsesom(
            corpus, edge, seed,
            binary=True,
            eval_corpus=eval_corpus,
            pca_init=pca_file,
            extra_flags=_sparsesom_flags(variant),
        )
        return r, r.get("metrics", {}) or {}

    elif impl in ("ssom-feat", "ssom-node"):
        layout = "feature" if impl == "ssom-feat" else "node"

        with tempfile.NamedTemporaryFile(suffix=".somw", delete=False) as tmp:
            weights_path = tmp.name

        # KL-stop needs a real epoch budget or StandardSparseSOM's default
        # --epochs 10 caps it before refinement, freezing quality (the
        # equal-quality confound). With --stop kl the sigma schedule is
        # progress-driven, so a high cap is safe — the plateau stop fires.
        extra = ["--save-weights", weights_path]
        if stop == "kl":
            extra += ["--epochs", "200"]

        r = run_standardsparsesom(
            corpus, edge, seed,
            layout=layout,
            neighbourhood=neighbourhood,
            stop=stop,
            precision="fp16",
            extra_flags=extra,
        )

        if r.get("oom"):
            return r, {}

        try:
            metrics = run_metrics(
                eval_corpus, weights_path,
                codebook_format="somw",
                rows=edge, cols=edge,
            )
        except Exception as e:
            print(f"    WARNING: metrics eval failed for {impl}: {e}")
            metrics = {}
        finally:
            Path(weights_path).unlink(missing_ok=True)

        return r, metrics

    else:
        raise ValueError(f"Unknown impl: {impl}")


def _sparsesom_flags(variant: str) -> list[str]:
    if variant == "gaussian":
        return ["--sigma-sched", "linear"]
    elif variant == "box":
        return ["--sigma-sched", "exp"]
    elif variant == "box+kl":
        return []
    return []
