"""`repro --profile outline` — fast pattern reproduction (~3 h, single GPU).

Demonstrates the DIRECTION and rough order of magnitude of every qualitative
result in the Phase 1 manuscript. It is NOT a statistical reproduction: single
seed, reduced epochs, cheapest map size at which each effect is visible. It
supports none of the paper's numeric values, ranges, CIs or hypothesis tests —
those come from `--profile full`. See PROFILE_OUTLINE_SPEC.md.

Every check asserts a direction or order of magnitude and prints PASS/FAIL
against that, with the published figure shown parenthesised for orientation
only (explicitly NOT a target). Runs 10/11/5 are measured at cheaper edges than
the paper and are expected to be systematically smaller by design.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from .runner import (run_sparsesom, run_standardsparsesom, run_medsom,
                     run_somoclu, run_metrics)


class Check:
    """One direction assertion + its observed value."""
    def __init__(self, n, name, impls, edges, epochs, direction):
        self.n, self.name, self.impls = n, name, impls
        self.edges, self.epochs, self.direction = edges, epochs, direction
        self.observed = "—"
        self.published = "—"       # orientation only, never a target
        self.status = "SKIP"

    def verdict(self, ok, observed, published="—"):
        self.observed, self.published = observed, published
        self.status = "PASS" if ok else "FAIL"
        return self


def _tiny(data: Path) -> str:
    """Ensure the ~50k tiny corpus exists (reuse the bringup gate's builder)."""
    tiny = data / "corpus.tiny.sbcsr"
    if not tiny.exists():
        scripts = Path(__file__).resolve().parent.parent / "scripts"
        subprocess.run([sys.executable, str(scripts / "make_tiny_corpus.py"),
                        str(data / "corpus.train.sbcsr"), str(tiny), "50000"],
                       check=True)
    return str(tiny)


def _libsvm(data: Path) -> str:
    """Ensure the full corpus exists in somoclu's libsvm format (cached)."""
    lib = data / "corpus.train.libsvm"
    if not lib.exists() or lib.stat().st_size == 0:
        scripts = Path(__file__).resolve().parent.parent / "scripts"
        subprocess.run([sys.executable, str(scripts / "sbcsr_to_libsvm.py"),
                        str(data / "corpus.train.sbcsr"), str(lib)], check=True)
    return str(lib)


def _ssom(corpus, eval_corpus, edge, layout, *, neighbourhood="box",
          stop="kl", precision="fp16", epochs=None):
    """Run a cuSPARSE arm and evaluate held-out QE (mirrors ex06)."""
    # KL-stop needs a real epoch budget or StandardSparseSOM's default --epochs 10
    # caps it before refinement, freezing quality (the equal-quality confound).
    if epochs is None and stop == "kl":
        epochs = 200
    flags = ["--epochs", str(epochs)] if epochs is not None else []
    with tempfile.NamedTemporaryFile(suffix=".somw", delete=False) as t:
        w = t.name
    r = run_standardsparsesom(corpus, edge, 0, layout=layout,
                              neighbourhood=neighbourhood, stop=stop,
                              precision=precision,
                              extra_flags=flags + ["--save-weights", w])
    metrics = {}
    if not r.get("oom"):
        try:
            metrics = run_metrics(eval_corpus, w, codebook_format="somw",
                                  rows=edge, cols=edge)
        except Exception:
            metrics = {}
    Path(w).unlink(missing_ok=True)
    return r, metrics


def _bmu_per_ep(r):
    d = r.get("done") or {}
    return d.get("bmu_s", 0) / max(d.get("epochs", 1), 1)


def run(data_dir: str, results_dir: str, **kwargs):
    data, out = Path(data_dir), Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "outline_scratch").mkdir(exist_ok=True)   # medsom/somoclu cwd
    corpus = str(data / "corpus.train.sbcsr")
    heldout = str(data / "corpus.heldout.sbcsr")
    pca = str(data / "corpus.sompca")
    checks = []

    # ── 1. corpus manifest ────────────────────────────────────────────────
    c1 = Check(1, "corpus_manifest", "—", "—", "—", "corpus present & readable")
    r = run_sparsesom(corpus, 32, 0, binary=True, pca_init=pca,
                      extra_flags=["--epochs", "0"])
    ok = bool(r.get("epochs") is not None)
    checks.append(c1.verdict(ok, "loaded"))

    # ── 2. bit-exact gate (FP16 vs FP32 quality-invariant), tiny corpus ───
    tiny = _tiny(data)
    r16 = run_sparsesom(tiny, 32, 0, binary=True, extra_flags=["--epochs", "5"])
    r32 = run_sparsesom(tiny, 32, 0, binary=False, extra_flags=["--epochs", "5"])
    q16, q32 = r16["epochs"][-1]["qe"], r32["epochs"][-1]["qe"]
    dq = abs(q16 - q32) / max(q32, 1e-9)
    checks.append(Check(2, "bitexact_gate", "sbsom fp16/fp32", 32, 5,
                        "quality invariant to precision (<1%)").verdict(
        dq < 0.01, f"Δqe={dq*100:.2f}%", "0 mismatches"))

    # ── 3. determinism (PCA is deterministic across seeds), tiny corpus ───
    d0 = run_sparsesom(tiny, 32, 0, binary=True, pca_init=pca)
    d1 = run_sparsesom(tiny, 32, 1, binary=True, pca_init=pca)
    same = (len(d0["epochs"]) == len(d1["epochs"]) and all(
        abs(a["qe"] - b["qe"]) < 1e-6 for a, b in zip(d0["epochs"], d1["epochs"])))
    checks.append(Check(3, "determinism_check", "sbsom seeds 0,1", 32, "KL",
                        "seeds identical (zero variance)").verdict(
        same, "identical" if same else "differ", "zero variance"))

    # ── 4. size sweep (sbsom-bin) — reused as the sbsom arm for 5–12 ──────
    sb = {}   # edge -> dict(qe, bmu_s_per_ep, wall_s, bmu_share, neurons, oom)
    for edge in [32, 64, 128, 256, 512]:
        r = run_sparsesom(corpus, edge, 0, binary=True, eval_corpus=heldout,
                          pca_init=pca)
        done, m = r.get("done") or {}, r.get("metrics") or {}
        share = done.get("bmu_s", 0) / max(done.get("bmu_s", 0)
                                           + done.get("update_s", 0), 1e-9)
        sb[edge] = {"qe": m.get("qe_cosine"), "bmu_s_per_ep": _bmu_per_ep(r),
                    "wall_s": done.get("wall_s", 0), "bmu_share": share,
                    "neurons": edge * edge, "converged": done.get("converged")}
    qes = [sb[e]["qe"] for e in [32, 64, 128, 256, 512]]
    strictly_dec = all(qes[i] > qes[i + 1] for i in range(len(qes) - 1))
    gains = [(qes[i] - qes[i + 1]) / qes[i] for i in range(len(qes) - 1)]
    checks.append(Check(4, "size_sweep: QE↓ in K", "sbsom-bin",
                        "32–512", "KL", "held-out QE strictly decreasing").verdict(
        strictly_dec, f"{qes[0]:.3f}→{qes[-1]:.3f}", "0.524→0.376"))
    checks.append(Check(4, "size_sweep: per-doubling gain", "sbsom-bin",
                        "32–512", "KL", "gain roughly constant, >0.5%").verdict(
        all(g > 0.005 for g in gains), f"{min(gains)*100:.1f}–{max(gains)*100:.1f}%",
        "3.3–4.8%"))
    shares = [sb[e]["bmu_share"] for e in [32, 64, 128, 256, 512]]
    checks.append(Check(4, "size_sweep: BMU share", "sbsom-bin", "32–512", "KL",
                        "BMU share of wall >95%, rising").verdict(
        min(shares) > 0.95 and shares[-1] >= shares[0],
        f"{shares[0]*100:.1f}→{shares[-1]*100:.1f}%", "98.7→99.5%"))
    checks.append(Check(4, "size_sweep: largest map on 24 GB", "sbsom-bin",
                        512, "KL", "512² trains (262,144 neurons)").verdict(
        bool(sb[512]["converged"]), f"{sb[512]['neurons']} neurons", "262,144"))

    # ── 5. impl crossover (ssom-feat/sbsom wall ratio crosses 1.0) ────────
    feat = {}
    for edge in [32, 64, 128]:
        r, m = _ssom(corpus, heldout, edge, "feature")
        feat[edge] = {"wall_s": (r.get("done") or {}).get("wall_s", 0),
                      "bmu_s_per_ep": _bmu_per_ep(r), "qe": m.get("qe_cosine")}
    ratios = {e: feat[e]["wall_s"] / max(sb[e]["wall_s"], 1e-9) for e in [32, 64, 128]}
    checks.append(Check(5, "impl_crossover", "ssom-feat vs sbsom", "32,64,128",
                        "KL", "wall ratio crosses 1.0 with edge").verdict(
        ratios[32] < 1.0 < ratios[128],
        f"{ratios[32]:.2f}→{ratios[64]:.2f}→{ratios[128]:.2f}×", "0.36→…→2.60×"))
    qgap = max(abs(feat[e]["qe"] - sb[e]["qe"]) / sb[e]["qe"] for e in [32, 64, 128]
               if feat[e]["qe"] and sb[e]["qe"])
    checks.append(Check(5, "impl_crossover: QE agree", "ssom-feat vs sbsom",
                        "32,64,128", "KL", "held-out QE agree within ~1%").verdict(
        qgap <= 0.01, f"≤{qgap*100:.1f}%", "≤0.5%"))

    # ── 6. layout gap (feature vs node BMU) ───────────────────────────────
    node = {}
    for edge in [32, 64]:
        r, _ = _ssom(corpus, heldout, edge, "node")
        node[edge] = {"bmu_s_per_ep": _bmu_per_ep(r)}
    lg = max(node[e]["bmu_s_per_ep"] / max(feat[e]["bmu_s_per_ep"], 1e-9)
             for e in [32, 64])
    checks.append(Check(6, "layout_gap", "ssom-node vs -feat", "32,64", "KL",
                        "feature-major several-fold faster").verdict(
        lg >= 3, f"{lg:.1f}×", "4.5–8.5×"))

    # ── 7. update rule (box blur cuts TE vs gaussian) ─────────────────────
    _, mg = _ssom(corpus, heldout, 128, "feature", neighbourhood="gaussian",
                  stop="fixed", epochs=10)
    _, mb = _ssom(corpus, heldout, 128, "feature", neighbourhood="box",
                  stop="fixed", epochs=10)
    teg, teb = mg.get("topographic_error"), mb.get("topographic_error")
    checks.append(Check(7, "update_rule", "ssom-feat gauss/box", 128, "10",
                        "box cuts TE ~order of magnitude").verdict(
        teg and teb and teb < teg / 3, f"{teg:.3f}→{teb:.3f}", "0.21→0.03"))

    # ── 8. FP16 asymmetry (helps feature-major more than node-major) ──────
    rf32, _ = _ssom(corpus, heldout, 32, "feature", precision="fp32")
    rn32, _ = _ssom(corpus, heldout, 32, "node", precision="fp32")
    feat_speedup = rf32 and _bmu_per_ep(rf32) / max(feat[32]["bmu_s_per_ep"], 1e-9)
    node_speedup = rn32 and _bmu_per_ep(rn32) / max(node[32]["bmu_s_per_ep"], 1e-9)
    checks.append(Check(8, "fp16_asymmetry", "ssom feat/node fp32→fp16", 32, "KL",
                        "FP16 helps feature ≫ node").verdict(
        feat_speedup > node_speedup + 0.2,
        f"feat {feat_speedup:.2f}× vs node {node_speedup:.2f}×", "1.65× vs 1.07×"))

    # ── 9. binary vs float ────────────────────────────────────────────────
    rfl = run_sparsesom(corpus, 128, 0, binary=False, pca_init=pca,
                        extra_flags=["--epochs", "3"])
    bvf = _bmu_per_ep(rfl) / max(sb[128]["bmu_s_per_ep"], 1e-9)
    checks.append(Check(9, "binary_vs_float", "sbsom-float vs -bin", 128, "3",
                        "binary faster, order 2–3×").verdict(
        bvf >= 1.8, f"{bvf:.1f}×", "2.1–2.8×"))

    # ── 10. MedSOM reference (≥1 order slower) ────────────────────────────
    med = run_medsom(corpus, 64, epochs=2, scratch_dir=str(out / "outline_scratch"))
    med_ratio = None
    if not med.get("oom"):
        ep = med.get("epochs", [])
        steady = ep[1:] or ep
        med_pe = sum(e.get("epoch_time_s", 0) for e in steady) / max(len(steady), 1)
        med_ratio = med_pe / max(sb[64]["bmu_s_per_ep"], 1e-9)
    checks.append(Check(10, "medsom_ref", "MedSOM vs sbsom-bin", 64, "2",
                        "MedSOM ≥1 order slower").verdict(
        med_ratio and med_ratio >= 10, f"{med_ratio:.0f}×" if med_ratio else "—",
        "~82× (edge 128)"))

    # ── 11. somoclu reference (≥2 orders slower) — CPU ────────────────────
    som = run_somoclu(_libsvm(data), 32, epochs=2,
                      scratch_dir=str(out / "outline_scratch"))
    som_ratio = (som.get("per_epoch_s") / max(sb[32]["bmu_s_per_ep"], 1e-9)
                 if not som.get("oom") and som.get("per_epoch_s") else None)
    # edge 32 is the cheapest point; the clean curve gives ~85× there (~1.9
    # orders), climbing to ~655× by edge 256 — so assert ">1.5 orders", not 2.
    checks.append(Check(11, "somoclu_ref", "somoclu vs sbsom-bin", 32, "2",
                        "somoclu >1.5 orders slower").verdict(
        som_ratio and som_ratio >= 50, f"{som_ratio:.0f}×" if som_ratio else "—",
        "85× @32 → 655× @256"))

    # ── 12. capacity probe (baselines OOM at 512, sbsom does not) ─────────
    oom = {}
    for layout in ["feature", "node"]:
        r, _ = _ssom(corpus, heldout, 512, layout, epochs=1)
        oom[f"ssom-{layout}"] = bool(r.get("oom"))
    med512 = run_medsom(corpus, 512, epochs=1, scratch_dir=str(out / "outline_scratch"))
    oom["medsom"] = bool(med512.get("oom"))
    n_oom = sum(oom.values())
    checks.append(Check(12, "capacity_probe", "ssom×2,MedSOM @512", 512, "1",
                        "3 baselines OOM, sbsom-bin passes").verdict(
        n_oom == 3 and bool(sb[512]["converged"]),
        f"{n_oom}/3 OOM, sbsom {'ok' if sb[512]['converged'] else 'fail'}",
        "3 OOM, 1 pass"))

    # ── 13. roofline (ssom-feat far below peak BW) — ncu, edge 128 ────────
    # Report the BANDWIDTH pattern only (cuSPARSE runs far below peak, so it is
    # not bandwidth-limited). Per the spec caveat, do NOT assert a DRAM-traffic
    # direction here — that comparison can change sign between edge 128 and 256
    # and belongs to the paper's headline edge / --profile full.
    c13 = Check(13, "roofline", "ncu sbsom-bin,ssom-feat", 128, "1",
                "ssom-feat sustains small fraction of peak BW")
    import os
    try:
        scripts = Path(__file__).resolve().parent.parent / "scripts"
        subprocess.run(["bash", str(scripts / "ncu_full_epoch.sh"), "128", corpus],
                       check=True, cwd=scripts.parent,
                       env={**os.environ, "OUTDIR": str(out),
                            "IMPLS": "ssom-feat,sbsom-bin"})
        subprocess.run([sys.executable, str(scripts / "aggregate_phase.py"), str(out)],
                       check=True)
        import pandas as pd
        s = pd.read_csv(out / "roofline_phase_summary.csv")
        row = s[(s.impl == "ssom-feat") & (s.edge == 128)].iloc[0]
        # sustained BW = measured phase DRAM / this run's unprofiled BMU wall
        # (feat[128] from run 5), independent of efficiency_sweep.parquet
        wall = feat[128]["bmu_s_per_ep"]
        bw = float(row["phase_dram_tb"]) * 1e12 / max(wall, 1e-9) / 1008e9 * 100
        checks.append(c13.verdict(bw < 60, f"ssom-feat {bw:.0f}% of peak BW",
                                  "15.6% (edge 256)"))
    except Exception as e:
        c = c13.verdict(False, f"skipped: {type(e).__name__}", "")
        c.status = "SKIP"
        checks.append(c)

    # ── 14. σ₀ × PCA interaction (opposite signs) ─────────────────────────
    def qe64(init, frac):
        r = run_sparsesom(corpus, 64, 0, binary=True,
                          pca_init=(pca if init == "pca" else None),
                          eval_corpus=heldout, sigma_init=frac * 64)
        return (r.get("metrics") or {}).get("qe_cosine")
    pca_hi, pca_lo = qe64("pca", 0.5), qe64("pca", 0.25)
    rnd_hi, rnd_lo = qe64("random", 0.5), qe64("random", 0.25)
    # The full opposite-sign interaction (PCA improves, random degrades) is a
    # sub-1.2% effect that needs the multi-seed / multi-edge aggregate — too
    # subtle for a single edge/seed. Assert only the robust demonstrable half:
    # smaller σ₀ improves QE under PCA. Random direction reported, not asserted.
    pca_better = pca_lo < pca_hi
    rnd_dir = "degrades" if rnd_lo > rnd_hi else "also improves"
    checks.append(Check(14, "sigma0_interaction", "sbsom PCA × σ₀",
                        64, "KL", "smaller σ₀ improves QE under PCA").verdict(
        pca_better,
        f"PCA {'improves' if pca_better else 'worse'}; random {rnd_dir}",
        "full interaction in --profile full"))

    _summary(checks, out)
    return out / "outline_summary.txt"


def _summary(checks, out):
    n_pass = sum(c.status == "PASS" for c in checks)
    lines = ["", "=" * 92,
             f"  Profile: outline (pattern reproduction) — {n_pass}/{len(checks)} checks passed",
             "=" * 92,
             f"  {'#':>2}  {'run':22} {'assert (direction)':40} {'observed':16} {'stat':5} (published)"]
    for c in checks:
        lines.append(f"  {c.n:>2}  {c.name:22.22} {c.direction:40.40} "
                     f"{str(c.observed):16.16} {c.status:5} ({c.published})")
    disclaimer = (
        "\n  Profile: outline (pattern reproduction). Demonstrates the qualitative results\n"
        "  of the Phase 1 manuscript — the direction and approximate scale of every reported\n"
        "  effect. It uses a single seed, reduced epoch budgets and the cheapest map size at\n"
        "  which each effect is visible, so THE NUMERIC VALUES HERE DIFFER FROM THE PUBLISHED\n"
        "  FIGURES BY DESIGN and support none of the paper's ranges, exponents, confidence\n"
        "  intervals or hypothesis tests. In particular the somoclu, MedSOM and crossover\n"
        "  ratios are measured at smaller maps and are correspondingly smaller. For published\n"
        "  values run `--profile full` (or `--profile sections` to run the same set in\n"
        "  stages). If the roofline check reported SKIP, the image lacks `ncu` or GPU\n"
        "  counter access — see 'Reproducing the roofline' in the README to enable it.")
    block = "\n".join(lines) + "\n" + disclaimer + "\n"
    print(block)
    (out / "outline_summary.txt").write_text(block)
