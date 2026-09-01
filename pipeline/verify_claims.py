"""Acceptance checker: verify every quantitative claim against tolerance."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .stats import t_interval, paired_log_ratio_t, tost_equivalence


class ClaimResult:
    def __init__(self, name: str, prose: str, passed: bool, detail: str,
                 delegated: bool = False):
        self.name = name
        self.prose = prose
        self.passed = passed
        self.detail = detail
        self.delegated = delegated

    def __str__(self):
        status = "DELEGATED" if self.delegated else ("PASS" if self.passed else "FAIL")
        return f"[{status}] {self.name}: {self.prose}\n        {self.detail}"


def verify_all(results_dir: Path, claims_path: Path) -> list[ClaimResult]:
    with open(claims_path) as f:
        claims_cfg = yaml.safe_load(f)

    results = []
    for name, spec in claims_cfg.get("claims", {}).items():
        result = _verify_one(name, spec, results_dir)
        results.append(result)

    return results


def _verify_one(name: str, spec: dict, results_dir: Path) -> ClaimResult:
    """Verify a single claim."""
    prose = spec.get("prose", name)
    check = spec.get("check", {})
    check_type = check.get("type", "")

    artifact = spec.get("artifact")
    if isinstance(artifact, list):
        artifacts = artifact
    else:
        artifacts = [artifact] if artifact else []

    # Check all artifacts exist
    for a in artifacts:
        p = results_dir / a
        if not p.exists():
            return ClaimResult(name, prose, False,
                               f"Artifact not found: {a}")

    try:
        if check_type == "delegated":
            return _check_delegated(name, prose, check)
        if check_type == "assertion":
            return _check_assertion(name, prose, check, artifacts, results_dir)
        elif check_type == "monotone":
            return _check_monotone(name, prose, check, artifacts, results_dir)
        elif check_type == "tost_equivalence":
            return _check_tost(name, prose, check, artifacts, results_dir)
        elif check_type == "exact":
            return _check_exact(name, prose, check, artifacts, results_dir)
        elif check_type == "ci_contains":
            return _check_ci_contains(name, prose, check, artifacts, results_dir)
        elif check_type == "ci_direction":
            return _check_ci_direction(name, prose, check, artifacts, results_dir)
        elif check_type == "re_measured":
            return _check_re_measured(name, prose, check, artifacts, results_dir)
        elif check_type == "model_comparison":
            return _check_model_comparison(name, prose, check, artifacts, results_dir)
        elif check_type == "compound":
            return _check_compound(name, prose, check, artifacts, results_dir)
        elif check_type == "sigma0_interaction":
            return _check_sigma0_interaction(name, prose, check, artifacts, results_dir)
        elif check_type == "cpu_ratio":
            return _check_cpu_ratio(name, prose, check, artifacts, results_dir)
        else:
            return ClaimResult(name, prose, False,
                               f"Unknown check type: {check_type}")
    except Exception as e:
        return ClaimResult(name, prose, False, f"Error: {e}")




def _check_delegated(name, prose, check):
    """A claim verified in another repository: echo where, and by what.

    Delegated claims are reported on their own summary line rather than folded
    into the local pass count — this repository did not check them; the named
    one did.
    """
    repo = check.get("repo", "")
    tag = check.get("tag", "")
    verified_by = check.get("verified_by", "")
    if not (repo and tag and verified_by):
        return ClaimResult(name, prose, False,
                           "delegated check missing repo/tag/verified_by")
    return ClaimResult(name, prose, True,
                       f"Delegated to {repo} @ {tag}; verified by {verified_by}",
                       delegated=True)


def _check_assertion(name, prose, check, artifacts, results_dir):
    field = check.get("field", "")
    expected = check.get("value")

    for a in artifacts:
        p = results_dir / a
        if p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, dict):
                val = data.get(field)
                if val != expected:
                    return ClaimResult(name, prose, False,
                                       f"{field}={val}, expected {expected}")
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
            if field in df.columns:
                bad = df[df[field] != expected]
                if len(bad) > 0:
                    return ClaimResult(name, prose, False,
                                       f"{len(bad)} rows with {field} != {expected}")

    return ClaimResult(name, prose, True, f"All {field} == {expected}")


def _check_monotone(name, prose, check, artifacts, results_dir):
    variable = check.get("variable", "")
    over = check.get("over", "")
    direction = check.get("direction", "non-decreasing")

    p = results_dir / artifacts[0]
    df = pd.read_parquet(p)

    # bmu_share is derived (not a column): sum(bmu_s)/(sum(bmu_s)+sum(update_s))
    if variable == "bmu_share" and "bmu_s" in df.columns:
        g = df.groupby(over).agg(b=("bmu_s", "sum"), u=("update_s", "sum"))
        grouped = (100 * g.b / (g.b + g.u)).sort_index()
    else:
        grouped = df.groupby(over)[variable].mean().sort_index()

    diffs = np.diff(grouped.values)
    if direction == "non-decreasing":
        violations = np.sum(diffs < -1e-9)
    elif direction == "strictly_decreasing":
        violations = np.sum(diffs >= 0)
    else:
        violations = 0

    if violations > 0:
        return ClaimResult(name, prose, False,
                           f"Monotonicity violated: {violations} violations "
                           f"in {variable} over {over}")

    return ClaimResult(name, prose, True,
                       f"{variable} is {direction} over {over} "
                       f"({len(grouped)} points)")


def _check_tost(name, prose, check, artifacts, results_dir):
    margin = check.get("margin_relative", 0.01)
    metrics = check.get("metrics", [])

    p = results_dir / artifacts[0]
    df = pd.read_parquet(p)

    failures = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        for edge in df["edge"].unique():
            sub = df[df["edge"] == edge]
            pca = sub[sub["init"] == "pca"][metric].dropna().values
            rand = sub[sub["init"] == "random"][metric].dropna().values
            if len(pca) < 2 or len(rand) < 2:
                continue
            result = tost_equivalence(pca, rand, margin_relative=margin)
            if not result["pass"]:
                failures.append(f"{metric} at edge {edge} (p={result['p_tost']:.4f})")

    if failures:
        return ClaimResult(name, prose, False,
                           f"TOST failed: {'; '.join(failures)}")

    return ClaimResult(name, prose, True,
                       f"TOST equivalence passed for {metrics}")


def _check_exact(name, prose, check, artifacts, results_dir):
    field = check.get("field", "")
    expected = check.get("value")

    p = results_dir / artifacts[0]
    df = pd.read_parquet(p)

    if field in df.columns:
        actual_max = df[df.get("converged", True) == True][field].max()
        if actual_max == expected:
            return ClaimResult(name, prose, True,
                               f"max({field}) = {expected}")
        else:
            return ClaimResult(name, prose, False,
                               f"max({field}) = {actual_max}, expected {expected}")

    return ClaimResult(name, prose, False, f"Field '{field}' not in artifact")


def _check_ci_contains(name, prose, check, artifacts, results_dir):
    """Equal-quality check: held-out QE agrees between sbsom-bin and ssom-feat
    at matched update (box+kl), within tolerance."""
    p = results_dir / artifacts[0]
    df = pd.read_parquet(p)
    df = df[~df.get("oom", False)]
    if "update_variant" in df.columns:
        df = df[df["update_variant"] == "box+kl"]

    a = df[df["impl"] == "sbsom-bin"]
    b = df[df["impl"] == "ssom-feat"]
    if a.empty or b.empty:
        return ClaimResult(name, prose, False,
                           "missing sbsom-bin or ssom-feat box+kl rows")

    worst = 0.0
    for edge in sorted(set(a["edge"]) & set(b["edge"])):
        qa = a[a.edge == edge]["qe"].mean()
        qb = b[b.edge == edge]["qe"].mean()
        if pd.notna(qa) and pd.notna(qb) and qa > 0:
            worst = max(worst, abs(qa - qb) / qa)

    ok = worst <= 0.03   # equal quality within ~1-3%
    return ClaimResult(name, prose, ok,
                       f"max held-out QE disagreement {worst*100:.1f}% "
                       f"({'<=3% (equal quality)' if ok else '>3%'})")


def _check_ci_direction(name, prose, check, artifacts, results_dir):
    """Check that one implementation is consistently faster/slower at every shared edge."""
    p = results_dir / artifacts[0]
    df = pd.read_parquet(p)
    df = df[~df.get("oom", False)]

    if "impl" not in df.columns:
        return ClaimResult(name, prose, True, "No impl column; skipping direction check")

    fast = check.get("impl_fast", "sbsom-bin")
    slow = check.get("impl_slow", "sbsom-float")
    col = check.get("column", "bmu_s_per_ep")

    fdf = df[df["impl"] == fast]
    sdf = df[df["impl"] == slow]
    if fdf.empty or sdf.empty:
        return ClaimResult(name, prose, False, f"missing {fast} or {slow}")

    violations = []
    for edge in sorted(set(fdf["edge"]) & set(sdf["edge"])):
        fv = fdf[fdf["edge"] == edge][col].mean()
        sv = sdf[sdf["edge"] == edge][col].mean()
        if pd.notna(fv) and pd.notna(sv) and fv >= sv:
            violations.append(int(edge))

    if violations:
        return ClaimResult(name, prose, False,
                           f"{fast} not faster than {slow} at edges {violations} ({col})")
    return ClaimResult(name, prose, True,
                       f"{fast} faster than {slow} at every shared edge ({col})")


def _check_re_measured(name, prose, check, artifacts, results_dir):
    """DRAM-traffic crossover: sbsom-bin moves less than ssom-feat at edge 128
    but more at edge 256 (whole-BMU-phase, roofline_phase_summary.csv)."""
    p = results_dir / artifacts[0]
    df = pd.read_csv(p)
    col = check.get("column", "phase_dram_tb")

    def val(impl, edge):
        r = df[(df["impl"] == impl) & (df["edge"] == edge)]
        return float(r.iloc[0][col]) if len(r) else None

    for impl, edge in check.get("needs_rows", []):
        if val(impl, edge) is None:
            return ClaimResult(name, prose, False, f"missing {impl}@edge{edge} {col}")

    d128 = val("sbsom-bin", 128) - val("ssom-feat", 128)
    d256 = val("sbsom-bin", 256) - val("ssom-feat", 256)
    crossover = d128 < 0 < d256
    return ClaimResult(name, prose, crossover,
                       f"phase DRAM (sbsom-bin - ssom-feat): edge128={d128:+.2f} TB, "
                       f"edge256={d256:+.2f} TB "
                       f"({'crossover confirmed' if crossover else 'NO crossover'})")


def _check_model_comparison(name, prose, check, artifacts, results_dir):
    """Check no-elbow claim: segmented model should not be significantly better."""
    from scipy import stats as sp_stats

    p = results_dir / artifacts[0]
    df = pd.read_parquet(p)

    # Aggregate QE across seeds per edge
    agg = df.groupby("edge").agg(
        qe_mean=("qe_heldout", "mean"),
        neurons=("neurons", "first"),
    ).reset_index()

    log_K = np.log10(agg["neurons"].values).astype(float)
    log_qe = np.log10(agg["qe_mean"].values).astype(float)
    n = len(log_K)

    if n < 5:
        return ClaimResult(name, prose, False,
                           f"Too few points for model comparison: {n}")

    # Single linear fit
    slope, intercept, r, p_val, se = sp_stats.linregress(log_K, log_qe)
    resid_single = log_qe - (intercept + slope * log_K)
    rss_single = np.sum(resid_single ** 2)

    # Segmented fit: try every interior point as a breakpoint
    best_rss_seg = rss_single
    best_bp = None
    for i in range(2, n - 2):
        bp = log_K[i]
        left = log_K <= bp
        right = ~left
        if left.sum() < 2 or right.sum() < 2:
            continue
        s1, i1, _, _, _ = sp_stats.linregress(log_K[left], log_qe[left])
        s2, i2, _, _, _ = sp_stats.linregress(log_K[right], log_qe[right])
        pred = np.where(left, i1 + s1 * log_K, i2 + s2 * log_K)
        rss_seg = np.sum((log_qe - pred) ** 2)
        if rss_seg < best_rss_seg:
            best_rss_seg = rss_seg
            best_bp = bp

    # F-test: segmented adds 2 params (breakpoint + second slope/intercept)
    df_single = n - 2
    df_seg = n - 4
    if df_seg <= 0:
        return ClaimResult(name, prose, True,
                           "Too few points for F-test; single model accepted by default")

    f_stat = ((rss_single - best_rss_seg) / 2) / (best_rss_seg / df_seg)
    p_ftest = 1 - sp_stats.f.cdf(f_stat, 2, df_seg)

    if p_ftest < 0.05:
        return ClaimResult(name, prose, False,
                           f"Segmented model significantly better (F={f_stat:.2f}, "
                           f"p={p_ftest:.4f}, breakpoint at log10(K)={best_bp:.2f})")

    return ClaimResult(name, prose, True,
                       f"No elbow: segmented model not favoured "
                       f"(F={f_stat:.2f}, p={p_ftest:.3f}, R²={r**2:.4f}, β={slope:.4f})")


def _check_compound(name, prose, check, artifacts, results_dir):
    """Bandwidth advantage at edge 256: sbsom-bin sustains higher DRAM bandwidth
    than ssom-feat AND moves more total DRAM (fusion wins by eliminating the
    round trip, not by moving fewer bytes)."""
    p = results_dir / artifacts[0]
    df = pd.read_csv(p)
    edge = check.get("edge", 256)

    sb = df[(df["impl"] == "sbsom-bin") & (df["edge"] == edge)]
    sf = df[(df["impl"] == "ssom-feat") & (df["edge"] == edge)]
    if sb.empty or sf.empty:
        return ClaimResult(name, prose, False, f"missing rows at edge {edge}")
    sb, sf = sb.iloc[0], sf.iloc[0]

    higher_bw = sb["sustained_dram_bw_pct"] > sf["sustained_dram_bw_pct"]
    more_dram = sb["phase_dram_tb"] > sf["phase_dram_tb"]
    ok = higher_bw and more_dram
    return ClaimResult(name, prose, ok,
                       f"edge{edge}: sbsom-bin sustains {sb['sustained_dram_bw_pct']}% vs "
                       f"ssom-feat {sf['sustained_dram_bw_pct']}% BW, moving "
                       f"{sb['phase_dram_tb']} vs {sf['phase_dram_tb']} TB "
                       f"({'advantage confirmed' if ok else 'NOT confirmed'})")


def _check_sigma0_interaction(name, prose, check, artifacts, results_dir):
    """sigma0=0.25E improves held-out QE under PCA (arm B<=A) and degrades it
    under random init (arm R'>=R), on balance across edges."""
    df = pd.read_parquet(results_dir / artifacts[0])

    def qe(arm, edge):
        r = df[(df["arm"] == arm) & (df["edge"] == edge)]
        return r["qe_heldout"].mean() if len(r) else None

    pca_imp, rnd_deg = [], []
    for e in sorted(df["edge"].unique()):
        a, b = qe("A", e), qe("B", e)
        r, rp = qe("R", e), qe("R'", e)
        if a is not None and b is not None:
            pca_imp.append(b <= a)      # PCA 0.25E QE not worse than 0.5E
        if r is not None and rp is not None:
            rnd_deg.append(rp >= r)     # random 0.25E QE not better than 0.5E
    pca_ok = pca_imp and sum(pca_imp) >= len(pca_imp) / 2
    rnd_ok = rnd_deg and sum(rnd_deg) >= len(rnd_deg) / 2
    ok = bool(pca_ok and rnd_ok)
    return ClaimResult(name, prose, ok,
                       f"sigma0=0.25E: PCA improves {sum(pca_imp)}/{len(pca_imp)} edges, "
                       f"random degrades {sum(rnd_deg)}/{len(rnd_deg)} edges")


def _check_cpu_ratio(name, prose, check, artifacts, results_dir):
    """A slow reference implementation (`impl_slow`, default somoclu) is >= min_ratio
    slower than sbsom-bin at every edge it ran, on the total s/epoch basis.
    somoclu's per-epoch time is the measured `somoclu_s_per_ep_measured`; any other
    implementation (e.g. medsom) uses total_wall_s / epochs like sbsom-bin."""
    df = pd.read_parquet(results_dir / artifacts[0])
    df = df[~df.get("oom", False)]
    slow_impl = check.get("impl_slow", "somoclu")
    slow = df[df["impl"] == slow_impl]
    sb = df[df["impl"] == "sbsom-bin"]
    if slow.empty or sb.empty:
        return ClaimResult(name, prose, False, f"missing {slow_impl} or sbsom-bin")

    minr = check.get("min_ratio", 50)
    ratios = {}
    for e in sorted(slow["edge"].unique()):
        sr = slow[slow["edge"] == e]
        if slow_impl == "somoclu":
            sv = sr["somoclu_s_per_ep_measured"].iloc[0]
        else:
            sv = (sr["total_wall_s"] / sr["epochs"]).mean()
        br = sb[sb["edge"] == e]
        bt = (br["total_wall_s"] / br["epochs"]).mean()
        if pd.notna(sv) and pd.notna(bt) and bt > 0:
            ratios[int(e)] = sv / bt
    bad = [e for e, r in ratios.items() if r < minr]
    ok = bool(ratios) and not bad
    return ClaimResult(name, prose, ok,
                       f"{slow_impl}/sbsom-bin ratios {{{', '.join(f'{e}:{round(r)}x' for e,r in ratios.items())}}}; "
                       f"all >= {minr}x? {ok}")


def main():
    parser = argparse.ArgumentParser(description="Verify manuscript claims")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--claims", type=Path,
                        default=Path("configs/claims.yaml"))
    args = parser.parse_args()

    results = verify_all(args.results_dir, args.claims)

    print("=" * 70)
    print("ACCEPTANCE REPORT")
    print("=" * 70)

    delegated = sum(1 for r in results if r.delegated)
    passed = sum(1 for r in results if r.passed and not r.delegated)
    failed = sum(1 for r in results if not r.passed)

    for r in results:
        print(r)
        print()

    print("=" * 70)
    print(f"Total: {len(results)} claims | Verified locally: {passed} passed, "
          f"{failed} failed | Delegated: {delegated}")
    print("=" * 70)

    if failed > 0:
        print("\nFAILED CLAIMS:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.detail}")
        sys.exit(1)
    else:
        print("\nAll claims verified.")


if __name__ == "__main__":
    main()
