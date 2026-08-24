"""Statistical aggregation: t-intervals, paired log-ratio t-intervals, TOST equivalence."""

import numpy as np
from scipy import stats


def t_interval(values, confidence=0.95):
    """Mean ± t-based CI for small samples."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return {"mean": float(arr[0]), "ci_lo": float(arr[0]), "ci_hi": float(arr[0]),
                "sd": 0.0, "n": n, "half_width": 0.0}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    hw = t_crit * sd / np.sqrt(n)
    return {"mean": mean, "ci_lo": mean - hw, "ci_hi": mean + hw,
            "sd": sd, "n": n, "half_width": float(hw)}


def paired_log_ratio_t(numerator, denominator, confidence=0.95):
    """Paired log-ratio t-interval for speed-up ratios.

    Forms per-seed ratio, takes log, builds t-interval, exponentiates back.
    Returns an asymmetric CI on the ratio scale.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    assert len(num) == len(den), "Paired comparison requires equal-length arrays"
    n = len(num)

    ratios = num / den
    log_ratios = np.log(ratios)

    m = float(np.mean(log_ratios))
    if n < 2:
        return {"ratio": float(np.exp(m)), "ci_lo": float(np.exp(m)),
                "ci_hi": float(np.exp(m)), "n": n}

    s = float(np.std(log_ratios, ddof=1))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    hw = t_crit * s / np.sqrt(n)

    return {
        "ratio": float(np.exp(m)),
        "ci_lo": float(np.exp(m - hw)),
        "ci_hi": float(np.exp(m + hw)),
        "n": n,
        "log_mean": m,
        "log_sd": s,
    }


def tost_equivalence(group_a, group_b, margin_relative=0.01, confidence=0.90):
    """Two one-sided test for equivalence.

    Tests whether |mean(A) - mean(B)| < margin (margin = margin_relative × mean(A)).
    Returns pass/fail and both one-sided p-values.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    margin = abs(margin_relative * np.mean(a))
    diff = np.mean(a) - np.mean(b)

    n_a, n_b = len(a), len(b)
    se = np.sqrt(np.var(a, ddof=1) / n_a + np.var(b, ddof=1) / n_b)

    if se == 0:
        return {"pass": abs(diff) < margin, "diff": float(diff),
                "margin": float(margin), "p_lo": 0.0, "p_hi": 0.0}

    df = n_a + n_b - 2

    t_lo = (diff - (-margin)) / se
    t_hi = (margin - diff) / se

    p_lo = 1.0 - stats.t.cdf(t_lo, df)
    p_hi = 1.0 - stats.t.cdf(t_hi, df)
    p_tost = max(p_lo, p_hi)

    alpha = 1.0 - confidence
    return {
        "pass": bool(p_tost < alpha),
        "diff": float(diff),
        "margin": float(margin),
        "p_lo": float(p_lo),
        "p_hi": float(p_hi),
        "p_tost": float(p_tost),
        "alpha": alpha,
    }


def format_ci(result, precision=3):
    """Format a t_interval result as 'mean ± half_width'."""
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(result['mean'])} ± {fmt.format(result['half_width'])}"


def format_ratio_ci(result, precision=1):
    """Format a paired_log_ratio_t result as 'ratio× [lo, hi]'."""
    fmt = f"{{:.{precision}f}}"
    return (f"{fmt.format(result['ratio'])}× "
            f"[{fmt.format(result['ci_lo'])}, {fmt.format(result['ci_hi'])}]")
