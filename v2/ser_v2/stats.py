"""Statistics used by the scorer: UAR, speaker pairing, bootstrap, Wilcoxon / sign test,
Holm, TOST, and the audit interval family (Wilson, partial identification, exact
hypergeometric frame envelope, Cohen's kappa with bootstrap CI)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def uar(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted average recall over the classes PRESENT in y_true."""
    recalls = []
    for c in range(n_classes):
        m = y_true == c
        if m.any():
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) * 100.0


def accuracy(y_true, y_pred) -> float:
    return float((y_true == y_pred).mean()) * 100.0


def macro_f1(y_true, y_pred, n_classes) -> float:
    f1s = []
    for c in range(n_classes):
        tp = float(((y_true == c) & (y_pred == c)).sum())
        fp = float(((y_true != c) & (y_pred == c)).sum())
        fn = float(((y_true == c) & (y_pred != c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) * 100.0 if f1s else float("nan")


def speaker_bootstrap_ci(diffs: np.ndarray, n_reps: int = 10000, seed: int = 20260903,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of the mean of speaker-level differences under whole-speaker resampling."""
    rng = np.random.RandomState(seed)
    n = len(diffs)
    idx = rng.randint(0, n, size=(n_reps, n))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)], method="linear")
    return float(lo), float(hi)


def paired_test(diffs: np.ndarray) -> dict:
    """Two-sided Wilcoxon (zero_method='wilcox'); exact sign test when >20% ties at zero."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    d_r = np.round(d, 10)                      # tie determinism (spec Section 5)
    n_zero = int((d_r == 0).sum())
    out = {"n": n, "n_nonzero": n - n_zero, "mean": float(d.mean()), "median": float(np.median(d)),
           "sd": float(d.std(ddof=1)) if n > 1 else float("nan")}
    if n - n_zero == 0:
        out.update({"wilcoxon_p": 1.0, "wilcoxon_stat": 0.0})
    else:
        res = sps.wilcoxon(d_r[d_r != 0], zero_method="wilcox", alternative="two-sided")
        out.update({"wilcoxon_p": float(res.pvalue), "wilcoxon_stat": float(res.statistic)})
    if n_zero > 0.2 * n:
        pos = int((d_r > 0).sum())
        res = sps.binomtest(pos, n - n_zero, 0.5, alternative="two-sided") if n - n_zero > 0 else None
        out["sign_test_p"] = float(res.pvalue) if res else 1.0
        out["sign_test_used"] = True
    else:
        out["sign_test_used"] = False
    out["p"] = out["wilcoxon_p"]
    return out


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down over the given family; keys are hypothesis ids. Missing p is 1.0."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 0.0
    rejected_so_far = True
    for rank, (hid, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        reject = rejected_so_far and running <= alpha
        if not reject:
            rejected_so_far = False
        out[hid] = {"p": float(p), "p_holm": float(running), "reject": bool(reject), "rank": rank + 1, "m": m}
    return out


def tost_paired(diffs: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided t-tests for equivalence of a paired mean difference within +/-margin."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    t_low = (mean + margin) / se
    t_high = (mean - margin) / se
    p_low = 1 - sps.t.cdf(t_low, n - 1)      # H0: mean <= -margin
    p_high = sps.t.cdf(t_high, n - 1)        # H0: mean >= +margin
    p = max(p_low, p_high)
    ci = sps.t.interval(1 - 2 * alpha, n - 1, loc=mean, scale=se)   # 90% CI for alpha 0.05
    return {"n": n, "mean": float(mean), "se": float(se), "margin": margin, "p_low": float(p_low),
            "p_high": float(p_high), "p": float(p), "ci90": [float(ci[0]), float(ci[1])], "equivalent": bool(p <= alpha)}


# ------------------------------------------------------------------ audit intervals

def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def partial_identification(y: int, u: int, n: int) -> tuple[float, float]:
    return (y / n, (y + u) / n)


def hypergeom_lower_bound(N: int, n: int, y: int, alpha_one_sided: float) -> int:
    """Smallest K such that P(X >= y | N, K, n) > alpha (exact, one-sided)."""
    for K in range(0, N + 1):
        if sps.hypergeom.sf(y - 1, N, K, n) > alpha_one_sided:
            return K
    return N


def hypergeom_upper_bound(N: int, n: int, y_plus_u: int, alpha_one_sided: float) -> int:
    """Largest K such that P(X <= y+u | N, K, n) > alpha (exact, one-sided)."""
    for K in range(N, -1, -1):
        if sps.hypergeom.cdf(y_plus_u, N, K, n) > alpha_one_sided:
            return K
    return 0


def frame_envelope(N: int, n: int, y: int, u: int, alpha: float = 0.05) -> dict:
    """Joint >=1-alpha envelope under partial classification: two one-sided exact
    hypergeometric bounds at alpha/2 each (Bonferroni), lower from Y, upper from Y+U."""
    lo = hypergeom_lower_bound(N, n, y, alpha / 2)
    hi = hypergeom_upper_bound(N, n, y + u, alpha / 2)
    return {"N": N, "n": n, "Y": y, "U": u, "K_low": lo, "K_high": hi, "pct_low": 100 * lo / N, "pct_high": 100 * hi / N}


def cohen_kappa(a: list, b: list) -> float:
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(axis=0) * m.sum(axis=1)).sum() / (n * n))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def kappa_bootstrap_ci(a: list, b: list, n_reps: int = 2000, seed: int = 20260903) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(a)
    vals = []
    for _ in range(n_reps):
        ix = rng.randint(0, n, size=n)
        vals.append(cohen_kappa([a[i] for i in ix], [b[i] for i in ix]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
