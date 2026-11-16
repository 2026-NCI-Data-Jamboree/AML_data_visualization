"""Two-group comparison matching compare_mTOR_RNA_groups.R::diff_test (Welch t, BH FDR)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (same as R p.adjust(..., 'BH')); NaN-safe."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    n = q.size
    o = np.argsort(q)[::-1]
    adj = np.minimum.accumulate(q[o] * n / np.arange(n, 0, -1))
    res = np.empty(n)
    res[o] = np.minimum(adj, 1)
    out[ok] = res
    return out


def welch(values: pd.DataFrame, group_a: list[str], group_b: list[str], min_n: int = 3) -> pd.DataFrame:
    """values: features x cell lines. Returns one row per testable feature, sorted by p.

    Columns follow the processing repo: diff = mean(A) - mean(B), named *_A / *_B here.
    """
    A = values.reindex(columns=[c for c in group_a if c in values.columns]).to_numpy(float)
    B = values.reindex(columns=[c for c in group_b if c in values.columns]).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty / single-line groups -> NaN, filtered below
        nA, nB = np.isfinite(A).sum(1), np.isfinite(B).sum(1)
        mA, mB = np.nanmean(A, 1), np.nanmean(B, 1)
        vA, vB = np.nanvar(A, 1, ddof=1), np.nanvar(B, 1, ddof=1)
        keep = (nA >= min_n) & (nB >= min_n) & ((vA + vB) > 0)
        sA, sB = vA / nA, vB / nB
        t = (mA - mB) / np.sqrt(sA + sB)
        df = (sA + sB) ** 2 / (sA**2 / (nA - 1) + sB**2 / (nB - 1))
        p = 2 * stats.t.sf(np.abs(t), df)
        sdp = np.sqrt(((nA - 1) * vA + (nB - 1) * vB) / (nA + nB - 2))
    res = pd.DataFrame({
        "feature": values.index, "n_A": nA, "n_B": nB, "mean_A": mA, "mean_B": mB,
        "diff_A_minus_B": mA - mB, "cohens_d": (mA - mB) / sdp, "t": t, "p_welch": p,
    })[keep]
    res["fdr_welch"] = bh(res["p_welch"].to_numpy())
    return res.sort_values("p_welch").reset_index(drop=True)


def spearman_rows(X: pd.DataFrame, y: pd.Series, min_n: int = 10) -> pd.DataFrame:
    """Spearman correlation of every row of X (features x samples) with y (indexed by sample).
    Same r and p as scipy.stats.spearmanr on the samples where both are present."""
    y = y.dropna()
    cols = [c for c in X.columns if c in y.index]
    A = X[cols].to_numpy(float)
    yv = y[cols].to_numpy(float)
    r = np.full(len(A), np.nan)
    n = np.isfinite(A).sum(1)
    complete = n == len(cols)
    if complete.any() and len(cols) >= min_n:                 # vectorised for complete rows
        R = pd.DataFrame(A[complete]).rank(axis=1).to_numpy()
        ry = pd.Series(yv).rank().to_numpy()
        R -= R.mean(1, keepdims=True)
        ry = ry - ry.mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            r[complete] = (R @ ry) / np.sqrt((R ** 2).sum(1) * (ry ** 2).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                          # constant rows -> NaN, dropped below
        for i in np.where(~complete & (n >= min_n))[0]:          # rows with missing values
            ok = np.isfinite(A[i])
            r[i] = stats.spearmanr(A[i, ok], yv[ok]).statistic
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
        p = 2 * stats.t.sf(np.abs(t), n - 2)
    out = pd.DataFrame({"feature": X.index, "n": n, "r": r, "p": p})
    return out[(out["n"] >= min_n) & np.isfinite(out["r"])].reset_index(drop=True)


def meta_correlation(per_cohort: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-cohort correlations (feature, n, r) by Fisher z, weighted by n - 3 (Stouffer).

    Returns feature, n_cohorts, n_total, r (pooled), z, p, fdr, concordant (all cohorts same sign),
    I2 (heterogeneity, %), plus r_<cohort> columns.
    """
    wide = None
    for c, d in per_cohort.items():
        d = d.set_index("feature")[["n", "r"]].rename(columns={"n": f"n_{c}", "r": f"r_{c}"})
        wide = d if wide is None else wide.join(d, how="outer")
    if wide is None or wide.empty:
        return pd.DataFrame(columns=["feature", "n_cohorts", "n_total", "r", "z", "p", "fdr", "concordant", "I2"])
    cs = list(per_cohort)
    R = wide[[f"r_{c}" for c in cs]].to_numpy(float)
    N = wide[[f"n_{c}" for c in cs]].to_numpy(float)
    W = np.where(np.isfinite(R) & (N > 3), N - 3, 0.0)
    Zf = np.arctanh(np.clip(np.nan_to_num(R), -0.999999, 0.999999))
    with np.errstate(invalid="ignore", divide="ignore"):
        sw = W.sum(1)
        zbar = (W * Zf).sum(1) / sw
        z = zbar * np.sqrt(sw)
        Q = (W * (Zf - zbar[:, None]) ** 2).sum(1)
        k = (W > 0).sum(1)
        I2 = np.where((k > 1) & (Q > 1e-9), np.clip((Q - (k - 1)) / Q, 0, 1) * 100, 0.0)   # 0 with one cohort
    sgn = np.sign(np.where(W > 0, R, np.nan))
    out = pd.DataFrame({
        "feature": wide.index, "n_cohorts": k, "n_total": np.where(W > 0, N, 0).sum(1).astype(int),
        "r": np.tanh(zbar), "z": z, "p": 2 * stats.norm.sf(np.abs(z)),
        "concordant": (np.nanmin(sgn, 1) == np.nanmax(sgn, 1)), "I2": I2,
    })
    for c in cs:
        out[f"r_{c}"] = wide[f"r_{c}"].to_numpy()
    out = out[out["n_cohorts"] > 0].copy()
    out["fdr"] = bh(out["p"].to_numpy())
    return out.sort_values("p").reset_index(drop=True)


def group_test(df: pd.DataFrame, value: str, group: str, levels: list[str], min_n: int = 3) -> tuple[str, float]:
    """Two groups: Welch t test; more: Kruskal-Wallis. Groups with < min_n values are dropped.
    Returns (test name, p); p is NaN when fewer than two groups are left."""
    vals = [df.loc[df[group] == lv, value].dropna().to_numpy(float) for lv in levels]
    vals = [v for v in vals if len(v) >= min_n]
    if len(vals) < 2:
        return ("Welch" if len(levels) == 2 else "Kruskal–Wallis"), np.nan
    if len(vals) == 2:
        if np.var(vals[0]) + np.var(vals[1]) == 0:
            return "Welch", np.nan
        return "Welch", float(stats.ttest_ind(vals[0], vals[1], equal_var=False).pvalue)
    return "Kruskal–Wallis", float(stats.kruskal(*vals).pvalue)
