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
