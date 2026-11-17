"""Statistics for the genomic cell-line tabs (co-occurrence, mutation frequency, mutation-CNV, similar lines)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from portal.stats import bh


def cooccurrence(M: pd.DataFrame, sided: str = "two-sided", min_lines: int = 3) -> pd.DataFrame:
    """Pairwise co-occurrence vs mutual exclusivity of binary alterations (as in cBioPortal).

    M: cell lines x alterations, 1/True = altered, 0/False = not altered, NaN = not annotated.
    Each pair uses the lines annotated for both. Fisher exact test on the 2x2 table; log2 odds
    ratio with a Haldane +0.5 correction so empty cells stay finite. sided = 'two-sided' or
    'one-sided' (in the direction of the observed tendency). BH FDR across the pairs tested.
    Pairs where either alteration has no altered line among the shared lines are not testable.
    """
    cols = list(M.columns)
    X = M.astype(float).to_numpy()
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            ok = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
            a, b = X[ok, i] > 0, X[ok, j] > 0
            if ok.sum() < min_lines or not a.any() or not b.any():
                continue
            n11, n10 = int((a & b).sum()), int((a & ~b).sum())
            n01, n00 = int((~a & b).sum()), int((~a & ~b).sum())
            table = [[n11, n10], [n01, n00]]
            lor = np.log2((n11 + 0.5) * (n00 + 0.5) / ((n10 + 0.5) * (n01 + 0.5)))
            p_co = stats.fisher_exact(table, alternative="greater")[1]
            p_ex = stats.fisher_exact(table, alternative="less")[1]
            p = stats.fisher_exact(table)[1] if sided == "two-sided" else (p_co if lor > 0 else p_ex)
            rows.append({"A": cols[i], "B": cols[j], "n_lines": int(ok.sum()), "neither": n00,
                         "A_not_B": n10, "B_not_A": n01, "both": n11, "log2_odds_ratio": lor,
                         "p": min(p, 1.0), "p_cooccurrence": p_co, "p_exclusivity": p_ex,
                         "tendency": "co-occurrence" if lor > 0 else "mutual exclusivity"})
    out = pd.DataFrame(rows, columns=["A", "B", "n_lines", "neither", "A_not_B", "B_not_A", "both",
                                      "log2_odds_ratio", "p", "p_cooccurrence", "p_exclusivity", "tendency"])
    out.insert(out.columns.get_loc("p") + 1, "q", bh(out["p"].to_numpy()))
    return out.sort_values(["p", "A", "B"]).reset_index(drop=True)


def set_exclusivity(M: pd.DataFrame, n_perm: int = 10000, seed: int = 0) -> dict:
    """Is a whole set of alterations more mutually exclusive than chance?

    Uses the lines annotated for every alteration in M. Statistic: coverage = lines with >= 1
    alteration. Null: each alteration's altered lines are shuffled independently across lines
    (frequencies kept). A set that avoids overlapping covers more lines than chance, so
    p = P(null coverage >= observed). Also returns the lines with exactly one alteration.
    """
    X = M.dropna().astype(float).to_numpy() > 0
    n, k = X.shape
    if n == 0 or k < 2:
        return {"n_lines": n, "n_alterations": k}
    rng = np.random.default_rng(seed)
    total = np.zeros((n_perm, n), dtype=np.int16)
    for c in range(k):
        total += rng.permuted(np.tile(X[:, c], (n_perm, 1)), axis=1)
    cov_null, one_null = (total > 0).sum(1), (total == 1).sum(1)
    cov, one = int(X.any(1).sum()), int((X.sum(1) == 1).sum())
    return {"n_lines": n, "n_alterations": k, "coverage": cov, "expected_coverage": float(cov_null.mean()),
            "exactly_one": one, "expected_exactly_one": float(one_null.mean()),
            "overlapping": int((X.sum(1) >= 2).sum()), "expected_overlapping": float((total >= 2).sum(1).mean()),
            "p_exclusive": float((1 + (cov_null >= cov).sum()) / (n_perm + 1)),
            "p_cooccurring": float((1 + (cov_null <= cov).sum()) / (n_perm + 1)), "n_perm": n_perm}


def mutation_enrichment(M: pd.DataFrame, lines_a: list[str], lines_b: list[str]) -> pd.DataFrame:
    """Per alteration: frequency in group A vs group B, Fisher exact test (two-sided), BH FDR.

    M: cell lines x alterations (1 / 0 / NaN = not annotated). Each alteration uses the lines of
    each group annotated for it; alterations altered in no line of either group are dropped."""
    rows = []
    for c in M.columns:
        a, b = M[c].reindex(lines_a).dropna() > 0, M[c].reindex(lines_b).dropna() > 0
        if len(a) == 0 or len(b) == 0 or not (a.any() or b.any()):
            continue
        ma, mb = int(a.sum()), int(b.sum())
        table = [[ma, len(a) - ma], [mb, len(b) - mb]]
        lor = np.log2((ma + 0.5) * (len(b) - mb + 0.5) / ((len(a) - ma + 0.5) * (mb + 0.5)))
        rows.append({"alteration": c, "altered_A": ma, "n_A": len(a), "freq_A": ma / len(a),
                     "altered_B": mb, "n_B": len(b), "freq_B": mb / len(b),
                     "diff_A_minus_B": ma / len(a) - mb / len(b), "log2_odds_ratio": lor,
                     "p": stats.fisher_exact(table)[1]})
    out = pd.DataFrame(rows, columns=["alteration", "altered_A", "n_A", "freq_A", "altered_B", "n_B", "freq_B",
                                      "diff_A_minus_B", "log2_odds_ratio", "p"])
    out["q"] = bh(out["p"].to_numpy())
    return out.sort_values(["p", "alteration"]).reset_index(drop=True)


def jaccard_similarity(X: pd.DataFrame) -> pd.DataFrame:
    """Lines x lines Jaccard similarity of binary profiles: shared aberrations / aberrations in
    either line, over the features annotated in both (NaN = not annotated). NaN when neither line
    has an aberration among their shared features."""
    ann = X.notna().to_numpy(float)
    A = X.fillna(0).to_numpy(float)
    inter = A @ A.T
    union = A @ ann.T + ann @ A.T - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, inter / union, np.nan)
    return pd.DataFrame(J, index=X.index, columns=X.index)


def correlate(X: pd.DataFrame, y: pd.Series, method: str = "pearson", min_n: int = 8) -> pd.DataFrame:
    """Correlation of every row of X (features x lines) with y (values per line), pairwise
    complete. Spearman = Pearson on ranks (ranked within the shared lines of each feature).
    Two-sided p from the t distribution; BH FDR across tested features."""
    lines = [c for c in X.columns if c in y.index]
    A = X[lines].to_numpy(float)
    b = y.reindex(lines).to_numpy(float)
    ok = np.isfinite(A) & np.isfinite(b)[None, :]
    n = ok.sum(1)
    if method == "spearman":
        Am = np.where(ok, A, np.nan)
        A = pd.DataFrame(Am).rank(axis=1).to_numpy()
        B = pd.DataFrame(np.where(ok, b[None, :], np.nan)).rank(axis=1).to_numpy()
    else:
        A, B = np.where(ok, A, np.nan), np.where(ok, b[None, :], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        A = A - np.nanmean(A, 1, keepdims=True)
        B = B - np.nanmean(B, 1, keepdims=True)
        r = np.nansum(A * B, 1) / np.sqrt(np.nansum(A**2, 1) * np.nansum(B**2, 1))
        r = np.clip(r, -1, 1)
        t = r * np.sqrt((n - 2) / np.maximum(1 - r**2, 1e-300))
        p = 2 * stats.t.sf(np.abs(t), n - 2)
    res = pd.DataFrame({"feature": X.index, "r": r, "n": n, "p": p})
    res = res[(n >= min_n) & np.isfinite(r)].copy()
    res["q"] = bh(res["p"].to_numpy())
    return res.sort_values("p").reset_index(drop=True)


def rowwise_correlate(A: pd.DataFrame, B: pd.DataFrame, method: str = "spearman", min_n: int = 6) -> pd.DataFrame:
    """Correlation of matching rows of A and B (e.g. a gene's protein vs its CRISPR effect) across their
    shared columns, pairwise complete. Rows = genes in both; BH FDR across genes."""
    genes = A.index.intersection(B.index)
    cols = A.columns.intersection(B.columns)
    a, b = A.loc[genes, cols].to_numpy(float), B.loc[genes, cols].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(ok, a, np.nan), np.where(ok, b, np.nan)
    if method == "spearman":
        a, b = pd.DataFrame(a).rank(axis=1).to_numpy(), pd.DataFrame(b).rank(axis=1).to_numpy()
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = a - np.nanmean(a, 1, keepdims=True)
        b = b - np.nanmean(b, 1, keepdims=True)
        r = np.clip(np.nansum(a * b, 1) / np.sqrt(np.nansum(a**2, 1) * np.nansum(b**2, 1)), -1, 1)
        t = r * np.sqrt((n - 2) / np.maximum(1 - r**2, 1e-300))
        p = 2 * stats.t.sf(np.abs(t), n - 2)
    res = pd.DataFrame({"gene": genes, "r": r, "n": n, "p": p})
    res = res[(n >= min_n) & np.isfinite(r)].copy()
    res["q"] = bh(res["p"].to_numpy())
    return res.sort_values("p").reset_index(drop=True)
