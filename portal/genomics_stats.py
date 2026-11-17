"""Statistics for the genomic cell-line tabs (co-occurrence, mutation frequency, mutation-CNV, similar lines)."""
from __future__ import annotations

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
