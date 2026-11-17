"""portal.stats against reference implementations, and against the prepared cell-line data."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portal.stats import bh, welch

RESULTS = Path(os.environ.get("AML_PORTAL_DATA",
               Path(__file__).resolve().parents[2] / "AML_data_processing/results"))
CL = RESULTS / "cell_lines"


def test_bh_matches_r():
    # R: p.adjust(c(0.01, 0.04, 0.03, 0.2, NA), "BH")
    out = bh(np.array([0.01, 0.04, 0.03, 0.2, np.nan]))
    np.testing.assert_allclose(out[:4], [0.04, 0.0533333333, 0.0533333333, 0.2], rtol=1e-6)
    assert np.isnan(out[4])


def test_welch_matches_scipy_with_missing_values():
    rng = np.random.default_rng(0)
    V = pd.DataFrame(rng.normal(size=(50, 20)), columns=[f"L{i}" for i in range(20)])
    V.iloc[::7, 3] = np.nan
    V.iloc[5, :8] = np.nan                       # too few values in group A -> dropped
    a, b = list(V.columns[:8]), list(V.columns[8:])
    res = welch(V, a, b).set_index("feature")
    assert 5 not in res.index and len(res) == 49
    for f in res.index:
        x, y = V.loc[f, a].dropna(), V.loc[f, b].dropna()
        ref = stats.ttest_ind(x, y, equal_var=False)
        assert res.loc[f, "p_welch"] == pytest.approx(ref.pvalue, rel=1e-9)
        assert res.loc[f, "diff_A_minus_B"] == pytest.approx(x.mean() - y.mean())


@pytest.mark.skipif(not CL.is_dir(), reason="results/cell_lines not prepared")
def test_values_and_features_agree():
    ds = pd.read_csv(CL / "datasets.tsv", sep="\t")
    samples = pd.read_csv(CL / "samples.tsv", sep="\t")
    for key in ds["key"]:
        V = pd.read_csv(CL / "values" / f"{key}.tsv.gz", sep="\t", index_col="feature")
        F = pd.read_csv(CL / "features" / f"{key}.tsv", sep="\t", index_col="feature")
        assert list(V.index) == list(F.index), key
        assert set(V.columns) <= set(samples["cell_line"]), key


def test_group_test_matches_scipy():
    import numpy as np
    import pandas as pd
    from scipy import stats as ss
    from portal.stats import group_test
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"v": rng.normal(size=40), "g": ["a"] * 15 + ["b"] * 15 + ["c"] * 8 + ["d"] * 2})
    name, p = group_test(df, "v", "g", ["a", "b"])
    assert name == "Welch" and p == ss.ttest_ind(df.v[:15], df.v[15:30], equal_var=False).pvalue
    name, p = group_test(df, "v", "g", ["a", "b", "c", "d"])          # d (n=2) is dropped
    assert name == "Kruskal–Wallis" and p == ss.kruskal(df.v[:15], df.v[15:30], df.v[30:38]).pvalue
    assert np.isnan(group_test(df, "v", "g", ["a", "d"])[1])


# synthetic survival data (R: set.seed(1); n = 60; tied times; x2 has missing values); see the tests below
SYN_TIME = [9, 13, 2, 2, 5, 30, 13, 6, 11, 2, 15, 9, 13, 45, 12, 11, 20, 8, 4, 7, 25, 7, 4, 7, 2, 2, 7, 41, 13, 11, 15, 1, 4, 14, 3, 11, 4, 8, 9, 3, 12, 11, 14, 14, 7, 4, 14, 11, 6, 21, 5, 23, 33, 7, 7, 11, 3, 4, 12, 9]
SYN_EVENT = [0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
SYN_X1 = [0.772142, -0.219516, 0.075190, 0.081020, 0.996987, -0.275778, 1.256019, 0.646674, 1.299312, -0.373262, 0.008371, -0.880872, 0.596259, 0.119718, -0.282174, 1.455988, 0.229020, 0.996544, 1.281859, -0.776777, -0.615990, 0.046580, -0.630386, 0.576719, -0.780749, 2.125447, -0.500697, 1.678297, -0.412520, -0.972287, 0.025383, 0.527475, -1.180183, 1.053751, -0.619599, 0.335617, 0.994796, 0.138053, -0.118792, 0.697684, -1.068693, -0.803213, -1.113765, 1.580092, 1.497819, 0.762645, -1.232901, -0.003724, 1.511672, -0.475698, 0.797916, -0.974003, 0.689373, -0.955839, -1.231707, -0.956892, -0.369783, -0.410681, 0.741276, 0.068512]
SYN_X2 = [-0.323751, -1.086503, np.nan, -0.767790, -1.119720, -0.448174, 0.471736, -1.180491, 1.470257, -1.311421, -0.096525, 2.369720, 0.890626, -0.252183, -0.865764, 0.582586, np.nan, -0.374855, 0.317886, -0.488806, 2.658658, 1.680278, 0.779584, 0.713241, -0.542882, 0.885778, -0.348595, -1.008055, 1.883183, -0.928971, -0.294196, -0.614950, -0.947076, 0.598975, -1.523615, -0.206189, -0.574295, -1.390166, -0.070417, np.nan, -0.592225, 0.981116, 0.532409, -0.090456, 0.156490, -0.737312, -0.201341, 1.102177, -0.016748, 0.161789, 2.024761, -0.703694, 0.960792, 1.790485, -1.064165, 0.017637, -0.389909, -0.490833, -1.045718, -0.896211]
SYN_GROUP = (["favorable", "intermediate", "adverse"] * 20)[:60]
SYN_GROUP[4] = SYN_GROUP[8] = None


def _syn():
    ids = [f"s{i}" for i in range(len(SYN_TIME))]
    X = pd.DataFrame([SYN_X1, SYN_X2], index=["x1", "x2"], columns=ids)
    return X, pd.Series(SYN_TIME, index=ids, dtype=float), pd.Series(SYN_EVENT, index=ids).astype(bool), ids


def test_cox_rows_matches_r_survival():
    """Reference: R survival::coxph(Surv(time, event) ~ scale(x), ties = "breslow") on the same data."""
    from portal.stats import cox_rows, meta_hazard
    X, time, event, _ = _syn()
    res = cox_rows(X, time, event, min_n=10, min_events=5).set_index("feature")
    assert res.loc["x1", "beta"] == pytest.approx(-0.072512, abs=1e-5)
    assert res.loc["x1", "se"] == pytest.approx(0.151749, abs=1e-5)
    assert res.loc["x2", "beta"] == pytest.approx(-0.111239, abs=1e-5)
    assert res.loc["x2", "se"] == pytest.approx(0.161342, abs=1e-5)
    assert res.loc["x2", "n"] == 57
    # meta-analysis of one cohort returns that cohort's estimate, with no heterogeneity
    m = meta_hazard({"A": res.reset_index()}).set_index("feature")
    assert m.loc["x1", "log_hr"] == pytest.approx(-0.072512, abs=1e-5)
    assert m.loc["x1", "I2"] == 0


def test_cox_rows_stratified_matches_r_survival():
    """Reference: k <- !is.na(grp); coxph(Surv(time[k], event[k]) ~ scale(x[k]) + strata(grp[k]),
    ties = "breslow"): stratified by group, patients without a group left out."""
    from portal.stats import cox_rows
    X, time, event, ids = _syn()
    res = cox_rows(X, time, event, min_n=10, min_events=5,
                   strata=pd.Series(SYN_GROUP, index=ids)).set_index("feature")
    assert res.loc["x1", "beta"] == pytest.approx(-0.097018, abs=1e-5)
    assert res.loc["x1", "se"] == pytest.approx(0.162627, abs=1e-5)
    assert res.loc["x2", "beta"] == pytest.approx(-0.043138, abs=1e-5)
    assert res.loc["x2", "se"] == pytest.approx(0.167799, abs=1e-5)
    assert res.loc["x1", "n"] == 58
