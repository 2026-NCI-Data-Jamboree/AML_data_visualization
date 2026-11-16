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
