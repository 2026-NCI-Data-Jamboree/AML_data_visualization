"""Cell lines page, Drug sensitivity tab: mutation-drug associations, target overexpression, CN-driven targets."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from streamlit.testing.v1 import AppTest

from portal.genomics_stats import correlate, rowwise_correlate
from tests.test_stats import CL

ROOT = Path(__file__).resolve().parents[1]
needs_data = pytest.mark.skipif(not CL.is_dir(), reason="results/cell_lines not prepared")


def test_correlate_matches_scipy_with_missing():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(30, 15)), columns=[f"L{i}" for i in range(15)])
    y = pd.Series(rng.normal(size=15), index=X.columns)
    X.iloc[0, :3] = np.nan
    X.iloc[1, :10] = np.nan                                   # n = 5 < min_n: dropped
    for method, ref in [("pearson", stats.pearsonr), ("spearman", stats.spearmanr)]:
        res = correlate(X, y, method).set_index("feature")
        assert 1 not in res.index and len(res) == 29
        for f in [0, 2, 7]:
            ok = X.loc[f].notna()
            r, p = ref(X.loc[f, ok], y[ok])
            assert res.loc[f, "r"] == pytest.approx(r, abs=1e-9) and res.loc[f, "p"] == pytest.approx(p, rel=1e-6)


def test_rowwise_correlate_matches_scipy():
    rng = np.random.default_rng(11)
    cols = [f"L{i}" for i in range(12)]
    A = pd.DataFrame(rng.normal(size=(5, 12)), index=list("abcde"), columns=cols)
    B = A * -0.8 + rng.normal(scale=0.5, size=(5, 12))
    B = B.drop(index="e").rename(index={"d": "z"})               # genes in both: a, b, c
    A.iloc[0, :2] = np.nan
    res = rowwise_correlate(A, B[cols[2:] + cols[:2]], "pearson").set_index("gene")
    assert set(res.index) == {"a", "b", "c"}
    ok = A.loc["a"].notna()
    r, p = stats.pearsonr(A.loc["a", ok], B.loc["a", ok])
    assert res.loc["a", "r"] == pytest.approx(r) and res.loc["a", "p"] == pytest.approx(p, rel=1e-6)
    assert res.loc["a", "n"] == 10 and (res["r"] < 0).all()


@needs_data
def test_drug_sensitivity_page():
    from portal import data
    from portal.drug_sensitivity import drug_layers
    at = AppTest.from_file(str(ROOT / "pages/cell_lines.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    if not drug_layers():
        return
    for screen in drug_layers():
        at.selectbox(key="dm_screen").set_value(screen).run()
        assert not at.exception, (screen, at.exception)
    for show in ["all", "sensitive only", "significant"]:
        at.segmented_control(key="dm_show").set_value(show).run()
    at.segmented_control(key="dm_sig").set_value("q < 0.1").run()
    assert not at.exception, at.exception
    keys = set(data.datasets()["key"])
    for layer in [k for k in ["expression", "surface_george"] if k in keys]:
        at.selectbox(key="dt_layer").set_value(layer).run()
        assert not at.exception, (layer, at.exception)
    at.number_input(key="dt_z").set_value(1.0).run()
    if "cn" in keys:
        at.toggle(key="dc_drug").set_value(False).run()
        at.number_input(key="dc_amp").set_value(1.3).run()
        assert not at.exception, at.exception
