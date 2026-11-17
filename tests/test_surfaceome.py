"""Highlights page, Cell surface proteins tab: key findings, marker expression, multi-antigen."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_stats import CL

ROOT = Path(__file__).resolve().parents[1]
needs_surface = pytest.mark.skipif(not (CL / "values" / "surface_george.tsv.gz").exists(),
                                   reason="surface_george layer not prepared")


@needs_surface
def test_surfaceome_page_and_controls():
    at = AppTest.from_file(str(ROOT / "pages/highlights.py"), default_timeout=600).run()
    assert not at.exception, at.exception
    assert any(h.value == "Featured findings" for h in at.subheader)
    for rule in ["above percentile", "above value", "detected"]:
        at.segmented_control(key="su_rule").set_value(rule).run()
        assert not at.exception, (rule, at.exception)
    at.segmented_control(key="hl_rule").set_value("p < 0.01").run()
    at.toggle(key="hl_surf").set_value(True).run()
    at.multiselect(key="hl_feat").set_value([]).run()
    at.selectbox(key="sm_m").set_value("FLT3").run()
    at.segmented_control(key="sm_scope").set_value("all markers").run()
    at.segmented_control(key="ma_logic").set_value("all (AND)").run()
    at.segmented_control(key="ma_size").set_value(3).run()
    assert not at.exception, at.exception


def test_positive_rules():
    import numpy as np
    import pandas as pd
    from portal.surfaceome import positive
    v = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0], index=list("abcde"))
    assert positive(v, "detected", 0).tolist() == [False, True, True, True, True]
    assert positive(v, "above value", 2.5).tolist() == [False, False, False, True, True]
    assert positive(v, "above percentile", 50).sum() == 2
