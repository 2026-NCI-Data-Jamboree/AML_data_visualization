"""Highlights page, Cell surface proteins tab: volcano plot and result table per subgroup comparison."""
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
    assert not any(h.value in ("Featured findings", "Key volcano plots", "Top findings") for h in at.subheader)
    assert not any(s.key == "su_rule" for s in at.segmented_control)            # no positivity rule
    assert not any(t.label in ("Marker expression", "Multi-antigen") for t in at.tabs)
    at.selectbox(key="hl_vc").set_value("TP53-mutant vs not").run()
    at.segmented_control(key="hl_vrule").set_value("q < 0.1").run()
    assert not at.exception, at.exception
    res = [d.value for d in at.dataframe if {"protein", "log2FC", "q"} <= set(d.value.columns)][0]
    assert len(res) > 100 and "direction" not in res.columns
    assert res["p"].is_monotonic_increasing
    at.toggle(key="hl_surf").set_value(True).run()
    assert not at.exception, at.exception
