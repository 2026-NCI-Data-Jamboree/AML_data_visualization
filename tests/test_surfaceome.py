"""Highlights page, Cell surface proteins tab: cell lines vs patients (George 2026 vs Leucegene surfaceome)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_stats import CL

ROOT = Path(__file__).resolve().parents[1]
needs_surface = pytest.mark.skipif(not (CL / "values" / "surface_george.tsv.gz").exists(),
                                   reason="surface_george layer not prepared")


@needs_surface
def test_surface_lines_vs_patients_tab():
    from portal.surface_cross import available, comparisons
    if not available():
        pytest.skip("Leucegene surfaceome cohort not prepared")
    at = AppTest.from_file(str(ROOT / "pages/highlights.py"), default_timeout=600).run()
    assert not at.exception, at.exception
    assert not {"Cell lines", "Cell lines vs patients"} & {t.label for t in at.tabs}   # one view, no sub-tabs
    assert at.selectbox(key="sx_cmp").value == "KMT2A-r vs not"                     # default
    assert not any("Abundance" in h.value for h in at.subheader)                  # abundance section removed
    assert any("Select one or more proteins" in i.value for i in at.info)         # boxes only for a selection
    assert not any("MYC" in o for o in at.selectbox(key="sx_cmp").options)       # no MYC / mTOR for patients
    assert "TP53-mutant vs not" in comparisons()                                  # TP53 from the Leucegene calls
    for name in comparisons():
        at.selectbox(key="sx_cmp").set_value(name).run()
        assert not at.exception, (name, at.exception)
        res = [d.value for d in at.dataframe if {"log2FC lines", "log2FC patients"} <= set(d.value.columns)][0]
        assert list(res.columns[:3]) == ["protein", "supported", "combined p"]      # both datasets first
        sup = res["supported"].tolist()
        assert sup == sorted(sup, reverse=True)                                    # supported proteins first
        for flag in (True, False):                                                 # then by combined p
            p = res.loc[res["supported"] == flag, "combined p"]
            assert p.is_monotonic_increasing
    at.toggle(key="sx_surf").set_value(True).run()
    assert not at.exception, at.exception


@needs_surface
def test_surface_selection_draws_boxplots():
    """Proteins picked on the scatter (or in the table) are drawn as box plots: cell lines and patients."""
    import json
    from portal.surface_cross import available
    if not available():
        pytest.skip("Leucegene surfaceome cohort not prepared")
    at = AppTest.from_string("""
import portal.surface_cross as sx
sx.selected_genes = lambda ev: ["CD38", "IL10RB"]          # as if picked on the scatter
sx.cross_section()
""", default_timeout=600).run()
    assert not at.exception, at.exception
    titles = [json.loads(c.proto.spec)["layout"].get("title", {}).get("text", "") for c in at.get("plotly_chart")]
    assert sum("· cell lines" in t for t in titles) == 2 and sum("· patients" in t for t in titles) == 2
