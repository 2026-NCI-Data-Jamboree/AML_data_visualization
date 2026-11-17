"""Smoke test: every page renders without an exception against the prepared results."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_stats import CL

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(str(p) for p in (ROOT / "pages").glob("*.py"))
needs_data = pytest.mark.skipif(not CL.is_dir(), reason="results/cell_lines not prepared")


@needs_data
@pytest.mark.parametrize("page", PAGES, ids=lambda p: Path(p).stem)
def test_page_renders(page):
    at = AppTest.from_file(page, default_timeout=180).run()
    assert not at.exception, at.exception


@needs_data
def test_cell_lines_every_grouping_and_layer():
    at = AppTest.from_file(str(ROOT / "pages/cell_lines.py"), default_timeout=180).run()
    from portal import data
    for grouping in data.grouping_labels():
        at.selectbox(key="grouping").set_value(grouping).run()
        assert not at.exception, (grouping, at.exception)
    for layer in data.datasets()["key"]:
        at.selectbox(key="layer").set_value(layer).run()
        assert not at.exception, (layer, at.exception)


@needs_data
def test_cell_line_heatmap_sorting():
    at = AppTest.from_file(str(ROOT / "pages/cell_lines.py"), default_timeout=180).run()
    for opt in ["mTOR_high", "disease_oncotree", "data availability"]:
        at.selectbox(key="sort_by").set_value(opt).run()
        assert not at.exception, (opt, at.exception)
    at.multiselect[0].set_value([]).run()                     # no tracks: availability only
    assert not at.exception, at.exception


@needs_data
def test_cell_line_feature_query_tab():
    at = AppTest.from_file(str(ROOT / "pages/cell_lines.py"), default_timeout=180).run()
    for gene in ["TP53", "BCL2", "KMT2A", "MTOR"]:
        at.selectbox(key="fq_gene").set_value(gene).run()
        assert not at.exception, (gene, at.exception)
    for grouping in ["KMT2A_rearranged", "MYC_mTOR_category", "mTOR_high"]:
        at.selectbox(key="fq_grouping").set_value(grouping).run()
        assert not at.exception, (grouping, at.exception)


@needs_data
def test_cell_line_gene_dependency_tab():
    from portal.dependency import CUSTOM, PRESETS
    at = AppTest.from_file(str(ROOT / "pages/cell_lines.py"), default_timeout=180).run()
    assert not any(t.key == "dep_filter" for t in at.text_input)        # rows are chosen in the table instead
    assert any("(top row)" in h.value for h in at.subheader)
    at.selectbox(key="dep_color").set_value("KMT2A_rearranged").run()
    assert not at.exception, at.exception
    for comparison in PRESETS:                                  # every featured comparison is testable
        at.selectbox(key="dep_comparison").set_value(comparison).run()
        assert not at.exception, (comparison, at.exception)
        assert any("CRISPR vs RNAi per cell line" in h.value for h in at.subheader), comparison
    at.selectbox(key="dep_comparison").set_value(CUSTOM).run()
    for grouping in ["disease_oncotree", "mut_FLT3", "population"]:  # too-small groups warn, no crash
        at.selectbox(key="dep_grouping").set_value(grouping).run()
        assert not at.exception, (grouping, at.exception)
    at.toggle(key="dep_shared").set_value(True).run()
    assert not at.exception, at.exception
    at.toggle(key="dep_shared").set_value(False).run()
    assert not at.exception, at.exception


def test_selected_genes_from_plot_event():
    from portal.dependency import selected_genes
    pts = [{"customdata": "BCL2"}, {"customdata": ["MCL1"]}, {"customdata": "BCL2"}, {"x": 1}]
    assert selected_genes({"selection": {"points": pts}}) == ["BCL2", "MCL1"]
    assert selected_genes({"selection": {"points": []}}) == []
    assert selected_genes(None) == []


@needs_data
def test_highlights_myc_mtor_dependency():
    at = AppTest.from_file(str(ROOT / "pages/highlights.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    assert any("MYC-High / mTOR-Low" in m.value and "MYC-Low / mTOR-High" in m.value for m in at.markdown)
    assert not any(s.key == "hl_dep_comparison" for s in at.selectbox)   # comparison is fixed
    at.toggle(key="hl_dep_shared").set_value(True).run()
    assert not at.exception, at.exception


@needs_data
def test_downloads_shows_data_readme():
    from portal import data
    if not data.results_readme():
        pytest.skip("results/README.md not prepared")
    at = AppTest.from_file(str(ROOT / "pages/downloads.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert at.tabs[0].label == "About the data"
    assert any("Folder layout" in m.value for m in at.markdown)
