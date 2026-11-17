"""Genomic Cell lines tabs: co-occurrence, mutation frequency, copy number, mutation-CNV, similar lines."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from streamlit.testing.v1 import AppTest

from portal.genomics_stats import cooccurrence, jaccard_similarity, mutation_enrichment, set_exclusivity
from portal.stats import bh
from tests.test_stats import CL

ROOT = Path(__file__).resolve().parents[1]
needs_data = pytest.mark.skipif(not CL.is_dir(), reason="results/cell_lines not prepared")


def test_cooccurrence_counts_and_fisher():
    # 12 lines; A and B never together (exclusive), A and C always together (co-occurring);
    # line L11 is not annotated for C, so the A-C pair uses 11 lines
    A = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    B = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    C = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, np.nan]
    M = pd.DataFrame({"A": A, "B": B, "C": C}, index=[f"L{i}" for i in range(12)])
    res = cooccurrence(M).set_index(["A", "B"])
    ab, ac = res.loc[("A", "B")], res.loc[("A", "C")]
    assert (ab["both"], ab["A_not_B"], ab["B_not_A"], ab["neither"], ab["n_lines"]) == (0, 4, 4, 4, 12)
    assert ab["tendency"] == "mutual exclusivity" and ab["log2_odds_ratio"] < 0
    assert ab["p"] == pytest.approx(stats.fisher_exact([[0, 4], [4, 4]])[1])
    assert ac["n_lines"] == 11 and ac["both"] == 4 and ac["tendency"] == "co-occurrence"
    assert ac["p"] == pytest.approx(stats.fisher_exact([[4, 0], [0, 7]])[1])
    one = cooccurrence(M, sided="one-sided").set_index(["A", "B"])
    assert one.loc[("A", "B"), "p"] == pytest.approx(stats.fisher_exact([[0, 4], [4, 4]], alternative="less")[1])
    assert one.loc[("A", "C"), "p"] == pytest.approx(stats.fisher_exact([[4, 0], [0, 7]], alternative="greater")[1])
    np.testing.assert_allclose(res["q"], bh(res["p"].to_numpy()))
    # an alteration with no altered line is not testable
    assert cooccurrence(M.assign(D=0))["A"].tolist().count("D") == 0


def test_set_exclusivity_direction():
    excl = pd.DataFrame({"A": [1, 1, 0, 0, 0, 0] * 4, "B": [0, 0, 1, 1, 0, 0] * 4, "C": [0, 0, 0, 0, 1, 0] * 4})
    r = set_exclusivity(excl, n_perm=2000)
    assert r["coverage"] == 20 and r["overlapping"] == 0 and r["p_exclusive"] < 0.01
    same = pd.DataFrame({"A": [1, 1, 0, 0] * 6, "B": [1, 1, 0, 0] * 6})
    r = set_exclusivity(same, n_perm=2000)
    assert r["coverage"] == 12 and r["p_exclusive"] > 0.5 and r["p_cooccurring"] < 0.01


def test_mutation_enrichment_matches_fisher():
    idx = [f"L{i}" for i in range(10)]
    M = pd.DataFrame({"X": [1, 1, 1, 0, 0, 0, 0, 0, 0, np.nan], "Y": [0] * 10}, index=idx)
    res = mutation_enrichment(M, idx[:5], idx[5:]).set_index("alteration")
    assert "Y" not in res.index                                  # never altered: dropped
    x = res.loc["X"]
    assert (x["altered_A"], x["n_A"], x["altered_B"], x["n_B"]) == (3, 5, 0, 4)   # L9 not annotated
    assert x["freq_A"] == pytest.approx(0.6) and x["freq_B"] == 0
    assert x["p"] == pytest.approx(stats.fisher_exact([[3, 2], [0, 4]])[1])


def test_jaccard_similarity_with_missing():
    X = pd.DataFrame({"a": [1, 1, 0, np.nan], "b": [1, 0, 0, 1], "c": [0, 1, 0, 1]}, index=list("PQRS"))
    J = jaccard_similarity(X)
    assert J.loc["P", "Q"] == pytest.approx(1 / 3)        # shared a; union a, b, c
    assert J.loc["P", "P"] == 1 and np.isnan(J.loc["R", "R"])  # R has no aberration
    assert J.loc["P", "S"] == pytest.approx(1 / 2)        # feature a not annotated in S: b shared, c only S
    assert J.loc["R", "P"] == 0


@needs_data
def test_cell_line_cooccurrence_tab():
    at = AppTest.from_file(str(ROOT / "pages/cell_lines.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert any(h.value == "Pairwise co-occurrence and mutual exclusivity" for h in at.subheader)
    for mode in ["exactly one", "any", "none", "all selected"]:
        at.segmented_control(key="co_mode_False").set_value(mode).run()
        assert not at.exception, (mode, at.exception)
    at.segmented_control(key="co_sided").set_value("one-sided").run()
    at.toggle(key="co_complete").set_value(True).run()
    assert not at.exception, at.exception
    from portal.cooccurrence import has_cn
    if has_cn():
        at.text_input(key="co_cn_genes").set_value("MYC, TP53, NOTAGENE").run()   # gene-level CN events
        opts = at.multiselect(key="co_alts").options
        assert any("MYC amp" in o for o in opts) and any("del(17p)" in o for o in opts), opts[:5]
        at.multiselect(key="co_alts").set_value(["mut_TP53", "cn_del_17p", "cn_dl_TP53"]).run()
        assert not at.exception, at.exception
    at.selectbox(key="co_subset_by").set_value("disease_oncotree").run()   # small subset, no crash
    assert not at.exception, at.exception
    at.selectbox(key="co_subset_by").set_value("all lines").run()
    at.multiselect(key="co_alts").set_value([]).run()
    assert not at.exception and at.info, at.exception


@needs_data
def test_cell_line_mutation_frequency_tab():
    at = AppTest.from_file(str(ROOT / "pages/cell_lines.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="mf_grouping").value == "KMT2A_rearranged"
    at.segmented_control(key="mf_sort").set_value("frequency").run()
    for grouping in ["mut_TP53", "MYC_mTOR_category", "disease_oncotree"]:
        at.selectbox(key="mf_grouping").set_value(grouping).run()
        assert not at.exception, (grouping, at.exception)
    at.multiselect(key="mf_alts").set_value([]).run()
    assert not at.exception and at.info, at.exception


@needs_data
def test_cell_line_copy_number_tab():
    from portal.cooccurrence import has_cn
    at = AppTest.from_file(str(ROOT / "pages/cell_lines.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    if not has_cn():
        pytest.skip("no copy-number layer in these results")
    at.segmented_control(key="cn_res").set_value("cytoband").run()
    at.toggle(key="cn_sex").set_value(True).run()
    at.selectbox(key="cn_order").set_value("KMT2A_rearranged").run()
    at.number_input(key="cn_amp").set_value(1.5).run()
    assert not at.exception, at.exception


@needs_data
def test_cell_line_mutation_cnv_tab():
    from portal.cooccurrence import has_cn
    at = AppTest.from_file(str(ROOT / "pages/cell_lines.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    if not has_cn():
        pytest.skip("no copy-number layer in these results")
    for event in ["amplification", "copy-number level", "deletion"]:
        at.segmented_control(key="mc_event").set_value(event).run()
        assert not at.exception, (event, at.exception)
    at.segmented_control(key="mc_res").set_value("cytoband").run()
    at.multiselect(key="mc_muts").set_value([]).run()
    assert not at.exception and at.info, at.exception


@needs_data
def test_cell_line_similar_lines_tab():
    at = AppTest.from_file(str(ROOT / "pages/cell_lines.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert any(h.value == "Most similar lines" for h in at.subheader)
    for line in ["MV411", "THP1", "HL60"]:
        at.selectbox(key="sim_line").set_value(line).run()
        assert not at.exception, (line, at.exception)
    at.pills(key="sim_parts").set_value(["mutations"]).run()
    assert not at.exception, at.exception
    at.pills(key="sim_parts").set_value([]).run()
    assert not at.exception and at.info, at.exception


def test_cytoband_parsing_and_order():
    from portal.copy_number import BAND
    assert BAND.match("17p13.1").groups() == ("17", "p", "13")
    assert BAND.match("Xq28").groups() == ("X", "q", "28")
    assert BAND.match("1q21.1-q21.2").groups() == ("1", "q", "21")
    assert BAND.match("17") is None
