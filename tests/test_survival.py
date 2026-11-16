"""Survival statistics vs R survival 3.8, and the Patients → Survival tab on a synthetic cohort."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from portal.survival import kaplan_meier, logrank, median_survival
from tests.test_stats import CL, RESULTS

ROOT = Path(__file__).resolve().parents[1]

# survival::aml
TIME = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161, 5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
STATUS = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
ARM = ["Maintained"] * 11 + ["Nonmaintained"] * 12


def test_logrank_matches_r_survdiff():
    r = logrank(TIME, np.array(STATUS) == 1, ARM)       # survdiff(Surv(time, status) ~ x, aml)
    assert r["chi2"] == pytest.approx(3.396389, abs=1e-6)
    assert r["observed"] == {"Maintained": 7, "Nonmaintained": 11}
    assert r["expected"]["Maintained"] == pytest.approx(10.68934, abs=1e-5)
    assert r["p"] == pytest.approx(0.0653, abs=1e-4)
    r2 = logrank(TIME, np.array(STATUS) == 1, ARM, order=["Nonmaintained", "Maintained"])
    assert r2["hr"] == pytest.approx(1 / r["hr"])                     # HR follows the given order
    assert r2["chi2"] == pytest.approx(r["chi2"])


def test_km_matches_r_survfit_loglog():
    km = kaplan_meier(TIME, np.array(STATUS) == 1)      # survfit(..., conf.type = "log-log")
    ev = km[km["n_event"] > 0].head(3)
    assert ev["surv"].round(4).tolist() == [0.9130, 0.8261, 0.7826]
    assert ev["lower"].round(4).tolist() == [0.6949, 0.6006, 0.5542]
    assert ev["upper"].round(4).tolist() == [0.9775, 0.9309, 0.9032]
    assert median_survival(km) == 27


@pytest.fixture
def synthetic_results(tmp_path, monkeypatch):
    """results/ with the real catalog, schema and cell lines plus one fake patient cohort."""
    if not CL.is_dir():
        pytest.skip("results/cell_lines not prepared")
    for name in ["catalog", "schema", "cell_lines"]:
        (tmp_path / name).symlink_to(RESULTS / name)
    coh = tmp_path / "patients" / "TCGA_LAML"
    (coh / "features").mkdir(parents=True)
    (coh / "values").mkdir()
    rng = np.random.default_rng(0)
    n = 200
    tp53 = rng.random(n) < 0.15
    expr = rng.normal(size=n)
    days = rng.exponential(700 / np.exp(0.9 * tp53 + 1.0 * expr))
    ids = [f"P{i:03d}" for i in range(n)]
    pd.DataFrame({
        "study_id": "TCGA_LAML", "sample_id": ids, "participant_id": ids, "sample_type": "patient",
        "population": "adult", "disease_oncotree": "AML",
        "mutations": [";".join(m for m, x in [("TP53", t), ("FLT3", r < 0.3), ("NPM1", r > 0.7)] if x)
                      for t, r in zip(tp53, rng.random(n))],
        "KMT2A_rearranged": np.where(rng.random(n) < 0.1, "TRUE", "FALSE"),
        "risk_group": rng.choice(["favorable", "intermediate", "adverse"], n),
        "overall_survival_days": days.round(), "os_event": np.where(rng.random(n) < 0.7, "TRUE", "FALSE"),
    }).to_csv(coh / "samples.tsv", sep="\t", index=False)
    pd.DataFrame({"key": ["expression"], "label": ["RNA expression"], "modality": ["transcriptomics"],
                  "unit": ["log2 TPM"], "source": ["synthetic"], "lower_is_stronger": [False],
                  "n_features": [2], "n_samples": [n]}).to_csv(coh / "datasets.tsv", sep="\t", index=False)
    pd.DataFrame({"feature": ["MAP1A", "BCL2"], "label": ["MAP1A", "BCL2"], "gene_symbol": ["MAP1A", "BCL2"]}
                 ).to_csv(coh / "features" / "expression.tsv", sep="\t", index=False)
    vals = pd.DataFrame([expr, rng.normal(size=n)], index=pd.Index(["MAP1A", "BCL2"], name="feature"), columns=ids)
    with gzip.open(coh / "values" / "expression.tsv.gz", "wt") as fh:
        vals.to_csv(fh, sep="\t")
    monkeypatch.setenv("AML_PORTAL_DATA", str(tmp_path))
    st.cache_data.clear()
    yield tmp_path
    st.cache_data.clear()


def test_survival_tab_placeholder_without_patients(tmp_path, monkeypatch):
    if not CL.is_dir():
        pytest.skip("results/cell_lines not prepared")
    for name in ["catalog", "schema", "cell_lines"]:
        (tmp_path / name).symlink_to(RESULTS / name)
    monkeypatch.setenv("AML_PORTAL_DATA", str(tmp_path))
    st.cache_data.clear()
    at = AppTest.from_file(str(ROOT / "pages/patients.py"), default_timeout=120).run()
    st.cache_data.clear()
    assert not at.exception, at.exception
    assert any("No patient cohort has survival data" in i.value for i in at.info)


def test_survival_tab_on_synthetic_cohort(synthetic_results):
    at = AppTest.from_file(str(ROOT / "pages/patients.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.radio(key="surv_by").options == ["Mutation", "Gene of interest", "Clinical"]
    muts = at.selectbox(key="surv_mut").options
    assert any(o.startswith("TP53 (") for o in muts) and any(o.startswith("FLT3 (") for o in muts), muts
    at.selectbox(key="surv_mut").set_value("mut:TP53").run()
    assert not at.exception, at.exception
    at.radio(key="surv_by").set_value("Clinical").run()
    assert {"Risk group", "KMT2A rearrangement"} <= set(at.selectbox(key="surv_clin").options)
    for key in ["risk_group", "KMT2A_rearranged"]:
        at.selectbox(key="surv_clin").set_value(key).run()
        assert not at.exception, (key, at.exception)
    at.radio(key="surv_by").set_value("Gene of interest").run()
    at.selectbox(key="surv_gene").set_value("MAP1A").run()
    p = {m.label: m.value for m in at.metric}["Log-rank p"]
    assert float(p) < 0.05                                       # MAP1A drives hazard in the fixture
    for split in ["tertiles (top vs bottom third)", "quartiles (top vs bottom quarter)"]:
        at.selectbox(key="surv_split").set_value(split).run()
        assert not at.exception, (split, at.exception)
    at.selectbox(key="surv_unit").set_value("years").run()
    at.toggle(key="surv_ci").set_value(False).run()
    assert not at.exception, at.exception
