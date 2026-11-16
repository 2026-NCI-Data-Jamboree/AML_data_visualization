"""Patient cohorts: correlation statistics, and every Patients tab on the prepared cohorts."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from streamlit.testing.v1 import AppTest

from portal.stats import meta_correlation, spearman_rows
from tests.test_stats import RESULTS

ROOT = Path(__file__).resolve().parents[1]
PATIENTS = RESULTS / "patients"
needs_patients = pytest.mark.skipif(not (PATIENTS / "BEAT_AML").is_dir(), reason="results/patients not prepared")


def test_spearman_rows_matches_scipy():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(40, 30)), columns=[f"s{i}" for i in range(30)])
    X.iloc[5, [2, 7, 11]] = np.nan                         # a row with missing values
    y = pd.Series(X.iloc[0].to_numpy() + rng.normal(size=30), index=X.columns)
    y.iloc[[0, 4]] = np.nan
    res = spearman_rows(X, y, min_n=10).set_index("feature")
    for i in [0, 1, 5, 17]:
        ok = X.iloc[i].notna() & y.notna()
        ref = stats.spearmanr(X.iloc[i][ok], y[ok])
        assert res.loc[i, "r"] == pytest.approx(ref.statistic, abs=1e-10)
        assert res.loc[i, "p"] == pytest.approx(ref.pvalue, rel=1e-6)
        assert res.loc[i, "n"] == ok.sum()


def test_meta_correlation_fisher_z():
    a = pd.DataFrame({"feature": ["g1", "g2"], "n": [103, 103], "r": [0.3, -0.2]})
    b = pd.DataFrame({"feature": ["g1", "g2"], "n": [53, 53], "r": [0.1, 0.25]})
    m = meta_correlation({"A": a, "B": b}).set_index("feature")
    zbar = (100 * np.arctanh(0.3) + 50 * np.arctanh(0.1)) / 150
    assert m.loc["g1", "r"] == pytest.approx(np.tanh(zbar))
    assert m.loc["g1", "z"] == pytest.approx(zbar * np.sqrt(150))
    assert bool(m.loc["g1", "concordant"]) and not bool(m.loc["g2", "concordant"])
    assert m.loc["g1", "n_total"] == 156


@needs_patients
def test_patient_results_follow_contract():
    for d in sorted(PATIENTS.iterdir()):
        s = pd.read_csv(d / "samples.tsv", sep="\t", low_memory=False)
        assert s["sample_id"].is_unique, d.name
        ds = pd.read_csv(d / "datasets.tsv", sep="\t")
        g = pd.read_csv(d / "groupings.tsv", sep="\t")
        assert set(g["column"]) <= set(s.columns), d.name
        for key in ds["key"]:
            f = pd.read_csv(d / "features" / f"{key}.tsv", sep="\t", index_col=0)
            v = pd.read_csv(d / "values" / f"{key}.tsv.gz", sep="\t", index_col=0, nrows=50)
            assert set(v.columns) <= set(s["sample_id"]), (d.name, key)
            assert list(v.index) == list(f.index[:len(v)]), (d.name, key)


@needs_patients
def test_drug_biomarkers_tab():
    at = AppTest.from_file(str(ROOT / "pages/patients.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="db_drug").value == "VENETOCLAX"
    assert any("**MAP1A**" in m.value for m in at.markdown)          # proof-of-concept default
    for drug in ["QUIZARTINIB", "CYTARABINE", "TRAMETINIB"]:
        if drug in at.selectbox(key="db_drug").options:
            at.selectbox(key="db_drug").set_value(drug).run()
            assert not at.exception, (drug, at.exception)
    at.selectbox(key="db_stage").set_value("diagnosis only").run()
    assert not at.exception, at.exception
    at.selectbox(key="db_drug").set_value("VENETOCLAX").run()
    at.radio(key="db_measure_MALANI_2022").set_value("drug_DSS").run()
    assert not at.exception, at.exception
    at.text_input(key="db_filter").set_value("BCL2").run()
    assert not at.exception, at.exception


@needs_patients
def test_patient_feature_query_and_cohorts_tabs():
    at = AppTest.from_file(str(ROOT / "pages/patients.py"), default_timeout=300).run()
    for gene in ["BCL2", "FLT3", "TP53"]:
        at.selectbox(key="pfq_gene").set_value(gene).run()
        assert not at.exception, (gene, at.exception)
    for grouping in ["mut_TP53", "KMT2A_rearranged", "mTOR_high", "timepoint"]:
        at.selectbox(key="pfq_grouping").set_value(grouping).run()
        assert not at.exception, (grouping, at.exception)
    from portal import data
    for cohort in data.patient_cohorts():
        at.selectbox(key="pt_cohort").set_value(cohort).run()
        assert not at.exception, (cohort, at.exception)


@needs_patients
def test_survival_tab_on_prepared_cohorts():
    at = AppTest.from_file(str(ROOT / "pages/patients.py"), default_timeout=300).run()
    from portal.survival_view import ENDPOINTS, _grouping_options, _patients
    names = at.selectbox(key="surv_cohort").options                  # display names
    assert not any("Lee" in n for n in names)                         # no event indicator
    for cohort in ["BEAT_AML", "MALANI_2022", "WANG_2022_PED"]:
        at.selectbox(key="surv_cohort").set_value(cohort).run()
        for ep in at.selectbox(key="surv_endpoint").options:
            at.selectbox(key="surv_endpoint").set_value(ep).run()
            opts = _grouping_options(cohort, _patients(cohort, *ENDPOINTS[ep]))
            assert not {"MYC_high", "mTOR_high", "MYC_mTOR_category"} & set(opts.get("Clinical", {}))
            for by, keys in opts.items():
                at.radio(key="surv_by").set_value(by).run()
                widget = {"Mutation": "surv_mut", "Clinical": "surv_clin"}.get(by)
                for k in (list(keys)[:5] if widget else []):
                    at.selectbox(key=widget).set_value(k).run()
                    assert not at.exception, (cohort, ep, k, at.exception)
                assert not at.exception, (cohort, ep, by, at.exception)
    at.selectbox(key="surv_cohort").set_value("BEAT_AML").run()
    at.selectbox(key="surv_endpoint").set_value("Overall survival").run()
    at.radio(key="surv_by").set_value("Mutation").run()
    at.selectbox(key="surv_mut").set_value("mut_TP53").run()
    m = {x.label: x.value for x in at.metric}
    assert float(m["Log-rank p"]) < 1e-10                             # TP53-mutant AML: worse OS
    assert float(m["HR TP53-mutant vs TP53-wild-type"]) > 2           # HR in the labelled direction


def test_age_groups_nci_bands():
    from portal.plots import age_groups
    g = age_groups(pd.Series([0.5, 14.9, 15, 39.9, 40, np.nan]))
    assert g.tolist() == ["pediatric (<15)", "pediatric (<15)", "AYA (15-39)", "AYA (15-39)",
                          "adult (≥40)", "age unknown"]
