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
    one = meta_correlation({"A": a}).set_index("feature")
    assert (one["I2"] == 0).all()                                      # no heterogeneity with one cohort


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
    at = AppTest.from_file(str(ROOT / "pages/highlights.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="db_drug_expression").value == "VENETOCLAX"
    assert any("**MAP1A**" in m.value for m in at.markdown)          # proof-of-concept default
    for drug in ["QUIZARTINIB", "CYTARABINE", "TRAMETINIB"]:
        if drug in at.selectbox(key="db_drug_expression").options:
            at.selectbox(key="db_drug_expression").set_value(drug).run()
            assert not at.exception, (drug, at.exception)
    at.selectbox(key="db_stage").set_value("diagnosis only").run()
    assert not at.exception, at.exception
    at.selectbox(key="db_drug_expression").set_value("VENETOCLAX").run()
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
    assert not {"MYC RNA signature", "mTOR RNA signature", "MYC x mTOR RNA category",
                "Cell state (Primitive / Committed)"} & set(
        at.selectbox(key="pfq_grouping").options)
    for grouping in ["mut_TP53", "KMT2A_rearranged", "risk_group", "timepoint"]:
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


@needs_patients
def test_age_association_tab():
    at = AppTest.from_file(str(ROOT / "pages/patients.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    t = [d.value for d in at.dataframe if "concordant" in d.value.columns and "gene" in d.value.columns][-1]
    lrrn3 = t.set_index("gene").loc["LRRN3"]                    # known to fall with age in blood
    assert lrrn3["r"] < 0 and lrrn3["fdr"] < 1e-6
    show = [m for m in at.multiselect if m.label == "Show genes"][0]
    assert show.value == [t["gene"].iloc[0]]                            # top hit by default
    show.set_value(["MYH11", "HOXA9"]).run()
    assert not at.exception, at.exception
    assert any(m.value.startswith("**HOXA9**") for m in at.markdown)
    at.selectbox(key="age_what").set_value("Drug response").run()
    assert not at.exception, at.exception
    [m for m in at.multiselect if m.label == "Show drugs"][0].set_value(["Venetoclax"]).run()
    assert any(m.value.startswith("**Venetoclax**") for m in at.markdown)
    at.multiselect(key="age_cohorts").set_value(["BEAT_AML", "WANG_2022_PED"]).run()
    assert not at.exception, at.exception
    at.selectbox(key="age_what").set_value("Gene expression").run()
    at.text_input(key="age_filter").set_value("HOXA").run()
    assert not at.exception, at.exception
    at.multiselect(key="age_cohorts").set_value(["BEAT_AML", "LEE_2018", "MALANI_2022", "WANG_2022_PED"]).run()
    at.radio(key="age_how").set_value("Age groups").run()
    assert not at.exception, at.exception
    note = [c.value for c in at.caption if c.value.startswith("Cohorts compared")][0]
    assert "Not used" in note and "Pediatric AML pharmacogenomics." in note   # no adults >= 40 for AYA vs adult
    assert not any("median age" in d.value.columns for d in at.dataframe)
    for ga, gb in [("pediatric (<15)", "all other ages"), ("pediatric (<15)", "AYA (15-39)"),
                   ("adult (≥40)", "pediatric (<15)")]:
        at.selectbox(key="age_ga").set_value(ga).run()
        at.selectbox(key="age_gb").set_value(gb).run()
        assert not at.exception, (ga, gb, at.exception)
    at.selectbox(key="age_what").set_value("Drug response").run()
    assert not at.exception, at.exception


def test_age_group_target_is_binary():
    from portal.age_association import OTHERS, _target
    if not (PATIENTS / "BEAT_AML").is_dir():
        pytest.skip("results/patients not prepared")
    y = _target("BEAT_AML", ("groups", "pediatric (<15)", OTHERS))
    assert set(y.unique()) <= {0.0, 1.0} and y.eq(1).sum() >= 3 and y.eq(0).sum() > 100


@needs_patients
def test_proteomics_cohorts():
    from portal import data
    for c in ["PTRC_BEAT_AML", "CPTAC_AML_ADULT"]:
        if c not in data.patient_cohorts():
            pytest.skip(f"{c} not prepared")
        keys = set(data.patient_datasets(c)["key"])
        assert {"protein", "phospho"} <= keys, (c, keys)
    names = data.feature_names("CPTAC_AML_ADULT", "phospho")
    assert names.str.contains("_").all()                                 # sites, not genes
    at = AppTest.from_file(str(ROOT / "pages/patients.py"), default_timeout=300).run()
    for dtype in ["protein", "phospho"]:
        at.selectbox(key="pfq_type").set_value(dtype).run()
        at.selectbox(key="pfq_gene").set_value("STAT5A").run()
        assert not at.exception, (dtype, at.exception)
    sites = at.multiselect(key="pfq_sites_STAT5A")
    assert 0 < len(sites.value) <= 6 and all(v.startswith("STAT5A_") for v in sites.value)
    sites.set_value(["STAT5A_S128"]).run()                           # one chosen site
    assert not at.exception, at.exception
    at.selectbox(key="pfq_grouping").set_value("mut_FLT3").run()
    import json
    titles = [json.loads(c.proto.spec).get("layout", {}).get("title", {}).get("text", "") for c in at.get("plotly_chart")]
    assert any("STAT5A_S128" in t and "Welch p =" in t for t in titles), titles
    at.selectbox(key="pfq_type").set_value("expression").run()
    hl = AppTest.from_file(str(ROOT / "pages/highlights.py"), default_timeout=300).run()
    hl.selectbox(key="db_layer").set_value("protein").run()
    assert not hl.exception, hl.exception
    info = [d.value for d in hl.dataframe if "samples with both" in d.value.columns][0]
    assert info["used"].all() and info["samples with both"].min() >= 100   # PTRC protein + Beat AML drug AUC
    t = [d.value for d in hl.dataframe if "concordant" in d.value.columns and "gene" in d.value.columns][0]
    assert t.set_index("gene").loc["MAP1A", "r"] > 0                    # MAP1A protein: venetoclax sensitivity
    hl.selectbox(key="db_layer").set_value("phospho").run()
    assert not hl.exception, hl.exception
    at.selectbox(key="age_what").set_value("Protein").run()
    assert not at.exception, at.exception
    at.selectbox(key="surv_cohort").set_value("PTRC_BEAT_AML").run()
    at.radio(key="surv_by").set_value("Gene expression").run()
    at.selectbox(key="surv_layer").set_value("phospho").run()
    assert not at.exception, at.exception


@needs_patients
def test_drug_marker_lookup_tab():
    at = AppTest.from_file(str(ROOT / "pages/patients.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="dm_marker_expression").value == "MAP1A"
    assert at.selectbox(key="dm_drug_expression").value == "VENETOCLAX"
    m = {x.label: x.value for x in at.metric}
    assert float(m["Pooled r"]) > 0 and m["Direction"] == "consistent"
    tab = [d.value for d in at.dataframe if "r (sensitivity)" in d.value.columns][0]
    import json
    titles = [json.loads(c.proto.spec).get("layout", {}).get("title", {}).get("text", "") for c in at.get("plotly_chart")]
    assert all(any(t.startswith(f"{c}<br><sup>n=") for t in titles) for c in tab["cohort"])  # a plot per cohort
    for layer, marker in [("protein", "BCL2"), ("phospho", "STAT5A_S749")]:
        at.selectbox(key="dm_layer").set_value(layer).run()
        at.selectbox(key=f"dm_marker_{layer}").set_value(marker).run()
        assert not at.exception, (layer, at.exception)
    at.selectbox(key="dm_stage").set_value("diagnosis only").run()
    assert not at.exception, at.exception
