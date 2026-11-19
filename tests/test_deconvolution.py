"""Deconvolution page and the deconvolution files of results/ (AML_data_processing results contract)."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from portal import data
from portal.deconvolution import sample_flags

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "pages/deconvolution.py")
COHORTS = data.deconvolution_cohorts() if data.results_dir() else []
needs_deconvolution = pytest.mark.skipif(not COHORTS, reason="no deconvolution results prepared")


@needs_deconvolution
@pytest.mark.parametrize("cohort", COHORTS)
def test_layers_are_consistent(cohort):
    runs = data.deconvolution_runs(cohort)
    samples = set(data.patient_samples(cohort)["sample_id"])
    datasets = data.patient_datasets(cohort).set_index("key")
    for _, run in runs.iterrows():
        key = run["key"]
        assert key in datasets.index and datasets.loc[key, "modality"] == "deconvolution", key
        P = data.patient_values(cohort, key)
        assert set(P.columns) <= samples, f"{key}: columns that are not in samples.tsv"
        assert set(data.patient_features(cohort, key).index) == set(P.index), key
        assert (P.fillna(0) >= 0).all().all(), key
        sums = P.sum()
        if run["method"] != "instaprism_update_malignant" or run["level"] != "cell_state":
            assert sums.between(0.99, 1.01).all(), f"{key}: proportions do not sum to 1 ({sums.min()}, {sums.max()})"
    warnings = data.deconvolution_warnings(cohort)
    assert set(warnings["key"]) <= set(runs["key"]) | {""}, "a warning names a layer that does not exist"
    assert set(warnings.loc[warnings["scope"] == "sample", "sample_id"]) <= samples
    assert set(warnings["severity"]) <= {"info", "warning", "caution"}
    assert set(warnings["scope"]) <= {"run", "sample", "label"}


@needs_deconvolution
def test_reference_files_match_the_runs():
    for reference in data.single_cell_references():
        table = data.reference_table(reference)
        markers = data.reference_markers(reference)
        assert set(markers["label"]) <= set(table["label"])
        assert {"reference_name", "limitations"} <= set(data.reference_provenance(reference))
    used = {r for c in COHORTS for r in data.deconvolution_runs(c)["reference"]}
    assert used <= set(data.single_cell_references()), "a run uses a reference with no single_cell/ folder"


def test_sample_flags_takes_the_worst_severity():
    w = pd.DataFrame({"scope": ["sample", "sample", "run", "sample"], "sample_id": ["a", "a", "", "b"],
                      "severity": ["info", "caution", "caution", "warning"]})
    assert sample_flags(w, ["a", "b", "c"]).to_dict() == {"a": 3, "b": 2, "c": 0}


@needs_deconvolution
def test_every_cohort_method_level_renders():
    at = AppTest.from_file(PAGE, default_timeout=300).run()
    assert not at.exception, at.exception
    for cohort in COHORTS:
        at.selectbox(key="dc_cohort").set_value(cohort).run()
        runs = data.deconvolution_runs(cohort)
        for _, run in runs.iterrows():
            at.selectbox(key="dc_method").set_value(run["method"]).run()
            at.selectbox(key="dc_level").set_value(run["level"]).run()
            assert not at.exception, (cohort, run["key"], at.exception)
            for hide in (True, False):
                at.toggle(key="dc_hide").set_value(hide).run()
                assert not at.exception, (cohort, run["key"], hide, at.exception)


@needs_deconvolution
def test_grouping_sorting_and_reference_controls():
    at = AppTest.from_file(PAGE, default_timeout=300).run()
    at.toggle(key="dc_hide").set_value(False).run()
    for option in at.selectbox(key="dc_group").options:
        at.selectbox(key="dc_group").set_value(option).run()
        assert not at.exception, (option, at.exception)
    for option in at.selectbox(key="dc_sort").options[:4]:
        at.selectbox(key="dc_sort").set_value(option).run()
        assert not at.exception, (option, at.exception)
    for level in ("cell_type", "cell_state"):
        at.radio(key="dc_ref_level").set_value(level).run()
        for label in at.selectbox(key="dc_marker_label").options[:3]:
            at.selectbox(key="dc_marker_label").set_value(label).run()
            assert not at.exception, (level, label, at.exception)


@needs_deconvolution
def test_hiding_removes_flagged_samples():
    """On a layer with sample-level warnings, hiding drops exactly those samples from the views."""
    target = None
    for cohort in COHORTS:
        w = data.deconvolution_warnings(cohort)
        for _, run in data.deconvolution_runs(cohort).iterrows():
            flagged = set(w[(w["scope"] == "sample") & w["key"].isin(["", run["key"]])]["sample_id"])
            flagged &= set(data.patient_values(cohort, run["key"]).columns)
            if flagged:
                target = (cohort, run, len(flagged))
                break
        if target:
            break
    if target is None:
        pytest.skip("no sample-level warnings")
    cohort, run, n_flagged = target
    at = AppTest.from_file(PAGE, default_timeout=300).run()
    at.selectbox(key="dc_cohort").set_value(cohort).run()
    at.selectbox(key="dc_method").set_value(run["method"]).run()
    at.selectbox(key="dc_level").set_value(run["level"]).run()
    shown = {}
    for hide in (False, True):
        at.toggle(key="dc_hide").set_value(hide).run()
        shown[hide] = int(next(c.value for c in at.caption if "samples shown" in c.value).split()[0])
    assert shown[False] - shown[True] == n_flagged


DIFF_COHORTS = []
if data.results_dir():
    from portal.deconvolution import diffstage_cohorts, diffstage_table
    DIFF_COHORTS = diffstage_cohorts()


@pytest.mark.skipif(not DIFF_COHORTS, reason="no differentiation stage scores prepared")
def test_differentiation_stage_scores():
    S, A = diffstage_table(DIFF_COHORTS)
    assert S.shape[1] == 13 and S.columns[0] == "HSC / MPP"
    assert not S.index.duplicated().any() and set(S.index) == set(A.index)
    # standardised across the five cohorts together: pooled mean about 0, SD about 1
    assert S.mean().abs().max() < 0.05 and S.std().between(0.95, 1.05).all()
    for c in DIFF_COHORTS:
        assert data.patient_datasets(c).set_index("key").loc["diffstage", "modality"] == "differentiation_score"


@pytest.mark.skipif(not DIFF_COHORTS, reason="no differentiation stage scores prepared")
def test_differentiation_stage_tab_renders():
    at = AppTest.from_file(PAGE, default_timeout=180).run()
    assert not at.exception
    assert not [e for e in at.error if "differentiation" in str(e.value)]
    assert any("Mean score per group" in str(h.value) for h in at.subheader)


def test_page_without_differentiation_stage_scores():
    """Older data releases have no diffstage layer: the tab is left out and nothing breaks."""
    script = f"""
import portal.deconvolution as d
d.diffstage_cohorts = lambda: []
exec(open({PAGE!r}).read())
"""
    at = AppTest.from_string(script, default_timeout=180).run()
    assert not at.exception
    assert "Differentiation stage" not in [t.label for t in at.tabs]
    assert not any("BoneMarrowMap" in str(m.value) for m in at.markdown)
