"""Loading the processed, harmonized data from AML_data_processing/results.

The layout is defined in AML_data_processing/docs/results_contract.md. The portal reads nothing else.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

SUPPORTED_RESULTS_VERSION = "3"
# the published harmonized dataset (the results/ folder), for users who want all files in one download
FIGSHARE_URL = ("https://figshare.com/articles/dataset/Building_a_Harmonized_Multi-Omics_Resource_and_Web_Application_"
                "for_Biomarker_Discovery_and_Drug_Target_Nomination_in_Leukemia/35050459")
_REPO = Path(__file__).resolve().parent.parent
_CANDIDATES = [
    _REPO / "data" / "results",
    _REPO.parent / "AML_data_processing" / "results",
]


def results_dir() -> Path | None:
    """$AML_PORTAL_DATA, else ./data/results, else the sibling processing repo's results/."""
    env = os.environ.get("AML_PORTAL_DATA")
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    return next((p for p in _CANDIDATES if p.is_dir()), None)


def require_results() -> Path:
    p = results_dir()
    if p is None:
        st.error(
            "No processed data found. In AML_data_processing run `Rscript prepare_cell_line_data.R`, "
            "then point `AML_PORTAL_DATA` to its `results/` folder (or copy it to `data/results` here)."
        )
        st.stop()
    return p


def _read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kw)


@st.cache_data(show_spinner=False)
def catalog() -> pd.DataFrame:
    return _read(require_results() / "catalog" / "datasets.tsv", dtype=str).fillna("")


@st.cache_data(show_spinner=False)
def schema() -> dict[str, pd.DataFrame]:
    d = require_results() / "schema"
    return {f.stem: _read(f, dtype=str).fillna("") for f in sorted(d.glob("*.tsv"))}


# ----------------------------------------------------------------- cell lines ---
CELL_LINES = "cell_lines"


def _cl() -> Path:
    p = require_results() / CELL_LINES
    if not p.is_dir():
        st.error("`results/cell_lines/` is missing. Run `Rscript prepare_cell_line_data.R`.")
        st.stop()
    return p


def file_path(rel: str) -> Path:
    return _cl() / rel


@st.cache_data(show_spinner=False)
def provenance() -> dict[str, str]:
    f = _cl() / "provenance.tsv"
    return dict(_read(f, dtype=str).fillna("").values) if f.exists() else {}


@st.cache_data(show_spinner=False)
def manifest() -> pd.DataFrame:
    return _read(_cl() / "manifest.tsv", dtype=str).fillna("")


@st.cache_data(show_spinner=False)
def groupings() -> pd.DataFrame:
    """results/cell_lines/groupings.tsv, indexed by column: label, true_label, false_label, kind."""
    return _read(_cl() / "groupings.tsv", dtype=str).fillna("").set_index("column")


@st.cache_data(show_spinner=False)
def cell_lines() -> pd.DataFrame:
    """samples.tsv with every grouping column as string group labels (NA = not annotated).

    Binary columns (TRUE/FALSE) are renamed to the grouping's true_label / false_label,
    e.g. mTOR_high -> 'mTOR-High' / 'mTOR-Low', mut_TP53 -> 'TP53-mutant' / 'TP53-wild-type'.
    """
    g = groupings()
    cl = _read(_cl() / "samples.tsv", dtype={c: str for c in g.index})
    for col, r in g.iterrows():
        if r["kind"] == "binary":
            cl[col] = cl[col].str.upper().map({"TRUE": r["true_label"], "FALSE": r["false_label"]})
    return cl


@st.cache_data(show_spinner=False)
def datasets() -> pd.DataFrame:
    d = _read(_cl() / "datasets.tsv")
    d["lower_is_stronger"] = d["lower_is_stronger"].astype(str).str.upper() == "TRUE"
    return d


@st.cache_data(show_spinner=False)
def features(key: str) -> pd.DataFrame:
    return _read(_cl() / "features" / f"{key}.tsv", dtype=str).fillna("").set_index("feature")


@st.cache_data(show_spinner="Loading values…")
def values(key: str) -> pd.DataFrame:
    """Rows = features (index), columns = cell lines."""
    return _read(_cl() / "values" / f"{key}.tsv.gz", index_col="feature")


@st.cache_data(show_spinner="Checking data availability…")
def availability() -> pd.DataFrame:
    """cell lines x layer keys: True when the line has any value in that layer."""
    lines = cell_lines()["cell_line"]
    return pd.DataFrame({k: lines.isin(values(k).columns[values(k).notna().any()]).to_numpy()
                         for k in datasets()["key"]}, index=lines)


def feature_labels(key: str) -> pd.Series:
    f = features(key)
    return f["label"].where(f["label"] != "", f.index.to_series())


@st.cache_data(show_spinner=False)
def gene_index() -> pd.DataFrame:
    """One row per (HGNC symbol, layer, feature); drugs are indexed by their annotated targets."""
    rows = []
    for key in datasets()["key"]:
        f = features(key).reset_index()
        if "targets" in f:
            t = f.assign(gene=f["targets"].str.split(";")).explode("gene")
            t["gene"] = t["gene"].str.strip()
            rows.append(t.assign(dataset=key, via="drug target")[["gene", "dataset", "feature", "via"]])
        g = f[f["gene_symbol"] != ""]
        rows.append(pd.DataFrame({"gene": g["gene_symbol"], "dataset": key, "feature": g["feature"],
                                  "via": "feature"}))
    idx = pd.concat(rows, ignore_index=True)
    return idx[idx["gene"].notna() & (idx["gene"] != "")]


def grouping_labels() -> dict[str, str]:
    """column -> display label, in the order given by groupings.tsv."""
    return groupings()["label"].to_dict()


def default_pair(column: str, levels: list[str]) -> tuple[str, str]:
    """Binary: (true_label, false_label). Categorical: the two largest groups."""
    r = groupings().loc[column]
    if r["kind"] == "binary":
        return r["true_label"], r["false_label"]
    return levels[0], levels[1] if len(levels) > 1 else levels[0]


# ------------------------------------------------------------------- patients ---
PATIENTS = "patients"


def patients_dir() -> Path | None:
    p = results_dir()
    return p / PATIENTS if p is not None and (p / PATIENTS).is_dir() else None


def patient_cohorts() -> list[str]:
    """Cohort IDs with a prepared results/patients/<DATASET_ID>/samples.tsv."""
    p = patients_dir()
    return sorted(d.name for d in p.iterdir() if (d / "samples.tsv").is_file()) if p else []


@st.cache_data(show_spinner=False)
def patient_samples(cohort: str) -> pd.DataFrame:
    return _read(patients_dir() / cohort / "samples.tsv", dtype={"sample_id": str, "participant_id": str})


@st.cache_data(show_spinner=False)
def patient_groupings(cohort: str) -> pd.DataFrame:
    """Optional groupings.tsv (same columns as the cell-line one); empty when absent."""
    f = patients_dir() / cohort / "groupings.tsv"
    if not f.exists():
        return pd.DataFrame(columns=["label", "true_label", "false_label", "kind"]).rename_axis("column")
    return _read(f, dtype=str).fillna("").set_index("column")


@st.cache_data(show_spinner=False)
def patient_datasets(cohort: str) -> pd.DataFrame:
    f = patients_dir() / cohort / "datasets.tsv"
    return _read(f) if f.exists() else pd.DataFrame(columns=["key", "label", "modality", "unit", "source"])


@st.cache_data(show_spinner=False)
def patient_features(cohort: str, key: str) -> pd.DataFrame:
    return _read(patients_dir() / cohort / "features" / f"{key}.tsv", dtype=str).fillna("").set_index("feature")


@st.cache_data(show_spinner="Loading values…")
def patient_values(cohort: str, key: str) -> pd.DataFrame:
    """Rows = features (index), columns = sample_id."""
    return _read(patients_dir() / cohort / "values" / f"{key}.tsv.gz", index_col="feature")


def patient_cohort_names() -> dict[str, str]:
    names = catalog().set_index("dataset_id")["name"].to_dict()
    return {c: names.get(c, c) for c in patient_cohorts()}


@st.cache_data(show_spinner=False)
def patient_samples_labeled(cohort: str) -> pd.DataFrame:
    """samples.tsv with binary grouping columns renamed to their true / false labels (as cell_lines())."""
    g = patient_groupings(cohort)
    s = _read(patients_dir() / cohort / "samples.tsv", dtype={**{c: str for c in g.index},
                                                              "sample_id": str, "participant_id": str})
    for col, r in g.iterrows():
        if r["kind"] == "binary" and col in s:
            s[col] = s[col].str.upper().map({"TRUE": r["true_label"], "FALSE": r["false_label"]})
    return s


@st.cache_data(show_spinner=False)
def patient_availability(cohort: str) -> pd.DataFrame:
    """samples x layer keys: True when the sample has any value in that layer."""
    s = patient_samples(cohort)["sample_id"]
    ds = patient_datasets(cohort)
    return pd.DataFrame({k: s.isin(patient_values(cohort, k).columns[patient_values(cohort, k).notna().any()]).to_numpy()
                         for k in ds["key"]}, index=s)


@st.cache_data(show_spinner=False)
def patient_drug_index() -> pd.DataFrame:
    """One row per (cohort, drug layer, drug): label, compound_key, targets, lower_is_stronger."""
    rows = []
    for c in patient_cohorts():
        ds = patient_datasets(c)
        for _, d in ds[ds["modality"] == "drug_screen"].iterrows():
            f = patient_features(c, d["key"]).reset_index()
            dup = d.get("duplicate_of", "")
            rows.append(f.assign(cohort=c, layer=d["key"], layer_label=d["label"],
                                 lower_is_stronger=str(d["lower_is_stronger"]).upper() == "TRUE",
                                 duplicate_of="" if pd.isna(dup) else str(dup))
                        [["cohort", "layer", "layer_label", "feature", "label", "compound_key", "targets",
                          "lower_is_stronger", "duplicate_of"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["cohort", "layer", "layer_label", "feature", "label", "compound_key", "targets", "lower_is_stronger",
                 "duplicate_of"])


@st.cache_data(show_spinner=False)
def patient_gene_index() -> pd.DataFrame:
    """One row per (HGNC symbol, cohort, layer, feature); drugs are indexed by their targets."""
    rows = []
    for c in patient_cohorts():
        for key in patient_datasets(c)["key"]:
            f = patient_features(c, key).reset_index()
            if "targets" in f:
                t = f.assign(gene=f["targets"].str.split(";")).explode("gene")
                t["gene"] = t["gene"].str.strip()
                rows.append(t.assign(cohort=c, dataset=key, via="drug target")[["gene", "cohort", "dataset", "feature", "via"]])
            g = f[f.get("gene_symbol", pd.Series("", index=f.index)) != ""]
            if len(g):
                rows.append(pd.DataFrame({"gene": g["gene_symbol"], "cohort": c, "dataset": key,
                                          "feature": g["feature"], "via": "feature"}))
    idx = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["gene", "cohort", "dataset", "feature", "via"])
    return idx[idx["gene"].notna() & (idx["gene"] != "")]


@st.cache_data(show_spinner=False)
def patient_provenance(cohort: str) -> dict[str, str]:
    f = patients_dir() / cohort / "provenance.tsv"
    return dict(_read(f, dtype=str).fillna("").values) if f.exists() else {}


@st.cache_data(show_spinner=False)
def patient_manifest(cohort: str) -> pd.DataFrame:
    f = patients_dir() / cohort / "manifest.tsv"
    return _read(f, dtype=str).fillna("") if f.exists() else pd.DataFrame(columns=["path", "kind", "description"])


@st.cache_data(show_spinner=False)
def dataset_sources() -> pd.DataFrame:
    """results/dataset_sources.tsv: every integrated dataset, with source and NCI resource status."""
    p = results_dir()
    f = p / "dataset_sources.tsv" if p else None
    return _read(f, dtype=str).fillna("") if f is not None and f.exists() else pd.DataFrame()


# -------------------------------------------------------------- deconvolution ---
# Layers are ordinary patient layers (modality "deconvolution"); the extra files are described in
# AML_data_processing/docs/results_contract.md (Deconvolution).
DECONVOLUTION_RUN_COLUMNS = ["key", "reference", "method", "level", "quantity", "update_mode", "run_date"]


@st.cache_data(show_spinner=False)
def deconvolution_runs(cohort: str) -> pd.DataFrame:
    """deconvolution_runs.tsv of a cohort: one row per layer; empty when the cohort has none."""
    f = patients_dir() / cohort / "deconvolution_runs.tsv" if patients_dir() else None
    return _read(f, dtype=str).fillna("") if f is not None and f.exists() else pd.DataFrame(
        columns=DECONVOLUTION_RUN_COLUMNS)


def deconvolution_cohorts() -> list[str]:
    """Cohorts that have at least one deconvolution layer."""
    return [c for c in patient_cohorts() if len(deconvolution_runs(c))]


def has_deconvolution() -> bool:
    """True when any cohort has deconvolution results; the Deconvolution page is shown only then."""
    try:
        return bool(deconvolution_cohorts())
    except Exception:                       # no results folder, unreadable files, ...
        return False


@st.cache_data(show_spinner=False)
def deconvolution_warnings(cohort: str) -> pd.DataFrame:
    """deconvolution_warnings.tsv: key (empty = every layer), scope, sample_id, label, severity, code, message, source."""
    cols = ["key", "scope", "sample_id", "label", "severity", "code", "message", "source"]
    f = patients_dir() / cohort / "deconvolution_warnings.tsv" if patients_dir() else None
    return _read(f, dtype=str).fillna("") if f is not None and f.exists() else pd.DataFrame(columns=cols)


def single_cell_references() -> list[str]:
    """Reference IDs with a results/single_cell/<REFERENCE_ID>/ folder."""
    p = results_dir()
    d = p / "single_cell" if p else None
    return sorted(x.name for x in d.iterdir() if (x / "reference.tsv").is_file()) if d is not None and d.is_dir() else []


@st.cache_data(show_spinner=False)
def reference_table(reference: str) -> pd.DataFrame:
    """level, label, cell_type, n_cells, n_reference_samples, is_malignant."""
    t = _read(results_dir() / "single_cell" / reference / "reference.tsv", dtype={"label": str, "cell_type": str})
    t["is_malignant"] = t["is_malignant"].astype(str).str.upper() == "TRUE"
    return t


@st.cache_data(show_spinner=False)
def reference_markers(reference: str) -> pd.DataFrame:
    """level, label, rank, gene_id, gene_symbol, score."""
    return _read(results_dir() / "single_cell" / reference / "markers.tsv", dtype={"label": str}).fillna("")


@st.cache_data(show_spinner=False)
def reference_provenance(reference: str) -> dict[str, str]:
    return dict(_read(results_dir() / "single_cell" / reference / "provenance.tsv", dtype=str).fillna("").values)


LAYER_LABELS = {"expression": "RNA expression", "protein": "Protein", "phospho": "Phosphosite", "surface": "Surface protein",
                "diffstage": "Differentiation stage score"}


def layer_label(key: str) -> str:
    return LAYER_LABELS.get(key, key)


def feature_names(cohort: str, layer: str) -> pd.Series:
    """Display name per feature: the HGNC symbol for gene-level layers, the site (e.g. "STAT5A_Y694") for
    site-level layers such as phospho, the label otherwise (drugs)."""
    f = patient_features(cohort, layer)
    if "site" in f or "gene_symbol" not in f:
        return f["label"]
    return f["gene_symbol"].where(f["gene_symbol"] != "", f["label"])


def results_readme() -> str:
    """results/README.md: overview of the harmonized data ('' if missing)."""
    p = results_dir()
    f = p / "README.md" if p else None
    return f.read_text() if f is not None and f.exists() else ""
