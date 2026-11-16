import streamlit as st

from portal import data

st.title("AML Multi-Omics Portal")
st.markdown(
    "A harmonized multi-omics resource for **biomarker discovery and drug-target nomination in "
    "leukemia**, with acute myeloid leukemia (AML) as the first use case. It brings together "
    "public patient cohorts (CPTAC, TCGA, TARGET, Beat AML, ...) and cell-line resources "
    "(DepMap, CCLE) under one annotation schema (OncoTree / NCI Thesaurus, HGNC)."
)

cat = data.catalog()
dsets = data.datasets()
cl = data.cell_lines()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Datasets in catalog", len(cat))
c2.metric("Processed", int((cat["status"] == "processed").sum()))
c3.metric("AML cell lines", len(cl))
c4.metric("Cell-line omics layers", len(dsets))

st.subheader("What you can do")
st.markdown(
    "- **Cell lines**: 70 AML cell lines across 11 omics layers. See which data each line has, compare "
    "any two groups (mTOR/MYC-high, KMT2A-rearranged, TP53-mutant, ...) with volcano plots, and look "
    "up one gene across all layers *(built-in analysis 2, cell-line variability)*.\n"
    "- **Patients**: harmonized adult and pediatric cohorts, with drug response biomarkers "
    "*(built-in analysis 1)* and a feature query *(in development)*.\n"
    "- **Deconvolution**: differentiation-stage composition of patient and cell-line transcriptomes "
    "from a single-cell reference *(built-in analysis 3, in development)*.\n"
    "- **Downloads**: the dataset inventory, the annotation schema and every processed file."
)

prov = data.provenance()
if prov:
    st.caption(
        f"Cell-line data prepared {prov.get('prepared_at', '?')} by AML_data_processing "
        f"commit `{prov.get('processing_commit') or '?'}`"
        + (" (uncommitted changes)" if prov.get("working_tree_dirty") == "TRUE" else "")
        + f" · results v{prov.get('results_version', '?')}"
    )
    if prov.get("results_version") != data.SUPPORTED_RESULTS_VERSION:
        st.warning(f"This portal expects results v{data.SUPPORTED_RESULTS_VERSION}.")
