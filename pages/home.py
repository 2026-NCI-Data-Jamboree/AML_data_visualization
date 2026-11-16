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

pt = data.patient_cohort_names()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Datasets in catalog", len(cat))
c2.metric("Processed", int((cat["status"] == "processed").sum()))
c3.metric("AML cell lines", len(cl))
c4.metric("Patients (drug-response cohorts)", f"{sum(data.patient_samples(c)['participant_id'].nunique() for c in pt):,}",
          help=f"{len(pt)} cohorts: " + ", ".join(pt.values()))

st.subheader("What you can do")
st.markdown(
    "- **Cell lines**: 70 AML cell lines across 11 omics layers. See which data each line has, compare "
    "any two groups (mTOR/MYC-high, KMT2A-rearranged, TP53-mutant, ...) with volcano plots, and look "
    "up one gene across all layers *(built-in analysis 2, cell-line variability)*.\n"
    "- **Patients**: four adult and pediatric cohorts with ex vivo drug response (Beat AML, Lee 2018, "
    "Wang 2022, Malani 2022). Find expression markers of drug response that replicate across cohorts "
    "*(built-in analysis 1)*, look up one gene in every cohort, and compare survival by mutation, risk "
    "group, MYC/mTOR signature or expression.\n"
    "- **Deconvolution**: differentiation-stage composition of patient and cell-line transcriptomes "
    "from a single-cell reference *(built-in analysis 3, in development)*.\n"
    "- **Documentation**: where the data comes from (including which datasets are from NCI resources), "
    "how it was made comparable, and how each analysis works.\n"
    "- **Downloads**: the dataset inventory, the annotation schema and every processed file."
)

prov = data.provenance()
if prov:
    st.caption(f"Data last updated {prov.get('prepared_at', '?')[:10]}.")
    if prov.get("results_version") != data.SUPPORTED_RESULTS_VERSION:
        st.warning(f"This portal expects results v{data.SUPPORTED_RESULTS_VERSION}.")
