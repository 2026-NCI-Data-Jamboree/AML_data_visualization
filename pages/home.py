import streamlit as st

from portal import data

st.title("Building a Harmonized Multi-Omics Data Resource and Web Application for Biomarker Discovery "
         "and Drug Target Nomination in Leukemia")
st.caption("A 2026 NCI Data Jamboree project")
st.markdown(
    "A harmonized multi-omics resource for **biomarker discovery and drug-target nomination in "
    "leukemia**, with acute myeloid leukemia (AML) as the first use case. It brings together "
    "public patient cohorts (CPTAC, Beat AML, ...) and cell-line resources "
    "(DepMap, CCLE) under one annotation schema (OncoTree / NCI Thesaurus, HGNC)."
)

cat = data.catalog()
dsets = data.datasets()
cl = data.cell_lines()

pt = data.patient_cohort_names()
c1, c2, c3 = st.columns(3)
c1.metric("Datasets", len(cat))
c2.metric("Patients", f"{len(set().union(*[set(data.patient_samples(c)['participant_id'].astype(str)) for c in pt])):,}",
          help=f"{len(pt)} cohorts: " + ", ".join(pt.values()))
c3.metric("AML cell lines", len(cl))

st.subheader("What you can do")
st.markdown(
    "- **Highlights**: genes that MYC-high vs mTOR-high AML cell lines depend on (CRISPR and RNAi), surface "
    "proteins measured on AML cell lines (markers and antigen combinations for targeting), RNA, "
    "protein or phosphosite markers of drug response that replicate across patient cohorts *(built-in "
    "analysis 1)*, features that change with age, and genes linked to survival across cohorts.\n"
    "- **Cell lines**: 70 AML cell lines across 11 omics layers. See which data each line has, compare "
    "any two groups (mTOR/MYC-high, KMT2A-rearranged, TP53-mutant, ...) with volcano plots, and look "
    "up one gene across all layers *(built-in analysis 2, cell-line variability)*.\n"
    "- **Patients**: twelve adult and pediatric cohorts (Beat AML, Beat AML proteomics (PTRC), CPTAC adult AML, "
    "AML proteogenomics from Zenodo, Lee 2018, Wang 2022, Malani 2022, St. Jude pediatric AML, TCGA, TARGET, "
    "Leucegene, LUMC) with RNA, protein, phosphosites and ex vivo drug response. Check how a "
    "marker relates to a drug or to age in every cohort, look up a gene, and compare survival.\n"
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
