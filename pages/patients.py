import streamlit as st

from portal import data
from portal.feature_query import patient_feature_query

st.title("Patients")
st.markdown("Harmonized AML patient cohorts (adult and pediatric). Cohorts are prepared by "
            "`prepare_patient_data.R` in AML_data_processing; this page fills in as they are added.")

tab_overview, tab_fq, tab_drug = st.tabs(["Cohorts", "Feature query", "Drug response biomarkers"])

with tab_overview:
    cat = data.catalog()
    pts = cat[cat["sample_type"].str.split(";").apply(lambda t: "patient" in t)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Patient cohorts in catalog", len(pts))
    c2.metric("Pediatric cohorts", int((pts["population"] == "pediatric").sum()))
    c3.metric("Processed", int((pts["status"] == "processed").sum()))
    st.dataframe(pts, hide_index=True, width="stretch")

with tab_drug:
    st.subheader("Built-in analysis 1 · Drug response biomarkers")
    st.info("In development for the data jamboree.")
    st.markdown(
        "**Goal:** integrate patient cohorts that pair a molecular measurement (RNA or protein) with "
        "ex vivo drug sensitivity, and find features that predict drug response in more than one cohort. "
        "Proof of concept: **MAP1A** protein predicts response to several drugs, including venetoclax, "
        "in independent adult AML cohorts (Chu et al. 2026).\n\n"
        "**Planned views:** pick a drug → ranked features with per-cohort and meta-analysed "
        "associations; pick a feature → its associations across all drugs; scatter of feature vs AUC per cohort."
    )
    st.caption("Cohorts used by this analysis")
    st.dataframe(cat[cat["used_by"].str.contains("drug_response_biomarkers")], hide_index=True, width="stretch")

with tab_fq:
    patient_feature_query()
