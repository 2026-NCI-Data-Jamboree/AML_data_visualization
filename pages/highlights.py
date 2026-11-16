import streamlit as st

from portal.drug_biomarkers import drug_biomarkers

st.title("Highlights")
st.caption("Genome-wide analyses across the patient cohorts")

(tab_drug,) = st.tabs(["Drug response markers"])

with tab_drug:
    st.caption("Built-in analysis 1 · Drug response")
    drug_biomarkers()
