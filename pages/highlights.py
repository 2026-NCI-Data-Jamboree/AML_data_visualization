import streamlit as st

from portal.age_association import age_association
from portal.drug_biomarkers import drug_biomarkers

st.title("Highlights")
st.caption("Genome-wide analyses across the patient cohorts")

tab_drug, tab_age = st.tabs(["Drug response markers", "Age association"])

with tab_drug:
    st.caption("Built-in analysis 1 · Drug response")
    drug_biomarkers()

with tab_age:
    age_association()
