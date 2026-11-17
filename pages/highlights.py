import streamlit as st

from portal.age_association import age_association
from portal.dependency import gene_dependency
from portal.drug_biomarkers import drug_biomarkers

st.title("Highlights")
st.caption("Genome-wide analyses across the cell lines and patient cohorts")

tab_dep, tab_drug, tab_age = st.tabs(["MYC-high vs mTOR-high dependency", "Drug response markers", "Age association"])

with tab_dep:
    st.caption("Built-in analysis 2 · AML cell lines: which genes MYC-high and mTOR-high lines depend on, "
               "in CRISPR and RNAi screens")
    gene_dependency(kp="hl_dep", fixed="MYC-high vs mTOR-high")

with tab_drug:
    st.caption("Built-in analysis 1 · Drug response")
    drug_biomarkers()

with tab_age:
    age_association()
