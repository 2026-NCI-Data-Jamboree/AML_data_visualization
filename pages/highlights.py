import streamlit as st

from portal.age_association import age_association
from portal.dependency import gene_dependency
from portal.drug_biomarkers import drug_biomarkers
from portal.surfaceome import surfaceome_section
from portal.survival_meta import survival_association

st.title("Highlights")

tab_dep, tab_surf, tab_drug, tab_age, tab_surv = st.tabs(
    ["Gene dependency", "Cell surface proteins", "Drug response markers", "Age association",
     "Survival"])

with tab_dep:
    st.caption("Built-in analysis 2 · AML cell lines: which genes one group of lines depends on more than another "
               "(MYC-high vs mTOR-high, TP53-mutant vs wild-type), in CRISPR and RNAi screens")
    gene_dependency(kp="hl_dep", fixed=["MYC-high vs mTOR-high", "TP53-mutant vs TP53-wild-type"],
                    shared_toggle=False)

with tab_surf:
    st.caption("AML cell lines: measured cell-surface proteome, for surface markers and antigen targeting")
    surfaceome_section()

with tab_drug:
    st.caption("Built-in analysis 1 · Drug response")
    drug_biomarkers()

with tab_age:
    age_association()

with tab_surv:
    survival_association()
