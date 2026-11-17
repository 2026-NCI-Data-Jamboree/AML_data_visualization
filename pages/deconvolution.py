import streamlit as st

from portal.deconvolution import composition_tab, reference_tab

st.title("Deconvolution")
st.caption("Built-in analysis 3")
st.markdown(
    "Estimated cell type and cell state composition of the bulk RNA samples in the patient cohorts. Proportions "
    "are estimated offline against a single-cell reference of AML bone marrow (AML scAtlas) and shown here as "
    "they are in `results/`. They are estimates, not measurements: check the warnings and the reference's "
    "limitations before reading a sample."
)

tab_composition, tab_reference = st.tabs(["Composition", "Reference"])
with tab_composition:
    composition_tab()
with tab_reference:
    reference_tab()
