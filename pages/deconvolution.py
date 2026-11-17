import streamlit as st

from portal.deconvolution import composition_tab, reference_tab

st.title("Deconvolution")
st.caption("Built-in analysis 3")
st.markdown(
    "Estimated cell type and cell state composition of patient bulk RNA samples, from a single-cell reference of "
    "AML bone marrow (AML scAtlas). The proportions are estimates, not measurements; see *About these estimates* below."
)

tab_composition, tab_reference = st.tabs(["Composition", "Reference"])
with tab_composition:
    composition_tab()
with tab_reference:
    reference_tab()

st.divider()
st.subheader("About these estimates")
st.markdown(
    """
Bulk RNA from a patient sample is a mix of many cell types: leukemic blasts, normal blood and immune cells, and their
precursors. Deconvolution estimates how much of each cell type and cell state is in that mix, by comparing the sample
with the single-cell reference.

- **What a value means.** Each value is a proportion between 0 and 1, and the proportions of a sample add up to 1.
  They are shares of the sample's mRNA, not of its cells: cell types differ in how much RNA they contain.
- **Methods.** InstaPrism gives an initial fit, using the reference as built, and an updated fit, which adapts the
  reference to the cohort, so a sample's estimate then depends on the other samples in the cohort. NNLS is a simpler
  method, at cell type level only. The initial fit is the default.
- **Warnings.** Samples can be flagged, for example when hemoglobin genes dominate the RNA or when the initial
  malignant share is low (which makes the updated fit unreliable). Flagged samples are hidden by default.
- **Limits.** The reference's malignant profile under-represents monocytic and more committed AML, so the malignant
  share is likely underestimated in those cases.
"""
)
