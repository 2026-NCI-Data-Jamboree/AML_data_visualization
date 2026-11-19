import streamlit as st

from portal import data
from portal.deconvolution import composition_tab, diffstage_cohorts, diffstage_tab, reference_tab


def _safe(section: str, fn) -> None:
    """Run one part of the page; a missing or malformed file shows a message instead of breaking the page."""
    try:
        fn()
    except Exception as e:                          # noqa: BLE001 - any data problem should stay on this tab
        st.error(f"The {section} could not be shown: the deconvolution files look incomplete or out of date. "
                 "Re-run the deconvolution step of the processing pipeline.", icon=":material/error:")
        with st.expander("Technical details"):
            st.code(f"{type(e).__name__}: {e}")

st.title("Deconvolution")
st.caption("Built-in analysis 3")
try:
    ready, refs = data.has_deconvolution(), bool(data.single_cell_references())
except Exception:                                   # noqa: BLE001 - e.g. no results folder at all
    ready, refs = False, False
try:                                                # differentiation stage scores: only in newer data releases
    stages = bool(diffstage_cohorts())
except Exception:                                   # noqa: BLE001
    stages = False

st.markdown(
    "Estimated cell type and cell state composition of patient bulk RNA samples, from a single-cell reference of "
    "AML bone marrow (AML scAtlas)"
    + (", and differentiation stage scores of five cohorts from a single-cell atlas of normal bone marrow "
       "(BoneMarrowMap). Both are" if stages else ". The proportions are")
    + " estimates, not measurements; see *About these estimates* below."
)
if not ready:
    st.info("**Deconvolution results are not available yet.** They are produced by the deconvolution step of the "
            "processing pipeline (AML_data_processing) and appear here once that step has been run"
            + (". The single-cell reference can already be browsed in the *Reference* tab." if refs else
               ", together with the single-cell reference."), icon=":material/hourglass_empty:")

tabs = st.tabs(["Composition"] + (["Differentiation stage"] if stages else []) + ["Reference"])
with tabs[0]:
    _safe("composition", composition_tab)
if stages:                                          # the tab is left out when the data has no scores (older releases)
    with tabs[1]:
        _safe("differentiation stage scores", diffstage_tab)
with tabs[-1]:
    _safe("reference", reference_tab)

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
if stages:
    st.markdown("- **Differentiation stage scores** (BoneMarrowMap) are a separate estimate: how much a sample resembles "
                "each stage of normal blood cell development, standardised across five cohorts (0 = average sample). "
                "They are not proportions.")
