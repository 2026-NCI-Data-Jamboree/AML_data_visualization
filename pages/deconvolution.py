import streamlit as st

st.title("Deconvolution")
st.caption("Built-in analysis 3")
st.info("In development for the data jamboree, with the Childhood Cancer Data Lab (Alex's Lemonade "
        "Stand Foundation).")
st.markdown(
    "**Goal:** estimate the composition of AML differentiation stages from bulk transcriptomes "
    "(Karakaslar et al. 2024) of patients and cell lines, then relate it to drug response and clinical "
    "variables across cohorts.\n\n"
    "**Planned views:** composition bar charts per sample, grouped by cohort, subtype or cell-line "
    "grouping; association of each stage's fraction with drug AUC and survival; the single-cell "
    "reference (cell types, marker genes)."
)
