"""AML multi-omics portal. Run: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="AML Multi-Omics Portal", page_icon="🧬", layout="wide")

st.navigation([
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/cell_lines.py", title="Cell lines", icon="🧫"),
    st.Page("pages/patients.py", title="Patients", icon="🧑‍⚕️"),
    st.Page("pages/deconvolution.py", title="Deconvolution", icon="🧩"),
    st.Page("pages/downloads.py", title="Downloads", icon="📥"),
]).run()
