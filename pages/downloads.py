import pandas as pd
import streamlit as st

from portal import data

st.title("Downloads")
st.markdown("The list of datasets, the shared annotation vocabulary, and every processed data file "
            "behind the portal.")
st.info(f"The complete harmonized dataset is publicly available on [figshare]({data.FIGSHARE_URL}) as one download: "
        "unzip it into the portal's `data/results/` folder to run the portal yourself. The tabs below offer the same "
        "files one at a time.", icon=":material/cloud_download:")

tab_readme, tab_datasets, tab_files, tab_pfiles, tab_schema = st.tabs(
    ["About the data", "Datasets", "Cell-line files", "Patient files", "Annotation schema"])

with tab_readme:
    readme = data.results_readme()
    if readme:
        st.download_button("Download README (Markdown)", readme, file_name="README.md")
        with st.container(border=True):
            st.markdown(readme.split("\n", 1)[1] if readme.startswith("# ") else readme)   # page has its own title
    else:
        st.info("The description of the harmonized data is not available.")

with tab_datasets:
    cat = data.catalog().drop(columns=["status", "used_by"], errors="ignore")
    c2, c3 = st.columns(2)
    stype = c2.multiselect("Sample type", sorted({t for v in cat["sample_type"] for t in v.split(";") if t}))
    modality = c3.multiselect("Modality", sorted({m for v in cat["modalities"] for m in v.split(";") if m}))
    view = cat
    if stype:
        view = view[view["sample_type"].apply(lambda v: any(t in v.split(";") for t in stype))]
    if modality:
        view = view[view["modalities"].apply(lambda v: any(m in v.split(";") for m in modality))]
    st.dataframe(view, hide_index=True, width="stretch")
    st.caption(f"{len(view)} of {len(cat)} datasets.")
    st.download_button("Download catalog (TSV)", view.to_csv(sep="\t", index=False), file_name="datasets.tsv")

    st.subheader("Modality coverage")
    long = cat.assign(modality=cat["modalities"].str.split(";")).explode("modality")
    long = long[long["modality"] != ""]
    cov = pd.crosstab(long["dataset_id"], long["modality"]).gt(0)
    st.dataframe(cov.replace({True: "✓", False: ""}), width="stretch")

with tab_files:
    man = data.manifest()
    for kind, title in [("samples", "Cell-line annotation"), ("groupings", "Groupings"),
                        ("datasets", "Layer index"), ("values", "Values per cell line"),
                        ("features", "Feature annotation"), ("provenance", "Provenance")]:
        rows = man[man["kind"] == kind]
        if rows.empty:
            continue
        st.subheader(title)
        for _, r in rows.iterrows():
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"`{r['path']}`: {r['description']}")
            c2.download_button("Download", data.file_path(r["path"]).read_bytes(),
                               file_name=r["path"].split("/")[-1], key=r["path"])

with tab_pfiles:
    names = data.patient_cohort_names()
    if not names:
        st.info("No patient cohorts are available yet.")
    else:
        cohort = st.selectbox("Cohort", list(names), format_func=names.get, key="dl_cohort")
        man = data.patient_manifest(cohort)
        for kind, title in [("samples", "Sample annotation"), ("groupings", "Groupings"),
                            ("datasets", "Layer index"), ("values", "Values per sample"),
                            ("features", "Feature annotation"), ("provenance", "Provenance")]:
            rows = man[man["kind"] == kind]
            if rows.empty:
                continue
            st.subheader(title)
            for _, r in rows.iterrows():
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"`{r['path']}`: {r['description']}")
                c2.download_button("Download", (data.patients_dir() / cohort / r["path"]).read_bytes(),
                                   file_name=f"{cohort}_{r['path'].split('/')[-1]}", key=f"{cohort}/{r['path']}")

with tab_schema:
    for name, df in data.schema().items():
        st.subheader(name)
        st.dataframe(df, hide_index=True, width="stretch")
