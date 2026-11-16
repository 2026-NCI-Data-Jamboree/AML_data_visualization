import streamlit as st

from portal import data

st.title("Downloads")
st.markdown("The dataset inventory, the harmonized annotation schema, and every processed file behind "
            "the portal. File layout and columns: `docs/results_contract.md` in AML_data_processing.")

tab_datasets, tab_files, tab_schema = st.tabs(["Datasets", "Cell-line files", "Annotation schema"])

with tab_datasets:
    cat = data.catalog()
    c1, c2, c3 = st.columns(3)
    status = c1.multiselect("Status", sorted(cat["status"].unique()))
    stype = c2.multiselect("Sample type", sorted({t for v in cat["sample_type"] for t in v.split(";") if t}))
    modality = c3.multiselect("Modality", sorted({m for v in cat["modalities"] for m in v.split(";") if m}))
    view = cat
    if status:
        view = view[view["status"].isin(status)]
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
    st.dataframe(long.pivot_table(index="dataset_id", columns="modality", values="status",
                                  aggfunc="first").fillna(""), width="stretch")

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

with tab_schema:
    for name, df in data.schema().items():
        st.subheader(name)
        st.dataframe(df, hide_index=True, width="stretch")
