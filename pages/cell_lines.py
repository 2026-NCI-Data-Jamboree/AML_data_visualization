import pandas as pd
import streamlit as st

from portal import data, plots
from portal.dependency import gene_dependency
from portal.feature_query import cell_line_feature_query
from portal.stats import welch

st.title("Cell lines")
st.caption("Built-in analysis 2 · AML cell-line variability")
st.markdown(
    "How much do AML cell lines differ across omics layers, depending on how they are grouped? "
    "Pick a grouping and two groups. Differences are tested per feature with a Welch t test and "
    "Benjamini–Hochberg FDR (features need ≥ 3 lines with data per group)."
)

cl = data.cell_lines()
dsets = data.datasets().set_index("key")
GROUPINGS = data.grouping_labels()

tab_lines, tab_fq, tab_cmp, tab_dep = st.tabs(["Cell lines", "Feature query", "Compare groups", "Gene dependency"])

with tab_lines:
    G = data.groupings()
    avail = data.availability().rename(columns=dsets["label"].to_dict())
    default_tracks = [c for c in ["mTOR_high", "MYC_high", "KMT2A_rearranged", "mut_TP53", "mut_FLT3",
                                  "mut_NRAS", "disease_oncotree", "population", "sex"] if c in G.index]
    c1, c2 = st.columns([3, 1])
    track_cols = c1.multiselect("Annotation tracks", list(G.index), default=default_tracks,
                                format_func=GROUPINGS.get)
    sort_opts = ["data availability"] + track_cols
    sort_by = c2.selectbox("Order lines by", sort_opts, key="sort_by",
                           format_func=lambda c: c if c == "data availability" else GROUPINGS[c])

    # columns: chosen track first (NA last), then most data first, then name
    order = cl[["cell_line"]].assign(n_layers=avail.sum(axis=1).to_numpy())
    keys = ["n_layers", "cell_line"]
    asc = [False, True]
    if sort_by != "data availability":
        order["track"] = cl[sort_by].to_numpy()
        keys, asc = ["track"] + keys, [True] + asc
    lines = order.sort_values(keys, ascending=asc, na_position="last")["cell_line"].tolist()

    tracks = []
    for c in track_cols:
        r = G.loc[c]
        levels = sorted(cl[c].dropna().astype(str).unique())
        binary = (r["true_label"], r["false_label"]) if r["kind"] == "binary" else None
        tracks.append({"label": r["label"], "values": cl.set_index("cell_line")[c],
                       "colors": plots.track_colors(levels, binary)})
    avail.columns = [f"{c} ({int(avail[c].sum())})" for c in avail.columns]
    st.plotly_chart(plots.availability_heatmap(lines, tracks, avail), width="stretch")

    legend = []
    for t in tracks:
        chips = " ".join(f'<span style="display:inline-block;width:10px;height:10px;background:{col};'
                         f'border-radius:2px;margin:0 3px 0 8px"></span>{lv}' for lv, col in t["colors"].items())
        legend.append(f"<b>{t['label']}</b>{chips}")
    legend.append(f'<b>Layers</b><span style="display:inline-block;width:10px;height:10px;'
                  f'background:{plots.AVAIL_COLOR};border-radius:2px;margin:0 3px 0 8px"></span>available'
                  f'<span style="display:inline-block;width:10px;height:10px;background:{plots.NA_COLOR};'
                  f'border-radius:2px;margin:0 3px 0 8px"></span>no data / not annotated')
    st.markdown("<div style='font-size:0.85em;line-height:1.9'>" + "<br>".join(legend) + "</div>",
                unsafe_allow_html=True)

    st.subheader("Annotation")
    show = ["cell_line", "participant_id", "disease_oncotree", "OncotreeSubtype", "population", "sex",
            "age_years", "mutations", "KMT2A_rearranged", "KMT2A_partner", "MYC_high", "mTOR_high",
            "MYC_mTOR_category", "MYC_score", "mTOR_score", "Mito_high"]
    st.dataframe(cl[[c for c in show if c in cl]], hide_index=True, width="stretch")

with tab_fq:
    cell_line_feature_query()

with tab_dep:  # before the compare block, which may st.stop()
    gene_dependency()

with tab_cmp:
    c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
    grouping = c1.selectbox("Grouping", list(GROUPINGS), format_func=GROUPINGS.get, key="grouping")
    levels = cl[grouping].dropna().astype(str).value_counts()
    opts = list(levels.index)
    a0, b0 = data.default_pair(grouping, opts)
    fmt = lambda g: f"{g} (n={levels[g]})"  # noqa: E731
    ga = c2.selectbox("Group A (blue)", opts, index=opts.index(a0) if a0 in opts else 0, format_func=fmt)
    rest = [o for o in opts if o != ga] + ["all other lines"]
    gb = c3.selectbox("Group B (red)", rest, index=rest.index(b0) if b0 in rest else 0,
                      format_func=lambda g: g if g == "all other lines" else fmt(g))
    key = c4.selectbox("Omics layer", list(dsets.index), format_func=lambda k: dsets.loc[k, "label"],
                       key="layer")
    meta = dsets.loc[key]

    grp_col = cl[grouping].astype("string")
    lines_a = cl.loc[grp_col == ga, "cell_line"].tolist()
    lines_b = (cl.loc[grp_col.notna() & (grp_col != ga), "cell_line"] if gb == "all other lines"
               else cl.loc[grp_col == gb, "cell_line"]).tolist()

    V = data.values(key)
    res = welch(V, lines_a, lines_b)
    if res.empty:
        st.warning("No feature has ≥ 3 lines with data in both groups for this layer.")
        st.stop()
    labels = data.feature_labels(key)
    res.insert(1, "label", res["feature"].map(labels))

    fdr = st.select_slider("FDR threshold", [0.01, 0.05, 0.1, 0.25], value=0.05)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lines A / B with data", f"{int(res['n_A'].max())} / {int(res['n_B'].max())}")
    m2.metric("Features tested", f"{len(res):,}")
    m3.metric(f"FDR < {fdr}", int((res["fdr_welch"] < fdr).sum()))
    m4.metric("FDR < 0.25", int((res["fdr_welch"] < 0.25).sum()))
    st.caption(f"{meta['unit']} · {meta['source']}"
               + (" · lower = more dependent / sensitive, so the negative side favours group A"
                  if meta["lower_is_stronger"] else ""))

    st.plotly_chart(plots.volcano(res, labels, ga, gb, bool(meta["lower_is_stronger"]), fdr=fdr),
                    width="stretch")

    q = st.text_input("Filter features", placeholder="e.g. BCL2, venetoclax, H3K27")
    tab = res if not q else res[res["label"].str.contains(q, case=False, na=False, regex=False)
                                | res["feature"].str.contains(q, case=False, na=False, regex=False)]
    if tab.empty:
        st.info("No feature matches the filter.")
        st.stop()
    event = st.dataframe(
        tab, hide_index=True, width="stretch", height=360, on_select="rerun", selection_mode="single-row",
        column_config={c: st.column_config.NumberColumn(format="%.3g")
                       for c in ["mean_A", "mean_B", "diff_A_minus_B", "cohens_d", "t", "p_welch", "fdr_welch"]})
    st.download_button("Download table (TSV)", tab.to_csv(sep="\t", index=False),
                       file_name=f"{key}_{grouping}_{ga}_vs_{gb}.tsv".replace(" ", "_").replace("/", "-"))

    rows = event.selection.rows if event and event.selection else []
    feat = tab.iloc[rows[0]]["feature"] if rows else tab.iloc[0]["feature"]
    st.subheader(f"Per-line values: {labels.get(feat, feat)}")
    long = V.loc[feat].rename("value").rename_axis("cell_line").reset_index()
    grp = pd.Series("not compared", index=cl["cell_line"])
    grp[lines_b] = gb
    grp[lines_a] = ga
    long["group"] = long["cell_line"].map(grp)
    st.plotly_chart(plots.strip(long, "value", "group", [ga, gb, "not compared"], y_title=meta["unit"],
                                colors={ga: plots.COL_A, gb: plots.COL_B, "not compared": plots.COL_NS}),
                    width="stretch")
    st.caption("Select a table row to change the feature shown.")
