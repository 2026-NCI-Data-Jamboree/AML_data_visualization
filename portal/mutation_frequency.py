"""Mutation frequency comparison: is each mutation enriched in one group of cell lines over another?"""
from __future__ import annotations

import streamlit as st

from portal import data
from portal import genomics_plots as plots
from portal.cooccurrence import _keep_valid, alterations
from portal.genomics_stats import mutation_enrichment

FMT = {c: st.column_config.NumberColumn(format="%.3g") for c in ["log2_odds_ratio", "p", "q"]}
PCT = {c: st.column_config.NumberColumn(format="percent") for c in ["freq_A", "freq_B", "diff_A_minus_B"]}


def mutation_frequency_tab(kp: str = "mf") -> None:
    M_all, labels, genomic = alterations()
    G, GL = data.groupings(), data.grouping_labels()
    cl = data.cell_lines().set_index("cell_line")
    st.markdown("Are certain mutations more frequent in one group of lines than another? Per mutation, a "
                "two-sided Fisher exact test on the lines annotated for it; q = Benjamini–Hochberg FDR "
                "across the mutations shown.")

    groupings = [c for c in G.index if c in cl.columns]
    c1, c2, c3 = st.columns([3, 2, 2])
    grouping = c1.selectbox("Compare lines by", groupings, format_func=GL.get, key=f"{kp}_grouping",
                            index=groupings.index("KMT2A_rearranged") if "KMT2A_rearranged" in groupings else 0)
    levels = cl[grouping].dropna().astype(str).value_counts()
    opts = list(levels.index)
    a0, b0 = data.default_pair(grouping, opts)
    fmt = lambda g: f"{g} (n={levels[g]})"  # noqa: E731
    ga = c2.selectbox("Group A (blue)", opts, index=opts.index(a0) if a0 in opts else 0, format_func=fmt,
                      key=f"{kp}_a")
    rest = [o for o in opts if o != ga] + ["all other lines"]
    gb = c3.selectbox("Group B (red)", rest, index=rest.index(b0) if b0 in rest else 0, key=f"{kp}_b",
                      format_func=lambda g: g if g == "all other lines" else fmt(g))
    grp = cl[grouping].astype("string")
    lines_a = cl.index[grp == ga].tolist()
    lines_b = cl.index[grp.notna() & (grp != ga) if gb == "all other lines" else grp == gb].tolist()

    from portal.surfaceome import show_groups
    show_groups(lines_a, lines_b, ga, gb, colours=("blue", "red"))
    candidates = [c for c in genomic if c != grouping and (M_all[c].reindex(lines_a + lines_b) > 0).any()]
    _keep_valid(f"{kp}_alts", candidates, candidates)
    c4, c5, c6 = st.columns([4, 2, 2], vertical_alignment="bottom")
    chosen = c4.multiselect("Mutations", candidates, key=f"{kp}_alts", format_func=labels.get,
                            help="Hotspot or damaging mutations, FLT3-ITD and KMT2A rearrangement; "
                                 "only those altered in at least one compared line.")
    sort = c5.segmented_control("Order by", ["p value", "frequency"], default="p value", key=f"{kp}_sort")
    fdr = c6.select_slider("FDR threshold", [0.01, 0.05, 0.1, 0.25], value=0.1, key=f"{kp}_fdr")
    if not chosen:
        st.info("Pick at least one mutation.")
        return
    res = mutation_enrichment(M_all[chosen], lines_a, lines_b)
    if res.empty:
        st.info("None of the chosen mutations is annotated in both groups.")
        return
    if sort == "frequency":
        res = res.assign(_f=res[["freq_A", "freq_B"]].max(axis=1)).sort_values("_f", ascending=False).drop(columns="_f")

    m1, m2, m3 = st.columns(3)
    m1.metric("Lines A / B", f"{len(lines_a)} / {len(lines_b)}")
    m2.metric("Mutations tested", len(res))
    m3.metric(f"q < {fdr}", int((res["q"] < fdr).sum()))
    st.plotly_chart(plots.frequency_bars(res, labels, ga, gb, fdr=fdr), width="stretch")
    st.caption(f"** q < {fdr}, * p < 0.05. Hover a bar for altered / annotated lines.")

    show = res.assign(alteration=res["alteration"].map(labels))
    st.dataframe(show, hide_index=True, width="stretch", column_config={**FMT, **PCT})
    st.download_button("Download table (TSV)", show.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                       file_name=f"mutation_frequency_{grouping}_{ga}_vs_{gb}.tsv".replace(" ", "_").replace("/", "-"))
