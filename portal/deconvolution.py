"""Deconvolution page: composition of bulk samples (three views) and the reference summary.

Reads only the deconvolution files of results/ (AML_data_processing/docs/results_contract.md, Deconvolution):
layers `deconv_<reference>_<method>_<level>` in each cohort, deconvolution_runs.tsv,
deconvolution_warnings.tsv and single_cell/<REFERENCE_ID>/.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from portal import data

METHOD_LABEL = {
    "instaprism": "InstaPrism, initial fit",
    "instaprism_update_all": "InstaPrism, updated fit",
    "instaprism_update_malignant": "InstaPrism, updated fit (malignant profile only)",
    "instaprism_update_pooled": "InstaPrism, updated fit (pooled)",
    "nnls": "NNLS",
}
LEVEL_LABEL = {"cell_type": "Cell type", "cell_state": "Cell state"}
QUANTITY_LABEL = {"mrna_share": "mRNA share", "cell_share": "cell share"}
SEVERITY_RANK = {"info": 1, "warning": 2, "caution": 3}
SEVERITY_COLOR = {"info": "#4c8bd4", "warning": "#e69f00", "caution": "#d1362f"}
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#76b7b2", "#edc948", "#9c755f", "#bab0ac", "#86bcb6",
           "#d4a6c8", "#8cd17d", "#a0cbe8", "#ffbe7d", "#b6992d", "#499894", "#d37295", "#79706e", "#fabfd2",
           "#d62728", "#ff9896", "#17becf", "#9edae5", "#bcbd22", "#c5b0d5"]
MALIGNANT = "Malignant"


# ------------------------------------------------------------------------------------------ helpers
def sample_flags(warnings: pd.DataFrame, samples) -> pd.Series:
    """Highest severity rank (0 = none, 1 info, 2 warning, 3 caution) of the sample-scope warnings per sample."""
    sample_rows = warnings[warnings["scope"] == "sample"]
    worst = sample_rows.assign(rank=sample_rows["severity"].map(SEVERITY_RANK)).groupby("sample_id")["rank"].max()
    return pd.Series(list(samples), index=list(samples)).map(worst).fillna(0).astype(int)


def order_samples(P: pd.DataFrame, groups: dict[str, str], group_order: list[str], sort_by: str) -> list[str]:
    """Samples by group (in group_order), then by descending proportion of `sort_by`."""
    ordered = []
    for g in group_order:
        members = [s for s in P.columns if groups[s] == g]
        if sort_by in P.index:
            members.sort(key=lambda s: -P.loc[sort_by, s])
        ordered += members
    return ordered


def warning_banner(warnings: pd.DataFrame, n_hidden: int, container=st) -> None:
    run = warnings[warnings["scope"] == "run"]
    worst = run["severity"].map(SEVERITY_RANK).max() if len(run) else 0
    n_flagged = warnings.loc[warnings["scope"] == "sample", "sample_id"].nunique()
    text = f"{len(run)} run-level notes; {n_flagged} samples with a warning"
    text += f" ({n_hidden} hidden)." if n_hidden else " (all shown and marked)." if n_flagged else "."
    show = container.error if worst == 3 else container.warning if worst == 2 else container.info
    show(text, icon=":material/warning:")


def warning_details(warnings: pd.DataFrame) -> None:
    with st.expander("Warning details"):
        if warnings.empty:
            st.caption("No warnings for this layer.")
        else:
            st.dataframe(warnings[["severity", "scope", "sample_id", "code", "message", "source"]],
                         hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------ views
def sorted_stacks(P, flags, groups, group_order, sort_by, colors) -> None:
    """One stacked bar per sample, grouped; markers above the bars flag samples with a warning."""
    ids = order_samples(P, groups, group_order, sort_by)
    x, centers, pos = [], {}, 0
    for g in group_order:
        members = [s for s in ids if groups[s] == g]
        if members:
            centers[g] = pos + (len(members) - 1) / 2
            x += list(range(pos, pos + len(members)))
            pos += len(members) + 4
    fig = go.Figure()
    for label in P.index:
        fig.add_bar(x=x, y=P.loc[label, ids], name=label, marker_color=colors[label], marker_line_width=0,
                    customdata=ids, hovertemplate="%{customdata}<br>" + label + ": %{y:.1%}<extra></extra>")
    for severity, rank in SEVERITY_RANK.items():
        flagged = [(xi, s) for xi, s in zip(x, ids) if flags[s] == rank]
        if flagged:
            fig.add_scatter(x=[f[0] for f in flagged], y=[1.04] * len(flagged), mode="markers",
                            name=f"warning: {severity}", marker=dict(symbol="triangle-down", size=8,
                                                                    color=SEVERITY_COLOR[severity]),
                            hovertext=[f[1] for f in flagged], hoverinfo="text")
    fig.update_layout(
        barmode="stack", bargap=0, height=520, margin=dict(t=10, b=10),
        xaxis=dict(tickmode="array", tickvals=list(centers.values()),
                   ticktext=[f"{g} (n={sum(groups[s] == g for s in ids)})" for g in centers]),
        yaxis=dict(tickformat=".0%", range=[0, 1.08], title="proportion"))
    st.plotly_chart(fig, width="stretch")


def heatmap_with_tracks(P, flags, groups, group_order, sort_by, run_notes) -> None:
    """Cell types x samples heatmap with a group track and a warning track; run-level notes in a side panel."""
    ids = order_samples(P, groups, group_order, sort_by)
    rows = P.mean(axis=1).sort_values(ascending=False).index.tolist()
    k = len(group_order)
    group_colors = [PALETTE[i % len(PALETTE)] for i in range(k)]
    stops = [[b, c] for i, c in enumerate(group_colors) for b in (i / k, (i + 1) / k)]  # one flat band per group
    show_flags = bool(flags.max() > 0)

    main, side = st.columns([4, 1])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.05, 0.05 if show_flags else 0.001, 0.9],
                        vertical_spacing=0.01)
    fig.add_heatmap(z=[[group_order.index(groups[s]) for s in ids]], x=ids, zmin=-0.5, zmax=k - 0.5, colorscale=stops,
                    showscale=False, hovertext=[[groups[s] for s in ids]], hoverinfo="text", row=1, col=1)
    if show_flags:
        fig.add_heatmap(z=[[int(flags[s]) for s in ids]], x=ids, zmin=0, zmax=3, showscale=False, hoverinfo="x+z",
                        colorscale=[[0, "#f2f2f2"], [0.33, SEVERITY_COLOR["info"]], [0.66, SEVERITY_COLOR["warning"]],
                                    [1, SEVERITY_COLOR["caution"]]], row=2, col=1)
    fig.add_heatmap(z=P.loc[rows, ids].values, x=ids, y=rows, colorscale="Viridis", zmin=0,
                    zmax=float(P.values.max()), colorbar=dict(title="proportion", len=0.6, y=0.4), row=3, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=620, margin=dict(t=10, b=10))
    main.plotly_chart(fig, width="stretch")
    with side:
        st.markdown("**Groups**")
        for g in group_order:
            st.caption(f"{g}: n={sum(groups[s] == g for s in ids)}")
        st.markdown("**Run-level notes**")
        for _, row in run_notes.iterrows():
            st.caption(f"{row['severity'].upper()}: {row['message']}")
        if show_flags:
            st.caption("The second track marks samples with a warning; hover for the sample ID.")


def group_summary(P, flags, groups, group_order, colors) -> None:
    """Mean composition per group, then the distribution of one cell type or state per group."""
    means = pd.DataFrame({g: P[[s for s in P.columns if groups[s] == g]].mean(axis=1) for g in group_order})
    fig = go.Figure()
    for label in P.index:
        fig.add_bar(y=means.columns, x=means.loc[label], name=label, orientation="h", marker_color=colors[label],
                    hovertemplate=label + ": %{x:.1%}<extra></extra>")
    fig.update_layout(barmode="stack", height=80 + 42 * len(group_order), margin=dict(t=10, b=10),
                      xaxis=dict(tickformat=".0%", range=[0, 1]), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

    default = list(P.index).index(MALIGNANT) if MALIGNANT in P.index else 0
    label = st.selectbox("Distribution of", list(P.index), index=default, key="dc_drill")
    box = go.Figure()
    for i, g in enumerate(group_order):
        members = [s for s in P.columns if groups[s] == g]
        box.add_trace(go.Box(y=P.loc[label, members], name=g, boxpoints="all", jitter=0.4, pointpos=0, text=members,
                             marker=dict(color=PALETTE[i % len(PALETTE)], size=4), hoverinfo="text+y"))
    flagged = [s for s in P.columns if flags[s] > 0]
    if flagged:
        rank_name = {v: k for k, v in SEVERITY_RANK.items()}
        box.add_trace(go.Scatter(x=[groups[s] for s in flagged], y=P.loc[label, flagged], mode="markers",
                                 name="sample with a warning", text=flagged, hoverinfo="text+y",
                                 marker=dict(symbol="x", size=9, line=dict(width=1),
                                             color=[SEVERITY_COLOR[rank_name[flags[s]]] for s in flagged])))
    box.update_layout(height=380, margin=dict(t=10, b=10), yaxis=dict(tickformat=".0%", title=label), showlegend=False)
    st.plotly_chart(box, width="stretch")


# ------------------------------------------------------------------------------------------ composition tab
def composition_tab() -> None:
    cohorts = data.deconvolution_cohorts()
    if not cohorts:
        st.info("No deconvolution results found. In AML_data_processing run `uv run prepare_deconvolution.py` "
                "after the two `prepare_*.R` scripts.")
        return
    names = data.patient_cohort_names()

    c1, c2, c3, c4, c5 = st.columns([1.3, 1.4, 1, 1.3, 1.2])
    cohort = c1.selectbox("Cohort", cohorts, format_func=lambda c: names.get(c, c), key="dc_cohort")
    runs = data.deconvolution_runs(cohort)
    methods = [m for m in METHOD_LABEL if m in set(runs["method"])] or sorted(set(runs["method"]))
    method = c2.selectbox("Method", methods, format_func=lambda m: METHOD_LABEL.get(m, m), key="dc_method",
                          help="Each method and level is its own layer. The updated fit depends on the other "
                               "samples in the cohort and is flagged as such.")
    levels = [lv for lv in LEVEL_LABEL if lv in set(runs.loc[runs["method"] == method, "level"])]
    level = c3.selectbox("Level", levels, format_func=LEVEL_LABEL.get, key="dc_level")
    if levels == ["cell_type"]:
        c3.caption("This method has cell types only.")
    run = runs[(runs["method"] == method) & (runs["level"] == level)].iloc[0]
    key = run["key"]

    P = data.patient_values(cohort, key).dropna(axis=1, how="all")
    warnings = data.deconvolution_warnings(cohort)
    warnings = warnings[warnings["key"].isin(["", key])]
    flags = sample_flags(warnings, P.columns)

    samples = data.patient_samples_labeled(cohort)
    grouping = data.patient_groupings(cohort)
    group_cols = ["(none)"] + [c for c in grouping.index if c in samples]
    group_col = c4.selectbox("Group by", group_cols, key="dc_group",
                             format_func=lambda c: c if c == "(none)" else grouping.loc[c, "label"])
    sort_by = c5.selectbox("Sort samples by", ["(none)"] + list(P.index), key="dc_sort",
                           index=(list(P.index).index(MALIGNANT) + 1) if MALIGNANT in P.index else 0)

    hide = st.toggle("Hide samples with a warning", value=True, key="dc_hide",
                     help="Leaves out every sample that has a sample-level warning (any severity) for this layer, "
                          "from all three views. Turn it off to show them, marked.")
    n_hidden = int((flags > 0).sum()) if hide else 0
    if hide:
        P = P.loc[:, flags[flags == 0].index]
        flags = flags[P.columns]
    st.caption(f"{P.shape[1]} samples shown, {P.shape[0]} {LEVEL_LABEL[level].lower()}s. "
               f"Quantity: {QUANTITY_LABEL.get(run['quantity'], run['quantity'])}. Layer `{key}`, "
               f"run {run.get('run_date', '')}.")
    warning_banner(warnings, n_hidden)
    warning_details(warnings)
    if P.empty:
        st.warning("Every sample has a warning; turn off the hiding to see them.")
        return

    group_series = (samples.set_index("sample_id")[group_col].fillna("NA").astype(str)
                    if group_col != "(none)" else pd.Series("all", index=P.columns))
    groups = {s: group_series.get(s, "NA") for s in P.columns}
    group_order = sorted(set(groups.values()))
    colors = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(P.index)}

    tab_stacks, tab_heatmap, tab_summary = st.tabs(["Sorted stacks", "Heatmap", "Group summary"])
    with tab_stacks:
        sorted_stacks(P, flags, groups, group_order, sort_by, colors)
    with tab_heatmap:
        heatmap_with_tracks(P, flags, groups, group_order, sort_by, warnings[warnings["scope"] == "run"])
    with tab_summary:
        group_summary(P, flags, groups, group_order, colors)


# ------------------------------------------------------------------------------------------ reference tab
def reference_tab() -> None:
    references = data.single_cell_references()
    if not references:
        st.info("No reference summary found (`results/single_cell/<REFERENCE_ID>/`).")
        return
    reference = references[0] if len(references) == 1 else st.selectbox("Reference", references, key="dc_reference")
    provenance = data.reference_provenance(reference)
    table = data.reference_table(reference)

    st.subheader(provenance.get("reference_name", reference))
    st.markdown(f"{provenance.get('citation', '')}  \n{provenance.get('source', '')}")
    if provenance.get("limitations"):
        st.warning(provenance["limitations"], icon=":material/warning:")

    levels = [lv for lv in LEVEL_LABEL if lv in set(table["level"])]
    level = st.radio("Level", levels, format_func=LEVEL_LABEL.get, horizontal=True, key="dc_ref_level")
    shown = table[table["level"] == level].sort_values("n_cells", ascending=False)
    fig = go.Figure(go.Bar(
        x=shown["n_cells"], y=shown["label"], orientation="h", customdata=shown["n_reference_samples"],
        marker_color=["#d62728" if m else "#4e79a7" for m in shown["is_malignant"]],
        hovertemplate="%{y}: %{x:,} cells from %{customdata} reference samples<extra></extra>"))
    fig.update_layout(height=60 + 24 * len(shown), margin=dict(t=10, b=10), xaxis_title="cells in the reference",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Red: the malignant cell type. Hover for the number of reference samples contributing.")
    st.dataframe(shown.rename(columns={"n_cells": "cells", "n_reference_samples": "reference samples",
                                       "is_malignant": "malignant"}), hide_index=True, width="stretch")

    st.subheader("Marker genes")
    label = st.selectbox("Cell type or state", list(shown["label"]), key="dc_marker_label")
    markers = data.reference_markers(reference)
    markers = markers[(markers["level"] == level) & (markers["label"] == label)]
    st.dataframe(markers[["rank", "gene_symbol", "gene_id", "score"]], hide_index=True, width="stretch")
    st.caption("Display markers: the genes most specific to this label in the reference (log2 CPM difference to the "
               "next highest label). They are not necessarily the genes the deconvolution method used.")

    st.subheader("Provenance")
    skip = {"citation", "source", "limitations", "reference_name"}
    st.dataframe(pd.DataFrame([(k, v) for k, v in provenance.items() if k not in skip],
                              columns=["item", "value"]), hide_index=True, width="stretch")

    st.subheader("Layers built with this reference")
    names = data.patient_cohort_names()
    built = [data.deconvolution_runs(c).assign(cohort=names.get(c, c)) for c in data.deconvolution_cohorts()]
    if built:
        built = pd.concat(built)
        built = built[built["reference"] == reference]
        built = built.assign(method=built["method"].map(lambda m: METHOD_LABEL.get(m, m)),
                             level=built["level"].map(lambda lv: LEVEL_LABEL.get(lv, lv)))
        st.dataframe(built[["cohort", "method", "level", "run_date"]], hide_index=True, width="stretch")
