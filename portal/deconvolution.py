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
QUANTITY_LABEL = {"mrna_share": "shares of the sample's mRNA", "cell_share": "shares of the sample's cells"}
SEVERITY_RANK = {"info": 1, "warning": 2, "caution": 3}
SEVERITY_COLOR = {"info": "#4c8bd4", "warning": "#e69f00", "caution": "#d1362f"}
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#76b7b2", "#edc948", "#9c755f", "#bab0ac", "#86bcb6",
           "#d4a6c8", "#8cd17d", "#a0cbe8", "#ffbe7d", "#b6992d", "#499894", "#d37295", "#79706e", "#fabfd2",
           "#d62728", "#ff9896", "#17becf", "#9edae5", "#bcbd22", "#c5b0d5"]
MALIGNANT = "Malignant"
# malignant cell states from most primitive to most differentiated; unknown malignant labels follow, alphabetically
MALIGNANT_STATE_ORDER = ["LSC", "HSC-like", "Prog-like", "GMP-like", "ProMono-like", "Mono-like", "cDC-like"]
MALIGNANT_COLORS = ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a", "#fc9272", "#fcbba1", "#fee0d2"]
NORMAL_PALETTE = [c for c in PALETTE if c not in ("#d62728", "#ff9896", "#d37295", "#fabfd2")]
LEGEND_BELOW = dict(orientation="h", yanchor="top", y=-0.08, x=0, xanchor="left", title_text="")


# ------------------------------------------------------------------------------------------ helpers
def malignant_labels(reference: str, level: str, labels) -> list[str]:
    """Labels of the layer that are malignant in the reference, most primitive first."""
    try:
        table = data.reference_table(reference)
        flagged = set(table.loc[(table["level"] == level) & table["is_malignant"], "label"])
    except (FileNotFoundError, KeyError, TypeError):
        flagged = {MALIGNANT}
    found = [label for label in labels if label in flagged or label == MALIGNANT]
    rank = {label: i for i, label in enumerate(MALIGNANT_STATE_ORDER)}
    return sorted(found, key=lambda label: (rank.get(label, len(rank)), label))


def stack_order(labels, malignant: list[str]) -> list[str]:
    """Normal labels first (bottom of a stacked bar), then the malignant ones, most primitive on top."""
    return [label for label in sorted(labels) if label not in malignant] + list(reversed(malignant))


def label_colors(order: list[str], malignant: list[str]) -> dict[str, str]:
    """Reds for malignant labels (darkest = most primitive), the general palette for the rest."""
    normal = [label for label in order if label not in malignant]
    colors = {label: NORMAL_PALETTE[i % len(NORMAL_PALETTE)] for i, label in enumerate(normal)}
    colors.update({label: MALIGNANT_COLORS[min(i, len(MALIGNANT_COLORS) - 1)] for i, label in enumerate(malignant)})
    return colors



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
    text = f"{len(run)} notes on the whole cohort; {n_flagged} samples with a warning"
    text += f" ({n_hidden} hidden)." if n_hidden else " (all shown and marked)." if n_flagged else "."
    show = container.error if worst == 3 else container.warning if worst == 2 else container.info
    show(text, icon=":material/warning:")


def warning_details(warnings: pd.DataFrame) -> None:
    with st.expander("Warning details"):
        if warnings.empty:
            st.caption("No warnings for this cohort, method and level.")
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
        yaxis=dict(tickformat=".0%", range=[0, 1.08], title="proportion"), legend=LEGEND_BELOW)
    st.plotly_chart(fig, width="stretch")


def heatmap_with_tracks(P, flags, groups, group_order, sort_by, run_notes, malignant_rows=()) -> None:
    """Cell types x samples heatmap with a group track and a warning track; run-level notes in a side panel."""
    ids = order_samples(P, groups, group_order, sort_by)
    malignant = [label for label in reversed(P.index) if label in set(malignant_rows)]
    rows = malignant + [r for r in P.mean(axis=1).sort_values(ascending=False).index if r not in malignant]
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
        st.markdown("**Notes on the whole cohort**")
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
    fig.update_layout(barmode="stack", height=200 + 42 * len(group_order), margin=dict(t=10, b=10),
                      xaxis=dict(tickformat=".0%", range=[0, 1]), yaxis=dict(autorange="reversed"),
                      legend={**LEGEND_BELOW, "y": -0.15, "traceorder": "normal"})  # left to right, as the bars
    st.plotly_chart(fig, width="stretch")

    first_malignant = next((label for label in reversed(P.index) if label == MALIGNANT or label in MALIGNANT_STATE_ORDER), None)
    default = list(P.index).index(first_malignant) if first_malignant else 0
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
        st.info("No deconvolution estimates are available.")
        return
    names = data.patient_cohort_names()

    c1, c2, c3, c4, c5 = st.columns([1.3, 1.4, 1, 1.3, 1.2])
    cohort = c1.selectbox("Cohort", cohorts, format_func=lambda c: names.get(c, c), key="dc_cohort")
    runs = data.deconvolution_runs(cohort)
    methods = [m for m in METHOD_LABEL if m in set(runs["method"])] or sorted(set(runs["method"]))
    method = c2.selectbox("Method", methods, format_func=lambda m: METHOD_LABEL.get(m, m), key="dc_method",
                          help="The initial fit uses the reference as built. The updated fit adapts the reference to "
                               "the cohort, so a sample's estimate depends on the other samples in the cohort.")
    levels = [lv for lv in LEVEL_LABEL if lv in set(runs.loc[runs["method"] == method, "level"])]
    level = c3.selectbox("Level", levels, format_func=LEVEL_LABEL.get, key="dc_level")
    if levels == ["cell_type"]:
        c3.caption("This method has cell types only.")
    run = runs[(runs["method"] == method) & (runs["level"] == level)].iloc[0]
    key = run["key"]

    P = data.patient_values(cohort, key).dropna(axis=1, how="all")
    malignant = malignant_labels(run["reference"], level, P.index)
    P = P.loc[stack_order(P.index, malignant)]
    warnings = data.deconvolution_warnings(cohort)
    warnings = warnings[warnings["key"].isin(["", key])]
    flags = sample_flags(warnings, P.columns)

    samples = data.patient_samples_labeled(cohort)
    grouping = data.patient_groupings(cohort)
    group_cols = ["(none)"] + [c for c in grouping.index if c in samples]
    group_col = c4.selectbox("Group by", group_cols, key="dc_group",
                             format_func=lambda c: c if c == "(none)" else grouping.loc[c, "label"])
    sort_options = ["(none)"] + list(P.index)
    sort_by = c5.selectbox("Sort samples by", sort_options, key="dc_sort",
                           index=sort_options.index(malignant[0]) if malignant else 0)

    hide = st.toggle("Hide samples with a warning", value=True, key="dc_hide",
                     help="Leaves out every sample with a warning (any severity) for this method and level, from "
                          "all three views. Turn it off to show them, marked.")
    n_hidden = int((flags > 0).sum()) if hide else 0
    if hide:
        P = P.loc[:, flags[flags == 0].index]
        flags = flags[P.columns]
    st.caption(f"{P.shape[1]} samples shown, {P.shape[0]} {LEVEL_LABEL[level].lower()}s. "
               f"Values are {QUANTITY_LABEL.get(run['quantity'], run['quantity'])}.")
    warning_banner(warnings, n_hidden)
    warning_details(warnings)
    if P.empty:
        st.warning("Every sample has a warning; turn off the hiding to see them.")
        return

    group_series = (samples.set_index("sample_id")[group_col].fillna("NA").astype(str)
                    if group_col != "(none)" else pd.Series("all", index=P.columns))
    groups = {s: group_series.get(s, "NA") for s in P.columns}
    group_order = sorted(set(groups.values()))
    colors = label_colors(list(P.index), malignant)

    tab_stacks, tab_heatmap, tab_summary = st.tabs(["Sorted stacks", "Heatmap", "Group summary"])
    with tab_stacks:
        sorted_stacks(P, flags, groups, group_order, sort_by, colors)
    with tab_heatmap:
        heatmap_with_tracks(P, flags, groups, group_order, sort_by, warnings[warnings["scope"] == "run"], malignant)
    with tab_summary:
        group_summary(P, flags, groups, group_order, colors)


# ------------------------------------------------------------------------------------------ reference tab
def reference_tab() -> None:
    references = data.single_cell_references()
    if not references:
        st.info("No reference is available.")
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

    st.subheader("Cohorts estimated with this reference")
    names = data.patient_cohort_names()
    built = [data.deconvolution_runs(c).assign(cohort=names.get(c, c)) for c in data.deconvolution_cohorts()]
    if built:
        built = pd.concat(built)
        built = built[built["reference"] == reference]
        built = built.assign(method=built["method"].map(lambda m: METHOD_LABEL.get(m, m)),
                             level=built["level"].map(lambda lv: LEVEL_LABEL.get(lv, lv)))
        st.dataframe(built[["cohort", "method", "level"]], hide_index=True, width="stretch")


# ------------------------------------------------------------------------------------------ differentiation stage tab
DIFFSTAGE = "diffstage"                 # layer key (modality differentiation_score), from BoneMarrowMap


def diffstage_cohorts() -> list[str]:
    """Cohorts with BoneMarrowMap differentiation stage scores. Older data releases have none: the portal then
    leaves the Differentiation stage tab out. A cohort whose values file is missing is skipped."""
    root = data.patients_dir()
    if not root:
        return []
    return [c for c in data.patient_cohorts()
            if DIFFSTAGE in set(data.patient_datasets(c).get("key", []))
            and (root / c / "values" / f"{DIFFSTAGE}.tsv.gz").is_file()]


def diffstage_table(cohorts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores (samples x stages, stage order kept) and the samples' annotation, pooled over cohorts."""
    names = data.patient_cohort_names()
    scores, samples = [], []
    for c in cohorts:
        V = data.patient_values(c, DIFFSTAGE)
        try:
            labels = data.patient_features(c, DIFFSTAGE)["label"]
        except (FileNotFoundError, KeyError):          # no feature file: keep the stage IDs
            labels = pd.Series(dtype=str)
        scores.append(V.rename(index=labels).T.dropna(how="all"))
        s = data.patient_samples_labeled(c).set_index("sample_id")
        samples.append(s.loc[s.index.intersection(scores[-1].index)].assign(cohort=names.get(c, c)))
    return pd.concat(scores), pd.concat(samples)


def diffstage_tab() -> None:
    cohorts = diffstage_cohorts()
    if not cohorts:
        st.info("No differentiation stage scores are available.")
        return
    names = data.patient_cohort_names()
    st.markdown(
        "How much each sample's RNA resembles 13 stages of normal blood cell development, from stem cells (HSC / MPP) "
        "to mature cells. Scores come from **BoneMarrowMap** (Zeng et al. 2025, *Blood Cancer Discovery*), which "
        "scored diagnosis samples of five cohorts against its single-cell atlas of normal bone marrow. Each score was "
        "standardised across all five cohorts together, so 0 is the average sample and samples from different "
        "cohorts can be compared directly. Unlike the proportions in *Composition*, a sample's scores do not add up "
        "to anything.")
    c1, c2 = st.columns([3, 2])
    picked = c1.multiselect("Cohorts", cohorts, default=cohorts, format_func=lambda c: names.get(c, c), key="ds_cohorts")
    if not picked:
        st.info("Pick at least one cohort.")
        return
    S, A = diffstage_table(picked)
    labels = {"cohort": "Cohort"}
    for c in picked:
        for col, r in data.patient_groupings(c).iterrows():
            if col in A and A[col].notna().any():
                labels.setdefault(col, r["label"])
    opts = list(labels)
    group_col = c2.selectbox("Group by", opts, format_func=labels.get, key="ds_group",
                             index=opts.index("amlmap_cluster") if "amlmap_cluster" in opts else 0)
    groups = A[group_col].astype("string").reindex(S.index)
    counts = groups.value_counts()
    keep = counts.index[counts >= 3]
    S, groups = S[groups.isin(keep)], groups[groups.isin(keep)]
    order = sorted(keep, key=str)
    n_missing = int((A[group_col].reindex(A.index).isna()).sum())
    st.caption(f"{len(S)} samples in {len(order)} groups (groups need at least 3 samples)"
               + (f"; {n_missing} samples without this annotation are left out." if n_missing else "."))
    if S.empty:
        st.warning("No group has 3 or more samples.")
        return

    means = S.groupby(groups).mean().loc[order].T                      # stages x groups
    lim = float(max(1.0, abs(means.values).max()))
    fig = go.Figure(go.Heatmap(
        z=means.values, x=[f"{g} (n={counts[g]})" for g in order], y=list(means.index), zmin=-lim, zmax=lim,
        colorscale="RdBu_r", colorbar=dict(title="mean score"),
        hovertemplate="%{y}<br>%{x}<br>mean score %{z:.2f}<extra></extra>"))
    fig.update_layout(height=160 + 28 * len(means), margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"),
                      xaxis=dict(tickangle=-40))
    st.subheader("Mean score per group")
    st.caption("Red: the group resembles this stage more than the average sample; blue: less.")
    st.plotly_chart(fig, width="stretch")

    st.subheader("One stage by group")
    stages = list(S.columns)
    stage = st.selectbox("Stage", stages, index=stages.index("HSC / MPP") if "HSC / MPP" in stages else 0,
                         key="ds_stage")
    box = go.Figure()
    for i, g in enumerate(order):
        members = groups.index[groups == g]
        box.add_trace(go.Box(y=S.loc[members, stage], name=f"{g} (n={len(members)})", boxpoints="all", jitter=0.4,
                             pointpos=0, text=list(members), marker=dict(color=PALETTE[i % len(PALETTE)], size=4),
                             hoverinfo="text+y"))
    box.add_hline(y=0, line_dash="dot", line_color="grey")
    box.update_layout(height=420, margin=dict(t=10, b=10), yaxis=dict(title=f"{stage} score"), showlegend=False,
                      xaxis=dict(tickangle=-40))
    st.plotly_chart(box, width="stretch")
    st.download_button("Download scores (CSV)", S.join(A[list(dict.fromkeys(["cohort", group_col]))])
                       .to_csv().encode(), "differentiation_stage_scores.csv", "text/csv", key="ds_download")
