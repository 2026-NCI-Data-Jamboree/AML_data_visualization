"""Feature query: one gene across every omics layer of a built-in analysis, by any grouping."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portal import data, plots
from portal.stats import welch


def cell_line_feature_query(key_prefix: str = "fq") -> None:
    idx = data.gene_index()
    dsets = data.datasets().set_index("key")
    cl = data.cell_lines()
    groupings = data.grouping_labels()
    genes = sorted(idx["gene"].unique())

    st.markdown("Look up one gene across every cell-line layer, including drugs that target it.")
    c1, c2 = st.columns([2, 3])
    gene = c1.selectbox("Gene (HGNC symbol)", genes, index=genes.index("MTOR") if "MTOR" in genes else 0,
                        key=f"{key_prefix}_gene")
    grouping = c2.selectbox("Group lines by", list(groupings), format_func=groupings.get,
                            key=f"{key_prefix}_grouping")
    g = data.groupings().loc[grouping]
    levels = sorted(cl[grouping].dropna().astype(str).unique())
    if g["kind"] == "binary":                                  # TRUE group first, as in the volcano
        levels = [g["true_label"], g["false_label"]]
    two = len(levels) == 2

    hits = idx[idx["gene"] == gene]
    summary = []
    for key, h in hits.groupby("dataset", sort=False):
        meta = dsets.loc[key]
        V = data.values(key)
        labels = data.feature_labels(key)
        feats = [f for f in dict.fromkeys(h["feature"]) if f in V.index]
        if not feats:
            continue
        st.subheader(meta["label"])
        st.caption(f"{meta['unit']} · {meta['source']}"
                   + (" · lower = more dependent / sensitive" if meta["lower_is_stronger"] else ""))
        if len(feats) > 6:  # e.g. many CpG clusters or drugs: show the most variable ones
            st.caption(f"{len(feats)} features; showing the 6 most variable.")
            feats = list(V.loc[feats].std(axis=1).sort_values(ascending=False).index[:6])
        cols = st.columns(min(3, len(feats)))
        for i, f in enumerate(feats):
            long = (V.loc[f].rename("value").rename_axis("cell_line").reset_index()
                    .merge(cl[["cell_line", grouping]], on="cell_line"))
            long[grouping] = long[grouping].astype("string").fillna("not annotated")
            via = h.loc[h["feature"] == f, "via"].iloc[0]
            title = labels.get(f, f) + (f" → {gene}" if via == "drug target" else "")
            cols[i % len(cols)].plotly_chart(
                plots.strip(long, "value", grouping, levels + ["not annotated"], title=title,
                            y_title=meta["unit"]),
                width="stretch", key=f"{key_prefix}-{key}-{f}")
            row = {"layer": meta["label"], "feature": labels.get(f, f)}
            for lv in levels:
                row[f"mean {lv}"] = long.loc[long[grouping] == lv, "value"].mean()
            if two:
                r = welch(V.loc[[f]], long.loc[long[grouping] == levels[0], "cell_line"].tolist(),
                          long.loc[long[grouping] == levels[1], "cell_line"].tolist())
                row["p (Welch)"] = r["p_welch"].iloc[0] if len(r) else np.nan
            summary.append(row)

    if not summary:
        st.info("No measurements for this gene.")
        return
    st.subheader(f"Summary by {groupings[grouping]}")
    st.caption("p values are per feature and not corrected for multiple testing." if two else
               "Welch p values are shown when the grouping has exactly two groups.")
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch",
                 column_config={"p (Welch)": st.column_config.NumberColumn(format="%.2e")})


def patient_feature_query(kp: str = "pfq") -> None:
    idx = data.patient_gene_index()
    idx = idx[idx["via"] == "feature"]                 # molecular data only, no drug response
    if idx.empty:
        st.info("No patient cohorts are available yet.")
        return
    names = data.patient_cohort_names()
    genes = sorted(idx["gene"].unique())
    # groupings offered: union over cohorts, in groupings.tsv order, labelled as in the first cohort
    glabels = {}
    for c in names:
        for col, r in data.patient_groupings(c).iterrows():
            if col not in ("MYC_high", "mTOR_high", "MYC_mTOR_category", "cell_state"):   # not offered here
                glabels.setdefault(col, r["label"])
    st.markdown("Look up one gene in every patient cohort.")
    c1, c2 = st.columns([2, 3])
    gene = c1.selectbox("Gene (HGNC symbol)", genes, key=f"{kp}_gene",
                        index=genes.index("MAP1A") if "MAP1A" in genes else 0)
    opts = list(glabels) or [None]
    glabels.setdefault(None, "no grouping")
    grouping = c2.selectbox("Group samples by", opts, format_func=glabels.get, key=f"{kp}_grouping",
                            index=opts.index("risk_group") if "risk_group" in opts else 0)

    hits = idx[idx["gene"] == gene]
    summary = []
    top = st.container()               # summary table goes here, above the per-cohort plots
    for c in names:
        h = hits[hits["cohort"] == c]
        if h.empty:
            continue
        s = data.patient_samples_labeled(c)
        g = data.patient_groupings(c)
        if grouping is not None and grouping in g.index and g.loc[grouping, "kind"] == "binary":
            levels = [g.loc[grouping, "true_label"], g.loc[grouping, "false_label"]]
        elif grouping is not None and grouping in s:
            levels = list(s[grouping].dropna().astype(str).value_counts().index)
        else:
            levels = []
        grp = (s.set_index("sample_id")[grouping].astype("string") if grouping is not None and grouping in s
               else pd.Series(dtype="string"))
        ds = data.patient_datasets(c).set_index("key")
        st.subheader(names[c])
        if not levels and grouping is not None:
            st.caption(f"{glabels[grouping]} is not annotated in this cohort; all samples shown together.")
        for key, hk in h.groupby("dataset", sort=False):
            meta = ds.loc[key]
            V = data.patient_values(c, key)
            labels = data.patient_features(c, key)["label"]
            feats = [f for f in dict.fromkeys(hk["feature"]) if f in V.index]
            if not feats:
                continue
            lower = str(meta["lower_is_stronger"]).upper() == "TRUE"
            if len(feats) > 6:
                feats = list(V.loc[feats].std(axis=1).sort_values(ascending=False).index[:6])
            cols = st.columns(min(3, len(feats)))
            for i, f in enumerate(feats):
                long = V.loc[f].rename("value").rename_axis("cell_line").reset_index()
                long["group"] = long["cell_line"].map(grp).fillna("not annotated") if levels else "all samples"
                via = hk.loc[hk["feature"] == f, "via"].iloc[0]
                title = (f"{meta['label']}: " + labels.get(f, f) + (f" → {gene}" if via == "drug target" else ""))
                cols[i % len(cols)].plotly_chart(
                    plots.strip(long, "value", "group", (levels + ["not annotated"]) if levels else ["all samples"],
                                title=title, y_title=meta["unit"]),
                    width="stretch", key=f"{kp}-{c}-{key}-{f}")
                row = {"cohort": names[c], "layer": meta["label"], "feature": labels.get(f, f),
                       "n": int(long["value"].notna().sum())}
                for lv in levels:
                    row[f"mean {lv}"] = long.loc[long["group"] == lv, "value"].mean()
                if len(levels) == 2:
                    r = welch(V.loc[[f]], long.loc[long["group"] == levels[0], "cell_line"].tolist(),
                              long.loc[long["group"] == levels[1], "cell_line"].tolist())
                    row["p (Welch)"] = r["p_welch"].iloc[0] if len(r) else np.nan
                summary.append(row)
            st.caption(f"{meta['unit']} · {meta['source']}" + (" · lower = more sensitive" if lower else ""))
    if not summary:
        with top:
            st.info("No measurements for this gene in the patient cohorts.")
