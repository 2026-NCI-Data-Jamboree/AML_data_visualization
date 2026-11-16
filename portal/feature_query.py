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


def patient_feature_query(key_prefix: str = "pfq") -> None:
    patients = (data.results_dir() or data.require_results()) / "patients"
    if not patients.is_dir():
        st.info("Patient cohorts are not prepared yet. This tab will query a gene across the harmonized "
                "cohorts (RNA, protein, phosphosite, drug response) once `prepare_patient_data.R` in "
                "AML_data_processing writes `results/patients/`.")
        return
    st.warning("`results/patients/` exists but the patient feature query is not implemented yet.")
