"""Mutation–CNV association: are specific mutations associated with amplifications or deletions?

For each mutation x genomic region, mutated vs wild-type lines (lines annotated for the mutation
and with copy number for the region) are compared on either the frequency of the region being
deleted / amplified (Fisher exact) or its copy-number level (Welch t on log2 relative CN).
BH FDR across all mutation x region tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portal import data
from portal import genomics_plots as plots
from portal.cooccurrence import _keep_valid, alterations
from portal.copy_number import CN, region_cn
from portal.genomics_stats import mutation_enrichment
from portal.stats import bh, welch

EVENTS = ["deletion", "amplification", "copy-number level"]


def associate(L: pd.DataFrame, M: pd.DataFrame, event: str, amp: float, dele: float, min_mut: int = 2) -> pd.DataFrame:
    """L: lines x regions median relative CN. M: lines x mutations (1/0/NaN)."""
    lines = L.index.intersection(M.index)
    L, M = L.loc[lines], M.loc[lines]
    out = []
    for m in M.columns:
        mut, wt = M.index[M[m] == 1].tolist(), M.index[M[m] == 0].tolist()
        if len(mut) < min_mut or len(wt) < min_mut:
            continue
        if event == "copy-number level":
            r = welch(np.log2(L.clip(lower=1 / 64)).T, mut, wt, min_n=min_mut)
            if r.empty:
                continue
            out.append(pd.DataFrame({
                "mutation": m, "region": r["feature"], "n_mut": r["n_A"], "n_wt": r["n_B"],
                "mean_mut": r["mean_A"], "mean_wt": r["mean_B"], "effect": r["diff_A_minus_B"], "p": r["p_welch"],
                "detail": [f"mean log2 rel. CN {a:.2f} mutated vs {b:.2f} wild-type (n {x}/{y})"
                           for a, b, x, y in zip(r["mean_A"], r["mean_B"], r["n_A"], r["n_B"])]}))
        else:
            E = (L <= dele) if event == "deletion" else (L >= amp)
            r = mutation_enrichment(E.astype(float).where(L.notna()), mut, wt)
            if r.empty:
                continue
            out.append(pd.DataFrame({
                "mutation": m, "region": r["alteration"], "n_mut": r["n_A"], "n_wt": r["n_B"],
                "freq_mut": r["freq_A"], "freq_wt": r["freq_B"], "effect": r["diff_A_minus_B"], "p": r["p"],
                "detail": [f"{event}: {a}/{x} mutated vs {b}/{y} wild-type"
                           for a, x, b, y in zip(r["altered_A"], r["n_A"], r["altered_B"], r["n_B"])]}))
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    res["q"] = bh(res["p"].to_numpy())
    return res.sort_values("p").reset_index(drop=True)


def mutation_cnv_tab(kp: str = "mc") -> None:
    dsets = data.datasets().set_index("key")
    if CN not in dsets.index or "cytoband" not in data.features(CN).columns:
        st.info("Needs the copy-number layer (cn, with cytobands) in results/cell_lines.")
        return
    M_all, labels, genomic = alterations()
    st.markdown("Are specific mutations associated with amplifications or deletions? For each mutation and "
                "genomic region, mutated and wild-type lines are compared: how often the region is deleted or "
                "amplified (Fisher exact test), or its copy-number level (Welch t test). q = Benjamini–Hochberg "
                "FDR across all mutation × region tests.")

    c1, c2, c3 = st.columns([3, 2, 2], vertical_alignment="bottom")
    event = c1.segmented_control("Compare", EVENTS, default="deletion", key=f"{kp}_event") or "deletion"
    res_lvl = c2.segmented_control("Regions", ["chromosome arm", "cytoband"], default="chromosome arm",
                                   key=f"{kp}_res") or "chromosome arm"
    fdr = c3.select_slider("FDR threshold", [0.01, 0.05, 0.1, 0.25], value=0.1, key=f"{kp}_fdr")
    c4, c5, c6, c7 = st.columns([4, 2, 2, 2], vertical_alignment="bottom")
    med, _ = region_cn("arm" if res_lvl == "chromosome arm" else "band")
    lines = med.index.intersection(M_all.index)
    n_mut = (M_all.reindex(lines) > 0).sum()
    min_mut = c5.number_input("Min. mutated lines", 2, 20, 3, key=f"{kp}_min")
    options = [c for c in genomic if n_mut[c] >= min_mut]
    _keep_valid(f"{kp}_muts", options, options)
    muts = c4.multiselect("Mutations", options, key=f"{kp}_muts",
                          format_func=lambda c: f"{labels[c]} ({int(n_mut[c])})")
    amp = c6.number_input("Amplified if rel. CN ≥", 1.05, 4.0, 1.3, 0.05, key=f"{kp}_amp",
                          disabled=event != "amplification")
    dele = c7.number_input("Deleted if rel. CN ≤", 0.0, 0.95, 0.7, 0.05, key=f"{kp}_del",
                           disabled=event != "deletion")
    sex = st.toggle("Include X and Y", value=False, key=f"{kp}_sex")
    if not muts:
        st.info("Pick at least one mutation.")
        return
    if not sex:
        med = med[[r for r in med.columns if not r.startswith(("X", "Y"))]]
    res = associate(med, M_all[muts], event, amp, dele, min_mut=min_mut)
    if res.empty:
        st.info("Nothing testable: each mutation needs enough mutated and wild-type lines with copy number.")
        return

    what = "CN" if event == "copy-number level" else f"{event} rate"
    m1, m2, m3 = st.columns(3)
    m1.metric("Lines with CN and mutation calls", len(lines))
    m2.metric("Tests", f"{len(res):,}")
    m3.metric(f"q < {fdr}", int((res["q"] < fdr).sum()))
    rows = [m for m in muts if m in set(res["mutation"])]
    st.plotly_chart(plots.assoc_heatmap(res, rows, list(med.columns), labels, fdr=fdr, what=what),
                    width="stretch")
    st.caption(f"Red = higher {what} in mutated lines, blue = lower; colour = signed −log10 p. "
               f"** q < {fdr}, * p < 0.05. Hover a cell for the counts.")

    show = res.drop(columns="detail").assign(mutation=res["mutation"].map(labels))
    cfg = {c: st.column_config.NumberColumn(format="%.3g") for c in ["effect", "p", "q", "mean_mut", "mean_wt"]}
    cfg.update({c: st.column_config.NumberColumn(format="percent") for c in ["freq_mut", "freq_wt"]})
    st.dataframe(show, hide_index=True, width="stretch", height=360, column_config=cfg)
    st.download_button("Download table (TSV)", show.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                       file_name=f"mutation_cnv_{event.replace(' ', '_')}_{res_lvl.replace(' ', '_')}.tsv")
