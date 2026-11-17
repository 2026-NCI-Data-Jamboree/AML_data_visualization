"""Genetically similar cell lines: lines sharing mutation, fusion and copy-number profiles.

Each line is a binary profile of aberrations (hotspot / damaging mutations and FLT3-ITD; KMT2A
fusion with its partner; arm-level copy-number gains and losses). Similarity between two lines is
the Jaccard index over the aberrations annotated in both: shared / present in either.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from portal import data
from portal import genomics_plots as plots
from portal.cooccurrence import alterations, cn_events, has_cn
from portal.genomics_stats import jaccard_similarity

COMPONENTS = ["mutations", "fusions", "copy number"]


def profiles(parts: list[str], amp: float, dele: float) -> tuple[pd.DataFrame, dict[str, str]]:
    """lines x aberrations (1 / 0 / NaN) for the chosen components."""
    M, labels, genomic = alterations()
    cl = data.cell_lines().set_index("cell_line")
    blocks, lab = [], {}
    if "mutations" in parts:
        cols = [c for c in genomic if c != "KMT2A_rearranged"]
        blocks.append(M[cols])
        lab.update({c: labels[c] for c in cols})
    if "fusions" in parts and "KMT2A_rearranged" in M.columns:
        F = M[["KMT2A_rearranged"]].copy()
        lab["KMT2A_rearranged"] = "KMT2A-r"
        if "KMT2A_partner" in cl.columns:
            partners = cl["KMT2A_partner"].reindex(F.index).fillna("").astype(str)
            for p in sorted({x for v in partners for x in v.split(";") if x}):
                F[f"fus_{p}"] = partners.str.split(";").map(lambda v, p=p: float(p in v)).where(F["KMT2A_rearranged"].notna())
                lab[f"fus_{p}"] = f"KMT2A::{p}"
        blocks.append(F)
    if "copy number" in parts and has_cn():
        E, elab = cn_events(amp, dele)
        blocks.append(E.reindex(M.index))
        lab.update(elab)
    if not blocks:
        return pd.DataFrame(index=M.index), lab
    X = pd.concat(blocks, axis=1)
    X = X.loc[:, (X > 0).any()]                                  # drop aberrations no line has
    return X, lab


def similar_lines_tab(kp: str = "sim") -> None:
    st.markdown("Which cell lines share a similar genetic make-up? Each line's aberrations (mutations, KMT2A "
                "fusions with their partner, arm-level copy-number gains and losses) are compared; similarity "
                "is the **Jaccard index**: aberrations both lines have ÷ aberrations either has, counting only "
                "data available for both. 1 = identical profiles, 0 = nothing shared.")
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    avail = COMPONENTS if has_cn() else COMPONENTS[:2]
    parts = c1.pills("Compare on", avail, selection_mode="multi", default=avail, key=f"{kp}_parts")
    amp = c2.number_input("Gain if rel. CN ≥", 1.05, 4.0, 1.3, 0.05, key=f"{kp}_amp",
                          disabled="copy number" not in (parts or []))
    dele = c3.number_input("Loss if rel. CN ≤", 0.0, 0.95, 0.7, 0.05, key=f"{kp}_del",
                           disabled="copy number" not in (parts or []))
    if not parts:
        st.info("Pick at least one kind of aberration.")
        return
    X, lab = profiles(parts, amp, dele)
    X = X[X.notna().any(axis=1)]
    if X.shape[1] == 0 or len(X) < 2:
        st.info("Not enough annotated lines or aberrations.")
        return
    S = jaccard_similarity(X)
    A = X.fillna(0).astype(bool)
    n_ab = A.sum(axis=1)

    def shared(a: str, b: str) -> tuple[list[str], list[str], list[str]]:
        both_ann = X.loc[a].notna() & X.loc[b].notna()
        xa, xb = A.loc[a] & both_ann, A.loc[b] & both_ann
        name = lambda m: [lab.get(c, c) for c in m.index[m]]  # noqa: E731
        return name(xa & xb), name(xa & ~xb), name(~xa & xb)

    # ---- one line: its nearest neighbours
    st.subheader("Most similar lines")
    lines = sorted(X.index)
    c4, c5 = st.columns([3, 1], vertical_alignment="bottom")
    q = c4.selectbox("Cell line", lines, key=f"{kp}_line",
                     index=lines.index("MOLM13") if "MOLM13" in lines else 0,
                     format_func=lambda x: f"{x} ({int(n_ab[x])} aberrations)")
    top_n = c5.number_input("Show top", 3, 40, 10, key=f"{kp}_top")
    st.caption("Aberrations of " + q + ": " + (", ".join(lab.get(c, c) for c in A.columns[A.loc[q]]) or "none"))
    s = S.loc[q].drop(q).dropna().sort_values(ascending=False).head(int(top_n))
    if s.empty:
        st.info(f"{q} has no aberration to compare.")
    else:
        rows = []
        for o, v in s.items():
            sh, only_q, only_o = shared(q, o)
            rows.append({"cell_line": o, "similarity": v, "n_shared": len(sh), "shared": ", ".join(sh),
                         f"only_{q}": ", ".join(only_q), "only_other": ", ".join(only_o)})
        nb = pd.DataFrame(rows)
        st.plotly_chart(plots.neighbour_bars(nb), width="stretch")
        st.dataframe(nb, hide_index=True, width="stretch",
                     column_config={"similarity": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                  format="%.2f")})
        st.download_button("Download (TSV)", nb.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                           file_name=f"similar_to_{q}.tsv")

    # ---- all lines: clustered similarity
    st.subheader("All lines, clustered by similarity")
    ok = S.index[S.notna().sum(axis=1) > 1]
    S2 = S.loc[ok, ok]
    D = (1 - S2.fillna(0)).to_numpy(copy=True)
    np.fill_diagonal(D, 0)
    order = ok[leaves_list(linkage(squareform(D, checks=False), "average"))] if len(ok) > 2 else ok
    S2 = S2.loc[order, order]
    Ab = (A & X.notna()).loc[order].to_numpy()
    names = np.array([lab.get(c, c) for c in X.columns], dtype=object)
    hover = np.array([["shared: " + (", ".join(names[Ab[i] & Ab[j]]) or "—") for j in range(len(order))]
                      for i in range(len(order))])
    st.plotly_chart(plots.similarity_heatmap(S2, hover), width="stretch")
    st.caption(f"{len(order)} lines, {X.shape[1]} aberrations; average-linkage clustering on 1 − Jaccard. "
               "Hover a cell for the shared aberrations. Lines with no aberration are left out.")

