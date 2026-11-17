"""Copy-number alteration landscape: which genomic regions are recurrently amplified or deleted?

The cn layer holds DepMap gene-level relative copy number as log2(relative CN + 1), with each gene's
chromosome and cytoband. Genes are grouped into chromosome arms or cytobands; a line's value for a
region is the median relative CN of its genes, called amplified / deleted against the thresholds.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from portal import data
from portal import genomics_plots as plots

CN = "cn"
CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
BAND = re.compile(r"^(\d+|X|Y)([pq])(\d+)?")


@st.cache_data(show_spinner=False)
def regions() -> pd.DataFrame:
    """feature -> arm ('17p') and band ('17p13'), with a genomic sort key (pter -> qter)."""
    f = data.features(CN)
    loc = f["cytoband"].str.split("|").str[0].str.split("-").str[0].str.strip()
    m = loc.str.extract(BAND)
    m.columns = ["chrom", "arm", "band"]
    m = m[m["chrom"].isin(CHROMS) & m["arm"].notna()].copy()
    b = pd.to_numeric(m["band"], errors="coerce").fillna(0)
    m["arm_region"] = m["chrom"] + m["arm"]
    m["band_region"] = m["arm_region"] + m["band"].fillna("")
    m["key_chrom"] = m["chrom"].map({c: i for i, c in enumerate(CHROMS)})
    m["key_arm"] = (m["arm"] == "q").astype(int)
    m["key_band"] = np.where(m["arm"] == "p", -b, b)          # p arm runs pter (high) -> cen (low)
    return m


@st.cache_data(show_spinner="Summarising copy number by region…")
def region_cn(resolution: str) -> tuple[pd.DataFrame, pd.Series]:
    """lines x regions median relative CN (1 = neutral), in genomic order; genes per region."""
    R = regions()
    V = data.values(CN).reindex(R.index)
    ratio = (2 ** V - 1).clip(lower=0)                        # back from log2(relative CN + 1)
    col = "arm_region" if resolution == "arm" else "band_region"
    med = ratio.groupby(R[col]).median().T
    order = R.drop_duplicates(col).sort_values(["key_chrom", "key_arm", "key_band"])[col]
    return med[order.tolist()], R[col].value_counts()


def copy_number_tab(kp: str = "cn") -> None:
    dsets = data.datasets().set_index("key")
    if CN not in dsets.index or "cytoband" not in data.features(CN).columns:
        st.info("Needs the copy-number layer (cn, with cytobands) in results/cell_lines. "
                "Run prepare_cell_line_data.R with DepMap OmicsCNGene.csv and NCBI gene_info present.")
        return
    cl = data.cell_lines().set_index("cell_line")
    G, GL = data.groupings(), data.grouping_labels()
    st.markdown(f"Which genomic regions are recurrently amplified or deleted? Gene-level relative copy number "
                f"({dsets.loc[CN, 'source']}) is summarised per region as the median over its genes "
                "(1 = no change relative to the line's ploidy), then called amplified or deleted.")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2], vertical_alignment="bottom")
    res = c1.segmented_control("Regions", ["chromosome arm", "cytoband"], default="chromosome arm",
                               key=f"{kp}_res") or "chromosome arm"
    amp = c2.number_input("Amplified if relative CN ≥", 1.05, 4.0, 1.3, 0.05, key=f"{kp}_amp",
                          help="1.5 ≈ one extra copy in a diploid line.")
    dele = c3.number_input("Deleted if relative CN ≤", 0.0, 0.95, 0.7, 0.05, key=f"{kp}_del",
                           help="0.5 ≈ one copy lost in a diploid line.")
    sex = c4.toggle("Include X and Y", value=False, key=f"{kp}_sex",
                    help="Off by default: X/Y copy number mostly reflects the donor's sex.")
    groupings = [c for c in G.index if c in cl.columns]
    order_by = st.selectbox("Order lines by", ["number of altered regions"] + groupings, key=f"{kp}_order",
                            format_func=lambda c: c if c == "number of altered regions" else GL[c])

    med, n_genes = region_cn("arm" if res == "chromosome arm" else "band")
    if not sex:
        med = med[[r for r in med.columns if not r.startswith(("X", "Y"))]]
    med = med.dropna(how="all")
    is_amp, is_del = med >= amp, med <= dele
    n_ok = med.notna().sum()
    pct_amp, pct_del = 100 * is_amp.sum() / n_ok, 100 * is_del.sum() / n_ok

    burden = (is_amp | is_del).sum(axis=1)
    o = pd.DataFrame({"burden": burden})
    if order_by != "number of altered regions":
        o["g"] = cl[order_by].reindex(o.index).astype("string")
        lines = o.sort_values(["g", "burden"], ascending=[True, False], na_position="last").index
    else:
        lines = o.sort_values("burden", ascending=False).index
    L = np.log2(med.loc[lines].clip(lower=1 / 64))
    if order_by != "number of altered regions":
        L.index = [f"{x} · {cl.loc[x, order_by] if pd.notna(cl.loc[x, order_by]) else 'NA'}" for x in L.index]

    m1, m2, m3 = st.columns(3)
    m1.metric("Lines", len(med))
    m2.metric("Regions", med.shape[1])
    m3.metric("Median altered regions per line", f"{burden.median():.0f}")
    st.plotly_chart(plots.cnv_heatmap(L, pct_amp, pct_del), width="stretch")
    st.caption("Top: % of lines with the region amplified (red) or deleted (blue). "
               "Bottom: log2 relative copy number per line and region; gray = no data.")

    st.subheader("Recurrent regions")
    tab = pd.DataFrame({"region": med.columns, "genes": n_genes.reindex(med.columns).to_numpy(),
                        "pct_amplified": pct_amp.to_numpy(), "pct_deleted": pct_del.to_numpy(),
                        "lines_amplified": [", ".join(is_amp.index[is_amp[r]]) for r in med.columns],
                        "lines_deleted": [", ".join(is_del.index[is_del[r]]) for r in med.columns]})
    tab = tab.assign(_m=tab[["pct_amplified", "pct_deleted"]].max(axis=1)).sort_values("_m", ascending=False)
    tab = tab.drop(columns="_m")
    st.dataframe(tab, hide_index=True, width="stretch", height=360,
                 column_config={c: st.column_config.NumberColumn(format="%.0f%%")
                                for c in ["pct_amplified", "pct_deleted"]})
    st.download_button("Download table (TSV)", tab.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                       file_name=f"cnv_recurrence_{res.replace(' ', '_')}.tsv")
