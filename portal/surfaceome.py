"""Surfaceome: surface-protein differences between AML subgroups on the measured cell-surface
proteome of AML cell lines (George et al. 2026, bioRxiv, doi 10.64898/2026.09.16.752145; surface biotinylation,
19 AML lines). Not detected = absent from the surface. Statistics: Welch t per protein, BH FDR."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portal import data, surface_plots as sp
from portal.plots import COL_A, COL_B
from portal.stats import welch

LAYER = "surface_george"


def available() -> bool:
    return LAYER in set(data.datasets()["key"])


@st.cache_data(show_spinner="Loading surface proteome…")
def surface_matrix(min_spc: int = 0) -> pd.DataFrame:
    """proteins (gene symbol) x lines, log2 abundance; NaN = not detected. Duplicated symbols: the row
    measured in most lines."""
    V, f = data.values(LAYER), data.features(LAYER)
    keep = f.index[pd.to_numeric(f.get("spc", 0), errors="coerce").fillna(0) >= min_spc]
    sym = f.loc[keep, "gene_symbol"].replace("", np.nan).dropna()
    V = V.loc[V.index.intersection(sym.index)]
    V = V.assign(_s=sym.reindex(V.index).to_numpy(), _n=V.notna().sum(axis=1).to_numpy())
    V = V.sort_values("_n", ascending=False).drop_duplicates("_s").set_index("_s").drop(columns="_n")
    V.index.name = None
    return V


def annotated_surface() -> set[str]:
    f = data.features(LAYER)
    if "surfaceome" not in f:
        return set()
    return set(f.loc[f["surfaceome"].astype(str).str.upper() == "TRUE", "gene_symbol"])


# ------------------------------------------------------------------ shared pieces ---


def _flag(col: str, lines: list[str]) -> pd.Series:
    """1 / 0 / NaN for a binary sample column (grouping label or TRUE / FALSE)."""
    cl = data.cell_lines().set_index("cell_line")
    if col not in cl:
        return pd.Series(np.nan, index=lines)
    v = cl[col].reindex(lines)
    G = data.groupings()
    if col in G.index and G.loc[col, "kind"] == "binary":
        hit = v.eq(G.loc[col, "true_label"])
    else:
        hit = v.astype(str).str.upper().eq("TRUE")
    return hit.astype(float).where(v.notna())


# ------------------------------------------------------------------ positivity ---


# ------------------------------------------------------------------ 1. highlights ---
def _comparisons(lines: list[str]) -> list[tuple[str, list[str], list[str], str, str]]:
    out = []

    def add(name, a, b, na, nb):
        if len(a) >= 2 and len(b) >= 2:
            out.append((name, sorted(a), sorted(b), na, nb))

    cl = data.cell_lines().set_index("cell_line")
    if "MYC_mTOR_category" in cl:                      # the MYC-high vs mTOR-high split used across the portal
        cat = cl["MYC_mTOR_category"].reindex(lines)
        add("MYC-high vs mTOR-high", list(cat.index[cat == "MYC-High / mTOR-Low"]),
            list(cat.index[cat == "MYC-Low / mTOR-High"]), "MYC-high", "mTOR-high")
    for col, name in [("FLT3_ITD", "FLT3-ITD"), ("KMT2A_rearranged", "KMT2A-r"), ("mut_TP53", "TP53-mutant"),
                      ("mut_NRAS", "NRAS-mutant"), ("mut_KRAS", "KRAS-mutant"), ("mut_NPM1", "NPM1-mutant")]:
        f = _flag(col, lines)
        add(f"{name} vs not", list(f.index[f == 1]), list(f.index[f == 0]), name, f"no {name}")
    d, t = _flag("mut_DNMT3A", lines), _flag("mut_TET2", lines)
    epi = [x for x in lines if d.get(x) == 1 or t.get(x) == 1]
    add("DNMT3A/TET2-mutant vs not", epi, [x for x in lines if x not in epi and pd.notna(t.get(x))],
        "DNMT3A/TET2-mutant", "WT")
    return out


@st.cache_data(show_spinner="Finding the strongest surface-marker differences…")
def highlights() -> tuple[pd.DataFrame, dict]:
    M = surface_matrix()
    surf = annotated_surface()
    rows, per = [], {}
    for name, a, b, na, nb in _comparisons(list(M.columns)):
        r = welch(M, a, b, min_n=2)
        if r.empty:
            continue
        per[name] = (r, a, b, na, nb)
        for x in r.itertuples():
            rows.append({"comparison": name, "protein": x.feature,
                         "direction": f"↑ in {na}" if x.diff_A_minus_B > 0 else f"↓ in {na}",
                         "log2FC": x.diff_A_minus_B, "p": x.p_welch, "q": x.fdr_welch, "lines": f"{x.n_A} vs {x.n_B}",
                         "annotated surface": x.feature in surf})
    F = pd.DataFrame(rows)
    if len(F):
        F["key"] = F["protein"] + " · " + F["comparison"]
    return F, per


def _box(M, protein, a, b, na, nb, subtitle, key):
    long = pd.DataFrame({"cell_line": a + b, "value": M.loc[protein].reindex(a + b).to_numpy(),
                         "group": [na] * len(a) + [nb] * len(b)}).dropna()
    st.plotly_chart(sp.group_box(long, [na, nb], protein, "surface abundance (log2)", subtitle,
                                 colors={na: COL_B, nb: COL_A}), width="stretch", key=key)


def volcano_section(F: pd.DataFrame, per: dict, M: pd.DataFrame, kp: str = "hl") -> None:
    """Interactive volcano for one subgroup comparison, its result table (filtered by the volcano selection),
    and box plots of the table rows picked (else the volcano selection, else the top rows)."""
    from portal.dependency import selected_genes
    names = list(per)
    n_hits = F[F["p"] < 0.01].groupby("comparison").size()
    default = max(names, key=lambda n: (n_hits.get(n, 0), -names.index(n)))   # most findings first
    c1, c2, c4 = st.columns([3, 2, 2], vertical_alignment="bottom")
    name = c1.selectbox("Comparison", names, index=names.index(default), key=f"{kp}_vc",
                        format_func=lambda n: f"{n} ({len(per[n][1])} vs {len(per[n][2])} lines)")
    rule = c2.segmented_control("Significant if", ["q < 0.1", "p < 0.01"], default="p < 0.01",
                                key=f"{kp}_vrule") or "p < 0.01"
    surf_only = c4.toggle("Annotated surface only", value=False, key=f"{kp}_surf",
                          help="UniProt places the protein at the cell surface; biotinylation can also capture abundant "
                               "intracellular proteins (e.g. ribosomal proteins).")
    r, a, b, na, nb = per[name]
    surf = annotated_surface()
    if surf_only:
        r = r[r["feature"].isin(surf)]
    if r.empty:
        st.info("No annotated surface protein was tested in this comparison.")
        return
    sig = (r["fdr_welch"] < 0.1) if rule.startswith("q") else (r["p_welch"] < 0.01)
    fig = sp.volcano(r, set(r.loc[sig, "feature"]), na, nb, sig_by="q" if rule.startswith("q") else "p")
    fig.update_layout(height=480, dragmode="select", clickmode="event+select",
                      title=dict(text=f"<b>{name}</b> · {len(a)} vs {len(b)} lines · {len(r):,} proteins · "
                                      f"{int(sig.sum())} at {rule}", x=0, font=dict(size=13)))
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points", "box", "lasso"),
                         key=f"{kp}_volcano_{name}_{surf_only}")
    picked = [g for g in selected_genes(ev) if g in r["feature"].values]
    st.caption(f"Red / blue = higher / lower in {na} at the chosen threshold; every protein passing it is labelled. "
               "Click a point, or drag a box / lasso around several, to filter the table; double-click to clear.")

    table = pd.DataFrame({
        "protein": r["feature"], "log2FC": r["diff_A_minus_B"], "p": r["p_welch"], "q": r["fdr_welch"],
        "lines": r["n_A"].astype(int).astype(str) + " vs " + r["n_B"].astype(int).astype(str),
        "annotated surface": r["feature"].isin(surf)}).sort_values("p").reset_index(drop=True)
    view = table[table["protein"].isin(picked)].reset_index(drop=True) if picked else table
    tbl = st.dataframe(view, hide_index=True, width="stretch", height=min(38 + 35 * len(view), 360),
                       on_select="rerun", selection_mode="multi-row", key=f"{kp}_tab_{name}_{surf_only}_{len(view)}",
                       column_config={"log2FC": st.column_config.NumberColumn(format="%.2f"),
                                      "p": st.column_config.NumberColumn(format="%.2g"),
                                      "q": st.column_config.NumberColumn(format="%.2g"),
                                      "annotated surface": st.column_config.CheckboxColumn("surface (UniProt)")})
    st.caption("Select rows (tick boxes on the left), or points on the volcano, to plot those proteins below (up to 6). "
               "log2FC = " + na + " − " + nb + " (positive = higher in " + na + ").")
    st.download_button("Download table (TSV)", view.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                       file_name=f"surface_{name.replace(' ', '_').replace('/', '-')}.tsv")
    rows = list(tbl.selection.rows) if tbl and tbl.selection else []
    show = (view.iloc[rows]["protein"].tolist() if rows else picked)[:6]
    if not show:
        st.info("Select one or more proteins in the table or on the volcano plot to see their abundance by group.")
        return
    res = r.set_index("feature")
    for i in range(0, len(show), 3):
        cols = st.columns(3)
        for col, prot in zip(cols, show[i:i + 3]):
            x = res.loc[prot]
            with col, st.container(border=True):
                _box(M, prot, a, b, na, nb, f"{name} · p = {x['p_welch']:.2g}, q = {x['fdr_welch']:.2g}",
                     key=f"{kp}_vb_{name}_{prot}")


def highlights_tab(kp: str = "hl") -> None:
    F, per = highlights()
    if F.empty:
        st.info("No comparison has at least 2 lines per group.")
        return
    M = surface_matrix()
    st.markdown("Surface-protein differences between AML subgroups: a Welch t test per protein (proteins detected in "
                "≥ 2 lines of each group), q = Benjamini–Hochberg FDR within each comparison. Groups of 2–9 lines: "
                "candidates to validate.")
    volcano_section(F, per, M, kp)


# ------------------------------------------------------------------ section (Cell lines tab) ---
def surfaceome_section() -> None:
    """Cell surface proteins: the content of the Highlights page tab."""
    if not available():
        st.info("Needs the surface_george layer in results/cell_lines (prepare_cell_line_data.R with the George 2026 "
                "supplementary files in data/cell_lines/surfaceome_George_2026/).")
        return
    M = surface_matrix()
    st.markdown(f"Surface proteins captured by cell-surface biotinylation in **{M.shape[1]} AML cell lines** "
                f"({M.shape[0]:,} proteins): George et al. 2026, bioRxiv, "
                "[doi:10.64898/2026.09.16.752145](https://doi.org/10.64898/2026.09.16.752145). Values are log2 mean "
                "abundance; *not detected* means not found on the surface.")
    highlights_tab()
