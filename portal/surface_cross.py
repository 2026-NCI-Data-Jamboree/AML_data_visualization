"""Highlights → Cell surface proteins → Cell lines vs patients: the measured surface proteome of AML cell lines
(George et al. 2026; surface biotinylation, 19 lines) set against primary AML specimens (Leucegene AML surfaceome,
Cell Reports 2024; 100 patients). Proteins are matched by HGNC symbol.

The same subgroup comparison in both datasets: Welch t per protein in each, then the two log2 fold changes side
by side. A protein is *supported by both* when it changes in the same direction at p < 0.05 in each dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from portal import data, plots, surface_plots as sp
from portal.dependency import selected_genes
from portal.stats import welch
from portal.surfaceome import _flag, annotated_surface, surface_matrix

COHORT, LAYER = "LEUCEGENE_SURFACEOME_2024", "surface"
ALPHA = 0.05
MAX_SHOWN = 3
# name -> (group A label, group B label, cell-line column, patient column): subgroups defined the same way (genomic
# findings) in both datasets
COMPARISONS = {
    "KMT2A-r vs not": ("KMT2A-r", "no KMT2A-r", "KMT2A_rearranged", "KMT2A_rearranged"),
    "TP53-mutant vs not": ("TP53-mutant", "TP53-wild-type", "mut_TP53", "mut_TP53"),
    "FLT3-ITD vs not": ("FLT3-ITD", "no FLT3-ITD", "FLT3_ITD", "FLT3_ITD"),
    "DNMT3A-mutant vs not": ("DNMT3A-mutant", "DNMT3A-wild-type", "mut_DNMT3A", "mut_DNMT3A"),
}


def available() -> bool:
    try:
        return COHORT in data.patient_cohorts() and LAYER in set(data.patient_datasets(COHORT)["key"])
    except Exception:                                   # noqa: BLE001 - no results / unreadable: not offered
        return False


@st.cache_data(show_spinner=False)
def patient_matrix() -> pd.DataFrame:
    """proteins (gene symbol) x patient specimens, log2 intensity; duplicated symbols: the most abundant row."""
    V, f = data.patient_values(COHORT, LAYER), data.patient_features(COHORT, LAYER)
    sym = f["gene_symbol"].reindex(V.index).replace("", np.nan)
    V = V[sym.notna().to_numpy()]
    V = V.assign(_s=sym.dropna().to_numpy(), _m=V.mean(axis=1).to_numpy())
    V = V.sort_values("_m", ascending=False).drop_duplicates("_s").set_index("_s").drop(columns="_m")
    V.index.name = None
    return V


def likely_surface() -> set[str]:
    """UniProt-annotated surface (cell-line data) or SPAT score >= 8 (patient data)."""
    f = data.patient_features(COHORT, LAYER)
    spat = set(f.loc[f.get("likely_surface", pd.Series(dtype=str)).astype(str).str.upper() == "TRUE", "gene_symbol"])
    return annotated_surface() | spat


def _patient_flag(col: str) -> pd.Series:
    """1 / 0 / NaN per specimen."""
    s = data.patient_samples(COHORT).set_index("sample_id")
    if col not in s:
        return pd.Series(np.nan, index=s.index)
    return s[col].astype(str).str.upper().map({"TRUE": 1.0, "FALSE": 0.0})


def _line_flag(col: str, lines: list[str]) -> pd.Series:
    return _flag(col, lines)


def _groups(flag: pd.Series) -> tuple[list[str], list[str]]:
    return sorted(flag.index[flag == 1]), sorted(flag.index[flag == 0])


def comparisons() -> dict[str, tuple]:
    """Comparisons with >= 2 lines and >= 3 patients in each group: name -> (A, B, lines A, lines B, pts A, pts B)."""
    L, P = surface_matrix(), patient_matrix()
    out = {}
    for name, (na, nb, lcol, pcol) in COMPARISONS.items():
        la, lb = _groups(_line_flag(lcol, list(L.columns)))
        pa, pb = _groups(_patient_flag(pcol).reindex(P.columns))
        if min(len(la), len(lb)) >= 2 and min(len(pa), len(pb)) >= 3:
            out[name] = (na, nb, la, lb, pa, pb)
    return out


@st.cache_data(show_spinner="Comparing cell lines and patients…")
def cross(name: str) -> pd.DataFrame:
    """One row per protein in both datasets: log2FC and p in lines and in patients, class, score."""
    na, nb, la, lb, pa, pb = comparisons()[name]
    rl = welch(surface_matrix(), la, lb, min_n=2).set_index("feature")
    rp = welch(patient_matrix(), pa, pb, min_n=3).set_index("feature")
    d = rl[["diff_A_minus_B", "p_welch", "n_A", "n_B"]].join(
        rp[["diff_A_minus_B", "p_welch", "fdr_welch"]], how="inner", lsuffix="_lines", rsuffix="_patients")
    d = d.rename(columns={"diff_A_minus_B_lines": "x", "diff_A_minus_B_patients": "y", "p_welch_lines": "p_lines",
                          "p_welch_patients": "p_patients", "fdr_welch": "q_patients", "n_A": "lines_A", "n_B": "lines_B"})
    same = np.sign(d["x"]) == np.sign(d["y"])
    both = same & (d["p_lines"] < ALPHA) & (d["p_patients"] < ALPHA)
    d["cls"] = np.where(both, np.where(d["x"] > 0, "A", "B"), "ns")
    # combined evidence (Stouffer, direction-aware): signed z per dataset, summed / sqrt(2); opposite directions cancel
    z = lambda p, fc: np.sign(fc) * stats.norm.isf(p.clip(1e-300, 1) / 2)          # noqa: E731
    d["z_combined"] = (z(d["p_lines"], d["x"]) + z(d["p_patients"], d["y"])) / np.sqrt(2)
    d["p_combined"] = 2 * stats.norm.sf(d["z_combined"].abs())
    d["score"] = np.log10(d["p_combined"].clip(lower=1e-300))                         # lower = stronger in both
    return d.rename_axis("gene").reset_index()


def _boxes(protein: str, M: pd.DataFrame, a: list[str], b: list[str], na: str, nb: str, where: str, p: float,
           key: str) -> None:
    long = pd.DataFrame({"cell_line": a + b, "value": M.loc[protein].reindex(a + b).to_numpy(),
                         "group": [na] * len(a) + [nb] * len(b)}).dropna()
    st.plotly_chart(sp.group_box(long, [na, nb], f"{protein} · {where}", "surface abundance (log2)",
                                 f"p = {p:.2g}" if np.isfinite(p) else "too few values for a test",
                                 colors={na: plots.COL_B, nb: plots.COL_A}), width="stretch", key=key)


def comparison_section(kp: str = "sx") -> None:
    comps = comparisons()
    if not comps:
        st.info("No subgroup comparison has enough cell lines and patients in both groups.")
        return
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    name = c1.selectbox("Comparison", list(comps), key=f"{kp}_cmp",
                        format_func=lambda n: f"{n} (lines {len(comps[n][2])} vs {len(comps[n][3])}, patients "
                                              f"{len(comps[n][4])} vs {len(comps[n][5])})")
    surf_only = c2.toggle("Likely surface proteins only", value=False, key=f"{kp}_surf",
                          help="Annotated at the cell surface by UniProt (cell-line data) or SPAT score ≥ 8 "
                               "(patient data); biotinylation also captures some intracellular proteins.")
    na, nb, la, lb, pa, pb = comps[name]
    if name.startswith("TP53"):
        st.caption("Patient TP53 status comes from the Leucegene mutation calls of the same patients (the surfaceome "
                   "study did not test TP53); specimens without those calls are left out. Most TP53-mutant patients "
                   "have a complex karyotype, so in patients this comparison partly reflects complex-karyotype AML.")
    d = cross(name)
    if surf_only:
        d = d[d["gene"].isin(likely_surface())]
    if d.empty:
        st.info("No protein is measured in both datasets for this comparison.")
        return
    m1, m2, m3 = st.columns(3)
    m1.metric("Proteins in both datasets", f"{len(d):,}")
    m2.metric(f"Higher in {na}, both datasets", int((d["cls"] == "A").sum()))
    m3.metric(f"Higher in {nb}, both datasets", int((d["cls"] == "B").sum()))
    style = {"ns": (plots.COL_NS, "not supported by both"),
             "B": (plots.COL_A, f"higher in {nb}, p < {ALPHA} in both"),
             "A": (plots.COL_B, f"higher in {na}, p < {ALPHA} in both")}
    fig = plots.platform_scatter(d.assign(score=np.where(d["cls"] != "ns", d["score"], 0)),
                                 f"cell lines: log2 fold change ({na} − {nb})",
                                 f"patients: log2 fold change ({na} − {nb})", style, no_count=("ns",),
                                 hover=("cell lines log2FC", "patients log2FC"),
                                 n_label=int((d["cls"] != "ns").sum().clip(max=15)))
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points", "box", "lasso"),
                         key=f"{kp}_scatter_{name}_{surf_only}")
    picked = [g for g in selected_genes(ev) if g in set(d["gene"])]
    st.caption("Each point is a protein measured in both datasets. Upper right / lower left = the same direction in "
               "cell lines and patients; coloured = p < 0.05 in both. Click or lasso points to filter the table.")

    table = d.assign(supported=d["cls"] != "ns").sort_values(["supported", "p_combined"], ascending=[False, True])
    table = table.rename(columns={"gene": "protein", "x": "log2FC lines", "y": "log2FC patients",
                                  "p_lines": "p lines", "p_patients": "p patients", "q_patients": "q patients",
                                  "p_combined": "combined p"})
    table["direction"] = np.where(table["z_combined"] > 0, f"higher in {na}", f"higher in {nb}")
    table["likely surface"] = table["protein"].isin(likely_surface())
    table = table[["protein", "supported", "combined p", "direction", "log2FC lines", "p lines", "log2FC patients",
                   "p patients", "q patients", "likely surface"]].reset_index(drop=True)
    view = table[table["protein"].isin(picked)].reset_index(drop=True) if picked else table
    tbl = st.dataframe(view, hide_index=True, width="stretch", height=min(38 + 35 * len(view), 360),
                       on_select="rerun", selection_mode="multi-row", key=f"{kp}_tab_{name}_{surf_only}_{len(view)}",
                       column_config={**{c: st.column_config.NumberColumn(format="%.2f")
                                         for c in ["log2FC lines", "log2FC patients"]},
                                      **{c: st.column_config.NumberColumn(format="%.2g")
                                         for c in ["combined p", "p lines", "p patients", "q patients"]},
                                      "supported": st.column_config.CheckboxColumn("both datasets"),
                                      "likely surface": st.column_config.CheckboxColumn("likely surface")})
    st.caption(f"Ranked by both datasets together: proteins with p < {ALPHA} in the same direction in cell lines and "
               "patients first, then by the combined p (Stouffer's method on the two signed tests, so opposite "
               f"directions weaken it). Then each dataset's own statistics; log2FC = {na} − {nb}; q = FDR within the "
               "patient cohort. Select rows to compare the protein in cell lines and patients below (up to 3).")
    st.download_button("Download table (TSV)", view.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                       file_name=f"surface_lines_vs_patients_{name.replace(' ', '_')}.tsv")
    rows = list(tbl.selection.rows) if tbl and tbl.selection else []
    show = (view.iloc[rows]["protein"].tolist() if rows else picked)[:MAX_SHOWN]
    if not show:
        st.info("Select one or more proteins in the table or on the plot to compare cell lines and patients.")
        return
    L, P = surface_matrix(), patient_matrix()
    res = d.set_index("gene")
    st.markdown(f"**Selected proteins: {na} vs {nb} in cell lines (left) and patients (right)**")
    for prot in show:
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            _boxes(prot, L, la, lb, na, nb, "cell lines", res.loc[prot, "p_lines"], f"{kp}_bl_{name}_{prot}")
        with c2, st.container(border=True):
            _boxes(prot, P, pa, pb, na, nb, "patients", res.loc[prot, "p_patients"], f"{kp}_bp_{name}_{prot}")


def cross_section() -> None:
    if not available():
        st.info("Needs the Leucegene AML surfaceome cohort (LEUCEGENE_SURFACEOME_2024) in results/patients.")
        return
    P = patient_matrix()
    st.markdown("By integrating cell-surface proteomics of AML cell lines and primary patient specimens, this "
                "analysis reveals **cell-surface target candidates for genomic subtypes of AML**: proteins that are "
                "higher (or lower) in a subtype, such as KMT2A-rearranged or TP53-mutant AML, in the same direction in "
                "both the cell lines and the patients.")
    st.caption(f"Data: {surface_matrix().shape[1]} AML cell lines profiled by cell-surface proteomics "
               "(George et al. 2026, bioRxiv, [doi:10.64898/2026.09.16.752145]"
               "(https://doi.org/10.64898/2026.09.16.752145)) and "
               f"{P.shape[1]} primary AML specimens from the Leucegene AML surfaceome (Bordeleau et al. 2024, "
               "*Immunotherapeutic targeting of surfaceome heterogeneity in AML*, Cell Reports 43:114260, "
               "[doi:10.1016/j.celrep.2024.114260](https://doi.org/10.1016/j.celrep.2024.114260)). Proteins are "
               "matched by gene symbol; *likely surface* uses UniProt (cell lines) and the SPAT score ≥ 8 (patients).")
    comparison_section()
