"""Surfaceome helpers for the measured cell-surface
proteome of AML cell lines (George et al. 2026, bioRxiv, doi 10.64898/2026.09.16.752145; surface biotinylation,
19 AML lines). Not detected = absent from the surface. Statistics: Welch t per protein, BH FDR."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portal import data

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


# ------------------------------------------------------------------ section (Cell lines tab) ---
def surfaceome_section() -> None:
    """Cell surface proteins: the content of the Highlights page tab."""
    if not available():
        st.info("Needs the surface_george layer in results/cell_lines (prepare_cell_line_data.R with the George 2026 "
                "supplementary files in data/cell_lines/surfaceome_George_2026/).")
        return
    from portal.surface_cross import cross_section
    cross_section()
