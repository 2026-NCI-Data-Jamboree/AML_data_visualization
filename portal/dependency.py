"""Gene dependency: CRISPR (Chronos) and RNAi (DEMETER2) side by side, gene by gene.

Both scores are 'lower = more dependent'. Genes are matched across the two screens by HGNC symbol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portal import data, plots
from portal.stats import welch

CRISPR, RNAI = "crispr", "rnai"
MAX_GENES = 6   # per-line plots drawn for a selection
MEAN, CUSTOM = "All AML lines: mean dependency", "Custom grouping…"
# featured comparisons: label -> (grouping column, group A, group B); shown when present in the results
PRESETS = {
    "MYC-high vs mTOR-high": ("MYC_mTOR_category", "MYC-High / mTOR-Low", "MYC-Low / mTOR-High"),
    "mTOR-high vs MYC-high": ("MYC_mTOR_category", "MYC-Low / mTOR-High", "MYC-High / mTOR-Low"),
    "TP53-mutant vs TP53-wild-type": ("mut_TP53", "TP53-mutant", "TP53-wild-type"),
    "KMT2A-rearranged vs KMT2A wild-type": ("KMT2A_rearranged", "KMT2A-r", "KMT2A wild-type"),
}


@st.cache_data(show_spinner="Matching CRISPR and RNAi genes…")
def _by_symbol(key: str) -> pd.DataFrame:
    """Layer values indexed by HGNC symbol (genes x cell lines)."""
    V = data.values(key)
    sym = data.features(key)["gene_symbol"].reindex(V.index)
    V = V[(sym != "").to_numpy() & sym.notna().to_numpy()]
    V.index = sym[V.index].to_numpy()
    return V[~V.index.duplicated()]


def _corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    ok = pd.concat([x, y], axis=1).dropna()
    if len(ok) < 3 or ok.iloc[:, 0].std() == 0 or ok.iloc[:, 1].std() == 0:
        return np.nan, len(ok)
    return float(ok.iloc[:, 0].corr(ok.iloc[:, 1])), len(ok)


def gene_dependency(kp: str = "dep", fixed: str | None = None) -> None:
    """fixed: a PRESETS comparison to show without the comparison menu (used by Highlights)."""
    dsets = data.datasets().set_index("key")
    if not {CRISPR, RNAI} <= set(dsets.index):
        st.info("This view needs both CRISPR and RNAi screens, which are not available.")
        return
    cl = data.cell_lines()
    G, GL = data.groupings(), data.grouping_labels()
    C_all, R_all = _by_symbol(CRISPR), _by_symbol(RNAI)
    genes = C_all.index.intersection(R_all.index)
    shared = [c for c in C_all.columns if c in R_all.columns]
    xt = f"CRISPR · {dsets.loc[CRISPR, 'unit']}"
    yt = f"RNAi · {dsets.loc[RNAI, 'unit']}"

    st.markdown(
        f"CRISPR knockout ({dsets.loc[CRISPR, 'source']}) and RNAi knockdown ({dsets.loc[RNAI, 'source']}) "
        f"in one view: **{len(genes):,} genes** screened by both, **{len(shared)} lines** screened by both. "
        "Lower scores mean stronger dependency in both screens, so genes in the lower-left quadrant "
        "are supported by both technologies.")

    # comparison menu: overall view, the featured AML comparisons, then any grouping
    presets = {k: v for k, v in PRESETS.items()
               if v[0] in G.index and {v[1], v[2]} <= set(cl[v[0]].dropna().astype(str))}
    options = [MEAN] + list(presets) + [CUSTOM]
    c1, c2 = st.columns([3, 2])
    if fixed is not None:
        if fixed not in presets:
            st.info(f"{fixed} is not available in the processed cell-line data.")
            return
        comparison = fixed
        g, a_lab, b_lab = presets[fixed]
        n = cl[g].astype(str).value_counts()
        c1.markdown(f"**{a_lab}** (n={n.get(a_lab, 0)}, blue) vs **{b_lab}** (n={n.get(b_lab, 0)}, red)")
    else:
        comparison = c1.selectbox("Comparison", options, key=f"{kp}_comparison",
                                  on_change=lambda: st.session_state.pop(f"{kp}_gene", None))
    only_shared = c2.toggle(f"Only the {len(shared)} lines in both screens", key=f"{kp}_shared",
                            help="Off: each screen uses all its lines (CRISPR "
                                 f"{C_all.shape[1]}, RNAi {R_all.shape[1]}).")
    C, R = C_all.loc[genes], R_all.loc[genes]
    if only_shared:
        C, R = C[shared], R[shared]

    def levels_of(col: str) -> list[str]:
        lv = list(cl[col].dropna().astype(str).value_counts().index)
        g = G.loc[col]
        return [x for x in [g["true_label"], g["false_label"]] if x in lv] if g["kind"] == "binary" else lv

    if comparison == MEAN:
        grouping = None
    elif comparison in presets:
        grouping, ga, gb = presets[comparison]
    else:
        a0, a1, a2 = st.columns([3, 2, 2])
        grouping = a0.selectbox("Grouping", list(GL), format_func=GL.get, key=f"{kp}_grouping")
        lv = levels_of(grouping)
        counts = cl[grouping].dropna().astype(str).value_counts()
        d_a, d_b = data.default_pair(grouping, lv)
        fmt = lambda v: f"{v} (n={counts[v]})"  # noqa: E731
        ga = a1.selectbox("Group A (blue)", lv, index=lv.index(d_a) if d_a in lv else 0,
                          format_func=fmt, key=f"{kp}_a")
        rest = [o for o in lv if o != ga] + ["all other lines"]
        gb = a2.selectbox("Group B (red)", rest, index=rest.index(d_b) if d_b in rest else 0,
                          format_func=lambda v: v if v == "all other lines" else fmt(v), key=f"{kp}_b")

    if grouping is None:
        thr = st.slider("Dependency threshold (applied to both scores)", -1.5, 0.0, -0.5, 0.05,
                        key=f"{kp}_thr")
        df = pd.DataFrame({"gene": genes, "x": C.mean(axis=1).to_numpy(), "y": R.mean(axis=1).to_numpy(),
                           "n_crispr": C.notna().sum(axis=1).to_numpy(),
                           "n_rnai": R.notna().sum(axis=1).to_numpy()}).dropna(subset=["x", "y"])
        dx, dy = df["x"] < thr, df["y"] < thr
        df["cls"] = np.select([dx & dy, dx, dy], ["both", "crispr", "rnai"], "ns")
        df["score"] = df["x"] + df["y"]
        style = {"ns": (plots.COL_NS, "neither"), "rnai": (plots.TINT_B, "RNAi only"),
                 "crispr": (plots.TINT_A, "CRISPR only"), "both": (plots.COL_B, "both screens")}
        r, n = _corr(df["x"], df["y"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Genes", f"{len(df):,}")
        m2.metric(f"Dependent in both (< {thr})", int((df["cls"] == "both").sum()))
        m3.metric("CRISPR only / RNAi only", f"{int((df['cls'] == 'crispr').sum())} / "
                                             f"{int((df['cls'] == 'rnai').sum())}")
        m4.metric("Pearson r (gene means)", f"{r:.2f}")
        x_title, y_title, thresholds = f"mean {xt}", f"mean {yt}", (thr, thr)
        table = df.rename(columns={"x": "mean_crispr", "y": "mean_rnai"}).drop(columns="score")
        table["class"] = table.pop("cls").map({k: v[1] for k, v in style.items()})
        table = table.sort_values("mean_crispr")
        file_tag = "mean"
    else:
        grp_col = cl.set_index("cell_line")[grouping].astype("string")
        lines_a = grp_col.index[grp_col == ga].tolist()
        lines_b = (grp_col.index[grp_col.notna() & (grp_col != ga)] if gb == "all other lines"
                   else grp_col.index[grp_col == gb]).tolist()
        rc, rr = welch(C, lines_a, lines_b), welch(R, lines_a, lines_b)
        if rc.empty or rr.empty:
            n_r = [sum(x in R.columns for x in lines) for lines in (lines_a, lines_b)]
            st.warning(f"{ga} vs {gb}: each group needs ≥ 3 lines in both screens; the RNAi screen has "
                       f"{n_r[0]} vs {n_r[1]}. Try another comparison, or switch off "
                       "'Only the lines in both screens'.")
            return
        file_tag = f"{grouping}_{ga}_vs_{gb}".replace(" ", "_").replace("/", "-")
        cols = ["feature", "n_A", "n_B", "diff_A_minus_B", "p_welch", "fdr_welch"]
        df = rc[cols].merge(rr[cols], on="feature", suffixes=("_crispr", "_rnai")).rename(
            columns={"feature": "gene", "diff_A_minus_B_crispr": "x", "diff_A_minus_B_rnai": "y"})
        # colour by agreement in direction only (no significance threshold); lower = more dependent
        same = np.sign(df["x"]) == np.sign(df["y"])
        df["cls"] = np.select([same & (df["x"] < 0), same & (df["x"] > 0)], ["A", "B"], "ns")
        df["score"] = np.where(same, np.log10(df["p_welch_crispr"]) + np.log10(df["p_welch_rnai"]), 0)
        style = {"ns": (plots.COL_NS, "screens disagree"), "B": (plots.COL_B, f"more dependent in {gb}, both screens"),
                 "A": (plots.COL_A, f"more dependent in {ga}, both screens")}
        m1, m2 = st.columns(2)
        m1.metric("Lines A / B (CRISPR · RNAi)", f"{int(df['n_A_crispr'].max())}/{int(df['n_B_crispr'].max())}"
                                                 f" · {int(df['n_A_rnai'].max())}/{int(df['n_B_rnai'].max())}")
        m2.metric("Genes compared", f"{len(df):,}")
        st.caption("Welch t test per screen (p and FDR in the table). Colour = direction only: both screens agree "
                   "the gene is more needed in group A (blue) or B (red). Negative difference = more dependent in "
                   "group A. Labels: the strongest genes by combined p in the two screens.")
        x_title, y_title, thresholds = f"CRISPR: mean {ga} − mean {gb}", f"RNAi: mean {ga} − mean {gb}", None
        table = df.rename(columns={"x": "diff_crispr", "y": "diff_rnai"})
        table["agree"] = table["cls"] != "ns"
        table = table.sort_values(["agree", "score"], ascending=[False, True])   # agreeing genes first, strongest first
        table["class"] = table.pop("cls").map({k: v[1] for k, v in style.items()})
        table = table.drop(columns=["score", "agree"])

    st.caption("Click a gene, or drag a box / lasso around several (toolbar, top right), to compare "
               "them line by line below. Double-click the plot to clear the selection.")
    event = st.plotly_chart(plots.platform_scatter(df, x_title, y_title, style, thresholds=thresholds,
                                                   no_count=("ns",) if grouping is not None else ()),
                            width="stretch", on_select="rerun", selection_mode=("points", "box", "lasso"),
                            key=f"{kp}_scatter_{file_tag}")
    selected = selected_genes(event)

    if selected:
        view = table[table["gene"].isin(selected)]
        st.markdown(f"**{len(selected)} selected gene{'s' if len(selected) > 1 else ''}**")
    else:
        view = table
    tbl = st.dataframe(view, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(view)),
                       on_select="rerun", selection_mode="multi-row", key=f"{kp}_table_{file_tag}_{len(view)}",
                       column_config={c: st.column_config.NumberColumn(format="%.3g") for c in view.columns
                                      if c.startswith(("mean_", "diff_", "p_", "fdr_"))})
    st.caption("Select rows in the table (tick boxes on the left) to plot those genes below; "
               "otherwise the plot selection, or the top row, is shown.")
    st.download_button("Download table (TSV)", view.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                       file_name=f"crispr_vs_rnai_{file_tag}.tsv")

    # per-line agreement for the selected genes (default: the top gene); only lines in both screens
    if grouping is None:                       # overall view: colour by any grouping
        col = st.selectbox("Colour lines by", list(GL), format_func=GL.get, key=f"{kp}_color",
                           index=list(GL).index("mTOR_high") if "mTOR_high" in GL else 0)
        grp = cl.set_index("cell_line")[col].astype("string").fillna("not annotated")
        order = levels_of(col) + ["not annotated"]
    else:                                      # comparison: blue = A, red = B, as in the scatter
        grp = pd.Series("not compared", index=cl["cell_line"])
        grp[lines_b], grp[lines_a] = gb, ga
        order = [ga, gb, "not compared"]

    rows = list(tbl.selection.rows) if tbl and tbl.selection else []
    picked = view.iloc[rows]["gene"].tolist() if rows else (selected or view["gene"].head(1).tolist())
    if not picked:
        return
    shown = picked[:MAX_GENES]
    st.subheader("CRISPR vs RNAi per cell line" + ("" if (rows or selected) else f": {shown[0]} (top row)"))
    if len(picked) > MAX_GENES:
        st.caption(f"Showing the first {MAX_GENES} of {len(picked)} chosen genes.")
    grid = st.columns(1 if len(shown) == 1 else 2)
    for i, gene in enumerate(shown):
        long = pd.DataFrame({"cell_line": shared, "x": C_all.loc[gene, shared].to_numpy(),
                             "y": R_all.loc[gene, shared].to_numpy()}).dropna()
        long["group"] = long["cell_line"].map(grp)
        r, n = _corr(long["x"], long["y"])
        stat = f"{n} lines · r = {r:.2f}" if np.isfinite(r) else f"{n} lines"
        grid[i % len(grid)].plotly_chart(
            plots.line_scatter(long, "group", order, xt, yt, title=f"{gene}  ({stat})"),
            width="stretch", key=f"{kp}_lines_{gene}")


def selected_genes(event) -> list[str]:
    """Genes picked on the scatter (click, box or lasso), in selection order, without duplicates."""
    try:
        points = event.selection.points
    except AttributeError:
        try:
            points = event["selection"]["points"]
        except (KeyError, TypeError):
            return []
    genes = [p.get("customdata") for p in points or []]
    genes = [g[0] if isinstance(g, (list, tuple)) else g for g in genes]
    return list(dict.fromkeys(g for g in genes if isinstance(g, str) and g))
