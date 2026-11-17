"""Surfaceome: highlights, marker expression, and multi-antigen targeting on the measured cell-surface
proteome of AML cell lines (George et al. 2026, bioRxiv, doi 10.64898/2026.09.16.752145; surface biotinylation,
19 AML lines). Not detected = absent from the surface. Statistics: Welch t per protein, BH FDR."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as ss

from portal import data, surface_plots as sp
from portal.plots import COL_A, COL_B
from portal.stats import welch

LAYER = "surface_george"
PANEL = ["CD33", "IL3RA", "CLEC12A", "CD70", "CD47", "FLT3", "KIT", "CD38", "CD34", "LILRB4", "ITGAM", "CD44",
         "FCGR1A", "CD4", "CD96", "HAVCR2", "IL10RB", "SIGLEC6"]
ALIASES = "CD123 = IL3RA, CLL-1 = CLEC12A, CD117 = KIT, CD135 = FLT3, CD11b = ITGAM, CD64 = FCGR1A, TIM-3 = HAVCR2"
LSC17 = ["DNMT3B", "ZBTB46", "NYNRIN", "ARHGAP22", "LAPTM4B", "MMRN1", "DPYSL3", "FAM30A", "CDK6", "CPXM1", "SOCS2",
         "SMIM24", "EMP1", "BEX3", "CD34", "AKR1C3", "ADGRG1"]
RULES = ["detected", "above percentile", "above value"]
FEATURED = ["IL10RB · FLT3-ITD vs not", "RPS3 · FLT3-ITD vs not"]


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
def show_groups(a: list[str], b: list[str], na: str, nb: str, colours: tuple[str, str] = ("red", "blue")) -> None:
    """Which cell lines are in each group (colours match the plots of the calling tab)."""
    with st.container(border=True):
        c1, c2 = st.columns(2)
        for col, lines, name, colour in ((c1, a, na, colours[0]), (c2, b, nb, colours[1])):
            col.markdown(f":{colour}[**{name}**] · {len(lines)} lines")
            col.caption(", ".join(sorted(lines)) or "none")


def _test(long: pd.DataFrame, order: list[str]) -> str:
    vals = [long.loc[long["group"] == g, "value"].dropna() for g in order]
    vals = [v for v in vals if len(v) >= 2]
    if len(vals) == 2:
        return f"Welch t test p = {ss.ttest_ind(vals[0], vals[1], equal_var=False).pvalue:.2g}"
    if len(vals) > 2:
        return f"Kruskal–Wallis p = {ss.kruskal(*vals).pvalue:.2g}"
    return ""


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


def colour_options() -> tuple[list[str], dict[str, str]]:
    G = data.groupings()
    cl = data.cell_lines()
    cols = [c for c in G.index if c in cl.columns]
    return ["none"] + cols, {"none": "none", **G["label"].to_dict()}


def colour_values(by: str, lines) -> pd.Series:
    if by == "none":
        return pd.Series("all lines", index=lines)
    return data.cell_lines().set_index("cell_line")[by].reindex(lines).astype("string").fillna("not annotated").astype(str)


def _order(g: pd.Series) -> list[str]:
    return sorted(set(g) - {"not annotated"}, key=lambda x: (-(g == x).sum(), x)) + \
        (["not annotated"] if (g == "not annotated").any() else [])


# ------------------------------------------------------------------ positivity ---
def positivity_picker(M: pd.DataFrame, kp: str = "su") -> tuple[str, float]:
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    rule = c1.segmented_control("A line is positive for a marker if", RULES, default=RULES[0], key=f"{kp}_rule") \
        or RULES[0]
    if rule == "above percentile":
        cut = c2.slider("Percentile of measured lines", 0, 90, 50, 10, key=f"{kp}_pct")
    elif rule == "above value":
        cut = c2.number_input("Value (log2)", value=round(float(np.nanmedian(M.to_numpy())), 1), step=0.5,
                              key=f"{kp}_val")
    else:
        cut = 0.0
    return rule, float(cut)


def positive(v: pd.Series, rule: str, cut: float) -> pd.Series:
    d = v.dropna()
    if rule == "detected":
        return v.notna()
    t = float(np.percentile(d, cut)) if rule == "above percentile" and len(d) else cut
    return (v >= t).fillna(False)


def positive_matrix(M: pd.DataFrame, rule: str, cut: float) -> pd.DataFrame:
    return pd.DataFrame({g: positive(M.loc[g], rule, cut) for g in M.index})


def marker_summary(M: pd.DataFrame, rule: str, cut: float) -> pd.DataFrame:
    pos = positive_matrix(M, rule, cut)
    return pd.DataFrame({"pct_positive": 100 * pos.mean(), "n_positive": pos.sum(), "n_measured": M.notna().sum(axis=1),
                         "median": M.median(axis=1), "sd": M.std(axis=1),
                         "fold_range": 2 ** (M.max(axis=1) - M.min(axis=1))})


# ------------------------------------------------------------------ 1. highlights ---
def _comparisons(lines: list[str]) -> list[tuple[str, list[str], list[str], str, str]]:
    out = []

    def add(name, a, b, na, nb):
        if len(a) >= 2 and len(b) >= 2:
            out.append((name, sorted(a), sorted(b), na, nb))

    for col, name in [("FLT3_ITD", "FLT3-ITD"), ("KMT2A_rearranged", "KMT2A-r"), ("mut_TP53", "TP53-mutant"),
                      ("mut_NRAS", "NRAS-mutant"), ("mut_KRAS", "KRAS-mutant"), ("mut_NPM1", "NPM1-mutant")]:
        f = _flag(col, lines)
        add(f"{name} vs not", list(f.index[f == 1]), list(f.index[f == 0]), name, f"no {name}")
    d, t = _flag("mut_DNMT3A", lines), _flag("mut_TET2", lines)
    epi = [x for x in lines if d.get(x) == 1 or t.get(x) == 1]
    add("DNMT3A/TET2-mutant vs not", epi, [x for x in lines if x not in epi and pd.notna(t.get(x))],
        "DNMT3A/TET2-mutant", "WT")
    if "expression" in set(data.datasets()["key"]):
        V, f = data.values("expression"), data.features("expression")
        E = V[f["gene_symbol"].reindex(V.index).isin(LSC17).to_numpy()]
        z = E.sub(E.mean(axis=1), axis=0).div(E.std(axis=1), axis=0).mean().reindex(lines).dropna()
        add("LSC17-high vs low", list(z.index[z > z.median()]), list(z.index[z <= z.median()]), "LSC17-high", "LSC17-low")
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


def highlights_tab(kp: str = "hl") -> None:
    F, per = highlights()
    if F.empty:
        st.info("No comparison has at least 2 lines per group.")
        return
    M = surface_matrix()
    st.markdown("The strongest surface-protein differences between the main AML subgroups: a Welch t test per protein "
                "(proteins detected in ≥ 2 lines of each group), q = Benjamini–Hochberg FDR within each comparison.")

    st.subheader("Featured findings")
    cand = F[F["p"] < 0.05].sort_values("p")
    featured = st.multiselect("Pinned observations", list(cand["key"]), max_selections=6, key=f"{kp}_feat",
                              default=[k for k in FEATURED if k in set(cand["key"])],
                              placeholder="Add a protein · comparison (p < 0.05)")
    for i in range(0, len(featured), 3):
        cols = st.columns(3)
        for col, k in zip(cols, featured[i:i + 3]):
            x = F[F["key"] == k].iloc[0]
            r, a, b, na, nb = per[x["comparison"]]
            with col, st.container(border=True):
                _box(M, x["protein"], a, b, na, nb, f"{x['comparison']} · p = {x['p']:.2g}, q = {x['q']:.2g}"
                     + (" · surface (UniProt)" if x["annotated surface"] else " · not annotated surface"),
                     key=f"{kp}_f_{k}")

    st.subheader("Top findings")
    c1, c2, c3 = st.columns([2, 2, 3], vertical_alignment="bottom")
    rule = c1.segmented_control("Finding if", ["q < 0.1", "p < 0.01"], default="q < 0.1", key=f"{kp}_rule") or "q < 0.1"
    surf_only = c2.toggle("Annotated surface only", value=False, key=f"{kp}_surf",
                          help="UniProt places the protein at the cell surface; biotinylation can also capture abundant "
                               "intracellular proteins (e.g. ribosomal proteins).")
    hits = F[(F["q"] < 0.1) if rule.startswith("q") else (F["p"] < 0.01)]
    if surf_only:
        hits = hits[hits["annotated surface"]]
    hits = hits.sort_values(["q", "p"])
    c3.caption(f"**{len(hits)} findings** in {hits['comparison'].nunique()} of {len(per)} comparisons. Groups of 2–9 "
               "lines: candidates to validate.")
    if hits.empty:
        st.info("Nothing passes this threshold; try p < 0.01.")
    else:
        show = hits.drop(columns="key").head(25).reset_index(drop=True)
        ev = st.dataframe(show, hide_index=True, width="stretch", height=min(38 + 35 * len(show), 460), on_select="rerun",
                          selection_mode="multi-row", key=f"{kp}_tab",
                          column_config={"log2FC": st.column_config.NumberColumn(format="%.2f"),
                                         "p": st.column_config.NumberColumn(format="%.2g"),
                                         "q": st.column_config.NumberColumn(format="%.2g"),
                                         "annotated surface": st.column_config.CheckboxColumn("surface (UniProt)")})
        rows = (ev.selection.rows if ev and ev.selection else []) or [0]
        picks = [show.iloc[r] for r in rows if r < len(show)][:9]
        st.caption("Select one or more findings to plot them (up to 9).")
        for i in range(0, len(picks), 3):
            cols = st.columns(3)
            for col, x in zip(cols, picks[i:i + 3]):
                r, a, b, na, nb = per[x["comparison"]]
                with col, st.container(border=True):
                    _box(M, x["protein"], a, b, na, nb, f"{x['comparison']} · p = {x['p']:.2g}, q = {x['q']:.2g}",
                         key=f"{kp}_b_{x['comparison']}_{x['protein']}")
        for name in dict.fromkeys(x["comparison"] for x in picks):
            r, a, b, na, nb = per[name]
            show_groups(a, b, na, nb)

    st.subheader("Key volcano plots")
    pinned = list(dict.fromkeys(F.loc[F["key"].isin(featured), "comparison"]))
    top_cmp = list(dict.fromkeys(pinned + hits["comparison"].value_counts().index.tolist()))[:4]
    for i in range(0, len(top_cmp), 2):
        cols = st.columns(2)
        for col, name in zip(cols, top_cmp[i:i + 2]):
            r, a, b, na, nb = per[name]
            hl = set(hits.loc[hits["comparison"] == name, "protein"].head(6)) | \
                set(F.loc[F["key"].isin(featured) & (F["comparison"] == name), "protein"])
            fig = sp.volcano(r, hl, na, nb, sig_by="q" if rule.startswith("q") else "p")
            fig.update_layout(title=dict(text=f"<b>{name}</b> · {len(a)} vs {len(b)} lines", x=0, font=dict(size=13)))
            with col, st.container(border=True):
                st.plotly_chart(fig, width="stretch", key=f"{kp}_v_{name}")
    st.caption("Featured comparisons first, then those with the most findings. Red / blue = higher / lower in the first "
               "group at the chosen threshold; ringed = featured and top findings. Groups of 2 lines give large, "
               "unstable fold changes.")


# ------------------------------------------------------------------ 2. marker expression ---
def marker_tab(M: pd.DataFrame, rule: str, cut: float, kp: str = "sm") -> None:
    markers = sorted(M.index)
    st.markdown("Which lines carry a surface marker, how often it is present across lines (frequency), and how much it "
                "varies between them (heterogeneity).")
    c1, c2 = st.columns(2)
    marker = c1.selectbox("Marker", markers, index=markers.index("CD33") if "CD33" in markers else 0, key=f"{kp}_m",
                          help=ALIASES)
    opts, labels = colour_options()
    by = c2.selectbox("Colour by", opts, format_func=labels.get, key=f"{kp}_by",
                      index=opts.index("KMT2A_rearranged") if "KMT2A_rearranged" in opts else 0)
    v = M.loc[marker]
    det, nd = v.dropna(), list(v.index[v.isna()])
    pos = positive(v, rule, cut)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Positive lines", f"{int(pos.sum())} / {len(v)}", f"{100 * pos.mean():.0f}% frequency", delta_color="off")
    m2.metric("Median", f"{det.median():.2f}" if len(det) else "—")
    m3.metric("Spread (SD)", f"{det.std():.2f}" if len(det) > 1 else "—")
    m4.metric("Fold range", f"{2 ** (det.max() - det.min()):.0f}×" if len(det) > 1 else "—")
    if len(det):
        g = colour_values(by, det.index)
        st.plotly_chart(sp.rank_bars(pd.DataFrame({"cell_line": det.index, "value": det.to_numpy(), "group": g.to_numpy()}),
                                     f"{marker} · surface abundance (log2)", _order(g)), width="stretch")
    if nd:
        st.caption(f"Not detected in {len(nd)} lines: {', '.join(nd)}.")

    st.subheader("Marker panel")
    panel = st.multiselect("Markers", markers, default=[m for m in PANEL if m in markers][:15], key=f"{kp}_panel")
    if panel:
        g = colour_values(by, M.columns)
        lines = sorted(M.columns, key=lambda x: (_order(g).index(g[x]), x))
        sub = M.loc[panel, lines]
        Z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
        st.plotly_chart(sp.panel_heatmap(Z, g.reindex(lines)), width="stretch")
        st.caption("z-score per marker across lines (red = high); blank = not detected on the surface.")

    st.subheader("Frequency and heterogeneity")
    scope = st.segmented_control("Markers", ["panel", "all markers"], default="panel", key=f"{kp}_scope") or "panel"
    S = marker_summary(M.loc[panel if scope == "panel" and panel else list(M.index)], rule, cut)
    S = S[S["n_measured"] >= 3]
    if S.empty:
        st.info("No marker measured in at least 3 lines.")
        return
    c3, c4 = st.columns([3, 2])
    with c3:
        st.plotly_chart(sp.frequency_bars(S.sort_values("pct_positive", ascending=False).head(40)), width="stretch")
        st.caption("% of lines positive (at most 40 shown); colour = SD of log2 values across measured lines.")
    with c4:
        st.plotly_chart(sp.freq_vs_het(S, {marker} | set(panel[:8])), width="stretch")
        st.caption("Bottom right = on most lines at a similar level; top = present but variable between lines.")
    st.dataframe(S.sort_values("pct_positive", ascending=False).rename_axis("marker").reset_index(), hide_index=True,
                 width="stretch", height=300,
                 column_config={"pct_positive": st.column_config.ProgressColumn("% positive", min_value=0, max_value=100,
                                                                                format="%.0f"),
                                **{c: st.column_config.NumberColumn(format="%.2f") for c in ["median", "sd"]},
                                "fold_range": st.column_config.NumberColumn("fold range", format="%.0f×")})


# ------------------------------------------------------------------ 3. multi-antigen ---
def multi_antigen_tab(M: pd.DataFrame, rule: str, cut: float, kp: str = "ma") -> None:
    markers = sorted(M.index)
    st.markdown("**All (AND)**: lines carrying every chosen antigen, e.g. for a logic-gated CAR or a bispecific that "
                "needs both. **Any (OR)**: lines carrying at least one, the coverage of dual targeting against "
                "heterogeneous antigen loss.")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    chosen = c1.multiselect("Antigens (2–4)", markers, max_selections=4, key=f"{kp}_set", help=ALIASES,
                            default=[g for g in ["CD33", "IL3RA", "CLEC12A"] if g in markers])
    logic = c2.segmented_control("Lines with", ["all (AND)", "any (OR)"], default="any (OR)", key=f"{kp}_logic") \
        or "any (OR)"
    if len(chosen) >= 2:
        pos = positive_matrix(M.loc[chosen], rule, cut)
        hit = pos.all(axis=1) if logic.startswith("all") else pos.any(axis=1)
        m1, m2, m3 = st.columns(3)
        m1.metric("Lines covered", f"{int(hit.sum())} / {len(hit)}", f"{100 * hit.mean():.0f}%", delta_color="off")
        m2.metric("All antigens", int(pos.all(axis=1).sum()))
        m3.metric("None of them", int((~pos.any(axis=1)).sum()))
        pat = pos[chosen].apply(tuple, axis=1)
        c = pat.value_counts().rename_axis("pattern").reset_index(name="count")
        c = c.assign(n_on=c["pattern"].map(sum)).sort_values(["count", "n_on"], ascending=False).reset_index(drop=True)
        c["name"] = [" + ".join(m for m, v in zip(chosen, p) if v) or "none of them" for p in c["pattern"]]
        st.plotly_chart(sp.upset(c, chosen, pos.sum()), width="stretch")
        st.caption("Each bar = lines with exactly that combination of positive antigens (dots).")
        st.markdown(f"**Covered** ({int(hit.sum())}): {', '.join(pos.index[hit]) or 'none'}  \n"
                    f"**Not covered**: {', '.join(pos.index[~hit]) or 'none'}")

    st.subheader("Best combinations from a panel")
    c3, c4 = st.columns([3, 1], vertical_alignment="bottom")
    panel = c3.multiselect("Candidate antigens", markers, default=[m for m in PANEL if m in markers][:12], key=f"{kp}_panel")
    size = c4.segmented_control("Combine", [2, 3], default=2, key=f"{kp}_size",
                                format_func=lambda k: "pairs" if k == 2 else "triples") or 2
    if len(panel) >= size:
        pos = positive_matrix(M.loc[panel], rule, cut)
        tab = pd.DataFrame([{"antigens": " + ".join(cmb), "any (OR) %": 100 * pos[list(cmb)].any(axis=1).mean(),
                             "all (AND) %": 100 * pos[list(cmb)].all(axis=1).mean(),
                             "lines missed (OR)": ", ".join(pos.index[~pos[list(cmb)].any(axis=1)])}
                            for cmb in combinations(panel, size)]).sort_values(["any (OR) %", "all (AND) %"], ascending=False)
        st.dataframe(tab, hide_index=True, width="stretch", height=320,
                     column_config={k: st.column_config.ProgressColumn(k, min_value=0, max_value=100, format="%.0f")
                                    for k in ["any (OR) %", "all (AND) %"]})
        st.caption(f"All {len(tab)} {'pairs' if size == 2 else 'triples'} of the candidates, ranked by coverage with any "
                   "antigen, then by lines carrying all.")
        st.download_button("Download (TSV)", tab.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                           file_name=f"antigen_{'pairs' if size == 2 else 'triples'}.tsv")


# ------------------------------------------------------------------ section (Cell lines tab) ---
def surfaceome_section() -> None:
    """Cell surface proteins: the content of the Cell lines page tab."""
    if not available():
        st.info("Needs the surface_george layer in results/cell_lines (prepare_cell_line_data.R with the George 2026 "
                "supplementary files in data/cell_lines/surfaceome_George_2026/).")
        return
    M = surface_matrix()
    st.markdown(f"Surface proteins captured by cell-surface biotinylation in **{M.shape[1]} AML cell lines** "
                f"({M.shape[0]:,} proteins): George et al. 2026, bioRxiv, "
                "[doi:10.64898/2026.09.16.752145](https://doi.org/10.64898/2026.09.16.752145). Values are log2 mean "
                "abundance; *not detected* means not found on the surface.")
    st.caption(f"Gene symbols: {ALIASES}.")
    rule, cut = positivity_picker(M)
    tab_hl, tab_mk, tab_ma = st.tabs(["Highlights", "Marker expression", "Multi-antigen"])
    with tab_hl:
        highlights_tab()
    with tab_mk:
        marker_tab(M, rule, cut)
    with tab_ma:
        multi_antigen_tab(M, rule, cut)
