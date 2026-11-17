"""Drug sensitivity: mutation-drug associations, target overexpression, and copy-number-driven targets.

Drug response = AUC from the harmonized DepMap screens (PRISM, CTRP, GDSC2): lower AUC = more sensitive.
Targets come from the drug annotation (`targets` column of the drug feature tables)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portal import data
from portal import genomics_plots as plots
from portal.cooccurrence import alterations, cn_events, has_cn
from portal.genomics_stats import rowwise_correlate
from portal.plots import COL_A, COL_B, line_scatter
from portal.stats import bh, welch
from portal.surfaceome import _order, _test as _group_test, colour_options, colour_values, show_groups


# ------------------------------------------------------------------ helpers ---
def aberration_table() -> tuple[pd.DataFrame, dict[str, str]]:
    """lines x aberrations (mutations, FLT3-ITD, KMT2A-r, arm-level CN gains / losses)."""
    M, labels, genomic = alterations()
    M, labels = M[genomic], {c: labels[c] for c in genomic}
    if has_cn():
        E, elab = cn_events(1.3, 0.7)
        M = M.join(E, how="left")
        labels.update(elab)
    return M, labels


@st.cache_data(show_spinner=False)
def sym_matrix(key: str) -> pd.DataFrame:
    """A layer as genes (HGNC symbol) x lines; duplicated symbols keep the row measured in most lines."""
    V = data.values(key)
    sym = data.features(key)["gene_symbol"].reindex(V.index).fillna("")
    V = V[(sym != "").to_numpy()]
    V = V.assign(_s=sym[V.index].to_numpy(), _n=V.notna().sum(axis=1).to_numpy()).sort_values("_n", ascending=False)
    V = V.drop_duplicates("_s").set_index("_s").drop(columns="_n")
    V.index.name = None
    return V.dropna(axis=1, how="all")


def _colour_options():
    opts, labels = colour_options()
    return opts, labels.get, colour_values

DRUG = "drug_screen"


def drug_layers() -> list[str]:
    d = data.datasets()
    return list(d.loc[d["modality"] == DRUG, "key"])


def screen_picker(kp: str, container=st) -> str:
    keys = drug_layers()
    dsets = data.datasets().set_index("key")
    return container.selectbox("Drug screen", keys, key=f"{kp}_screen",
                               index=keys.index("drug_GDSC2") if "drug_GDSC2" in keys else 0,
                               format_func=lambda k: f"{dsets.loc[k, 'label']} ({int(dsets.loc[k, 'n_features'])} drugs, "
                                                    f"{int(dsets.loc[k, 'n_lines'])} lines)")


@st.cache_data(show_spinner=False)
def targets(screen: str) -> pd.DataFrame:
    """drug feature -> compound label, target list, mechanism."""
    f = data.features(screen)
    t = f["targets"].fillna("") if "targets" in f else pd.Series("", index=f.index)
    return pd.DataFrame({"compound": data.feature_labels(screen), "mechanism": f.get("mechanism", ""),
                         "targets": t.map(lambda s: [x.strip() for x in str(s).split(";") if x.strip()])})


@st.cache_data(show_spinner=False)
def druggable() -> pd.DataFrame:
    """target gene -> drugs (all screens), one row per gene."""
    rows = []
    for k in drug_layers():
        T = targets(k)
        for d, r in T.iterrows():
            for g in r["targets"]:
                rows.append((g, r["compound"], k))
    df = pd.DataFrame(rows, columns=["gene", "compound", "screen"])
    return df.groupby("gene").agg(drugs=("compound", lambda x: ", ".join(sorted(set(x)))),
                                  n_drugs=("compound", "nunique"))


def _pair_box(values: pd.Series, a: list[str], b: list[str], na: str, nb: str, title: str, subtitle: str, y: str, key: str):
    long = pd.DataFrame({"cell_line": a + b, "value": values.reindex(a + b).to_numpy(),
                         "group": [na] * len(a) + [nb] * len(b)}).dropna()
    st.plotly_chart(plots.group_box(long, "value", "group", [na, nb], title, y, subtitle=subtitle or _group_test(long, [na, nb]),
                                    colors={na: COL_A, nb: COL_B}), width="stretch", key=key)


# ------------------------------------------------------------------ 1. mutation-drug ---
@st.cache_data(show_spinner="Testing every aberration against every drug…")
def mutation_drug(screen: str, aberrations: tuple[str, ...], min_n: int) -> pd.DataFrame:
    D = data.values(screen)
    M, labels = aberration_table()
    M = M.reindex(D.columns)
    T = targets(screen)
    out = []
    for ab in aberrations:
        a, b = list(M.index[M[ab] == 1]), list(M.index[M[ab] == 0])
        if len(a) < min_n or len(b) < min_n:
            continue
        r = welch(D, a, b, min_n=min_n)
        if r.empty:
            continue
        out.append(pd.DataFrame({"aberration": ab, "label": labels[ab], "drug": r["feature"],
                                 "compound": r["feature"].map(T["compound"]),
                                 "targets": r["feature"].map(lambda d: ", ".join(T["targets"].get(d, []))),
                                 "n_with": r["n_A"], "n_without": r["n_B"], "AUC_with": r["mean_A"],
                                 "AUC_without": r["mean_B"], "dAUC": r["diff_A_minus_B"], "p": r["p_welch"]}))
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    res["q"] = bh(res["p"].to_numpy())
    res["direction"] = np.where(res["dAUC"] < 0, "more sensitive with", "more resistant with")
    return res.sort_values("p").reset_index(drop=True)


def _section(title: str):
    """A light card with a small heading, for a calm, scannable page."""
    box = st.container(border=True)
    box.markdown(f"**{title}**")
    return box


def mutation_drug_tab(kp: str = "dm") -> None:
    st.caption("Which genotypes go with sensitivity to which drugs? Lines with vs without each aberration are compared "
               "on AUC (Welch t; q = BH FDR across all tests). ΔAUC < 0 = more sensitive with the aberration.")
    c1, c2, c3, c4 = st.columns([3, 3, 1, 2], vertical_alignment="bottom")
    screen = screen_picker(kp, c1)
    D = data.values(screen)
    M, labels = aberration_table()
    M = M.reindex(D.columns)
    min_n = c3.number_input("Min. lines", 2, 10, 3, key=f"{kp}_min", help="Per group, with drug data.")
    opts = [c for c in M.columns if (M[c] == 1).sum() >= min_n and (M[c] == 0).sum() >= min_n]
    genomic = [c for c in opts if not c.startswith("cn_")]
    chosen = c2.multiselect("Aberrations", opts, default=genomic, key=f"{kp}_abs", format_func=labels.get,
                            help="Mutations, FLT3-ITD and KMT2A-r by default; arm-level copy-number events can be added.")
    sig = c4.segmented_control("Significant if", ["p < 0.05", "q < 0.1"], default="p < 0.05", key=f"{kp}_sig") or "p < 0.05"
    if not chosen:
        st.info("Pick at least one aberration.")
        return
    res = mutation_drug(screen, tuple(chosen), int(min_n))
    if res.empty:
        st.info("No aberration has enough lines with drug data in this screen.")
        return
    hit = (res["p"] < 0.05) if sig.startswith("p") else (res["q"] < 0.1)
    m1, m2, m3 = st.columns(3)
    m1.metric("Tests", f"{len(res):,}", help="aberration × drug pairs")
    m2.metric("More sensitive with", int((hit & (res["dAUC"] < 0)).sum()), help=sig)
    m3.metric("More resistant with", int((hit & (res["dAUC"] > 0)).sum()), help=sig)

    with _section("1 · All associations"):
        st.plotly_chart(plots.drug_volcano(res, hit), width="stretch", key=f"{kp}_volc")
        st.caption(f"Each point is one aberration × drug pair; coloured when {sig}. The 6 strongest are labelled.")

    with _section("2 · Strongest pairs"):
        top = res.nsmallest(30, "p")
        drugs = list(dict.fromkeys(top["compound"]))[:15]
        abs_ = [a for a in chosen if a in set(top["aberration"])]
        hm = res[res["compound"].isin(drugs) & res["aberration"].isin(abs_)].rename(
            columns={"aberration": "mutation", "compound": "region", "dAUC": "effect"})
        hm["detail"] = [f"AUC {w:.2f} with ({nw}) vs {wo:.2f} without ({nwo})" for w, nw, wo, nwo in
                        zip(hm["AUC_with"], hm["n_with"], hm["AUC_without"], hm["n_without"])]
        fig = plots.assoc_heatmap(hm, abs_, drugs, labels, fdr=0.1, what="AUC")
        fig.update_traces(textfont=dict(size=15, color="#222"), xgap=3, ygap=3)
        fig.update_xaxes(tickangle=-40, tickfont=dict(size=11))
        fig.update_yaxes(tickfont=dict(size=12))
        fig.update_layout(height=170 + 46 * len(abs_), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch", key=f"{kp}_hm")
        st.caption("Blue = lower AUC (more sensitive) with the aberration, red = higher. ** q < 0.1, * p < 0.05.")

    with _section("3 · Table"):
        c5, _ = st.columns([2, 3])
        view = c5.segmented_control("Show", ["significant", "sensitive only", "all"], default="significant",
                                    key=f"{kp}_show", label_visibility="collapsed")
        tab = res[hit] if view == "significant" else res[hit & (res["dAUC"] < 0)] if view == "sensitive only" else res
        tab = tab.reset_index(drop=True)
        ev = st.dataframe(tab[["label", "compound", "targets", "n_with", "n_without", "dAUC", "p", "q"]]
                          .rename(columns={"label": "aberration", "n_with": "lines with", "n_without": "lines without"}),
                          hide_index=True, width="stretch", height=300, on_select="rerun", selection_mode="multi-row",
                          key=f"{kp}_tab",
                          column_config={"dAUC": st.column_config.NumberColumn("ΔAUC", format="%+.3f"),
                                         "p": st.column_config.NumberColumn(format="%.2g"),
                                         "q": st.column_config.NumberColumn(format="%.2g")})
        c6, c7 = st.columns([3, 1], vertical_alignment="center")
        c6.caption("Select rows to plot them below (up to 9).")
        c7.download_button("Download (TSV)", tab.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                           file_name=f"mutation_drug_{screen}.tsv", width="stretch")

    rows = ev.selection.rows if ev and ev.selection else []
    picks = [tab.iloc[r] for r in rows if r < len(tab)][:9]
    if picks:
        with _section("4 · Selected pairs"):
            for i in range(0, len(picks), 3):
                cols = st.columns(3)
                for col, x in zip(cols, picks[i:i + 3]):
                    a, b = list(M.index[M[x["aberration"]] == 1]), list(M.index[M[x["aberration"]] == 0])
                    with col:
                        _pair_box(D.loc[x["drug"]], a, b, x["label"], f"no {x['label']}", x["compound"],
                                  f"ΔAUC {x['dAUC']:+.2f} · p = {x['p']:.2g}", "AUC",
                                  key=f"{kp}_b_{x['aberration']}_{x['drug']}")
            for ab in dict.fromkeys(x["aberration"] for x in picks):
                show_groups(list(M.index[M[ab] == 1]), list(M.index[M[ab] == 0]), labels[ab], f"no {labels[ab]}", colours=("blue", "red"))


# ------------------------------------------------------------------ 2. target overexpression ---
def expression_layers() -> list[str]:
    d = data.datasets()
    return list(d.loc[d["modality"].isin(["transcriptomics", "proteomics"]), "key"])


def target_tab(kp: str = "dt") -> None:
    dsets = data.datasets().set_index("key")
    st.caption("Which drug targets are highly expressed in AML lines, and does a line's target level predict its "
               "sensitivity to the drugs that hit it?")
    c1, c2 = st.columns(2)
    layers = expression_layers()
    layer = c1.selectbox("Expression", layers, key=f"{kp}_layer", format_func=lambda k: dsets.loc[k, "label"],
                         index=layers.index("expression") if "expression" in layers else 0)
    screen = screen_picker(kp, c2)
    X = sym_matrix(layer)
    unit = dsets.loc[layer, "unit"]
    DG = druggable()
    tg = DG.index.intersection(X.index)

    st.markdown("**1 · Druggable targets by expression**")
    pct = X.median(axis=1).rank(pct=True)
    Z = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1).replace(0, np.nan), axis=0)
    rank = pd.DataFrame({"gene": tg, "median": X.loc[tg].median(axis=1).to_numpy(),
                         "percentile among all genes": 100 * pct.reindex(tg).to_numpy(),
                         "lines measured": X.loc[tg].notna().sum(axis=1).to_numpy(),
                         "n_drugs": DG.loc[tg, "n_drugs"].to_numpy(), "drugs": DG.loc[tg, "drugs"].to_numpy()})
    rank = rank.sort_values("median", ascending=False).reset_index(drop=True)
    c3, c4 = st.columns([2, 3])
    with c3:
        top = rank.head(25)
        st.plotly_chart(plots.target_bars(top, unit), width="stretch", key=f"{kp}_bars")
        st.caption(f"The 25 most highly expressed of {len(rank)} drug targets measured ({dsets.loc[layer, 'label']}).")
    with c4:
        st.dataframe(rank, hide_index=True, width="stretch", height=420,
                     column_config={"median": st.column_config.NumberColumn(format="%.2f"),
                                    "percentile among all genes": st.column_config.ProgressColumn(
                                        min_value=0, max_value=100, format="%.0f")})

    st.divider()
    st.markdown("**2 · Does target expression predict sensitivity?**")
    T = targets(screen)
    D = data.values(screen)
    pairs = [(d, g) for d, gs in T["targets"].items() if d in D.index for g in gs if g in X.index]
    if not pairs:
        st.info("No drug target is measured in this expression layer.")
    else:
        P = pd.DataFrame(pairs, columns=["drug", "gene"])
        r = rowwise_correlate(X.reindex(P["gene"]).set_axis(range(len(P))), D.reindex(P["drug"]).set_axis(range(len(P))),
                              "spearman", 6)
        if r.empty:
            st.info("Too few lines with both data.")
        else:
            r = r.join(P, on="gene", rsuffix="_").drop(columns="gene").rename(columns={"gene_": "target"})
            r.insert(0, "compound", r["drug"].map(T["compound"]))
            r = r.sort_values("r")
            c5, c6 = st.columns([3, 2])
            with c5:
                ev = st.dataframe(r[["compound", "target", "r", "n", "p", "q"]], hide_index=True, width="stretch",
                                  height=380, on_select="rerun", selection_mode="single-row", key=f"{kp}_pairs",
                                  column_config={c: st.column_config.NumberColumn(format="%.3g") for c in ["r", "p", "q"]})
                st.caption(f"{len(r)} drug–target pairs. Negative r = higher target level, lower AUC (more sensitive): "
                           "the target may mark sensitivity.")
            sel = ev.selection.rows if ev and ev.selection else []
            row = r.iloc[sel[0]] if sel else r.iloc[0]
            with c6:
                opts, label, values = _colour_options()
                by = st.selectbox("Colour lines by", opts, key=f"{kp}_by", format_func=label)
                df = pd.DataFrame({"x": X.loc[row["target"]], "y": D.loc[row["drug"]]}).dropna() \
                    .rename_axis("cell_line").reset_index()
                df["group"] = values(by, df["cell_line"]).to_numpy()
                st.plotly_chart(line_scatter(df, "group", _order(df["group"]), f"{row['target']} · {unit}",
                                                   f"{row['compound']} AUC"), width="stretch", key=f"{kp}_sc")

    st.divider()
    st.markdown("**3 · One line: its overexpressed druggable targets**")
    lines = sorted(X.columns)
    c7, c8 = st.columns([2, 1])
    line = c7.selectbox("Cell line", lines, key=f"{kp}_line", index=lines.index("MOLM13") if "MOLM13" in lines else 0)
    zt = c8.number_input("Overexpressed if z ≥", 0.5, 4.0, 1.5, 0.25, key=f"{kp}_z",
                         help="z-score of the line vs the other AML lines.")
    zl = Z.loc[tg, line].dropna().sort_values(ascending=False)
    over = zl[zl >= zt]
    if over.empty:
        st.info(f"No druggable target is at z ≥ {zt} in {line}.")
        return
    auc_pct = D.rank(axis=1, pct=True)[line] if line in D.columns else pd.Series(dtype=float)
    rows = []
    for g, z in over.items():
        ds = [d for d, gs in T["targets"].items() if g in gs]
        resp = [f"{T.loc[d, 'compound']} ({100 * auc_pct[d]:.0f}th pct)" for d in ds if d in auc_pct.index and pd.notna(auc_pct[d])]
        rows.append({"target": g, "z-score": z, unit: X.loc[g, line], "drugs (all screens)": DG.loc[g, "drugs"],
                     f"{line} AUC percentile in {dsets.loc[screen, 'label']}": "; ".join(resp) or "not screened"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"z-score": st.column_config.NumberColumn(format="%.2f"),
                                unit: st.column_config.NumberColumn(format="%.2f")})
    st.caption("AUC percentile among the screened AML lines: low = this line is among the most sensitive to that drug.")


# ------------------------------------------------------------------ 3. CN-driven targets ---
@st.cache_data(show_spinner="Matching copy number to expression…")
def cn_expression(amp: float, z_thr: float) -> pd.DataFrame:
    C = (2 ** sym_matrix("cn") - 1).clip(lower=0)          # relative CN, 1 = neutral
    X = sym_matrix("expression")
    genes, lines = C.index.intersection(X.index), C.columns.intersection(X.columns)
    C, X = C.loc[genes, lines], X.loc[genes, lines]
    Z = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1).replace(0, np.nan), axis=0)
    ampm, over = C >= amp, Z >= z_thr
    both = ampm & over
    r = rowwise_correlate(C, X, "spearman", 8).set_index("gene")
    DG = druggable()
    out = pd.DataFrame({"lines amplified": ampm.sum(axis=1), "amplified + overexpressed": both.sum(axis=1),
                        "lines": [", ".join(lines[both.loc[g].to_numpy()]) for g in genes]}, index=genes)
    out = out.join(r[["r", "q"]].rename(columns={"r": "CN–RNA r", "q": "CN–RNA q"}))
    out = out.join(DG[["n_drugs", "drugs"]])
    out["druggable"] = out["n_drugs"].fillna(0) > 0
    return out.rename_axis("gene").reset_index()


def cn_targets_tab(kp: str = "dc") -> None:
    keys = set(data.datasets()["key"])
    if not {"cn", "expression"} <= keys:
        st.info("Needs the copy-number (cn) and RNA expression layers.")
        return
    st.caption("Genes whose copy-number gain drives their expression: amplified and overexpressed in the same lines, "
               "with a positive copy-number vs RNA correlation across lines (dosage), flagged when a drug targets them.")
    c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
    amp = c1.number_input("Amplified if relative CN ≥", 1.1, 4.0, 1.5, 0.1, key=f"{kp}_amp",
                          help="1.5 ≈ one extra copy in a diploid line; 2 ≈ two.")
    zt = c2.number_input("Overexpressed if RNA z ≥", 0.5, 4.0, 1.0, 0.25, key=f"{kp}_z")
    only = c3.toggle("Druggable only", value=True, key=f"{kp}_drug")
    R = cn_expression(float(amp), float(zt))
    R = R[R["amplified + overexpressed"] > 0]
    if only:
        R = R[R["druggable"]]
    R = R.sort_values(["amplified + overexpressed", "CN–RNA r"], ascending=False).reset_index(drop=True)
    st.markdown("**1 · Amplified and overexpressed genes**")
    m1, m2 = st.columns(2)
    m1.metric("Genes amplified + overexpressed in ≥ 1 line", len(R))
    m2.metric("with CN–RNA dosage (r > 0, q < 0.1)", int(((R["CN–RNA r"] > 0) & (R["CN–RNA q"] < 0.1)).sum()))
    if R.empty:
        st.info("No gene passes these thresholds.")
        return
    ev = st.dataframe(R, hide_index=True, width="stretch", height=360, on_select="rerun", selection_mode="single-row",
                      key=f"{kp}_tab",
                      column_config={"CN–RNA r": st.column_config.NumberColumn(format="%.2f"),
                                     "CN–RNA q": st.column_config.NumberColumn(format="%.2g"),
                                     "n_drugs": st.column_config.NumberColumn("drugs (n)", format="%d")})
    st.download_button("Download (TSV)", R.to_csv(sep="\t", index=False), key=f"{kp}_dl", file_name="cn_driven_targets.tsv")
    st.divider()
    sel = ev.selection.rows if ev and ev.selection else []
    gene = R.iloc[sel[0]]["gene"] if sel else R.iloc[0]["gene"]
    st.markdown(f"**2 · {gene}: copy number, expression and drug response** (select a table row to change the gene)")
    C = (2 ** sym_matrix("cn") - 1).clip(lower=0)
    X = sym_matrix("expression")
    df = pd.DataFrame({"x": C.loc[gene], "y": X.loc[gene]}).dropna().rename_axis("cell_line").reset_index()
    df["group"] = np.where(df["x"] >= amp, "amplified", "not amplified")
    c4, c5 = st.columns(2)
    with c4:
        st.plotly_chart(line_scatter(df, "group", ["amplified", "not amplified"], f"{gene} relative copy number",
                                           f"{gene} RNA log2(TPM+1)"), width="stretch", key=f"{kp}_sc")
    with c5:
        drugs = [(k, d) for k in drug_layers() for d, gs in targets(k)["targets"].items() if gene in gs]
        if not drugs:
            st.info(f"No drug in the screens targets {gene}.")
        else:
            a, b = list(df.loc[df["group"] == "amplified", "cell_line"]), list(df.loc[df["group"] == "not amplified", "cell_line"])
            k, d = drugs[0] if len(drugs) == 1 else st.selectbox(
                "Drug targeting " + gene, drugs, key=f"{kp}_d",
                format_func=lambda kd: f"{targets(kd[0]).loc[kd[1], 'compound']} ({data.datasets().set_index('key').loc[kd[0], 'label']})")
            with st.container(border=True):
                _pair_box(data.values(k).loc[d], a, b, "amplified", "not amplified", targets(k).loc[d, "compound"], "",
                          "AUC (lower = more sensitive)", key=f"{kp}_box")
    st.divider()
    st.markdown("**3 · One line: its amplified, overexpressed genes**")
    lines = sorted(C.columns.intersection(X.columns))
    line = st.selectbox("Cell line", lines, key=f"{kp}_line", index=lines.index("KASUMI1") if "KASUMI1" in lines else 0)
    full = cn_expression(float(amp), float(zt))
    mine = full[full["lines"].str.split(", ").map(lambda v: line in v)]
    if only:
        mine = mine[mine["druggable"]]
    st.dataframe(mine.drop(columns=["lines"]).sort_values("CN–RNA r", ascending=False), hide_index=True, width="stretch",
                 height=300, column_config={"CN–RNA r": st.column_config.NumberColumn(format="%.2f"),
                                            "CN–RNA q": st.column_config.NumberColumn(format="%.2g")})
    st.caption(f"Genes amplified (relative CN ≥ {amp}) and overexpressed (RNA z ≥ {zt}) in {line}.")


def drug_sensitivity_section() -> None:
    """Drug sensitivity: the content of the Cell lines page tab."""
    if not drug_layers():
        st.info("No drug-response layer in results/cell_lines.")
        return
    st.caption("Drug response from PRISM, CTRP and GDSC2 (AUC; lower = more sensitive).")
    tab_mut, tab_tgt, tab_cn = st.tabs(["Mutation–drug associations", "Target overexpression",
                                         "Copy-number-driven targets"])
    with tab_mut:
        mutation_drug_tab()
    with tab_tgt:
        target_tab()
    with tab_cn:
        cn_targets_tab()
