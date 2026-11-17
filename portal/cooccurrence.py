"""Co-occurrence and mutual exclusivity of genomic aberrations across AML cell lines.

Pick aberrations (mutations, FLT3-ITD, KMT2A-r, copy-number gains / losses, binary annotations) and
see how many lines carry each combination (UpSet), list them, and test the set (permutation) and
every pair (Fisher exact test, log2 odds ratio, BH FDR across pairs).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from portal import data
from portal import genomics_plots as plots
from portal.copy_number import CN, region_cn
from portal.genomics_stats import cooccurrence, set_exclusivity

SHORT = {"FLT3_ITD": "FLT3-ITD", "KMT2A_rearranged": "KMT2A-r"}
FMT = {c: st.column_config.NumberColumn(format="%.3g") for c in ["log2_odds_ratio", "p", "q"]}
DEFAULT = ["mut_TP53", "mut_NRAS", "mut_KRAS", "FLT3_ITD", "mut_NPM1", "KMT2A_rearranged"]
MAX_COMBOS = 40


def _as_bool(s: pd.Series, true_label: str | None) -> pd.Series:
    """TRUE / true_label -> 1.0, anything else -> 0.0, not annotated stays NaN."""
    hit = s.eq(true_label) if true_label else s.astype(str).str.upper().eq("TRUE")
    return hit.astype(float).where(s.notna())


@st.cache_data(show_spinner=False)
def alterations() -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """lines x alterations (1 / 0 / NaN), display labels, and which columns are genomic.

    Genomic: every mut_<GENE> column in samples.tsv (also genes too rare to be a grouping),
    FLT3-ITD and KMT2A rearrangement. Other binary groupings (MYC-High, Mito-High, ...) are
    offered too, so an alteration can be tested against a phenotype."""
    cl = data.cell_lines().set_index("cell_line")
    G = data.groupings()
    cols, labels = {}, {}
    genomic = [c for c in cl.columns if c.startswith("mut_")] + [c for c in SHORT if c in cl.columns]
    for c in genomic:
        tl = G.loc[c, "true_label"] if c in G.index and G.loc[c, "kind"] == "binary" else None
        cols[c] = _as_bool(cl[c], tl)
        labels[c] = c[4:] if c.startswith("mut_") else SHORT[c]
    for c, r in G.iterrows():
        if r["kind"] == "binary" and c not in cols:
            cols[c] = _as_bool(cl[c], r["true_label"])
            labels[c] = r["true_label"]
    return pd.DataFrame(cols), labels, genomic


def has_cn() -> bool:
    return CN in set(data.datasets()["key"]) and "cytoband" in data.features(CN).columns


@st.cache_data(show_spinner="Calling copy-number events…")
def cn_events(amp: float, dele: float, genes: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict[str, str]]:
    """Copy-number aberrations as binary columns: arm-level gain(8q) / del(17p) from the median
    relative CN of the arm's genes, and gene-level amp / del for the genes asked for.
    NaN for lines without copy number. X and Y are left out (they mostly reflect sex)."""
    med, _ = region_cn("arm")
    med = med[[r for r in med.columns if not r.startswith(("X", "Y"))]]
    cols, labels = {}, {}
    for r in med.columns:
        cols[f"cn_gain_{r}"], labels[f"cn_gain_{r}"] = (med[r] >= amp).astype(float).where(med[r].notna()), f"gain({r})"
        cols[f"cn_del_{r}"], labels[f"cn_del_{r}"] = (med[r] <= dele).astype(float).where(med[r].notna()), f"del({r})"
    if genes:
        f = data.features(CN)
        ratio = (2 ** data.values(CN) - 1).clip(lower=0)
        for g in genes:
            fid = f.index[f["gene_symbol"] == g]
            if len(fid):
                v = ratio.loc[fid[0]].reindex(med.index)
                cols[f"cn_amp_{g}"], labels[f"cn_amp_{g}"] = (v >= amp).astype(float).where(v.notna()), f"{g} amp"
                cols[f"cn_dl_{g}"], labels[f"cn_dl_{g}"] = (v <= dele).astype(float).where(v.notna()), f"{g} del"
    return pd.DataFrame(cols, index=med.index), labels


def _keep_valid(key: str, options: list[str], default: list[str]) -> None:
    """Default on first use; afterwards only drop selections no longer offered (options change
    with the filters), so a deliberately emptied selection stays empty."""
    if key not in st.session_state:
        st.session_state[key] = default
    else:
        st.session_state[key] = [v for v in st.session_state[key] if v in options]


def _table(res: pd.DataFrame, labels: dict[str, str], fdr: float, name: str) -> None:
    show = res.assign(A=res["A"].map(labels), B=res["B"].map(labels))
    show = show[["A", "B", "both", "A_not_B", "B_not_A", "neither", "n_lines", "log2_odds_ratio", "p", "q"]]
    st.caption(f"{int((res['q'] < fdr).sum())} of {len(res)} pairs at q < {fdr}. "
               "Counts are cell lines: altered in both, only A, only B, neither.")
    st.dataframe(show, hide_index=True, width="stretch", height=min(38 + 35 * len(show), 360),
                 column_config=FMT)
    st.download_button("Download table (TSV)", show.to_csv(sep="\t", index=False), file_name=f"{name}.tsv",
                       key=f"co_dl_{name}")


def _combos(X: pd.DataFrame, members: list[str], labels: dict[str, str]) -> pd.DataFrame:
    """Observed combinations of the members (complete cases), largest first."""
    pat = X[members].astype(bool).apply(tuple, axis=1)
    c = pat.value_counts().rename_axis("pattern").reset_index(name="count")
    c["n_on"] = c["pattern"].map(sum)
    c = c.sort_values(["count", "n_on"], ascending=[False, False]).reset_index(drop=True)
    c["id"] = [str(i) for i in range(len(c))]
    c["name"] = [" + ".join(labels[m] for m, v in zip(members, p) if v) or "none of them" for p in c["pattern"]]
    return c


def cooccurrence_tab(kp: str = "co") -> None:
    M_base, labels, genomic = alterations()
    labels = dict(labels)
    G, GL = data.groupings(), data.grouping_labels()
    cl = data.cell_lines().set_index("cell_line")
    st.markdown(
        "Pick aberrations to see how many cell lines carry them, alone or together, and whether they "
        "**co-occur** or are **mutually exclusive**. Copy-number events are called per chromosome arm "
        "(or per gene, if you add genes) from DepMap relative copy number.")

    # ---- aberration universe
    M_all, cn_cols = M_base, []
    with st.expander("Lines and copy-number settings", icon=":material/tune:"):
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        subset_by = c1.selectbox("Lines to include", ["all lines"] + [c for c in G.index if c in cl.columns],
                                 format_func=lambda c: c if c == "all lines" else f"only lines by {GL[c]}",
                                 key=f"{kp}_subset_by")
        lines = cl.index
        if subset_by != "all lines":
            lv = sorted(cl[subset_by].dropna().astype(str).unique())
            _keep_valid(f"{kp}_subset_levels", lv, lv[:1])
            keep = c1.multiselect(f"{GL[subset_by]} levels", lv, key=f"{kp}_subset_levels")
            lines = cl.index[cl[subset_by].astype(str).isin(keep)]
        min_alt = c2.number_input("Min. altered lines", 1, 20, 1, key=f"{kp}_min",
                                  help="Aberrations with fewer altered lines (among the included lines) are not offered.")
        if has_cn():
            c3, c4, c5 = st.columns([1, 1, 2], vertical_alignment="bottom")
            amp = c3.number_input("Gain if relative CN ≥", 1.05, 4.0, 1.3, 0.05, key=f"{kp}_amp",
                                  help="Median over the arm's genes (or the gene's value); 1.5 ≈ +1 copy in a diploid line.")
            dele = c4.number_input("Loss if relative CN ≤", 0.0, 0.95, 0.7, 0.05, key=f"{kp}_del",
                                   help="0.5 ≈ one copy lost in a diploid line.")
            gtxt = c5.text_input("Gene-level copy number for", placeholder="e.g. MYC, TP53, KMT2A, CDKN2A",
                                 key=f"{kp}_cn_genes")
            genes = tuple(dict.fromkeys(g.strip().upper() for g in gtxt.replace(";", ",").split(",") if g.strip()))
            E, elab = cn_events(amp, dele, genes)
            cn_cols = list(E.columns)
            M_all = M_base.join(E, how="left")
            labels.update(elab)
            missing = [g for g in genes if f"cn_amp_{g}" not in E.columns]
            if missing:
                st.caption(f"No copy number for: {', '.join(missing)}")
        else:
            st.caption("No copy-number layer in the results, so only mutations and annotations are offered.")

    M = M_all.reindex(lines)
    n_alt = (M > 0).sum()
    annot = [c for c in M.columns if c not in genomic and c not in cn_cols]
    gene_cn = [c for c in cn_cols if c.startswith(("cn_amp_", "cn_dl_"))]
    options = [c for c in genomic + gene_cn + [c for c in cn_cols if c not in gene_cn] + annot if n_alt[c] >= min_alt]
    _keep_valid(f"{kp}_alts", options, [c for c in DEFAULT if c in options])
    kind = lambda c: (" · annotation" if c in annot else " · copy number" if c in cn_cols else "")  # noqa: E731
    chosen = st.multiselect(
        "Aberrations", options, key=f"{kp}_alts", placeholder="Type a gene, FLT3-ITD, KMT2A-r, gain(8q), del(17p)…",
        format_func=lambda c: f"{labels[c]} ({int(n_alt[c])}/{int(M[c].notna().sum())}){kind(c)}",
        help="In brackets: altered / annotated lines. Copy-number events need lines with copy number.")
    if not chosen:
        st.info("Pick one or more aberrations.")
        return

    # ---- lines with the selected aberrations
    X = M[chosen].dropna()
    n_missing = int(M[chosen].notna().any(axis=1).sum()) - len(X)
    k = X.astype(bool).sum(axis=1)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Lines annotated for all", len(X), help=f"{n_missing} more lines lack data for at least one selected aberration.")
    m2.metric("With all selected", int((k == len(chosen)).sum()), help="Co-occurring: every selected aberration.")
    m3.metric("With exactly one", int((k == 1).sum()), help="Exclusive: one selected aberration and no other.")
    m4.metric("With any", int((k >= 1).sum()))
    m5.metric("With none", int((k == 0).sum()))
    if len(X) == 0:
        st.info("No line is annotated for every selected aberration.")
        return

    combos = _combos(X, chosen, labels)
    shown = combos.head(MAX_COMBOS)
    sizes = X.astype(bool).sum()
    event = st.plotly_chart(plots.upset(shown, chosen, labels, sizes), width="stretch", key=f"{kp}_upset",
                            on_select="rerun", selection_mode="points")
    st.caption("Each bar is one combination of the selected aberrations (dots below) and the number of lines "
               "with exactly that combination. Gray = none, blue = one alone (exclusive), red = two or more "
               f"together (co-occurring). Click a bar to list its lines"
               + (f"; the {MAX_COMBOS} largest of {len(combos)} combinations are shown." if len(combos) > MAX_COMBOS else "."))

    picked = []
    if event and event.selection:
        picked = [p.get("customdata") for p in event.selection.get("points", []) if p.get("customdata") is not None]
    modes = ["all selected", "exactly one", "any", "none"]
    mode = st.segmented_control("List lines with", modes + (["clicked combination"] if picked else []),
                                default="clicked combination" if picked else "all selected",
                                key=f"{kp}_mode_{bool(picked)}") or "all selected"
    if mode == "clicked combination":
        pats = set(combos.loc[combos["id"].isin([str(x) for x in picked]), "pattern"])
        sel = X.index[X[chosen].astype(bool).apply(tuple, axis=1).isin(pats)]
    else:
        sel = X.index[{"all selected": k == len(chosen), "exactly one": k == 1, "any": k >= 1, "none": k == 0}[mode]]
    show = pd.DataFrame({"cell_line": sel,
                         "aberrations": [", ".join(labels[c] for c in chosen if X.loc[x, c]) for x in sel]})
    extra = [c for c in ["disease_oncotree", "population", "KMT2A_partner", "mutations"] if c in cl.columns]
    show = show.join(cl[extra], on="cell_line")
    st.dataframe(show, hide_index=True, width="stretch", height=min(38 + 35 * len(show), 320))
    st.download_button("Download lines (TSV)", show.to_csv(sep="\t", index=False), key=f"{kp}_dl_lines",
                       file_name="cell_lines_" + "_".join(labels[c] for c in chosen)[:80].replace(" ", "") + ".tsv")

    if len(chosen) < 2:
        return

    # ---- is the set exclusive or co-occurring?
    st.subheader("Co-occurrence or exclusivity of the set")
    r = set_exclusivity(X)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Lines with any", r["coverage"], f"{r['coverage'] - r['expected_coverage']:+.1f} vs chance",
              delta_color="off")
    s2.metric("Lines with ≥ 2", r["overlapping"], f"{r['overlapping'] - r['expected_overlapping']:+.1f} vs chance",
              delta_color="off")
    s3.metric("p (mutually exclusive)", f"{r['p_exclusive']:.3g}")
    s4.metric("p (co-occurring)", f"{r['p_cooccurring']:.3g}")
    st.caption("Compared with 10,000 shuffles that keep each aberration's frequency: an exclusive set covers more "
               "lines with fewer overlaps than chance, a co-occurring set fewer lines with more overlaps.")

    # ---- every pair
    st.subheader("Pairwise co-occurrence and mutual exclusivity")
    c6, c7, c8 = st.columns(3, vertical_alignment="bottom")
    sided = c6.segmented_control("Test", ["two-sided", "one-sided"], default="two-sided", key=f"{kp}_sided",
                                 help="One-sided tests in the direction of each pair's tendency (as cBioPortal).")
    fdr = c7.select_slider("FDR threshold", [0.01, 0.05, 0.1, 0.25], value=0.1, key=f"{kp}_fdr")
    complete = c8.toggle("Only lines annotated for all", key=f"{kp}_complete",
                         help="Off: each pair uses every line annotated for both aberrations.")
    Mp = M[chosen].dropna() if complete else M[chosen]
    res = cooccurrence(Mp, sided=sided or "two-sided")
    if res.empty:
        st.info("No pair is testable: each needs ≥ 3 shared lines and ≥ 1 altered line per aberration.")
        return
    co, ex = res[res["tendency"] == "co-occurrence"], res[res["tendency"] == "mutual exclusivity"]
    order = list((Mp > 0).sum().sort_values(ascending=False).index)
    st.plotly_chart(plots.pair_matrix(res, order, labels, fdr=fdr), width="stretch")
    st.caption(f"Red = co-occurrence, blue = mutual exclusivity (log2 odds ratio). ** q < {fdr}, * p < 0.05. "
               "Fisher exact test per pair; hover a cell for the counts.")
    st.markdown("**Co-occurring pairs**")
    _table(co, labels, fdr, "co_occurrence")
    st.markdown("**Mutually exclusive pairs**")
    _table(ex, labels, fdr, "mutual_exclusivity")
