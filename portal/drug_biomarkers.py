"""Patients → Drug response biomarkers (built-in analysis 1).

Pick a drug; every cohort that tested it correlates each molecular feature with the drug response
(Spearman), and the cohorts are combined by a Fisher-z meta-analysis. The sign is set so that a
positive association always means "higher feature, more sensitive", whatever the cohort's measure
(AUC: lower = sensitive; DSS: higher = sensitive).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portal import data, plots
from portal.dependency import selected_genes
from portal.stats import meta_correlation, spearman_rows

MIN_N = 10
WITH = {"expression": "RNA", "protein": "protein", "phospho": "phosphosites"}
MAX_GENES = 6
# when a cohort has several measures of the same drug, prefer the one that is relative to healthy cells
MEASURE_ORDER = ["drug_sDSS", "drug_auc", "drug_DSS"]


@st.cache_data(show_spinner="Correlating features with drug response…")
def _cohort_assoc(cohort: str, layer: str, drug_layer: str, drug_feature: str, timepoints: tuple,
                  min_n: int) -> pd.DataFrame:
    s = data.patient_samples(cohort)
    keep = s["sample_id"] if not timepoints else s.loc[s["timepoint"].isin(timepoints), "sample_id"]
    y = data.patient_values(cohort, drug_layer).loc[drug_feature]
    y = y[y.index.isin(set(keep))]
    return spearman_rows(data.patient_values(cohort, layer), y, min_n=min_n)


def _symbols(cohort: str, layer: str) -> pd.Series:
    return data.feature_names(cohort, layer)


def drug_biomarkers(kp: str = "db") -> None:
    idx = data.patient_drug_index()
    if idx.empty:
        st.info("No prepared patient cohort has drug response data yet.")
        return
    names = data.patient_cohort_names()

    st.markdown(
        "Which molecular features predict ex vivo response to a drug, consistently across cohorts? "
        "Each cohort correlates every feature with the drug response (Spearman), and the cohorts are "
        "combined by meta-analysis (Fisher z, weighted by n − 3). **Positive = higher feature, more sensitive.** "
        "Proof of concept: MAP1A and venetoclax.")

    # data type first; then the drugs, counting only cohorts that have both the drug and that data type
    has_layer = {c: {k for k, m in data.patient_datasets(c)[["key", "modality"]].itertuples(index=False)
                     if m != "drug_screen"} for c in idx["cohort"].unique()}
    order = list(data.LAYER_LABELS)
    layers = sorted(set().union(*has_layer.values()), key=lambda k: (order.index(k) if k in order else 99, k))
    if not layers:
        st.info("No molecular data in the cohorts with drug response.")
        return
    c1, c2, c3 = st.columns([2, 3, 2])
    layer = c1.selectbox("Features", layers, key=f"{kp}_layer", format_func=data.layer_label,
                         index=layers.index("protein") if "protein" in layers else 0)
    usable = idx[idx["cohort"].map(lambda c: layer in has_layer[c])]
    per_key = usable.groupby("compound_key").agg(label=("label", "first"), cohorts=("cohort", lambda c: sorted(set(c))))
    per_key["k"] = per_key["cohorts"].apply(len)
    per_key = per_key[per_key.index != ""].sort_values(["k", "label"], ascending=[False, True])
    keys = list(per_key.index)
    key = c2.selectbox(
        "Drug", keys, index=keys.index("VENETOCLAX") if "VENETOCLAX" in keys else 0, key=f"{kp}_drug_{layer}",
        format_func=lambda k: f"{per_key.loc[k, 'label']}  ·  {per_key.loc[k, 'k']} cohort"
                              f"{'s' if per_key.loc[k, 'k'] > 1 else ''} with {WITH.get(layer, data.layer_label(layer))}",
        on_change=lambda: [st.session_state.pop(k, None) for k in list(st.session_state) if str(k).startswith(f"{kp}_volcano")])
    hits = usable[usable["compound_key"] == key]
    targets = sorted({t for v in hits["targets"] for t in v.split(";") if t})
    stage = c3.selectbox("Samples", ["all samples", "diagnosis only"], key=f"{kp}_stage")
    timepoints = ("diagnosis",) if stage == "diagnosis only" else ()
    if targets:
        st.caption("Annotated targets: " + ", ".join(targets))

    # one drug measure per cohort
    choice = {}
    for c, h in hits.groupby("cohort", sort=False):
        opts = sorted(h["layer"].unique(), key=lambda l: MEASURE_ORDER.index(l) if l in MEASURE_ORDER else 99)
        choice[c] = opts[0]
    multi = {c: sorted(hits.loc[hits["cohort"] == c, "layer"].unique()) for c in choice
             if hits.loc[hits["cohort"] == c, "layer"].nunique() > 1}
    if multi:
        with st.expander("Drug measure per cohort"):
            for c, opts in multi.items():
                lab = hits.drop_duplicates("layer").set_index("layer")["layer_label"]
                choice[c] = st.radio(names.get(c, c), opts, index=opts.index(choice[c]), horizontal=True,
                                     format_func=lambda l: lab.get(l, l), key=f"{kp}_measure_{c}")

    cohorts = [c for c in choice if layer in set(data.patient_datasets(c)["key"])]
    picked = st.multiselect("Cohorts", cohorts, default=cohorts, format_func=lambda c: names.get(c, c),
                            key=f"{kp}_cohorts_{key}_{layer}")
    per, info = {}, []
    for c in picked:
        row = hits[(hits["cohort"] == c) & (hits["layer"] == choice[c])].iloc[0]
        res = _cohort_assoc(c, layer, choice[c], row["feature"], timepoints, MIN_N)
        sym = _symbols(c, layer)
        res["feature"] = res["feature"].map(sym).fillna(res["feature"])
        res = res.drop_duplicates("feature")
        if row["lower_is_stronger"]:
            res["r"] = -res["r"]                       # positive = higher feature, more sensitive
        n = int(res["n"].max()) if len(res) else 0
        info.append({"cohort": names.get(c, c), "drug": row["label"], "measure": row["layer_label"],
                     "samples with both": n, "used": n >= MIN_N})
        if n >= MIN_N:
            per[c] = res
    st.dataframe(pd.DataFrame(info), hide_index=True, width="stretch")
    if not per:
        st.warning(f"No cohort has ≥ {MIN_N} samples with both the drug response and {layer}.")
        return

    meta = meta_correlation(per).rename(columns={"feature": "gene"})
    fdr = st.select_slider("FDR threshold", [0.01, 0.05, 0.1, 0.25], value=0.05, key=f"{kp}_fdr")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cohorts", len(per))
    m2.metric("Features tested", f"{len(meta):,}")
    m3.metric(f"Sensitivity, FDR < {fdr}", int(((meta["fdr"] < fdr) & (meta["r"] > 0) & meta["concordant"]).sum()))
    m4.metric(f"Resistance, FDR < {fdr}", int(((meta["fdr"] < fdr) & (meta["r"] < 0) & meta["concordant"]).sum()))
    st.caption("Click a feature, or drag a box / lasso around several, to see the per-cohort data below. "
               "Double-click the plot to clear the selection.")
    event = st.plotly_chart(
        plots.meta_volcano(meta, "pooled Spearman r (positive = higher feature, more sensitive)", fdr=fdr, y="fdr"),
        width="stretch", on_select="rerun", selection_mode=("points", "box", "lasso"),
        key=f"{kp}_volcano_{key}_{layer}_{stage}_{'-'.join(picked)}")
    selected = selected_genes(event)

    rcols = [f"r_{c}" for c in per]
    table = meta[["gene", "r", "p", "fdr", "n_cohorts", "n_total", "concordant", "I2"] + rcols].rename(
        columns={f"r_{c}": f"r {names.get(c, c)}" for c in per})
    view = table[table["gene"].isin(selected)] if selected else table
    tbl = st.dataframe(view, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(view)),
                       on_select="rerun", selection_mode="multi-row",
                       key=f"{kp}_table_{key}_{layer}_{stage}_{'-'.join(picked)}_{len(view)}",
                       column_config={c: st.column_config.NumberColumn(format="%.3g") for c in view.columns
                                      if c in ("r", "p", "fdr", "I2") or c.startswith("r ")})
    st.caption("Select rows in the table (tick boxes on the left) to show those features below; "
               "otherwise the plot selection, or the top row, is shown.")
    st.download_button("Download table (TSV)", view.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                       file_name=f"drug_biomarkers_{key}_{layer}.tsv")

    rows = list(tbl.selection.rows) if tbl and tbl.selection else []
    table_pick = view.iloc[rows]["gene"].tolist() if rows else []
    shown = (table_pick or selected or view["gene"].head(1).tolist())[:MAX_GENES]
    if not shown:
        return
    st.subheader("Per-cohort data" + ("" if (table_pick or selected) else f": {shown[0]} (top row)"))
    for gene in shown:
        r = meta.set_index("gene").loc[gene]
        st.markdown(f"**{gene}**: pooled r = {r['r']:.2f}, p = {r['p']:.2g}, FDR = {r['fdr']:.2g}"
                    f"{'' if r['concordant'] else ' (cohorts disagree on direction)'}")
        cols = st.columns(min(len(per), 4))
        for i, c in enumerate(per):
            row = hits[(hits["cohort"] == c) & (hits["layer"] == choice[c])].iloc[0]
            sym = _symbols(c, layer)
            feats = sym.index[sym == gene]
            if not len(feats):
                cols[i % len(cols)].caption(f"{names.get(c, c)}: {gene} not measured")
                continue
            V = data.patient_values(c, layer)
            y = data.patient_values(c, choice[c]).loc[row["feature"]]
            x = V.loc[[f for f in feats if f in V.index]].mean(axis=0)
            keep = set(data.patient_samples(c)["sample_id"] if not timepoints else
                       data.patient_samples(c).query("timepoint in @timepoints")["sample_id"])
            d = pd.DataFrame({"x": x, "y": y}).dropna()
            d = d[d.index.isin(keep)].rename_axis("sample_id").reset_index()
            rc = per[c].set_index("feature")["r"].get(gene, np.nan)
            pc = per[c].set_index("feature")["p"].get(gene, np.nan)
            ds = data.patient_datasets(c).set_index("key")
            cols[i % len(cols)].plotly_chart(
                plots.assoc_scatter(d, f"{gene} ({ds.loc[layer, 'unit']})", f"{row['label']} {ds.loc[choice[c], 'unit']}",
                                    title=f"{names.get(c, c)}<br><sup>n={len(d)} · r={rc:+.2f} · p={pc:.2g}</sup>"),
                width="stretch", key=f"{kp}_sc_{gene}_{c}")
    st.caption("r is shown with the sensitivity sign (positive = higher feature, more sensitive); "
               "the y axes show each cohort's own measure (AUC: lower = more sensitive; DSS: higher = more sensitive).")


def drug_marker_lookup(kp: str = "dm") -> None:
    """One marker (gene, protein or phosphosite) x one drug, in every cohort that has both."""
    idx = data.patient_drug_index()
    if idx.empty:
        st.info("No patient cohort has drug response data yet.")
        return
    names = data.patient_cohort_names()
    st.markdown("Pick a marker and a drug to see how the marker relates to drug response in each cohort. "
                "**Positive r = higher marker, more sensitive**, whatever the cohort's measure. "
                "For a genome-wide search across all markers, see **Highlights**.")

    has_layer = {c: {k for k, m in data.patient_datasets(c)[["key", "modality"]].itertuples(index=False)
                     if m != "drug_screen"} for c in idx["cohort"].unique()}
    order = list(data.LAYER_LABELS)
    layers = sorted(set().union(*has_layer.values()), key=lambda k: (order.index(k) if k in order else 99, k))
    c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
    layer = c1.selectbox("Marker type", layers, format_func=data.layer_label, key=f"{kp}_layer")
    cohorts = [c for c in has_layer if layer in has_layer[c]]
    markers = sorted(set().union(*[set(data.feature_names(c, layer)) for c in cohorts]))
    default = {"expression": "MAP1A", "protein": "MAP1A", "phospho": "STAT5A_S749"}.get(layer)
    marker = c2.selectbox({"phospho": "Phosphosite"}.get(layer, "Gene"), markers, key=f"{kp}_marker_{layer}",
                          index=markers.index(default) if default in markers else 0)
    with_marker = [c for c in cohorts if marker in set(data.feature_names(c, layer))]
    usable = idx[idx["cohort"].isin(with_marker)]
    per_key = usable.groupby("compound_key").agg(label=("label", "first"), k=("cohort", "nunique"))
    per_key = per_key[per_key.index != ""].sort_values(["k", "label"], ascending=[False, True])
    if per_key.empty:
        st.warning(f"No cohort has both {marker} and drug response.")
        return
    keys = list(per_key.index)
    key = c3.selectbox("Drug", keys, index=keys.index("VENETOCLAX") if "VENETOCLAX" in keys else 0,
                       key=f"{kp}_drug_{layer}",
                       format_func=lambda k: f"{per_key.loc[k, 'label']}  ·  {per_key.loc[k, 'k']} cohort"
                                             f"{'s' if per_key.loc[k, 'k'] > 1 else ''}")
    stage = c4.selectbox("Samples", ["all samples", "diagnosis only"], key=f"{kp}_stage")
    hits = usable[usable["compound_key"] == key]

    rows, plots_in = [], []
    for c, h in hits.groupby("cohort", sort=False):
        opts = sorted(h["layer"].unique(), key=lambda l: MEASURE_ORDER.index(l) if l in MEASURE_ORDER else 99)
        row = h[h["layer"] == opts[0]].iloc[0]
        sym = data.feature_names(c, layer)
        feats = [f for f in sym.index[sym == marker] if f in data.patient_values(c, layer).index]
        x = data.patient_values(c, layer).loc[feats].mean(axis=0)
        y = data.patient_values(c, row["layer"]).loc[row["feature"]]
        sm = data.patient_samples(c)
        keep = set(sm["sample_id"] if stage == "all samples" else sm.loc[sm["timepoint"].eq("diagnosis"), "sample_id"])
        d = pd.DataFrame({"x": x, "y": y}).dropna()
        d = d[d.index.isin(keep)].rename_axis("sample_id").reset_index()
        r = spearman_rows(pd.DataFrame([d["x"].to_numpy()], columns=d["sample_id"]),
                          d.set_index("sample_id")["y"], min_n=MIN_N)
        rr, pp = (float(r["r"].iloc[0]), float(r["p"].iloc[0])) if len(r) else (np.nan, np.nan)
        if row["lower_is_stronger"] and np.isfinite(rr):
            rr = -rr
        rows.append({"cohort": names.get(c, c), "drug measure": row["layer_label"], "n": len(d),
                     "r (sensitivity)": rr, "p": pp})
        ds = data.patient_datasets(c).set_index("key")
        plots_in.append((c, d, rr, pp, f"{marker} ({ds.loc[layer, 'unit']})", f"{row['label']} {ds.loc[row['layer'], 'unit']}"))

    tab = pd.DataFrame(rows)
    ok = {names.get(c, c): pd.DataFrame({"feature": [marker], "n": [r_["n"]], "r": [r_["r (sensitivity)"]]})
          for c, r_ in zip([p[0] for p in plots_in], rows) if r_["n"] >= MIN_N and np.isfinite(r_["r (sensitivity)"])}
    if ok:
        m = meta_correlation(ok).iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cohorts", f"{int(m['n_cohorts'])}", help=f"with ≥ {MIN_N} samples having both")
        m2.metric("Pooled r", f"{m['r']:+.2f}", help="positive = higher marker, more sensitive")
        m3.metric("Pooled p", f"{m['p']:.2g}")
        m4.metric("Direction", "consistent" if m["concordant"] else "cohorts disagree")
    else:
        st.warning(f"No cohort has ≥ {MIN_N} samples with both {marker} and this drug.")
    st.dataframe(tab, hide_index=True, width="stretch",
                 column_config={"r (sensitivity)": st.column_config.NumberColumn(format="%+.2f"),
                                "p": st.column_config.NumberColumn(format="%.2g")})
    cols = st.columns(min(len(plots_in), 3) or 1)
    for i, (c, d, rr, pp, xt, yt) in enumerate(plots_in):
        title = f"{names.get(c, c)}<br><sup>n={len(d)} · r={rr:+.2f} · p={pp:.2g}</sup>" if np.isfinite(rr) else \
                f"{names.get(c, c)}<br><sup>n={len(d)} · too few samples</sup>"
        cols[i % len(cols)].plotly_chart(plots.assoc_scatter(d, xt, yt, title=title), width="stretch",
                                         key=f"{kp}_sc_{c}")
    st.caption("Spearman correlation per cohort; cohorts combined by Fisher-z meta-analysis (weighted by n − 3). "
               "y axes show each cohort's own measure (AUC / IC50: lower = more sensitive; DSS: higher = more sensitive).")
