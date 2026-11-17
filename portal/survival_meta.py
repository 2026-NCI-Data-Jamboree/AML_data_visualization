"""Highlights → Survival: which genes (or proteins, phosphosites) are linked to survival across cohorts.

Each cohort fits a Cox regression of survival on every feature (standardized, so the hazard ratio is per
1 SD), one sample per patient (the diagnosis sample when there is one); the log hazard ratios are combined
by fixed-effect inverse-variance meta-analysis, and p values are adjusted for the number of features (FDR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portal import data, plots
from portal.dependency import selected_genes
from portal.stats import cox_rows, meta_hazard
from portal.survival import kaplan_meier, logrank
from portal.survival_view import ENDPOINTS, _true

EXPR, PROT, PHOS = "Gene expression", "Protein", "Phosphosite"
MODALITY = {EXPR: "transcriptomics", PROT: "proteomics", PHOS: "phosphoproteomics"}
MIN_N, MIN_EVENTS = 20, 10      # per cohort
MAX_SHOWN = 3
YEAR = 365.25
RISK_NONE, RISK_ADJ, RISK_WITHIN = "Not used", "Adjusted for risk group", "Within one risk group"
RISK_LEVELS = ["favorable", "intermediate", "adverse"]


def _patients(cohort: str, layer: str, endpoint: str, risk: tuple = (RISK_NONE,)) -> pd.DataFrame:
    """sample_id, time (days), event, risk_group: one sample per patient with data in the layer (diagnosis
    first). risk = (RISK_ADJ,): only patients with a risk group; (RISK_WITHIN, level): only that group."""
    t_col, e_col = ENDPOINTS[endpoint]
    s = data.patient_samples(cohort)
    if not {t_col, e_col} <= set(s.columns):
        return pd.DataFrame(columns=["sample_id", "time", "event", "risk_group"])
    cols = set(data.patient_values(cohort, layer).columns)
    s = s[s["sample_id"].isin(cols)].copy()
    s["time"] = pd.to_numeric(s[t_col], errors="coerce")
    s["event"] = _true(s[e_col])
    s = s[s["time"].notna() & s["event"].notna()]
    if "timepoint" in s:
        s = s.sort_values("timepoint", key=lambda t: t.ne("diagnosis"), kind="stable")
    s = s.drop_duplicates("participant_id")
    rg = s["risk_group"] if "risk_group" in s else pd.Series(pd.NA, index=s.index, dtype=object)
    s = s.assign(risk_group=rg.where(rg.isin(RISK_LEVELS)))
    if risk[0] == RISK_ADJ:
        s = s[s["risk_group"].notna()]
    elif risk[0] == RISK_WITHIN:
        s = s[s["risk_group"] == risk[1]]
    return s[["sample_id", "time", "event", "risk_group"]].assign(event=lambda d: d["event"].astype(bool))


def _layer(cohort: str, what: str) -> str | None:
    ds = data.patient_datasets(cohort)
    keys = list(ds.loc[ds["modality"] == MODALITY[what], "key"])
    return keys[0] if keys else None


@st.cache_data(show_spinner="Fitting Cox models…")
def _cox(cohort: str, layer: str, endpoint: str, risk: tuple = (RISK_NONE,)) -> pd.DataFrame:
    """Cox regression of the endpoint on every feature of the layer, keyed by gene symbol / site label.
    Adjusted for risk group: stratified Cox (own baseline hazard per risk group)."""
    p = _patients(cohort, layer, endpoint, risk).set_index("sample_id")
    if len(p) < MIN_N or p["event"].sum() < MIN_EVENTS:
        return pd.DataFrame(columns=["feature", "n", "events", "beta", "se", "z", "p"])
    V = data.patient_values(cohort, layer)
    rna = data.patient_datasets(cohort).set_index("key").loc[layer, "modality"] == "transcriptomics"
    res = cox_rows(V[p.index], p["time"], p["event"], min_n=MIN_N, min_events=MIN_EVENTS,
                   expressed_above=1.0 if rna else None,       # RNA: expressed (log2 >= 1) in >= 20%
                   strata=p["risk_group"] if risk[0] == RISK_ADJ else None)
    res["feature"] = res["feature"].map(data.feature_names(cohort, layer)).fillna(res["feature"])
    return res.drop_duplicates("feature")


def _km_by_risk(p: pd.DataFrame, item: str):
    """Kaplan-Meier curves per risk group x high / low (median split within each risk group), and a
    per-risk-group table: patients, events, log-rank HR (high vs low) and p."""
    curves, colors, dashes, rows = {}, {}, {}, []
    for rg in RISK_LEVELS:
        d = p[p["risk_group"] == rg]
        if len(d) < 4:
            continue
        hi = d["v"] >= d["v"].median()
        labs = {True: f"{rg} · high", False: f"{rg} · low"}
        g = hi.map(labs)
        for k in (True, False):
            lab = labs[k]
            sel = g == lab
            if sel.any():
                curves[lab] = kaplan_meier(d.loc[sel, "years"], d.loc[sel, "event"])
                colors[lab] = plots.RISK_COLORS[rg]
                dashes[lab] = "solid" if k else "dash"
        lr = logrank(d["years"], d["event"], g, order=[labs[True], labs[False]])
        rows.append({"risk group": rg, "patients": len(d), "events": int(d["event"].sum()),
                     "HR high vs low": lr["hr"], "log-rank p": lr["p"]})
    fig = plots.kaplan_meier(curves, colors, "Years", ci=False, dashes=dashes)
    return fig, pd.DataFrame(rows)


def survival_association(kp: str = "sv") -> None:
    names = data.patient_cohort_names()
    st.markdown(
        "Which genes are linked to survival? In each cohort, every gene's level is related to survival with a "
        "Cox regression (one sample per patient; hazard ratio per standard deviation), and the cohorts are "
        "combined by meta-analysis. **Hazard ratio above 1 = higher level, shorter survival.**")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    what = c1.selectbox("Features", [EXPR, PROT, PHOS], key=f"{kp}_what")
    endpoint = c2.selectbox("Endpoint", list(ENDPOINTS), key=f"{kp}_endpoint")
    rmode = c3.selectbox("Risk group", [RISK_NONE, RISK_ADJ, RISK_WITHIN], index=1, key=f"{kp}_risk",
                         help="Adjusted: is the gene linked to survival beyond the risk group (Cox model stratified "
                              "by risk group)? Within one risk group: only patients in that group. Both use "
                              "only cohorts that report a risk group (ELN or cytogenetic).")
    risk = (rmode,)
    if rmode == RISK_WITHIN:
        risk = (rmode, c4.selectbox("Risk group to analyse", RISK_LEVELS, index=2, key=f"{kp}_risk_level"))
    avail, no_risk = {}, []
    for c in names:
        layer = _layer(c, what)
        if layer:
            p = _patients(c, layer, endpoint, risk)
            if len(p) >= MIN_N and p["event"].sum() >= MIN_EVENTS:
                avail[c] = (layer, len(p), int(p["event"].sum()))
            elif rmode != RISK_NONE and len(_patients(c, layer, endpoint)) >= MIN_N:
                no_risk.append(names[c])
    if no_risk:
        st.caption(("Not used (no risk group, or too few patients / events in it): " if rmode == RISK_WITHIN
                    else "Not used (no risk group reported): ") + ", ".join(no_risk) + ".")
    if not avail:
        st.warning(f"No cohort has {endpoint.lower()} for at least {MIN_N} patients ({MIN_EVENTS} events) "
                   f"with {what.lower()} data" + (" and this risk group." if rmode == RISK_WITHIN else "."))
        return
    picked = st.multiselect("Cohorts", list(avail), default=list(avail),
                            format_func=lambda c: f"{names[c]} (n={avail[c][1]}, {avail[c][2]} events)",
                            key=f"{kp}_cohorts_{what}_{endpoint}_{'-'.join(risk)}")
    per = {c: _cox(c, avail[c][0], endpoint, risk) for c in picked}
    per = {c: d for c, d in per.items() if not d.empty}
    if not per:
        st.info("Pick at least one cohort.")
        return

    meta = meta_hazard(per).rename(columns={"feature": "gene"})
    meta["r"] = np.log2(meta["hr"])                          # x axis of the volcano
    fdr = st.select_slider("FDR threshold", [0.01, 0.05, 0.1, 0.25], value=0.05, key=f"{kp}_fdr")
    pos, neg = "shorter survival", "longer survival"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cohorts", len(per))
    m2.metric({EXPR: "Genes tested", PROT: "Proteins tested", PHOS: "Sites tested"}[what], f"{len(meta):,}")
    m3.metric(f"Higher level, shorter survival (FDR < {fdr})",
              int(((meta["fdr"] < fdr) & (meta["hr"] > 1) & meta["concordant"]).sum()))
    m4.metric(f"Higher level, longer survival (FDR < {fdr})",
              int(((meta["fdr"] < fdr) & (meta["hr"] < 1) & meta["concordant"]).sum()))
    st.caption("Click a point, or drag a box / lasso around several, to filter the table. "
               "Double-click the plot to clear the selection.")
    event = st.plotly_chart(
        plots.meta_volcano(meta, "pooled log2 hazard ratio per SD (positive = higher level, shorter survival)",
                           fdr=fdr, pos=pos, neg=neg, y="fdr", x_label="pooled log2 HR"),
        width="stretch", on_select="rerun", selection_mode=("points", "box", "lasso"),
        key=f"{kp}_volcano_{what}_{endpoint}_{'-'.join(risk)}_{'-'.join(per)}")
    selected = selected_genes(event)

    key_col = "site" if what == PHOS else "gene"
    table = meta[["gene", "hr", "p", "fdr", "n_cohorts", "n_total", "events", "concordant", "I2"]
                 + [f"hr_{c}" for c in per]].rename(
        columns={**{f"hr_{c}": f"HR {names[c]}" for c in per}, "gene": key_col, "hr": "HR per SD"})
    view = table[table[key_col].isin(selected)] if selected else table
    tbl = st.dataframe(view, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(view)),
                       on_select="rerun", selection_mode="multi-row",
                       key=f"{kp}_table_{what}_{endpoint}_{'-'.join(risk)}_{'-'.join(per)}_{len(view)}",
                       column_config={c: st.column_config.NumberColumn(format="%.3g") for c in view.columns
                                      if c in ("HR per SD", "p", "fdr", "I2") or c.startswith("HR ")})
    st.caption("Select rows in the table (tick boxes on the left) to show those features below; "
               "otherwise the top row is shown. *concordant* = same direction in every cohort; "
               "I2 = how much the cohorts disagree (%).")
    st.download_button("Download table (TSV)", view.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                       file_name=f"survival_{what.lower().replace(' ', '_')}_{endpoint.lower().replace(' ', '_')}"
                       + {RISK_NONE: "", RISK_ADJ: "_risk_adjusted"}.get(rmode, f"_{risk[-1]}_risk") + ".tsv")

    rows = list(tbl.selection.rows) if tbl and tbl.selection else []
    picked_rows = view.iloc[rows][key_col].tolist() if rows else []
    shown = (picked_rows or view[key_col].head(1).tolist())[:MAX_SHOWN]
    if not shown:
        return
    if len(picked_rows) > MAX_SHOWN:
        st.caption(f"Showing the first {MAX_SHOWN} selected rows.")
    st.subheader("Per-cohort survival" + ("" if picked_rows else f": {shown[0]} (top row)"))
    for item in shown:
        r = meta.set_index("gene").loc[item]
        lo, hi = np.exp(r["log_hr"] - 1.96 * r["se"]), np.exp(r["log_hr"] + 1.96 * r["se"])
        st.markdown(f"**{item}**: pooled HR per SD = {r['hr']:.2f} (95% CI {lo:.2f}–{hi:.2f}), p = {r['p']:.2g}, "
                    f"FDR = {r['fdr']:.2g}{'' if r['concordant'] else ' (cohorts disagree on direction)'}")
        frows = []
        for c, d in per.items():
            x = d.set_index("feature")
            if item in x.index:
                b, se = x.loc[item, "beta"], x.loc[item, "se"]
                frows.append({"label": names[c], "hr": np.exp(b), "lo": np.exp(b - 1.96 * se),
                              "hi": np.exp(b + 1.96 * se), "n": int(x.loc[item, "n"]),
                              "events": int(x.loc[item, "events"])})
        st.plotly_chart(plots.forest(pd.DataFrame(frows), {"hr": r["hr"], "lo": lo, "hi": hi}), width="stretch",
                        key=f"{kp}_forest_{item}")
        cols = [st] if rmode == RISK_ADJ else st.columns(min(len(per), 3))   # 6 curves: one cohort per row
        for i, c in enumerate(per):
            layer = avail[c][0]
            sym = data.feature_names(c, layer)
            V = data.patient_values(c, layer)
            feats = [f for f in sym.index[sym == item] if f in V.index]
            col = cols[i % len(cols)]
            if not feats:
                col.caption(f"{names[c]}: not measured")
                continue
            p = _patients(c, layer, endpoint, risk).set_index("sample_id")
            v = V.loc[feats].mean(axis=0).reindex(p.index)
            p = p[v.notna()].assign(v=v.dropna())
            p["years"] = p["time"] / YEAR
            x = per[c].set_index("feature")
            if rmode == RISK_ADJ:
                fig, by_risk = _km_by_risk(p, item)
                cox = (f"risk-adjusted Cox HR {np.exp(x.loc[item, 'beta']):.2f}/SD, p={x.loc[item, 'p']:.2g}"
                       if item in x.index else "Cox: too few patients")
                sub = f"n={len(p)} · {cox}"
            else:
                hi_lab, lo_lab = f"{item}-high", f"{item}-low"
                p["group"] = np.where(p["v"] >= p["v"].median(), hi_lab, lo_lab)
                curves = {g: kaplan_meier(p.loc[p["group"] == g, "years"], p.loc[p["group"] == g, "event"])
                          for g in (hi_lab, lo_lab)}
                lr = logrank(p["years"], p["event"], p["group"], order=[hi_lab, lo_lab])
                cox = (f"Cox HR {np.exp(x.loc[item, 'beta']):.2f}/SD, p={x.loc[item, 'p']:.2g}"
                       if item in x.index else "Cox: too few patients")
                fig = plots.kaplan_meier(curves, {hi_lab: plots.COL_B, lo_lab: plots.COL_A}, "Years", ci=False)
                sub = f"n={len(p)} · {cox} · log-rank p={lr['p']:.2g}"
                by_risk = None
            fig.update_layout(height=330 if by_risk is None else 460,
                              title=dict(text=f"{names[c]}<br><sup>{sub}</sup>", font=dict(size=13)),
                              margin=dict(t=60, l=10, r=10, b=10), legend=dict(y=-0.25))
            col.plotly_chart(fig, width="stretch", key=f"{kp}_km_{item}_{c}")
            if by_risk is not None:
                col.dataframe(by_risk, hide_index=True, width="stretch",
                              column_config={"HR high vs low": st.column_config.NumberColumn(format="%.2f"),
                                             "log-rank p": st.column_config.NumberColumn(format="%.2g")})
    st.caption("Kaplan–Meier curves split each cohort at its median level (high = red, low = blue), for display; "
               "the statistics use the continuous level (Cox regression). Each patient counts once."
               + (" Adjusted for risk group: colour = risk group (green favorable, amber intermediate, red adverse); "
                  "solid = high, dashed = low, split at the median within each risk group, so each curve pair "
                  "compares patients of the same risk. The table under each plot gives that comparison per risk "
                  "group (log-rank); the title gives the risk-adjusted Cox result (stratified by risk group)."
                  if rmode == RISK_ADJ else "")
               + (f" Only {risk[1]}-risk patients." if rmode == RISK_WITHIN else ""))
