"""Patients → Survival: Kaplan-Meier curves and log-rank tests for any patient grouping.

Reads results/patients/<DATASET_ID>/samples.tsv (survival fields from schema/sample_annotation.tsv)
and, for expression splits, that cohort's values/<key>.tsv.gz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portal import data, plots
from portal.survival import at_risk, kaplan_meier, logrank, median_survival

# endpoint -> (time in days, event) columns of samples.tsv
ENDPOINTS = {
    "Overall survival": ("overall_survival_days", "os_event"),
    "Event-free survival": ("event_free_survival_days", "efs_event"),
}
# clinical annotation from the schema usable as groupings, with labels for binary ones
SCHEMA_GROUPINGS = {
    "risk_group": ("Risk group", None),
    "KMT2A_rearranged": ("KMT2A rearrangement", ("KMT2A-r", "KMT2A wild-type")),
    "treatment_response": ("Induction response", None),
    "disease_oncotree": ("OncoTree disease", None),
    "population": ("Population", None),
    "sex": ("Sex", None),
}
UNITS = {"months": 30.4375, "years": 365.25, "days": 1.0}
SPLITS = {"median (high vs low)": (0.5, 0.5), "tertiles (top vs bottom third)": (1 / 3, 2 / 3),
          "quartiles (top vs bottom quarter)": (0.25, 0.75)}
EXPRESSION, MUT = "High vs low expression of a gene", "mut:"
SKIP = {"MYC_high", "mTOR_high", "MYC_mTOR_category", "timepoint"}   # not survival groupings
MIN_MUTANT = 5


def _true(v: pd.Series) -> pd.Series:
    return v.astype(str).str.upper().map({"TRUE": True, "FALSE": False, "1": True, "0": False})


def _patients(cohort: str, t_col: str, e_col: str) -> pd.DataFrame:
    """One row per participant with survival data (diagnosis sample preferred)."""
    s = data.patient_samples(cohort).copy()
    s = s[pd.to_numeric(s[t_col], errors="coerce").notna() & _true(s[e_col]).notna()]
    if "timepoint" in s:
        s = s.sort_values("timepoint", key=lambda c: c.ne("diagnosis"), kind="stable")
    s = s.drop_duplicates("participant_id")
    s["_time_days"] = pd.to_numeric(s[t_col])
    s["_event"] = _true(s[e_col]).astype(bool)
    return s.reset_index(drop=True)


def _grouping_options(cohort: str, s: pd.DataFrame) -> dict[str, dict[str, str]]:
    """category -> {option key: label}. Mutation: genes with >= MIN_MUTANT mutant patients here (most
    frequent first); Gene of interest: high vs low expression; Clinical: annotation columns.
    The MYC / mTOR RNA signature groupings are not offered for survival."""
    g = data.patient_groupings(cohort)
    muts = {}
    for col in [c for c in g.index if c.startswith("mut_")] + [c for c in s if c.startswith("mut_")]:
        if col in s and col not in muts:
            n = int(_true(s[col].dropna()).sum())
            if n >= MIN_MUTANT:
                muts[col] = (n, f"{col[4:]} ({n} mutant)")
    if "mutations" in s:                       # genes outside the 20 harmonized mut_<GENE> columns
        counts = s["mutations"].dropna().astype(str).str.split(";").explode().str.strip()
        for gene, n in counts[counts != ""].value_counts().items():
            if n >= MIN_MUTANT and f"mut_{gene}" not in muts:
                muts[f"{MUT}{gene}"] = (int(n), f"{gene} ({n} mutant)")
    clin = {}
    for col, r in g.iterrows():
        if col in s and not col.startswith("mut_") and col not in SKIP and s[col].notna().any():
            clin[col] = r["label"]
    for col, (label, _) in SCHEMA_GROUPINGS.items():
        if col in s and col not in clin and s[col].notna().sum() >= 2 and s[col].nunique() > 1:
            clin[col] = label
    opts = {}
    if muts:
        opts["Mutation"] = {k: v[1] for k, v in sorted(muts.items(), key=lambda kv: -kv[1][0])}
    if _expression_layers(cohort):
        opts["Gene of interest"] = {EXPRESSION: EXPRESSION}
    if clin:
        opts["Clinical"] = clin
    return opts


def _expression_layers(cohort: str) -> list[str]:
    ds = data.patient_datasets(cohort)
    return list(ds.loc[ds["modality"] != "drug_screen", "key"])


def _groups(cohort: str, s: pd.DataFrame, key: str, kp: str) -> tuple[pd.Series, list[str], str]:
    """Group label per patient (NA = not grouped), level order, short description."""
    if key.startswith(MUT):
        gene = key[len(MUT):]
        has = s[[c for c in s if c.startswith("mut_")]].notna().any(axis=1)   # "" mutations = none found
        mutant = s["mutations"].fillna("").astype(str).str.split(";").apply(lambda m: gene in m)
        g = pd.Series(np.where(mutant, f"{gene}-mutant", f"{gene}-wild-type"), index=s.index).where(has)
        return g, [f"{gene}-mutant", f"{gene}-wild-type"], f"{gene} mutation"
    if key == EXPRESSION:
        ds = data.patient_datasets(cohort).set_index("key")
        layers = _expression_layers(cohort)
        c1, c2, c3 = st.columns([2, 2, 3])
        layer = c1.selectbox("Layer", layers, format_func=lambda k: ds.loc[k, "label"], key=f"{kp}_layer")
        f = data.patient_features(cohort, layer)
        sym = f["gene_symbol"] if "gene_symbol" in f else f.index.to_series()
        sym = sym.where(sym != "", f["label"])
        genes = sorted(sym.dropna().unique())
        gene = c2.selectbox("Gene of interest", genes, key=f"{kp}_gene",
                            index=genes.index("MAP1A") if "MAP1A" in genes else 0)
        split = c3.selectbox("Split patients at", list(SPLITS), key=f"{kp}_split")
        V = data.patient_values(cohort, layer)
        feats = [x for x in sym.index[sym == gene] if x in V.index]
        v = V.loc[feats].mean(axis=0) if feats else pd.Series(dtype=float)   # several probes: average
        v = s["sample_id"].map(v)
        lo_q, hi_q = SPLITS[split]
        lo, hi = v.quantile(lo_q), v.quantile(hi_q)
        high, low = f"{gene}-high", f"{gene}-low"
        g = pd.Series(pd.NA, index=s.index, dtype="object")
        g[v >= hi] = high
        g[v <= lo] = low
        if lo_q == hi_q:                                     # median: ties go to the high group
            g[(v > lo) & (v < hi)] = high
        return g, [high, low], f"{gene} ({ds.loc[layer, 'label']}, {split})"
    labels = None
    r = data.patient_groupings(cohort)
    if key in r.index and r.loc[key, "kind"] == "binary":
        labels = (r.loc[key, "true_label"], r.loc[key, "false_label"])
    elif key in SCHEMA_GROUPINGS:
        labels = SCHEMA_GROUPINGS[key][1]
    v = s[key]
    if labels or (v.notna().any() and _true(v.dropna()).notna().all()):
        labels = labels or (f"{key}: yes", f"{key}: no")
        g = _true(v).map({True: labels[0], False: labels[1]})
        return g, list(labels), key
    g = v.astype("string")
    return g.astype(object).where(g.notna()), list(g.dropna().value_counts().index), key


def patient_survival(kp: str = "surv") -> None:
    def has(c: str, t: str, e: str) -> bool:
        s = data.patient_samples(c)
        return {t, e} <= set(s.columns) and (s[t].notna() & s[e].notna()).sum() >= 10
    cohorts = [c for c in data.patient_cohorts() if any(has(c, t, e) for t, e in ENDPOINTS.values())]
    if not cohorts:
        st.info("No patient cohort has survival data yet.")
        return

    names = data.catalog().set_index("dataset_id")["name"].to_dict()
    c1, c2, c3, c4 = st.columns([3, 2, 3, 1])
    cohort = c1.selectbox("Cohort", cohorts, format_func=lambda c: names.get(c, c), key=f"{kp}_cohort")
    endpoints = [k for k, (t, e) in ENDPOINTS.items() if has(cohort, t, e)]
    endpoint = c2.selectbox("Endpoint", endpoints, key=f"{kp}_endpoint")
    s = _patients(cohort, *ENDPOINTS[endpoint])
    opts = _grouping_options(cohort, s)
    if not opts:
        st.warning("This cohort has no annotation to group patients by.")
        return
    by = c3.radio("Group patients by", list(opts), horizontal=True, key=f"{kp}_by")
    unit = c4.selectbox("Time in", list(UNITS), key=f"{kp}_unit")
    if by == "Mutation":
        m = opts[by]
        key = st.selectbox("Mutated gene (mutant vs wild-type)", list(m), format_func=m.get, key=f"{kp}_mut",
                           index=list(m).index("mut_TP53") if "mut_TP53" in m else 0)
    elif by == "Clinical":
        cl = opts[by]
        key = st.selectbox("Annotation", list(cl), format_func=cl.get, key=f"{kp}_clin")
    else:
        key = EXPRESSION

    grp, levels, desc = _groups(cohort, s, key, kp)
    counts = grp.value_counts()
    levels = [lv for lv in levels if counts.get(lv, 0) > 0]
    big = [lv for lv in levels if counts[lv] >= 3]
    shown = st.multiselect("Groups", levels, default=big, key=f"{kp}_levels_{key}_{desc}",
                           format_func=lambda lv: f"{lv} (n={counts[lv]})")
    if len(shown) < 1:
        st.info("Pick at least one group.")
        return
    d = s.assign(group=grp)
    d = d[d["group"].isin(shown)]
    d["time"] = d["_time_days"] / UNITS[unit]

    palette = [plots.COL_A, plots.COL_B] + [c for c in plots.CAT if c != plots.COL_B]
    colors = {lv: palette[i % len(palette)] for i, lv in enumerate(shown)}
    curves = {lv: kaplan_meier(d.loc[d["group"] == lv, "time"], d.loc[d["group"] == lv, "_event"]) for lv in shown}
    test = logrank(d["time"], d["_event"], d["group"], order=shown) if len(shown) > 1 else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Patients", f"{len(d):,}", help=f"of {len(s):,} with {endpoint.lower()} data")
    m2.metric("Events", int(d["_event"].sum()))
    m3.metric("Log-rank p", "–" if test is None or not np.isfinite(test["p"]) else f"{test['p']:.2g}")
    m4.metric(f"HR {shown[0]} vs {shown[1]}" if len(shown) == 2 else "Hazard ratio",
              f"{test['hr']:.2f}" if test and np.isfinite(test["hr"]) else "–",
              help="Log-rank estimate (O/E in the first group ÷ O/E in the second); > 1 = worse in the first group.")

    ci = st.toggle("95% confidence bands", value=len(shown) <= 3, key=f"{kp}_ci")
    st.plotly_chart(plots.kaplan_meier(curves, colors, f"Time ({unit})", ci=ci), width="stretch",
                    key=f"{kp}_km")

    ticks = np.linspace(0, np.nanpercentile(d["time"], 95), 6).round(0 if unit != "years" else 1)
    st.caption("Number at risk")
    st.dataframe(at_risk(d["time"], d["group"], ticks).reindex(shown), width="stretch")

    summary = pd.DataFrame([{
        "group": lv, "patients": int((d["group"] == lv).sum()), "events": int(d.loc[d["group"] == lv, "_event"].sum()),
        f"median ({unit})": median_survival(curves[lv]),
        **({"observed": test["observed"].get(lv), "expected": test["expected"].get(lv)} if test else {}),
    } for lv in shown])
    st.dataframe(summary, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.1f")
                                for c in [f"median ({unit})", "observed", "expected"]})
    if test:
        st.caption(f"{endpoint}, {names.get(cohort, cohort)}, grouped by {desc}. Log-rank χ² = {test['chi2']:.2f} "
                   f"on {test['df']} df. Median = not reached when survival stays above 50%.")
    out = pd.concat([km.assign(group=lv) for lv, km in curves.items()])[["group"] + list(curves[shown[0]].columns)]
    st.download_button("Download Kaplan–Meier table (TSV)", out.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                       file_name=f"KM_{cohort}_{endpoint.replace(' ', '_')}_{desc}.tsv".replace(" ", "_")
                       .replace("/", "-").replace(",", "").replace("(", "").replace(")", ""))
