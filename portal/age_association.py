"""Patients → Age: which genes (or drug responses) change with age at diagnosis, across cohorts.

Each cohort correlates every feature with age (Spearman, one sample per patient, the diagnosis sample
when there is one); cohorts are combined by the same Fisher-z meta-analysis as the drug response tab.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portal import data, plots
from portal.dependency import selected_genes
from portal.stats import meta_correlation, spearman_rows

MIN_N = 10        # continuous age: patients per cohort
MIN_GROUP = 3     # age groups: patients per group per cohort
MAX_SHOWN = 6
EXPR, PROT, PHOS, DRUG = "Gene expression", "Protein", "Phosphosite", "Drug response"
MODALITY = {EXPR: "transcriptomics", PROT: "proteomics", PHOS: "phosphoproteomics"}
CONT, GROUPS = "Continuous age", "Age groups"
OTHERS = "all other ages"


def one_per_patient(cohort: str) -> pd.DataFrame:
    """sample_id, participant_id, age_years: one sample per patient, diagnosis sample preferred."""
    s = data.patient_samples(cohort)
    if "age_years" not in s:
        return pd.DataFrame(columns=["sample_id", "participant_id", "age_years"])
    if "timepoint" in s:
        s = s.sort_values("timepoint", key=lambda t: t.ne("diagnosis"), kind="stable")
    s = s.drop_duplicates("participant_id")
    s = s[pd.to_numeric(s["age_years"], errors="coerce").notna()]
    return s[["sample_id", "participant_id", "age_years"]].assign(age_years=lambda d: d["age_years"].astype(float))


def _target(cohort: str, mode: tuple) -> pd.Series:
    """Per-sample y: age in years ("continuous",), or 1 = group A / 0 = group B ("groups", A, B)."""
    p = one_per_patient(cohort).set_index("sample_id")
    if mode[0] == "continuous":
        return p["age_years"]
    g = plots.age_groups(p["age_years"])
    a, b = mode[1], mode[2]
    in_b = (g != a) & (g != plots.AGE_GROUPS[3]) if b == OTHERS else (g == b)
    y = pd.Series(np.nan, index=p.index)
    y[g == a] = 1.0
    y[in_b] = 0.0
    return y.dropna()


@st.cache_data(show_spinner="Testing association with age…")
def _assoc(cohort: str, layer: str, mode: tuple, drug: bool, lower_is_stronger: bool) -> pd.DataFrame:
    """Spearman r of every feature with age (or with group A vs B), keyed by gene symbol / compound_key.
    Drugs: positive = more sensitive with older age / in group A."""
    V = data.patient_values(cohort, layer)
    y = _target(cohort, mode)
    y = y[y.index.isin(V.columns)]
    min_n = MIN_N if mode[0] == "continuous" else 2 * MIN_GROUP
    if mode[0] != "continuous" and (y.eq(1).sum() < MIN_GROUP or y.eq(0).sum() < MIN_GROUP):
        return pd.DataFrame(columns=["feature", "n", "r", "p"])
    res = spearman_rows(V, y, min_n=min_n)
    f = data.patient_features(cohort, layer)
    if drug:
        key = f["compound_key"].where(f["compound_key"] != "", f.index.to_series())
        if lower_is_stronger:
            res["r"] = -res["r"]
    else:
        key = data.feature_names(cohort, layer)
    res["feature"] = res["feature"].map(key).fillna(res["feature"])
    return res.drop_duplicates("feature")


def age_association(kp: str = "age") -> None:
    names = data.patient_cohort_names()
    cohorts = [c for c in names if len(one_per_patient(c)) >= MIN_N]
    if not cohorts:
        st.info("No patient cohort has age information.")
        return
    st.markdown(
        "Which genes, or responses to which drugs, change with age at diagnosis? Each cohort correlates every "
        "feature with age (Spearman, one sample per patient), and the cohorts are combined by meta-analysis. "
        "The pediatric cohort (Wang 2022) adds the youngest patients; the others are mostly adults.")

    c1, c2, c3 = st.columns([2, 2, 3])
    what = c1.selectbox("Features", [EXPR, PROT, PHOS, DRUG], key=f"{kp}_what")
    how = c2.radio("Age as", [CONT, GROUPS], horizontal=True, key=f"{kp}_how")
    picked = c3.multiselect("Cohorts", cohorts, default=cohorts, format_func=names.get, key=f"{kp}_cohorts")
    bands = plots.AGE_GROUPS[:3]
    if how == GROUPS:
        g1, g2, _ = st.columns([2, 2, 3])
        ga = g1.selectbox("Group A (blue)", bands, index=bands.index("AYA (15-39)"), key=f"{kp}_ga")
        rest = [b for b in bands if b != ga] + [OTHERS]
        gb = g2.selectbox("Group B (red)", rest, index=rest.index("adult (≥40)") if "adult (≥40)" in rest else 0,
                          key=f"{kp}_gb")
        mode = ("groups", ga, gb)
    else:
        mode = ("continuous",)

    if how == GROUPS:                         # say which cohorts have both groups
        def enough(c: str) -> bool:
            y = _target(c, mode)
            return y.eq(1).sum() >= MIN_GROUP and y.eq(0).sum() >= MIN_GROUP
        used = [names[c] for c in picked if enough(c)]
        skipped = [names[c] for c in picked if not enough(c)]
        st.caption(f"Cohorts compared: {', '.join(used) or 'none'}."
                   + (f" Not used (fewer than {MIN_GROUP} patients in a group): {', '.join(skipped)}." if skipped else ""))

    per, labels, units = {}, {}, {}
    for c in picked:
        ds = data.patient_datasets(c)
        if what != DRUG:
            keys = list(ds.loc[ds["modality"] == MODALITY[what], "key"])
            if keys:
                per[c] = _assoc(c, keys[0], mode, False, False)
                units[c] = (keys[0], ds.set_index("key").loc[keys[0], "unit"])
        else:
            d = ds[ds["modality"] == "drug_screen"]
            if "duplicate_of" in d:                     # e.g. PTRC drug AUC = Beat AML's; count it once
                src = d["duplicate_of"].fillna("").astype(str)
                d = d[src.eq("") | ~src.str.split("/").str[0].isin(picked)]
            if len(d):                                  # prefer sDSS > AUC > DSS, as in the drug response tab
                order = {"drug_sDSS": 0, "drug_auc": 1, "drug_DSS": 2}
                row = d.assign(o=d["key"].map(order).fillna(9)).sort_values("o").iloc[0]
                lower = str(row["lower_is_stronger"]).upper() == "TRUE"
                per[c] = _assoc(c, row["key"], mode, True, lower)
                units[c] = (row["key"], row["unit"])
                f = data.patient_features(c, row["key"])
                labels.update(f.drop_duplicates("compound_key").set_index("compound_key")["label"].to_dict())
        if c in per and (per[c].empty or int(per[c]["n"].max()) < (MIN_N if how == CONT else 2 * MIN_GROUP)):
            per.pop(c)
    if not per:
        st.warning(f"No cohort has enough patients with both age and {what.lower()}"
                   + (f" in {mode[1]} and {mode[2]}." if how == GROUPS else "."))
        return

    meta = meta_correlation(per).rename(columns={"feature": "gene"})
    if what == DRUG:
        meta["gene"] = meta["gene"].map(lambda k: labels.get(k, k))
    fdr = st.select_slider("FDR threshold", [0.01, 0.05, 0.1, 0.25], value=0.05, key=f"{kp}_fdr")
    if how == CONT:
        pos, neg = ("higher with age", "lower with age") if what != DRUG else ("more sensitive with age", "less sensitive with age")
    else:
        short = {b: b.split(" (")[0] for b in bands} | {OTHERS: "other ages"}
        A, B = short[mode[1]], short[mode[2]]
        pos, neg = ((f"higher in {A}", f"higher in {B}") if what != DRUG
                    else (f"more sensitive in {A}", f"more sensitive in {B}"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cohorts", len(per))
    m2.metric({EXPR: "Genes tested", PROT: "Proteins tested", PHOS: "Sites tested"}.get(what, "Drugs tested"),
              f"{len(meta):,}")
    m3.metric(pos[0].upper() + pos[1:] + f", FDR < {fdr}", int(((meta["fdr"] < fdr) & (meta["r"] > 0) & meta["concordant"]).sum()))
    m4.metric(neg[0].upper() + neg[1:] + f", FDR < {fdr}", int(((meta["fdr"] < fdr) & (meta["r"] < 0) & meta["concordant"]).sum()))
    st.caption("Click a point, or drag a box / lasso around several, to see the per-cohort data below. "
               "Double-click the plot to clear the selection.")
    if how == CONT:
        x_title = ("pooled Spearman r with age (positive = higher in older patients)" if what != DRUG else
                   "pooled Spearman r with age (positive = more sensitive in older patients)")
    else:
        x_title = (f"pooled rank effect, {A} vs {B} (positive = {pos})")
    event = st.plotly_chart(plots.meta_volcano(meta, x_title, fdr=fdr, pos=pos, neg=neg), width="stretch",
                            on_select="rerun", selection_mode=("points", "box", "lasso"),
                            key=f"{kp}_volcano_{what}_{'-'.join(mode)}_{'-'.join(picked)}")
    selected = selected_genes(event)

    table = meta[["gene", "r", "p", "fdr", "n_cohorts", "n_total", "concordant", "I2"] + [f"r_{c}" for c in per]].rename(
        columns={**{f"r_{c}": f"r {names[c]}" for c in per},
                 "gene": "drug" if what == DRUG else ("site" if what == PHOS else "gene")})
    key_col = "drug" if what == DRUG else ("site" if what == PHOS else "gene")
    if selected:
        view = table[table[key_col].isin(selected)]
    else:
        q = st.text_input("Filter table", placeholder="e.g. venetoclax" if what == DRUG else "e.g. HOXA9, MECOM",
                          key=f"{kp}_filter")
        view = table if not q else table[table[key_col].str.contains(q, case=False, na=False, regex=False)]
    st.dataframe(view, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(view)),
                 column_config={c: st.column_config.NumberColumn(format="%.3g") for c in view.columns
                                if c in ("r", "p", "fdr", "I2") or c.startswith("r ")})
    st.download_button("Download table (TSV)", view.to_csv(sep="\t", index=False), key=f"{kp}_dl",
                       file_name=f"age_association_{what.lower().replace(' ', '_')}.tsv")

    inv = {v: k for k, v in labels.items()}
    noun = {DRUG: "drug", PHOS: "site", PROT: "protein"}.get(what, "gene")
    st.subheader("Per-cohort data")
    options = meta["gene"].tolist()                              # ranked: strongest association first
    default = [x for x in selected if x in options][:MAX_SHOWN] or options[:1]
    shown = st.multiselect(
        f"Show {noun}s", options, default=default, max_selections=MAX_SHOWN, key=f"{kp}_show_{what}_{'-'.join(mode)}_{'|'.join(default)}",
        help=f"Type to search any {noun}. Starts with the {noun}s selected on the plot, or the top hit.")
    if not shown:
        st.info(f"Pick a {noun} to see its data in each cohort.")
        return
    for item in shown:
        r = meta.set_index("gene").loc[item]
        st.markdown(f"**{item}**: pooled r = {r['r']:.2f}, p = {r['p']:.2g}, FDR = {r['fdr']:.2g}"
                    f"{'' if r['concordant'] else ' (cohorts disagree on direction)'}")
        cols = st.columns(min(len(per), 4))
        for i, c in enumerate(per):
            layer, unit = units[c]
            f = data.patient_features(c, layer)
            if what != DRUG:
                sym = data.feature_names(c, layer)
                feats = list(sym.index[sym == item])
            else:
                feats = list(f.index[f["compound_key"] == inv.get(item, item)])
            V = data.patient_values(c, layer)
            feats = [x for x in feats if x in V.index]
            if not feats:
                cols[i % len(cols)].caption(f"{names[c]}: not measured")
                continue
            p = one_per_patient(c).set_index("sample_id")
            d = pd.DataFrame({"x": p["age_years"], "y": V.loc[feats].mean(axis=0)}).dropna()
            d = d.rename_axis("sample_id").reset_index()
            d["group"] = plots.age_groups(d["x"])
            fkey = inv.get(item, item) if what == DRUG else item
            rc = per[c].set_index("feature")["r"].get(fkey, np.nan)
            pc = per[c].set_index("feature")["p"].get(fkey, np.nan)
            title = f"{names[c]}<br><sup>n={len(d)} · r={rc:+.2f} · p={pc:.2g}</sup>"
            if how == CONT:
                fig = plots.assoc_scatter(d, "age at diagnosis (years)", f"{item} ({unit})", title=title,
                                          group_colors=plots.AGE_COLORS)
            else:
                y01 = _target(c, mode)
                d["cmp"] = d["sample_id"].map(y01).map({1.0: mode[1], 0.0: mode[2]}).fillna("not compared")
                d["cell_line"] = d["sample_id"]                           # hover label used by strip()
                fig = plots.strip(d, "y", "cmp", [mode[1], mode[2], "not compared"], title=title,
                                  y_title=f"{item} ({unit})",
                                  colors={mode[1]: plots.COL_A, mode[2]: plots.COL_B, "not compared": plots.COL_NS})
                fig.update_layout(title=dict(font=dict(size=13)), margin=dict(t=60))
            cols[i % len(cols)].plotly_chart(fig, width="stretch", key=f"{kp}_sc_{item}_{c}")
    st.caption(("Points are coloured by age group (pediatric < 15, AYA 15–39, adult ≥ 40)." if how == CONT else
                "Group comparison: Spearman correlation with group membership (1 = A, 0 = B), equivalent to a "
                "rank-sum test; r is the rank effect size.")
               + (" For drugs, r is shown with the sensitivity sign (positive = more sensitive in older patients); "
                  "the y axes show each cohort's own measure (AUC: lower = more sensitive; DSS: higher = more sensitive)."
                  if what == DRUG else ""))
