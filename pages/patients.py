import numpy as np
import pandas as pd
import streamlit as st

from portal import data, plots
from portal.age_association import age_association
from portal.drug_biomarkers import drug_marker_lookup
from portal.feature_query import patient_feature_query
from portal.survival_view import patient_survival

st.title("Patients")
st.markdown("AML patient cohorts, adult and pediatric, with RNA expression, mutations, clinical "
            "information and drug response measured on patient cells in the lab.")

tab_overview, tab_fq, tab_drug, tab_age, tab_surv = st.tabs(
    ["Overview", "Feature query", "Drug response", "Age association", "Survival"])

with tab_overview:
    names = data.patient_cohort_names()
    c1, c3, c4 = st.columns(3)
    c1.metric("Cohorts", len(names))
    n_samples = sum(len(data.patient_samples(c)) for c in names)
    c3.metric("Patient samples", f"{n_samples:,}")
    c4.metric("Patients", f"{len(set().union(*[set(data.patient_samples(c)['participant_id'].astype(str)) for c in names])):,}",
              help="Counted once across cohorts (PTRC specimens come from Beat AML patients)")

    rows = []
    for c in names:
        s = data.patient_samples(c)
        ds = data.patient_datasets(c)
        surv = [lab for lab, (t, e) in {"OS": ("overall_survival_days", "os_event"),
                                         "EFS": ("event_free_survival_days", "efs_event")}.items()
                if {t, e} <= set(s.columns) and (s[t].notna() & s[e].notna()).sum() >= 10]
        rows.append({"cohort": names[c], "id": c,
                     "population": ", ".join(sorted(s["population"].dropna().unique())),
                     "samples": len(s), "patients": s["participant_id"].nunique(),
                     **{lab: int(n) for lab, n in zip(ds["label"], ds["n_samples"])},
                     "survival": ", ".join(surv) or "–"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No patient cohort prepared yet.")

    if names:
        st.subheader("Samples and data availability")
        summ = {}
        for c in names:
            optional = ["risk_group", "KMT2A_rearranged", "overall_survival_days", "os_event",
                        "event_free_survival_days", "efs_event"]      # schema fields a cohort may lack
            sm = data.patient_samples(c)
            sm = sm.reindex(columns=list(sm.columns) + [x for x in optional if x not in sm])
            ds = data.patient_datasets(c)
            av = data.patient_availability(c)
            drug = av[ds.loc[ds["modality"] == "drug_screen", "key"]].any(axis=1)
            rna = av[ds.loc[ds["modality"] == "transcriptomics", "key"]].any(axis=1)
            prot = av[ds.loc[ds["modality"] == "proteomics", "key"]].any(axis=1)
            phos = av[ds.loc[ds["modality"] == "phosphoproteomics", "key"]].any(axis=1)
            pts = sm.drop_duplicates("participant_id")
            summ[names[c]] = {
                "samples": len(sm),
                "patients": len(pts),
                "RNA expression": int(rna.sum()),
                "drug response": int(drug.sum()),
                "protein": int(prot.sum()),
                "phosphosites": int(phos.sum()),
                "RNA + drug": int((rna & drug).sum()),
                "protein + drug": int((prot & drug).sum()),
                "mutation calls": int(sm[[x for x in sm if x.startswith("mut_")]].notna().any(axis=1).sum()),
                "risk group": int(sm["risk_group"].notna().sum()),
                "KMT2A status": int(sm["KMT2A_rearranged"].notna().sum()),
                "OS (patients)": int((pts["overall_survival_days"].notna() & pts["os_event"].notna()).sum()),
                "EFS (patients)": int((pts["event_free_survival_days"].notna() & pts["efs_event"].notna()).sum()),
            }
        summ = pd.DataFrame(summ).T
        st.plotly_chart(plots.availability_summary(summ), width="stretch")
        st.caption("Number of samples with each data type (survival: number of patients). Colour = share of the "
                   "cohort's samples (patients for survival). RNA or protein + drug = samples usable for the drug response analysis.")


    if names:
        st.subheader("Age composition")
        comp = []
        for c in names:
            p = data.patient_samples(c).drop_duplicates("participant_id")
            comp.append(plots.age_groups(p.get("age_years", pd.Series(index=p.index, dtype=float))).value_counts().reindex(plots.AGE_GROUPS, fill_value=0)
                        .rename(names[c]))
        comp = pd.DataFrame(comp)
        comp = comp.loc[:, (comp > 0).any()]
        pct = st.toggle("Show as % of patients", key="pt_age_pct")
        st.plotly_chart(plots.composition_bar(comp, percent=pct), width="stretch")
        st.caption("Age at diagnosis, one row per patient. NCI age bands: pediatric < 15, "
                   "AYA (adolescent and young adult) 15–39, adult ≥ 40.")

    if names:
        st.subheader("Risk and genomic classification")
        # one row per patient, diagnosis sample preferred
        pat = {}
        for c in names:
            sm = data.patient_samples(c)
            if "timepoint" in sm:
                sm = sm.sort_values("timepoint", key=lambda t: t.ne("diagnosis"), kind="stable")
            pat[c] = sm.drop_duplicates("participant_id")
        risk = pd.DataFrame({names[c]: p.get("risk_group", pd.Series(index=p.index, dtype=object))
                             .fillna("not classified").value_counts().reindex(plots.RISK_GROUPS, fill_value=0)
                             for c, p in pat.items()}).T
        risk = risk.loc[:, (risk > 0).any()]
        pct_r = st.toggle("Show as % of patients", key="pt_risk_pct")
        st.plotly_chart(plots.composition_bar(risk, percent=pct_r, colors=plots.RISK_COLORS), width="stretch")
        src = {names[c]: ", ".join(sorted(p["risk_classification"].dropna().unique()))
               for c, p in pat.items() if "risk_classification" in p}
        st.caption("Risk group per patient (diagnosis sample when available). Classification used: "
                   + "; ".join(f"{k}: {v or 'none'}" for k, v in src.items()) + ".")

        genes = [c for c in ["KMT2A_rearranged"] + [f"mut_{x}" for x in plots.AML_GENES]]
        n_mut, n_tested = {}, {}
        for c, p in pat.items():
            row_m, row_t = {}, {}
            for col in genes:
                v = p[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False}) if col in p else pd.Series(dtype=object)
                row_t[col] = int(v.notna().sum())
                row_m[col] = int((v == True).sum())  # noqa: E712
            n_mut[names[c]], n_tested[names[c]] = row_m, row_t
        n_mut, n_tested = pd.DataFrame(n_mut).T, pd.DataFrame(n_tested).T
        order = (n_mut.sum() / n_tested.sum().replace(0, np.nan)).sort_values(ascending=False).index
        order = ["KMT2A_rearranged"] + [g for g in order if g != "KMT2A_rearranged"]
        labels = {g: "KMT2A-r" if g == "KMT2A_rearranged" else g[4:] for g in order}
        st.plotly_chart(plots.frequency_heatmap(n_mut[order].rename(columns=labels),
                                                n_tested[order].rename(columns=labels)), width="stretch")
        st.caption("Share of patients with the alteration among patients tested for it (hover for counts); "
                   "– = not assayed in that cohort. Lee 2018 reports NPM1 and FLT3 only.")

    if names:
        st.subheader("Per cohort: samples and data availability")
        c1, c2 = st.columns([2, 3])
        cohort = c1.selectbox("Cohort", list(names), format_func=names.get, key="pt_cohort")
        g = data.patient_groupings(cohort)
        signature = ["MYC_high", "mTOR_high", "MYC_mTOR_category"]      # not shown in this overview
        track_opts = [c for c in g.index if c not in signature]
        default = [c for c in ["KMT2A_rearranged", "mut_TP53", "mut_FLT3", "mut_NPM1",
                               "risk_group", "timepoint", "cell_state"] if c in track_opts]
        track_cols = c2.multiselect("Annotation tracks", track_opts, default=default,
                                    format_func=lambda c: g.loc[c, "label"], key=f"pt_tracks_{cohort}")
        s = data.patient_samples_labeled(cohort)
        ds = data.patient_datasets(cohort).set_index("key")
        avail = data.patient_availability(cohort).rename(columns=ds["label"].to_dict())
        order = s[["sample_id"]].assign(n=avail.sum(axis=1).to_numpy())
        if track_cols:
            order["t"] = s[track_cols[0]].to_numpy()
            samples = order.sort_values(["t", "n", "sample_id"], ascending=[True, False, True],
                                        na_position="last")["sample_id"].tolist()
        else:
            samples = order.sort_values(["n", "sample_id"], ascending=[False, True])["sample_id"].tolist()
        tracks = []
        for col in track_cols:
            r = g.loc[col]
            levels = sorted(s[col].dropna().astype(str).unique())
            binary = (r["true_label"], r["false_label"]) if r["kind"] == "binary" else None
            tracks.append({"label": r["label"], "values": s.set_index("sample_id")[col],
                           "colors": plots.track_colors(levels, binary)})
        avail.columns = [f"{c} ({int(avail[c].sum())})" for c in avail.columns]
        st.plotly_chart(plots.availability_heatmap(samples, tracks, avail), width="stretch")
        st.caption(f"Samples ordered by {g.loc[track_cols[0], 'label'] if track_cols else 'data availability'}; "
                   "hover for sample IDs. Grey = no data / not annotated.")
        st.caption(f"Notes: {data.patient_provenance(cohort).get('notes', '')}")

with tab_fq:
    patient_feature_query()

with tab_drug:
    drug_marker_lookup()

with tab_age:
    age_association()

with tab_surv:
    patient_survival()
