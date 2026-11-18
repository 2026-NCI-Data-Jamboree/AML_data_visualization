import pandas as pd
import streamlit as st

from portal import data

st.title("Documentation")
st.markdown("What is in the portal, where the data comes from, and how each analysis is done.")

tab_about, tab_sources, tab_methods, tab_glossary = st.tabs(["Overview", "Data sources", "Methods", "Glossary"])

with tab_about:
    st.markdown(
        """
The portal brings public acute myeloid leukemia (AML) data together in one place, so that patient
cohorts and cell lines can be explored side by side. Every dataset is described with the same vocabulary
(disease codes from OncoTree, gene names from HGNC), so that a gene, a mutation or a drug means the same
thing everywhere in the portal.

**Built-in analyses**

1. **Drug response** (Highlights, and Patients → Drug response): which genes are linked to how patient cells respond to a drug, and
   does the link hold up across independent cohorts? Example: MAP1A and venetoclax.
2. **Cell-line variability** (Cell lines): how much AML cell lines differ across data types, depending on
   how they are grouped (TP53 mutation, KMT2A rearrangement, MYC- or mTOR-high, ...).
3. **Deconvolution** (in development): the mix of differentiation stages in patient and cell-line samples,
   estimated from a single-cell reference.

**Pages**

| page | what you can do |
|---|---|
| Highlights | gene dependency in cell lines (MYC-high vs mTOR-high, TP53-mutant vs wild-type); cell-surface proteins that differ between AML cell-line subgroups; genome-wide searches across patient cohorts: markers of drug response, features that change with age, and genes linked to survival |
| Cell lines | see which data each line has, which aberrations co-occur, recurrent copy-number changes and which lines are most alike; compare two groups of lines; look up a gene; compare CRISPR and RNAi gene dependency |
| Patients | see what each cohort contains; look up a gene's RNA, protein or phosphosites in every cohort; check one marker against one drug or against age; compare survival between groups |
| Deconvolution | planned |
| Downloads | download every processed file |
"""
    )
    st.caption("All analyses are computed live from the processed data; nothing is pre-selected or filtered "
               "for significance beforehand.")
    st.subheader("Code")
    st.markdown(
        """
The portal is open source, developed for the 2026 NCI Data Jamboree:

- [AML_data_processing](https://github.com/2026-NCI-Data-Jamboree/AML_data_processing): turns the public
  datasets into the harmonized data the portal reads (R).
- [AML_data_visualization](https://github.com/2026-NCI-Data-Jamboree/AML_data_visualization): this portal
  (Python, Streamlit).
"""
    )

with tab_sources:
    src = data.dataset_sources()
    if src.empty:
        st.info("The list of data sources is not available.")
    else:
        c1, c3 = st.columns(2)
        c1.metric("Datasets", len(src))
        c3.metric("From NCI resources", int(src["nci_resource"].isin(["yes", "partly"]).sum()),
                  help="Fully or partly from an NCI program or data commons (GDC, PDC, CPTAC, TCGA, TARGET, CTD²)")
        nci = st.multiselect("NCI resource", ["yes", "partly", "no"], key="doc_nci")
        view = src
        if nci:
            view = view[view["nci_resource"].isin(nci)]
        show = ["name", "category", "population", "data_types", "reference", "source", "accession",
                "nci_resource", "nci_program"]
        st.dataframe(view[[c for c in show if c in view]], hide_index=True, width="stretch",
                     column_config={"nci_resource": st.column_config.TextColumn("NCI resource"),
                                    "nci_program": st.column_config.TextColumn("NCI program / commons"),
                                    "data_types": st.column_config.TextColumn("data types")})
        st.download_button("Download list (TSV)", view.to_csv(sep="\t", index=False), file_name="dataset_sources.tsv")
        st.caption("The datasets integrated into the portal. NCI resource: GDC = Genomic Data Commons, PDC = Proteomic Data Commons, "
                   "CTD² = Cancer Target Discovery and Development Network. A blank accession is still to be confirmed.")

with tab_methods:
    st.subheader("How the data is made comparable")
    st.markdown(
        """
- **Proteomics**: protein abundance (one value per gene) and phosphosites (one value per site, e.g.
  STAT5A_Y694) from TMT mass spectrometry, as log2 ratios. Beat AML proteomics (PTRC) covers Beat AML
  specimens: for these, clinical data and drug response come from the matching Beat AML sample, and RNA
  is only in the Beat AML cohort, so no patient is counted twice in an analysis. CPTAC adult AML values
  are corrected for red-blood-cell contamination.
- **Genes** use current HGNC symbols. Cohorts that use older gene names are updated to the current name
  (e.g. HDGFRP3 → HDGFL3), so a gene can be found under one name everywhere. Patient RNA data keeps
  protein-coding genes; where a gene has several measurements (e.g. array probes) they are averaged.
- **Mutations** are recorded for 20 genes recurrently mutated in AML (TP53, FLT3, NPM1, NRAS, KRAS, PTPN11,
  KIT, IDH1, IDH2, DNMT3A, TET2, ASXL1, RUNX1, CEBPA, WT1, SRSF2, U2AF1, SF3B1, STAG2, BCOR). A gene is
  marked *not tested* when a cohort did not assay it, which is different from *not mutated*.
  Cell lines: DepMap hotspot and damaging calls. Patients: the cohort's sequencing calls plus clinical
  FLT3 and NPM1 tests.
- **KMT2A rearrangement**: curated status or fusion calls (cell lines); clinical fusion or karyotype
  (patients).
- **Risk group** (favorable / intermediate / adverse) uses what each cohort reports: ELN 2017 (Beat AML,
  FPM tumor board), ELN 2010 cytogenetic risk (Lee 2018) or cytogenetic risk (Wang 2022).
  St. Jude pediatric AML has no risk group; it has a molecular category (e.g. KMT2A-r, RUNX1::RUNX1T1, NUP98-r) instead.
  TCGA, TARGET, Leucegene and LUMC (from the AMLmap compendium) have no risk group; they have WHO 2022 and ICC 2022
  classes and the AMLmap transcriptomic cluster instead.
- **Age groups** follow the NCI bands: pediatric under 15, adolescent and young adult (AYA) 15–39,
  adult 40 and over, from age at diagnosis.
- **MYC and mTOR RNA signatures**: a score per sample, the average of the signature genes after
  standardizing each gene within the dataset; *high* = above the dataset's median. For cell lines the
  published classification is used. For patients the score is recomputed and closely, but not exactly,
  reproduces it (correlation 0.90 for MYC and 0.82 for mTOR on the cell lines).
- **Drugs** are matched across cohorts by name, ignoring salt forms and brackets, and annotated with their
  target genes from DepMap. Each cohort keeps its own measure: **AUC** (area under the dose–response
  curve; lower = more sensitive) or **DSS / sDSS** (drug sensitivity score; higher = more sensitive).
"""
    )
    st.subheader("Analyses")
    st.markdown(
        """
- **Compare groups** (Cell lines): each feature is compared between two groups of lines with a Welch
  t test, and p values are adjusted for the number of features tested (Benjamini–Hochberg false discovery
  rate, FDR). A feature needs at least 3 lines with data in each group.
- **Gene dependency** (Cell lines): CRISPR knockout (Chronos) and RNAi knockdown (DEMETER2) scores for the
  same genes, side by side. Lower scores mean the cells depend more on the gene. Group comparisons are run
  separately in each screen.
- **Feature query**: one gene across every data type, shown by group. p values there are per feature
  and not adjusted.
- **Drug response** (Patients: one marker and one drug; Highlights: every marker at once): in each cohort, the marker's RNA or protein level (or phosphosite) is correlated with response to
  the chosen drug (Spearman rank correlation, at least 10 samples). The sign is set so that a **positive
  value always means higher expression goes with more sensitivity**, whatever the cohort's measure. The
  cohorts are then combined, giving larger cohorts more weight (Fisher's z with Stouffer's method,
  weighted by sample size), and adjusted for the number of genes tested (FDR). *Cohorts agree* means the
  link has the same direction in every cohort.
- **Age association** (Patients: one feature; Highlights: every feature at once): every gene's expression (or response to every drug) is related to age
  at diagnosis, one sample per patient, either as **continuous age** (Spearman correlation with age in
  years; at least 10 patients) or between two **age groups** (e.g. AYA vs adult; a rank comparison of the
  two groups, at least 3 patients in each). Cohorts are combined as in the drug response analysis.
  Positive = higher in older patients, or in group A (for drugs: more sensitive). The pediatric cohorts
  (Wang 2022, St. Jude) add the youngest patients; they have no adults, so they are not used when the
  comparison needs them.
- **Survival** (Patients): Kaplan–Meier curves with 95% confidence bands, compared with the log-rank
  test. The hazard ratio compares the first group with the second (above 1 = worse outcome in the first
  group). Each patient counts once, using the diagnosis sample when there is one. Overall survival is
  available for Beat AML (also used for its PTRC proteomics samples), the Zenodo proteogenomics cohort, the
  FPM tumor board, TCGA and TARGET; event-free survival for the Zenodo cohort, the FPM tumor board and
  Wang 2022. Lee 2018 reports survival times without whether the patient died, so it is not included.
- **Survival across cohorts** (Highlights): in each cohort, every gene's RNA level (or protein, or
  phosphosite) is related to survival with a Cox regression, one sample per patient. Each feature is
  standardized within the cohort, so the hazard ratio is per standard deviation: above 1 = higher level,
  shorter survival. Cohorts need at least 20 patients and 10 events. For RNA, a gene is tested only where it
  is expressed (log2 level of at least 1) in at least 20% of patients, because barely expressed genes give
  unstable results. Cohorts are combined by inverse-variance meta-analysis of the log hazard ratios, and
  p values are adjusted for the number of features tested (FDR). The Kaplan–Meier curves below the table
  split each cohort at its median, for display only. **Risk group** can be taken into account in two ways,
  using only cohorts that report one (ELN or cytogenetic risk): *adjusted for risk group* compares patients
  within the same risk group (a Cox model stratified by risk group), so it shows whether a gene adds to
  the risk classification, and its Kaplan–Meier curves show high vs low within each risk group (split at the
  median of that risk group), with a log-rank test per risk group; *within one risk group* analyses
  favorable, intermediate or adverse patients only.
"""
    )
    st.subheader("Things to keep in mind")
    st.markdown(
        """
- Cohorts differ in how drug response was measured, which drugs were tested, and how samples were
  collected, so the size of an effect is best compared within a cohort.
- Drug response is measured on patient cells in the lab (ex vivo), not in the patient.
- Associations are correlations, not proof of cause.
"""
    )

with tab_glossary:
    terms = pd.DataFrame([
        ("AML", "Acute myeloid leukemia"),
        ("AYA", "Adolescents and young adults, age 15–39 at diagnosis"),
        ("AUC", "Area under the dose–response curve; lower = more sensitive to the drug"),
        ("DSS / sDSS", "Drug sensitivity score / selective DSS (relative to healthy cells); higher = more sensitive"),
        ("Chronos / DEMETER2", "Gene dependency scores from CRISPR and RNAi screens; lower = more dependent"),
        ("ELN", "European LeukemiaNet risk classification (favorable / intermediate / adverse)"),
        ("KMT2A-r", "KMT2A (MLL) rearrangement, a gene fusion common in pediatric AML"),
        ("FDR", "False discovery rate: the expected share of false positives among results called significant"),
        ("Spearman r", "Rank correlation between −1 and 1"),
        ("Hazard ratio (HR)", "Relative rate of events (e.g. death) in one group versus another"),
        ("OS / EFS", "Overall survival / event-free survival (time to relapse, refractory disease or death)"),
        ("Ex vivo", "Measured on patient cells outside the body"),
        ("GDC / PDC", "NCI Genomic Data Commons / Proteomic Data Commons"),
        ("CPTAC / TCGA / TARGET", "NCI programs: Clinical Proteomic Tumor Analysis Consortium / The Cancer Genome "
                                  "Atlas / Therapeutically Applicable Research to Generate Effective Treatments"),
        ("OncoTree / HGNC", "Standard vocabularies for cancer types / human gene names"),
    ], columns=["term", "meaning"])
    st.dataframe(terms, hide_index=True, width="stretch", height=38 + 35 * len(terms))
