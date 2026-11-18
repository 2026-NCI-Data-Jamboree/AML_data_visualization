# AML data jamboree: web portal

Streamlit portal for the harmonized AML multi-omics resource, built for the 2026 NCI Data Jamboree project
*Building a Harmonized Multi-Omics Data Resource and Web Application for Biomarker Discovery and Drug Target
Nomination in Leukemia*. It reads the processed, harmonized data (`results/`) made by
[AML_data_processing](https://github.com/2026-NCI-Data-Jamboree/AML_data_processing) and nothing else (see that
repo's `docs/results_contract.md`).

## Get the data

The portal needs one folder, `results/`: the harmonized data (about 230 MB). It is not committed here. Either:

- **Download it** (no R or raw data needed): the harmonized dataset is publicly available on figshare,
  [*Building a Harmonized Multi-Omics Resource and Web Application for Biomarker Discovery and Drug Target Nomination
  in Leukemia*](https://figshare.com/articles/dataset/Building_a_Harmonized_Multi-Omics_Resource_and_Web_Application_for_Biomarker_Discovery_and_Drug_Target_Nomination_in_Leukemia/35050459) (figshare article 35050459). Download it and unzip it to `data/results/` in this repo; or
- **Build it** with the processing repo next to this one:

  ```bash
  cd ../AML_data_processing
  Rscript prepare_cell_line_data.R
  Rscript prepare_patient_data.R
  uv run prepare_deconvolution.py      # last: the patient step rebuilds results/patients from scratch
  ```

The portal looks for the data, in order, at `$AML_PORTAL_DATA`, `data/results/`, then
`../AML_data_processing/results/`. The Home page warns when the data's `results_version` (in its `provenance.tsv`
files) differs from the one this portal expects (currently 3); use a portal version that matches the data release (both
repos are tagged together, see below).

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/streamlit run app.py
```

With [uv](https://docs.astral.sh/uv/), dependencies come from `pyproject.toml` (locked in `uv.lock`):

```bash
uv run streamlit run app.py
```

This creates `.venv/` in the repo. To keep the environment elsewhere, set `UV_PROJECT_ENVIRONMENT=/path/to/env`.
`requirements.txt` mirrors the dependencies for pip and hosts that read it: change both together.

## Pages

| page | content |
|---|---|
| Home | the project, what each page does, data provenance |
| Highlights | genome-wide analyses. *Gene dependency*: CRISPR vs RNAi in AML cell lines, MYC-high vs mTOR-high or TP53-mutant vs wild-type. *Cell surface proteins*: surface proteomics of AML cell lines (George et al. 2026) and patient specimens (Leucegene surfaceome, Bordeleau et al. 2024) compared within genomic subtypes (KMT2A-r, TP53, FLT3-ITD, DNMT3A), ranked by agreement of both datasets. *Drug response markers* (built-in analysis 1): every RNA / protein / phosphosite against a drug's response, meta-analysed across cohorts. *Age association*: every feature against age at diagnosis. *Survival*: every feature against overall or event-free survival (Cox regression per cohort, inverse-variance meta-analysis), optionally adjusted for or within risk group |
| Cell lines | built-in analysis 2, 70 AML lines, 13 data layers (incl. copy number and surface proteome). *Overview*: data availability with annotation tracks (incl. FLT3-ITD), co-occurrence of aberrations, recurrent copy-number changes, genetically similar lines. *Feature query*, *Compare groups* (any grouping × any layer), *Gene dependency* (CRISPR vs RNAi, any comparison) |
| Patients | 15 cohorts, adult and pediatric (Beat AML, PTRC, CPTAC, Zenodo 14160480, Lee 2018, Wang 2022, Malani 2022, St. Jude 2024, TCGA, TARGET, Leucegene, LUMC, Jayavelu 2022 discovery and validation, Leucegene surfaceome). *Overview* (cohort summary, availability, age and risk composition, mutation frequencies, per-cohort heatmap), *Feature query* (one gene's RNA, protein, phosphosites or surface protein in every cohort), *Drug response* (one marker × one drug), *Age association* (one feature), *Survival* (Kaplan–Meier, log-rank, hazard ratio) |
| Deconvolution | built-in analysis 3: estimated cell type and cell state proportions of bulk RNA samples (AML scAtlas reference) in 8 cohorts; the page shows a notice instead when the data has no deconvolution layers |
| Documentation | overview, data sources (with NCI resource status), methods and glossary in plain language |
| Downloads | the harmonized-data README, dataset catalog, every processed file, annotation schema |

Statistics are computed on the fly from `results/` (`portal/stats.py`: Welch t test, BH FDR, Spearman, Fisher-z
and inverse-variance meta-analysis, vectorised Cox regression validated against R `survival`).

## Layout

```
app.py                     navigation
pages/                     one file per page
portal/data.py             loading results/ (cached), groupings, gene index, deconvolution files
portal/stats.py            Welch t, BH FDR, Spearman, meta-analysis, Cox regression
portal/plots.py            shared figures (volcano, strip/box, heatmaps, Kaplan–Meier, forest)
portal/feature_query.py    feature query (cell lines and patients)
portal/dependency.py       CRISPR vs RNAi gene dependency
portal/cooccurrence.py, copy_number.py, similar_lines.py, genomics_stats.py, genomics_plots.py
                           cell-line genomics sections of the Overview
portal/surfaceome.py, surface_cross.py, surface_plots.py
                           cell surface proteins: cell lines vs patients
portal/drug_biomarkers.py  drug response (Highlights and Patients)
portal/age_association.py  age association (Highlights and Patients)
portal/survival.py         Kaplan–Meier and log-rank (validated against R survival)
portal/survival_view.py    Patients → Survival
portal/survival_meta.py    Highlights → Survival (multi-cohort Cox meta-analysis)
portal/deconvolution.py    Deconvolution page
tests/                     statistics vs scipy / R, results consistency, page smoke tests (AppTest)
```

## Test

```bash
.venv/bin/pytest -q
# or with uv
uv run pytest -q
```

Tests that need data skip when `results/` (or a part of it, such as deconvolution) is missing.

## Deploy

Streamlit Community Cloud (or any host): set `AML_PORTAL_DATA` to a copy of `results/`, or fetch the
[figshare release](https://figshare.com/articles/dataset/Building_a_Harmonized_Multi-Omics_Resource_and_Web_Application_for_Biomarker_Discovery_and_Drug_Target_Nomination_in_Leukemia/35050459) into `data/results/` at startup.

## Releasing the data

The harmonized data is published on figshare ([article 35050459](https://figshare.com/articles/dataset/Building_a_Harmonized_Multi-Omics_Resource_and_Web_Application_for_Biomarker_Discovery_and_Drug_Target_Nomination_in_Leukemia/35050459)). `results/` is self-contained (catalog, schema, sources, README, cell lines, patient cohorts, deconvolution and the
single-cell reference summary), so it is the unit to deposit: with this repo at the matching tag, it is all a user needs.

Before each upload of a new version:

1. **Redistribution rights.** `results/` holds processed per-sample data from about 20 sources, each with its own terms
   (see `results/dataset_sources.tsv`). Confirm that each may be redistributed in processed form. Open examples:
   DepMap (CC BY 4.0), TCGA / TARGET open-access data. To check: the CPTAC adult AML data freeze (v8; may not be
   released for redistribution yet), journal supplementary tables (Jayavelu 2022, Malani 2022, Lee 2018, Wang 2022,
   Bordeleau 2024), the George 2026 bioRxiv supplement, AMLmap and the cBioPortal St. Jude study. Leave out any cohort
   that may not be redistributed (its folder under `patients/` and its rows in `catalog/` and `dataset_sources.tsv`), and
   point users to the processing repo to build it themselves.
2. **Patient data.** All cohorts come from public, de-identified releases, but `samples.tsv` holds sample-level clinical
   data (age, sex, survival times, karyotype). Confirm that none of it comes from controlled-access tiers (e.g. dbGaP).
3. **Housekeeping.** Rebuild `results/` from the tagged processing commit (each `provenance.tsv` records it), run the
   deconvolution last, remove stray files (e.g. macOS `.DS_Store`), and check that this repo's tests pass on it.
4. **Versions.** Tag both repos together (e.g. `v1.0`), name the tags and `results_version` in the figshare record, and
   update the link in the *Get the data* section above if a new version gets a new address.
5. **License and citation.** Add a license for the processed data and a citation note to the record, and ask users to cite
   the original datasets (listed in `results/dataset_sources.tsv` and `results/README.md`).
