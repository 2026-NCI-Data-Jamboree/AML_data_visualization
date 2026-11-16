# AML data jamboree: web portal

Streamlit portal for the harmonized AML multi-omics resource. It reads the processed,
harmonized data in [AML_data_processing](../AML_data_processing)`/results` and nothing else (see that repo's
`docs/results_contract.md`).

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
# once, in the processing repo:  cd ../AML_data_processing && Rscript prepare_cell_line_data.R
.venv/bin/streamlit run app.py
```

The data is found at, in order: `$AML_PORTAL_DATA`, `data/results/`, then
`../AML_data_processing/results/`.

## Pages

| page | content |
|---|---|
| Home | resource overview and provenance |
| Cell lines | built-in analysis 2, 70 AML lines. Tabs: *Cell lines* (data-availability heatmap with annotation tracks), *Feature query* (one gene across all layers, drugs matched by target), *Compare groups* (any grouping from `groupings.tsv` × any layer: volcano, table, per-line plot), *Gene dependency* (CRISPR vs RNAi scatter per gene: mean dependency, or a comparison such as MYC-high vs mTOR-high, TP53-mutant, KMT2A-r, or any custom grouping; plus per-line agreement for one gene) |
| Patients | tabs: *Cohorts*, *Feature query*, *Drug response biomarkers* (built-in analysis 1). Placeholders until `results/patients/` exists |
| Deconvolution | built-in analysis 3; placeholder until `results/single_cell/` exists |
| Downloads | tabs: *Datasets* (catalog, modality coverage), *Cell-line files*, *Annotation schema* |

Statistics are computed on the fly (`portal/stats.py`: Welch t test, BH FDR). Nothing is
precomputed in the processing repo.

## Layout

```
app.py            navigation
pages/            one file per page
portal/data.py    loading results/ (cached), groupings, gene index
portal/stats.py   Welch t test, BH FDR
portal/plots.py   volcano, strip/box plots, availability heatmap
portal/feature_query.py  feature-query tab used by each built-in analysis
portal/dependency.py     gene-dependency tab (CRISPR vs RNAi) on the cell-lines page
tests/            stats vs scipy/R, results consistency, page smoke tests (AppTest)
```

## Test

```bash
.venv/bin/pytest -q
```

## Deploy

Streamlit Community Cloud (or any host): set `AML_PORTAL_DATA` to a copy of the processing repo's
`results/` (about 56 MB), or fetch it into `data/results/` at startup. It is not committed here.
