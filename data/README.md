# Data Sources and Setup

This project uses publicly available cancer cell-line gene-expression,
drug-response, and cancer-gene annotation data.

## Required input files

Place the following files in this directory.

| File | Description | Source |
|---|---|---|
| `Cell_line_RMA_proc_basalExp.txt` | RMA-normalized basal gene-expression profiles | GDSC |
| `GDSC1_dose_response.csv` | GDSC1 drug-response measurements | GDSC |
| `GDSC2_dose_response.csv` | GDSC2 drug-response measurements | GDSC |
| `cancerGeneList.tsv` | Cancer-gene reference used for biomarker enrichment | OncoKB |

### GDSC drug-response alternatives

Instead of the two CSV files, you can provide the original Excel files:

- `GDSC1_fitted_dose_response_27Oct23.xlsx`
- `GDSC2_fitted_dose_response_27Oct23.xlsx`

If the CSV files are missing, the pipeline reads the Excel files and
converts them to CSV.

## Optional input

| File | Description |
|---|---|
| `Model.csv` | Cell-line metadata and tissue annotations |
| `GDSC_cell_lines_details.csv` | Alternative cell-line metadata file |
| `cell_line_details.csv` | Alternative cell-line metadata file |

A tissue metadata file enables tissue-stratified analysis. Without one,
that portion of the analysis is skipped.

## Data download sources

- [GDSC Bulk Downloads](https://www.cancerrxgene.org/downloads/bulk_download)
- [GDSC](https://www.cancerrxgene.org/)
- [OncoKB Cancer Gene List](https://www.oncokb.org/cancerGenes)
- [OncoKB Licensing and Data Use](https://faq.oncokb.org/licensing)

Download the corresponding files and place them in this directory
using the filenames listed above.

## Generated cache files

The pipeline may generate these files automatically:

- `expr.pkl` — processed gene-expression matrix
- `hallmark.pkl` — cached MSigDB Hallmark 2020 gene sets

If the Hallmark cache is absent, the pipeline retrieves gene sets from
the [Enrichr API](https://maayanlab.cloud/Enrichr/). Internet access is
therefore required on the first run.

These cache files are not required downloads.

## Data availability and usage

Raw datasets are not included in this repository. Users must obtain
them from their respective sources and comply with the applicable
data access and usage terms.