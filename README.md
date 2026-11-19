# Cancer Drug Response Prediction and Biomarker Discovery

## Overview

This project asks whether gene-expression profiles of cancer cell lines can predict sensitivity to anti-cancer drugs, and which genes are associated with that response.

Using the Genomics of Drug Sensitivity in Cancer (GDSC) datasets, I built a machine learning pipeline that predicts drug sensitivity, measured as LN(IC50), for 464 drug–screen units. It combines predictive modeling, feature-importance analysis, and a statistical check of the identified genes against known cancer genes.

Headline findings:

- Gene expression carries real but modest predictive signal: median held-out R² of 0.157.
- Within individual tissue types, predictive performance is near zero or negative, which suggests much of the pooled signal reflects differences between tissues.
- An apparent ~4× enrichment of identified biomarkers for known cancer genes disappears under a fair null model (see [OncoKB comparison](#oncokb-comparison)).

## Research Objectives

1. Predict drug response from cancer cell-line gene expression.
2. Compare several regression models using a leakage-free evaluation.
3. Identify genes associated with drug response across compounds.
4. Examine pathway-level patterns among identified genes.
5. Examine predictive performance within individual cancer tissue types.
6. Compare identified genes with known cancer genes, using an appropriate null.

## Dataset

| Data | Description |
|---|---|
| Gene expression | RMA-normalized basal expression, 17,419 genes × 1,018 cell lines (GDSC) |
| Drug response | GDSC1 and GDSC2 fitted dose-response (LN_IC50) |
| Cell-line metadata | Tissue (OncoTree lineage) annotations |
| Cancer gene reference | OncoKB cancer gene list |

GDSC1 and GDSC2 use different assays and drug panels, so they are analyzed **separately**. Each (drug, screen) pair is one analysis unit, e.g. `Erlotinib [GDSC2]`. This gives 468 units, of which 464 had enough cell lines (≥ 200) to model. Units are not fully independent: the same drug can appear in both screens.

See `data/README.md` for sources and setup.

## Methodology

### 1. Features
For each drug, the models see only a drug-specific **candidate gene set**, not all 17k genes. It is built from the drug's annotated targets, curated gene lists for its pathway, and MSigDB Hallmark sets matched to that pathway. If fewer than 8 candidate genes are present in the expression data, a fixed list of 20 well-known cancer genes is used instead. Candidate sets are built without using drug-response values.

### 2. Models
Linear Regression, Lasso, ElasticNet, Random Forest, XGBoost, and a mean-predicting Dummy baseline. Each model is a single scikit-learn `Pipeline` that takes the raw gene matrix:

- Linear models and the Dummy baseline: standardization → PCA (up to 50 components) → model.
- Random Forest and XGBoost: standardization → model.

### 3. Evaluation protocol
- A single 80/20 train/test split per drug (fixed random seed).
- 5-fold cross-validation **on the training split only**. Scaling and PCA are fitted inside each fold, so there is no preprocessing leakage.
- The best non-dummy model per drug is chosen by **CV R²**. Test-set R² is reported afterwards and is never used for selection.
- Metrics: R², RMSE.

### 4. Biomarker analysis
- Feature importance for the selected model, always in gene space: built-in importance for tree models, permutation importance on the full pipeline for linear models.
- Pearson correlation between each candidate gene and LN_IC50 (descriptive, all samples; positive r = higher expression with higher IC50, i.e. resistance).
- Gene-level and pathway-level summaries, and tissue-stratified performance.

## Results

### Predictive performance
Across 464 drug–screen units:

- Median best-model **CV R²: 0.146** (mean 0.164).
- Median selected-model **test R²: 0.157** (mean 0.162); 444 of 464 units (95.7%) were above zero.

Gene expression contains predictive information about drug response, but a typical drug is only weakly predictable. Each estimate comes from one train/test split, so small differences between drugs should not be over-interpreted.

### Frequently selected genes
Genes most often among a drug's top-20 features, counted over drug–screen units:

| Gene | Units |
|---|---|
| TP53 | 198 |
| MYC | 192 |
| CCND1 | 134 |
| EGFR | 132 |
| MAPK1 | 114 |
| KRAS | 114 |
| BCL2 | 105 |
| MTOR | 89 |
| PIK3CA | 89 |
| AKT1 | 84 |

These counts largely reflect how many drugs' candidate gene sets contain each gene (TP53 and MYC appear in many pathway lists). They are not evidence of independent biological importance, and they are associations, not causal effects.

### Pathway-level summary
Pathways with the most biomarker associations: PI3K/mTOR signaling (2,440), RTK signaling (2,120), DNA replication (1,400), cell cycle (1,236), ERK/MAPK (1,010), apoptosis regulation (1,000), and mitosis (980). These totals scale with the number of drugs mapped to each pathway, so they describe where the analysis had the most data, not which pathways matter most.

### OncoKB comparison
Candidate genes come from curated cancer pathway lists, so they are enriched for known cancer genes by construction. Testing biomarkers against all ~17k expressed genes would therefore overstate enrichment. Instead I tested against the pool of **1,452 candidate genes** actually offered to the models (306 of them in OncoKB):

| Biomarker set | Genes | Overlap with OncoKB | Hypergeometric fold | Drug-aware permutation null |
|---|---|---|---|---|
| Model-based importance | 919 | 250 | 1.29× | 0.93× (empirical p = 1.0) |
| Pearson | 721 | 226 | 1.49× | 0.84× (empirical p = 1.0) |

The permutation null redraws genes from each drug's own candidate pool, preserving pool composition and the number of picks per drug (2,000 permutations). Under this null, the identified biomarkers are **not enriched** for known cancer genes beyond what the candidate lists already contain. Overlap with OncoKB is descriptive context, not independent validation of predictive or therapeutic utility.

### Tissue-specific performance
Models were trained across all tissues, then scored within each tissue on the held-out test set (14 tissue groups). Mean within-tissue R² was near zero or negative for nearly every tissue: Lung was highest (0.007; 55% of drug–tissue comparisons positive) and Myeloid lowest (−0.397). Pooled test R² is positive (median 0.157), so a substantial part of the predictive signal may reflect differences *between* tissue types rather than variation *within* a tissue. This has not been tested directly (see [Future Improvements](#future-improvements)). Groups with very few comparisons (e.g. Pancreas, n = 7; Bone, n = 17) are not interpretable, and per-tissue R² on small test subsets is noisy and biased downward.

## Visualizations
`results/figures/` contains:

- Figure 2A–2D: CV R² distribution, drugs ranked by performance, pathway × model heatmap, model comparison across drugs
- Figure 2E: predicted vs. actual LN_IC50 for top drugs, colored by tissue
- Figure 2F: feature importance for the top drug
- Figure 2G–2H: tissue-stratified R² heatmap, model comparison summary
- `biomarker/`: volcano plots and gene-correlation clustermaps for the ten best-performing drugs

`results/tables/` contains the corresponding model evaluation, feature-importance, biomarker and tissue tables.

## Technologies
Python · NumPy, Pandas, SciPy · scikit-learn, XGBoost · Matplotlib, Seaborn

## Repository Structure
```
cancer-drug-response-biomarker-analysis/
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── README.md        # raw data not included
└── results/
    ├── figures/
    └── tables/
```

## Reproducing the Analysis
1. Clone the repository.
2. Install dependencies.
3. Obtain the datasets listed in `data/README.md` and place them in `data/`.
4. Run the pipeline.

```bash
git clone <repository-url>
cd cancer-drug-response-biomarker-analysis

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python train.py
```

The first run needs internet access to download Hallmark gene sets (cached afterwards). A full run took about 35 minutes on 8 CPU cores. Results are cached per drug in `outputs/cache/`, so an interrupted run resumes where it stopped; delete that folder after changing the modeling code.