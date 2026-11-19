# Cancer Drug Response Prediction and Biomarker Discovery

## Overview

This project asks whether gene-expression profiles of cancer cell lines can predict sensitivity to anti-cancer drugs, and which genes are associated with that response.

Using the Genomics of Drug Sensitivity in Cancer (GDSC) datasets, I built a machine learning pipeline that predicts drug sensitivity, measured as LN(IC50), for 464 drug–screen units. It combines predictive modeling, feature-importance analysis, a tissue-only baseline, and a statistical check of the identified genes against known cancer genes.

Headline findings:

- Gene expression carries real but modest predictive signal: median held-out R² of 0.154.
- A baseline that knows only a cell line's tissue reaches a median R² of 0.136, so most of the signal reflects tissue of origin. The expression models beat that baseline for 60% of drug–screen units.
- Within individual tissues, predictive performance is near zero or negative.
- An apparent ~4× enrichment of identified biomarkers for known cancer genes disappears under a fair null model (see [OncoKB comparison](#oncokb-comparison)).

## Research Objectives

1. Predict drug response from cancer cell-line gene expression.
2. Compare several regression models, each under both raw and PCA features, using a leakage-free evaluation.
3. Quantify how much predictive signal is explained by tissue alone.
4. Identify genes associated with drug response across compounds.
5. Examine pathway-level patterns and within-tissue performance.
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
Linear Regression, Lasso, ElasticNet, Random Forest and XGBoost, each evaluated under **both** feature representations:

- **Raw:** standardized candidate genes → model.
- **PCA:** standardized candidate genes → PCA (up to 50 components) → model.

Two baselines are also reported and never selected as the best model: a mean-predicting **Dummy**, and a **TissueMean** model that predicts the training-set mean LN_IC50 of each cell line's tissue. Each configuration is a single scikit-learn `Pipeline`.

### 3. Evaluation protocol
- A single 80/20 train/test split per drug (fixed random seed).
- 5-fold cross-validation **on the training split only**. Scaling and PCA are fitted inside each fold, so there is no preprocessing leakage.
- The best non-baseline model per drug is chosen by **CV R²**. Test-set R² is reported afterwards and is never used for selection.
- Metrics: R², RMSE.

### 4. Biomarker analysis
- Feature importance for the selected model, always in gene space: built-in importance for tree models, permutation importance on the full pipeline for linear models.
- Pearson correlation between each candidate gene and LN_IC50 (descriptive, all samples; positive r = higher expression with higher IC50, i.e. resistance).
- Gene-level and pathway-level summaries, and tissue-stratified performance.

## Results

### Predictive performance
Across 464 drug–screen units:

- Median best-model **CV R²: 0.152** (mean 0.167).
- Median selected-model **test R²: 0.154** (mean 0.161); 444 of 464 units (95.7%) were above zero.

Gene expression contains predictive information about drug response, but a typical drug is only weakly predictable. Each estimate comes from one train/test split, so small differences between drugs should not be over-interpreted.

### Raw versus PCA features
Median CV R² across drugs, before selecting the best model per drug:

| Model | PCA | Raw |
|---|---|---|
| Linear | 0.118 | 0.078 |
| Lasso | 0.129 | 0.128 |
| ElasticNet | 0.130 | 0.132 |
| Random Forest | 0.129 | 0.142 |
| XGBoost | 0.091 | 0.109 |

Regularized linear models perform about the same under both representations. Plain linear regression does worse on raw genes (many correlated features, no penalty), and tree models do somewhat better on raw genes. Random Forest was selected for 266 of 464 units (57%), most often on raw genes (214 units); ElasticNet was selected for 98, Lasso for 56, XGBoost for 38, and Linear for 6. Random Forest and XGBoost use fixed hyperparameters and were not tuned.

### How much is tissue?
A baseline that predicts only each cell line's tissue mean reaches a median test R² of **0.136**, compared with **0.154** for the selected expression model. The expression model beats the tissue-only baseline for 60% of units. Most of the predictive signal is therefore tissue of origin, with expression adding a modest amount. Because expression itself differs by lineage, expression models can capture tissue implicitly, so this is best read as "expression adds only a little beyond tissue", not "expression is uninformative".

### Frequently selected genes
Genes most often among a drug's top-20 features, counted over drug–screen units:

| Gene | Units |
|---|---|
| TP53 | 196 |
| MYC | 187 |
| CCND1 | 131 |
| EGFR | 131 |
| KRAS | 117 |
| MAPK1 | 114 |
| BCL2 | 107 |
| PIK3CA | 90 |
| MTOR | 87 |
| AKT1 | 84 |

These counts largely reflect how many drugs' candidate gene sets contain each gene (TP53 and MYC appear in many pathway lists). They are not evidence of independent biological importance, and they are associations, not causal effects.

### Pathway-level summary
Pathways with the most biomarker associations: PI3K/mTOR signaling (2,360), RTK signaling (2,120), DNA replication (1,360), cell cycle (1,236), ERK/MAPK (1,010), mitosis (980), and apoptosis regulation (940). These totals scale with the number of drugs mapped to each pathway, so they describe where the analysis had the most data, not which pathways matter most.

### OncoKB comparison
Candidate genes come from curated cancer pathway lists, so they are enriched for known cancer genes by construction. Testing biomarkers against all ~17k expressed genes would therefore overstate enrichment. Instead I tested against the pool of **1,452 candidate genes** actually offered to the models (306 of them in OncoKB):

| Biomarker set | Genes | Overlap with OncoKB | Hypergeometric fold | Drug-aware permutation null |
|---|---|---|---|---|
| Model-based importance | 886 | 255 | 1.37× | 0.97× (empirical p = 0.98) |
| Pearson | 721 | 226 | 1.49× | 0.84× (empirical p = 1.0) |

The permutation null redraws genes from each drug's own candidate pool, preserving pool composition and the number of picks per drug (2,000 permutations). Under this null, the identified biomarkers are **not enriched** for known cancer genes beyond what the candidate lists already contain. Overlap with OncoKB is descriptive context, not independent validation of predictive or therapeutic utility.

### Tissue-specific performance
Selected models were trained across all tissues, then scored within each tissue on the held-out test set (14 tissue groups). Mean within-tissue R² was near zero or negative for nearly every tissue: Lung was highest (0.007; 56% of drug–tissue comparisons positive) and Myeloid lowest (−0.413). Together with the tissue-only baseline, this suggests that much of the pooled signal reflects differences *between* tissue types rather than variation *within* a tissue. Groups with very few comparisons (e.g. Pancreas, n = 7; Bone, n = 17) are not interpretable, and per-tissue R² on small test subsets is noisy and biased downward.

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
├── reference_data.py    # drug-target and pathway gene lists
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
python train.py                  # full run
python train.py --n-drugs 4      # quick test on 4 drugs
```

The first run needs internet access to download Hallmark gene sets (cached afterwards). A full run took about 2 hours on 8 CPU cores. Results are cached per drug in `outputs/cache/`, so an interrupted run resumes where it stopped; delete that folder after changing the modeling code.