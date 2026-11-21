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
- Genome-wide gene–response associations for the best-predicted drugs: Pearson r between each of the 17,419 genes and LN(IC50), with Benjamini–Hochberg FDR correction. Each association is computed twice: pooled across all cell lines, and within tissue (gene expression and LN(IC50) are centered on their tissue mean before correlating; tissues with fewer than 5 cell lines are merged into "Other"). A gene counts as a hit at FDR q < 0.05 and |r| ≥ 0.2.

### 5. External validation (PRISM)
Models were tested against the PRISM Repurposing secondary screen, an independent viability assay. Drugs were matched by name (213 of 464 units; PRISM's `crizotinib-(S)` was excluded as a different stereoisomer). For each matched drug and PRISM screen, all cell lines measured in both datasets were held out, the model type selected in the main run was refit on the remaining GDSC cell lines, and its predictions for the held-out lines were compared with PRISM AUC using within-drug Spearman correlation (response scales differ, so R² is not comparable). PRISM screens were analyzed separately.

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

### Gene–response associations
For the six best-predicted drug–screen units, I tested all 17,419 genes (not just the curated candidates) against LN(IC50):

| Drug (screen) | Cell lines | Hits, pooled | Hits, within tissue |
|---|---|---|---|
| Nutlin-3a (−) (GDSC2) | 940 | 2,721 | 58 |
| Venetoclax (GDSC2) | 930 | 4,536 | 242 |
| AZD5991 (GDSC2) | 711 | 4,531 | 315 |
| ABT-737 (GDSC2) | 929 | 3,962 | 564 |
| Navitoclax (GDSC2) | 939 | 3,633 | 708 |
| Alpha-lipoic acid (GDSC2) | 731 | 3,011 | 41 |

Adjusting for tissue removes roughly 80–99% of the pooled hits, so most apparent associations reflect which tissue a cell line comes from. The genes that remain are the more credible candidates, and several match known drug biology:

- **BH3-mimetic drugs.** BCL2 is among the top sensitivity genes for Venetoclax and ABT-737 (rank 1 for both) and appears in the top six for Navitoclax. For ABT-737 and Navitoclax, MCL1 is a top resistance gene; for the MCL1 inhibitor AZD5991, the top resistance gene is BCL2L1 (BCL-XL). This is consistent with the idea that other anti-apoptotic proteins compensate when one is inhibited.
- **Nutlin-3a.** The top sensitivity genes (EDA2R, MDM2, DDB2, RPS27L, AEN, ZMAT3) are, to my knowledge, p53 target genes, consistent with Nutlin-3a needing active p53.
- **Shared signal.** YAP1, LMNA, TNFRSF12A, BLVRB and CTSL1 appear among the top resistance genes for several BH3-mimetic drugs. These drugs are related, so this is not independent evidence, and it may reflect a general cell state.

Caveats: these six drugs were chosen because they are well predicted, so they are not representative of all drugs. Tissue adjustment is conservative and also removes real between-lineage biology. Associations are not causal, and some top genes (for example for alpha-lipoic acid) have no obvious explanation. Full ranked lists are in `results/tables/biomarker_top_genes.csv` and hit counts in `biomarker_hit_counts.csv`.

### External validation on PRISM
| | HTS002 (202 units) | MTS010 (76 units) |
|---|---|---|
| Model vs GDSC, same held-out cell lines | 0.244 | 0.247 |
| **Model vs PRISM** | **0.111** | **0.148** |
| Tissue-mean baseline vs PRISM | 0.074 | 0.091 |
| GDSC vs PRISM measurements (ceiling) | 0.234 | 0.297 |
| Model beats tissue baseline on PRISM | 69% of units | 80% of units |
| Drug-level correlation, GDSC vs PRISM performance | 0.48 | 0.68 |

(median within-drug Spearman across units; MTS006 had only 11 units and is in `external_validation.csv`.)

Predictions transfer to an independent assay but retain only about half of their GDSC-measured correlation. Part of this reflects assay disagreement: GDSC and PRISM measurements of the same cell lines correlate at only about 0.23–0.30, so absolute external performance is bounded. Models still beat a tissue-only baseline for most drugs, and drugs that were predictable in GDSC tended to be predictable in PRISM. Units are not independent (the same cell lines are held out for every drug, and drugs appear in both GDSC screens), so p-values are descriptive. Expression is GDSC microarray throughout, so this tests transfer of the drug-response measurement, not of the expression platform.

## Visualizations
`results/figures/` (generated by `make_figures.py` from the pipeline outputs):

- `fig1_performance_vs_baselines.png`: held-out R² of the selected models vs the tissue-mean and dummy baselines
- `fig2_raw_vs_pca.png`: raw vs PCA features for each model, per drug
- `fig3_model_vs_tissue_baseline.png`: selected model vs tissue-only baseline, per drug
- `fig4_top_drugs.png`: top drug–screen units by test R², alongside the tissue baseline
- `fig5_within_tissue_r2.png`: performance within each tissue
- `fig6_oncokb_null.png`: observed OncoKB overlap vs the drug-aware permutation null
- `fig7_biomarker_example.png`: one gene's expression vs drug response, colored by tissue
- `biomarker/`: per-drug gene–response associations (pooled and tissue-adjusted volcano plots, plus top genes)

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
python make_figures.py           # generate figures and biomarker table
python train.py --n-drugs 4      # quick test on 4 drugs
```

The first run needs internet access to download Hallmark gene sets (cached afterwards). A full run took about 2 hours on 8 CPU cores. Results are cached per drug in `outputs/cache/`, so an interrupted run resumes where it stopped; delete that folder after changing the modeling code.