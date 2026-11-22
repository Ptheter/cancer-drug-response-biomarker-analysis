# Cancer Drug Response Prediction and Biomarker Discovery


## Overview

This project asks whether gene-expression profiles of cancer cell lines can predict sensitivity to anti-cancer drugs, and which genes are associated with that response.

Using the Genomics of Drug Sensitivity in Cancer (GDSC) datasets, I built a machine learning pipeline that predicts drug sensitivity, measured as LN(IC50), for 464 drug–screen units. The evaluation is designed to avoid common pitfalls (preprocessing leakage, model selection on test data, circular enrichment tests), and every result is reported next to simple baselines. Predictions are then tested on an independent screen (PRISM) using cell lines held out of training.

**Interactive app:** run locally with `streamlit run app/streamlit_app.py`

### Headline findings

- Gene expression carries real but modest predictive signal: median held-out R² of 0.154.
- A baseline that knows only a cell line's tissue reaches a median R² of 0.136, so most of the signal reflects tissue of origin. Expression models beat that baseline for 60% of drug–screen units.
- Within individual tissues, predictive performance is near zero or negative.
- On an independent dataset (PRISM), predictions transfer with about half of their GDSC-measured correlation, bounded by how well the two assays agree with each other. Models still beat the tissue baseline for most drugs.
- An apparent ~4× enrichment of identified biomarkers for known cancer genes disappears under a fair null model, because the candidate genes were cancer-curated to begin with.
- Adjusting for tissue removes most pooled gene–response associations, and the genes that remain match known biology for several drugs.

## Research Objectives

1. Predict drug response from cancer cell-line gene expression.
2. Compare several regression models, each under both raw and PCA features, using a leakage-free evaluation.
3. Quantify how much predictive signal is explained by tissue alone.
4. Test whether the models generalize to an independent drug-screening dataset.
5. Identify genes associated with drug response, separating tissue effects from within-tissue effects.
6. Compare identified genes with known cancer genes using an appropriate null.

## Dataset

| Data | Description |
|---|---|
| Gene expression | RMA-normalized basal expression, 17,419 genes × 1,018 cell lines (GDSC) |
| Drug response | GDSC1 and GDSC2 fitted dose-response (LN_IC50) |
| Cell-line metadata | Tissue (OncoTree lineage) annotations and ID mapping (DepMap `Model.csv`) |
| Cancer gene reference | OncoKB cancer gene list |
| External validation | PRISM Repurposing secondary screen (DepMap) |

GDSC1 and GDSC2 use different assays and drug panels, so they are analyzed **separately**. Each (drug, screen) pair is one analysis unit, e.g. `Erlotinib [GDSC2]`. This gives 468 units, of which 464 had enough cell lines (≥ 200) to model. Units are not fully independent: the same drug can appear in both screens.

See `data/README.md` for sources and setup.

## Methodology

### 1. Features
For each drug, the models see only a drug-specific **candidate gene set**, not all 17k genes. It is built from the drug's annotated targets, curated gene lists for its pathway, and MSigDB Hallmark sets matched to that pathway (`reference_data.py`). If fewer than 8 candidate genes are present in the expression data, a fixed list of 20 well-known cancer genes is used instead. Candidate sets are built without using drug-response values.

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
- **Genome-wide gene–response associations** for the six best-predicted drug–screen units: Pearson r between each of the 17,419 genes and LN(IC50), with Benjamini–Hochberg FDR correction. Each association is computed twice: pooled across all cell lines, and within tissue (gene expression and LN(IC50) are centered on their tissue mean before correlating; tissues with fewer than 5 cell lines are merged into "Other"). A gene counts as a hit at FDR q < 0.05 and |r| ≥ 0.2.
- **OncoKB comparison** against the pool of candidate genes actually offered to the models, with a drug-aware permutation null.

### 5. External validation (PRISM)
Models were tested against the PRISM Repurposing secondary screen, an independent viability assay.

- Drugs were matched by name (213 of 464 units had a PRISM counterpart; PRISM's `crizotinib-(S)` was excluded as a different stereoisomer). PRISM cell lines were mapped to COSMIC IDs through `Model.csv`.
- For each matched drug and PRISM screen, **all cell lines measured in both datasets were held out**. The model type selected in the main run was refit on the remaining GDSC cell lines, and its predictions for the held-out lines were compared with PRISM AUC using **within-drug Spearman correlation** (response scales differ, so R² is not comparable).
- Each result is shown next to two references: the same predictions scored against GDSC's own measurements of the same lines, and the agreement between the GDSC and PRISM measurements themselves (a ceiling).
- PRISM screens (HTS002, MTS010, MTS006) were analyzed separately and never pooled. Expression is GDSC microarray throughout, so this tests transfer of the drug-response measurement, not of the expression platform.

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
A baseline that predicts only each cell line's tissue mean reaches a median test R² of **0.136**, compared with **0.154** for the selected expression model. The expression model beats the tissue-only baseline for 60% of units (paired Wilcoxon p = 3.5e-08, descriptive because units are not independent). Most of the predictive signal is therefore tissue of origin, with expression adding a modest amount. Because expression itself differs by lineage, expression models can capture tissue implicitly, so this is best read as "expression adds only a little beyond tissue", not "expression is uninformative".

### External validation on PRISM
Median within-drug Spearman correlation across units, on cell lines held out of training:

| | HTS002 (202 units) | MTS010 (76 units) |
|---|---|---|
| Model vs GDSC, same held-out cell lines | 0.244 | 0.247 |
| **Model vs PRISM** | **0.111** | **0.148** |
| Tissue-mean baseline vs PRISM | 0.074 | 0.091 |
| GDSC vs PRISM measurements (ceiling) | 0.234 | 0.297 |
| Model beats tissue baseline on PRISM | 69% of units | 80% of units |
| Drug-level correlation, GDSC vs PRISM performance | 0.48 | 0.68 |

MTS006 had only 11 units and is in `results/tables/external_validation.csv`. Where the two assays agree well (HTS002 units with assay agreement ≥ 0.3, n = 65), the model reaches 0.218 against a ceiling of 0.375 and beats the tissue baseline for 78% of units; this subset is defined from the two measurements only, not from model performance.

Predictions transfer to an independent assay but retain only about half of their GDSC-measured correlation. Part of this reflects assay disagreement: GDSC and PRISM measurements of the same cell lines correlate at only about 0.23–0.30, so absolute external performance is bounded. Models still beat a tissue-only baseline for most drugs, and drugs that were predictable in GDSC tended to be predictable in PRISM. Units are not independent (the same cell lines are held out for every drug, and drugs appear in both GDSC screens), so p-values are descriptive.

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

- **BH3-mimetic drugs.** BCL2 is the top sensitivity gene for Venetoclax and ABT-737 and is among the top six for Navitoclax. MCL1 is among the top resistance genes for ABT-737 and Navitoclax, and for the MCL1 inhibitor AZD5991 the top resistance gene is BCL2L1 (BCL-XL). This is consistent with other anti-apoptotic proteins compensating when one is inhibited.
- **Nutlin-3a.** The top sensitivity genes (EDA2R, MDM2, DDB2, RPS27L, AEN, ZMAT3) are p53 target genes, consistent with Nutlin-3a requiring active p53.
- **Shared signal.** YAP1, LMNA, TNFRSF12A, BLVRB and CTSL1 appear among the top resistance genes for several BH3-mimetic drugs. These drugs are related, so this is not independent evidence, and it may reflect a general cell state.

These six drugs were chosen because they are well predicted, so they are not representative of all drugs. Tissue adjustment is conservative and also removes real between-lineage biology. Associations are not causal, and some top genes (for example for alpha-lipoic acid) have no obvious explanation. Full ranked lists are in `results/tables/biomarker_top_genes.csv`.

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

These counts largely reflect how many drugs' candidate gene sets contain each gene (TP53 and MYC appear in many pathway lists). They are not evidence of independent biological importance.

### Pathway-level summary
Pathways with the most biomarker associations: PI3K/mTOR signaling (2,360), RTK signaling (2,120), DNA replication (1,360), cell cycle (1,236), ERK/MAPK (1,010), mitosis (980), and apoptosis regulation (940). These totals scale with the number of drugs mapped to each pathway, so they describe where the analysis had the most data, not which pathways matter most.

### OncoKB comparison
Candidate genes come from curated cancer pathway lists, so they are enriched for known cancer genes by construction. Testing biomarkers against all ~17k expressed genes would therefore overstate enrichment. Instead I tested against the pool of **1,452 candidate genes** actually offered to the models (306 of them in OncoKB):

| Biomarker set | Genes | Overlap with OncoKB | Hypergeometric fold | Drug-aware permutation null |
|---|---|---|---|---|
| Model-based importance | 886 | 255 | 1.37× | 0.97× (empirical p = 0.98) |
| Pearson | 721 | 226 | 1.49× | 0.84× (empirical p = 1.0) |

The permutation null redraws genes from each drug's own candidate pool, preserving pool composition and the number of picks per drug (2,000 permutations). Under this null, the identified biomarkers are **not enriched** for known cancer genes beyond what the candidate lists already contain. Overlap with OncoKB is descriptive context, not independent validation.

### Tissue-specific performance
Selected models were trained across all tissues, then scored within each tissue on the held-out test set (14 tissue groups). Mean within-tissue R² was near zero or negative for nearly every tissue: Lung was highest (0.007; 56% of drug–tissue comparisons positive) and Myeloid lowest (−0.413). Together with the tissue-only baseline, this suggests that much of the pooled signal reflects differences *between* tissue types rather than variation *within* a tissue. Groups with very few comparisons (e.g. Pancreas, n = 7; Bone, n = 17) are not interpretable, and per-tissue R² on small test subsets is noisy and biased downward.

## Visualizations
`results/figures/` (generated by `make_figures.py` from the pipeline outputs):

- `fig1_performance_vs_baselines.png`: held-out R² of the selected models vs the tissue-mean and dummy baselines
- `fig2_raw_vs_pca.png`: raw vs PCA features for each model, per drug
- `fig3_model_vs_tissue_baseline.png`: selected model vs tissue-only baseline, per drug
- `fig4_top_drugs.png`: top drug–screen units by test R², alongside the tissue baseline
- `fig5_within_tissue_r2.png`: performance within each tissue
- `fig6_oncokb_null.png`: observed OncoKB overlap vs the drug-aware permutation null
- `fig7_biomarker_example.png`: one gene's expression vs drug response, colored by tissue
- `fig8_external_validation.png`: GDSC-trained models scored against PRISM on held-out cell lines, with the assay-agreement ceiling and tissue baseline
- `biomarker/`: per-drug gene–response associations (pooled and tissue-adjusted volcano plots, plus top genes)

`results/tables/` contains the corresponding evaluation, feature-importance, biomarker, external-validation and tissue tables.

## Interactive App
`app/streamlit_app.py` lets you browse every drug–screen unit: the selected model beside its tissue and mean baselines, all 12 model configurations, top genes, within-tissue performance, PRISM validation, and the tissue-adjusted gene associations. It reads only the committed `results/` folder, so no raw data are needed.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Technologies
Python · NumPy, Pandas, SciPy · scikit-learn, XGBoost · Matplotlib, Seaborn · Streamlit · pytest, GitHub Actions

## Repository Structure
```
cancer-drug-response-biomarker-analysis/
├── train.py                 # main pipeline: models, CV, importance, OncoKB test
├── reference_data.py        # drug-target and pathway gene lists
├── make_figures.py          # figures and biomarker tables from pipeline outputs
├── audit_prism.py           # GDSC / PRISM overlap and assay-agreement audit
├── external_validation.py   # held-out-cell-line validation on PRISM
├── app/                     # Streamlit app
│   ├── streamlit_app.py
│   ├── app_data.py
│   └── requirements.txt
├── tests/                   # pytest suite
├── .github/workflows/ci.yml # lint + tests on every push
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── data/
│   └── README.md            # raw data not included
└── results/
    ├── figures/
    └── tables/
```

## Reproducing the Analysis
1. Clone the repository and install dependencies.
2. Obtain the datasets listed in `data/README.md` and place them in `data/`.
3. Run the steps below.

```bash
git clone <repository-url>
cd cancer-drug-response-biomarker-analysis

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py              # full run (~2 hours on 8 CPU cores); writes to outputs/
python audit_prism.py        # overlap with PRISM (needs the PRISM file in data/)
python external_validation.py
python make_figures.py       # figures and biomarker tables
```

`train.py` caches results per drug in `outputs/cache/`, so an interrupted run resumes where it stopped; delete that folder after changing the modeling code. The first run needs internet access to download Hallmark gene sets (cached afterwards). `make_figures.py` writes figures to `results/figures/` and the biomarker tables to `results/tables/`; copy the other tables you want to publish from `outputs/` to `results/tables/`.

### Tests and CI
```bash
pip install -r requirements-dev.txt
pyflakes train.py reference_data.py make_figures.py external_validation.py audit_prism.py app tests
pytest -q
```
The tests use small synthetic data (no GDSC or PRISM files). They check, among other things, that biomarkers are always labelled with genes, that the best model is chosen by CV and never a baseline, that GDSC1 and GDSC2 stay separate, and that the enrichment and tissue-adjusted association statistics are computed correctly. GitHub Actions runs the same checks on every push.

## Limitations
- Predictive performance is modest for most drugs (median test R² ≈ 0.15), and most of it is explained by tissue of origin.
- Models see only a curated candidate gene set per drug, which limits what they can discover and builds cancer-relevant genes into the biomarker list.
- Gene–response associations are shown for six well-predicted drugs only. They are associations, not causal effects, and tissue adjustment also removes real between-lineage biology.
- Within-tissue performance is poor, so predictions may not generalize within a cancer type.