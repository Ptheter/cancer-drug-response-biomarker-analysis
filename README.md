# Cancer Drug Response Prediction and Biomarker Discovery

## Overview

This project investigates whether gene expression profiles can predict cancer cell-line sensitivity to anti-cancer drugs and identify genes associated with drug response.

Using gene expression and drug-response data from the Genomics of Drug Sensitivity in Cancer (GDSC) datasets, I developed a machine learning pipeline to predict drug sensitivity measured by **LN(IC50)** across hundreds of compounds.

The project combines predictive modeling, feature importance analysis, and biological interpretation to explore potential molecular biomarkers associated with drug sensitivity.

## Research Objectives

- Predict drug response from cancer cell-line gene expression profiles.
- Compare the performance of multiple regression models.
- Identify genes associated with drug sensitivity across different compounds.
- Investigate pathway-level patterns among identified biomarkers.
- Examine differences in predictive performance across cancer tissue types.
- Compare discovered biomarkers with known cancer-associated genes.

## Dataset

The analysis integrates:

- **Gene expression:** Basal gene expression profiles from cancer cell lines.
- **Drug response:** GDSC1 and GDSC2 drug sensitivity measurements.
- **Cell-line metadata:** Tissue and cell-line annotations.
- **Cancer gene reference:** Known cancer-associated genes used for biological comparison.

Approximate project scale:

- ~1,000 cancer cell lines
- ~17,000 gene-expression features
- 363 anti-cancer compounds

See [`data/README.md`](data/README.md) for dataset sources and setup instructions.

## Methodology

### 1. Data preprocessing

- Loaded and integrated gene expression and drug-response datasets.
- Matched cell lines across expression and drug-response records.
- Prepared gene-expression features and drug-specific response targets.
- Used LN(IC50) as the regression target.

### 2. Machine learning models

Evaluated the following models:

- Linear Regression
- Lasso Regression
- ElasticNet
- Random Forest
- XGBoost
- Dummy Regression baseline

Linear models and the dummy baseline were evaluated using PCA-transformed features. Random Forest and XGBoost were evaluated using standardized gene-expression features without PCA.

### 3. Model evaluation

- Five-fold cross-validation
- R²
- Root Mean Squared Error (RMSE)
- Predicted-versus-actual analysis
- Model comparisons across individual drugs

The best-performing non-dummy model was identified separately for each drug based on cross-validation R².

### 4. Biomarker discovery

- Feature importance analysis
- Pearson correlation analysis
- Gene-level biomarker summaries
- Pathway-level analysis
- Tissue-specific performance analysis
- Comparison against known cancer-associated genes

## Results

### Predictive performance

Across 363 drugs:

- **361 of 363 drugs (99.45%)** had a positive cross-validation R² for their best-performing non-dummy model.
- Median best-model cross-validation R²: **0.169**
- Mean best-model cross-validation R²: **0.179**

The results indicate that gene expression contains predictive information for drug response, although predictive performance varies substantially across compounds. The median R² indicates that there is considerable room for improvement.

### Frequently identified biomarkers

| Gene | Number of drug associations |
|---|---:|
| TP53 | 155 |
| MYC | 146 |
| EGFR | 98 |
| CCND1 | 98 |
| KRAS | 90 |
| MAPK1 | 88 |
| MTOR | 71 |
| PIK3CA | 71 |
| AKT1 | 69 |
| BCL2 | 66 |

These counts represent the number of drug-specific biomarker associations in the analysis, not independent experimental validations.

### Pathway-level findings

Frequently represented pathways included:

- PI3K/MTOR signaling
- Receptor tyrosine kinase (RTK) signaling
- DNA replication
- Cell-cycle regulation
- ERK/MAPK signaling
- Apoptosis regulation

These findings highlight the relevance of signaling, proliferation, and cell-survival pathways in the drug-response analysis.

### OncoKB Cancer Gene Enrichment

To assess whether identified biomarkers were enriched for known cancer-associated genes, we performed a one-sided hypergeometric enrichment test using OncoKB.

- **269 of 969** biomarker genes overlapped with OncoKB cancer genes represented in the expression dataset.
- The observed overlap was approximately **4.39-fold higher** than expected under random sampling.
- The enrichment was statistically significant (**p = 2.65 × 10⁻¹⁰⁶**).

These findings indicate substantial overlap between the computationally identified biomarkers and established cancer-associated genes. This analysis provides external biological context but does not independently validate their predictive or therapeutic utility.

### Tissue-Specific Predictive Performance

Predictive performance varied across cancer tissue groups when evaluating drug–tissue combinations separately.

- **Lung:** Highest mean R² (0.020), with 62.2% positive comparisons across 360 drug–tissue combinations.
- **Bone:** 72.7% positive comparisons, although only 11 combinations were available.
- **Lymphoid:** 39.2% positive comparisons across 360 combinations.
- **Myeloid:** Lowest mean R² (-0.472).
- **Peripheral Nervous System:** Mean R² of -0.415, with only 28 comparisons.

Overall, tissue-stratified results suggest that the relationship between gene expression and drug response varies across cancer types. However, most tissue groups had negative mean R² values, and sample sizes varied substantially. These findings should therefore be interpreted cautiously.

## Visualizations

The [`results/figures`](results/figures/) directory contains figures illustrating:

- Cross-validation R² distributions
- Drug-level predictive performance
- Pathway-level model comparisons
- Predicted versus actual drug response
- Feature importance
- Tissue-specific model performance

The [`results/tables`](results/tables/) directory contains the corresponding model evaluation and biomarker summary tables.

## Technologies

**Programming:** Python

**Data analysis:** NumPy, Pandas, SciPy

**Machine learning:** Scikit-learn, XGBoost

**Dimensionality reduction:** PCA

**Visualization:** Matplotlib, Seaborn

**Statistical analysis:** Pearson correlation, feature importance, enrichment analysis

## Repository Structure

```text
cancer-drug-response-biomarker-analysis/
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── README.md
└── results/
    ├── figures/
    └── tables/
```

## Reproducing the Analysis

1. Clone the repository.
2. Install the required Python dependencies.
3. Obtain the datasets listed in `data/README.md`.
4. Place the required files in the `data/` directory.
5. Run the pipeline.

```bash
git clone YOUR_REPOSITORY_URL
cd cancer-drug-response-biomarker-analysis

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python train.py
```

The pipeline generates model evaluation tables, biomarker summaries, visualizations, and cached intermediate files in `outputs/`.

## Limitations

- Predictive performance varies across drugs, with a modest median cross-validation R².
- Gene-expression associations do not establish causal relationships with drug sensitivity.
- Identified biomarkers require independent experimental or clinical validation.
- The model comparison uses different preprocessing strategies for linear and tree-based models, so it is not a fully factorial comparison of every model under both feature representations.

## Future Improvements

- Benchmark all models using both PCA and standardized raw gene-expression features.
- Incorporate additional molecular data, such as mutation and copy-number profiles.
- Explore alternative feature selection and dimensionality reduction methods.
- Validate candidate biomarkers using independent datasets.
- Develop more robust drug-response prediction models.
