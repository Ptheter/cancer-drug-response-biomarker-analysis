# ══════════════════════════════════════════════════════════════════════════════
# Drug Response Prediction & Biomarker Discovery Pipeline
# ══════════════════════════════════════════════════════════════════════════════
#
# Overview
# --------
# This script builds a large-scale machine learning pipeline for predicting
# anti-cancer drug response (LN_IC50) from cancer cell line gene expression
# profiles using GDSC (Genomics of Drug Sensitivity in Cancer) datasets.
#
# The overall goal is to:
#   1. Predict how sensitive different cancer cell lines are to specific drugs
#   2. Identify biologically meaningful biomarkers associated with drug response
#   3. Compare multiple machine learning models across signaling pathways
#   4. Interpret important genes driving predictions
#   5. Explore tissue-specific differences in predictive performance
#
#
# Main Features
# -------------
# • Loads and preprocesses GDSC drug-response datasets
# • Loads cancer cell-line gene expression matrices
# • Maps drugs to pathways and putative targets
# • Expands pathway gene sets using MSigDB Hallmark signatures
# • Trains multiple regression models including:
#       - Linear Regression
#       - Lasso Regression
#       - ElasticNet
#       - Random Forest
#       - XGBoost
#       - Dummy baseline
#
# • Evaluates models using:
#       - Train/test R²
#       - RMSE
#       - 5-fold cross-validation
#
# • Performs:
#       - PCA dimensionality reduction
#       - Feature importance analysis
#       - Permutation importance
#       - Pearson correlation biomarker analysis
#       - Tissue-stratified evaluation
#       - Prediction interval estimation
#
# • Generates publication-style figures including:
#       - Performance distributions
#       - Heatmaps
#       - Predicted vs actual scatter plots
#       - Volcano plots
#       - Biomarker clustermaps

# • Performs external biological validation using:
#       - OncoKB cancer gene enrichment analysis
#       - Statistical enrichment testing against known cancer genes
#       - Known vs potentially novel biomarker identification
#
#
# Expected Outputs
# ----------------
# The pipeline saves:
#
#   /outputs
#       results_pca.csv
#       results_cv.csv
#       feature_importance.csv
#       stratified_r2.csv
#       prediction_intervals.csv
#       figure2*.png
#
# along with biomarker discovery visualizations and cached intermediate
# results for faster reruns.
#
#
# Long-Term Goal
# --------------
# This project aims to contribute toward interpretable precision oncology
# workflows by combining machine learning with biologically informed feature
# selection. The broader objective is to better understand how molecular
# expression patterns influence therapeutic response and to support future
# biomarker-driven cancer treatment strategies.
#
# ══════════════════════════════════════════════════════════════════════════════


import os
import re
import warnings
import logging
import urllib.request
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from scipy.stats import hypergeom
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from reference_data import (TARGET_TO_GENES, SKIP_WORDS, PATHWAY_TO_GENES,
                            FALLBACK_GENES, STOP_WORDS)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent

CFG = dict(
    data_dir        = str(PROJECT_ROOT / "data"),
    output_dir      = str(PROJECT_ROOT / "outputs"),

    min_samples     = 200,
    skip_pathways   = {"Other", "Unclassified", "Other, kinases"},

    test_size       = 0.20,
    random_state    = 42,

    n_splits        = 5,

    pca_max         = 50,
    pca_min         = 5,

    alpha_log_range = (-4, 1, 60),
    l1_ratios       = [0.1, 0.2, 0.5, 0.8, 0.95, 1.0],
    max_iter        = 10_000,

    min_avail_genes = 8,

    r2_threshold    = 0.3,
    dpi             = 300,

    # P1 — feature importance
    imp_top_n           = 15,       # genes shown in bar chart / clustermap
    perm_n_repeats      = 20,       # permutation importance repeats

    # P2 — stratification
    min_tissue_samples  = 10,       # min test-set samples per tissue to report R²

    # P4 — prediction intervals
    pi_n_estimators     = 100,
    pi_max_depth        = 4,
    pi_learning_rate    = 0.05,
)

Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
Path(os.path.join(CFG["output_dir"], "cache")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(CFG["output_dir"], "biomarker")).mkdir(parents=True, exist_ok=True)

TREE_MODELS   = {"XGBoost", "RandomForest"}
LINEAR_MODELS = {"Linear", "Lasso", "ElasticNet"}
BASE_MODELS = ["Linear", "Lasso", "ElasticNet", "RandomForest", "XGBoost"]
MODEL_ORDER = [f"{b}-{i}" for b in BASE_MODELS for i in ("Raw", "PCA")]
BASELINES   = {"Dummy", "TissueMean"}

class TissueMeanRegressor(RegressorMixin, BaseEstimator):
    """Predict the training mean LN_IC50 of the sample's tissue
    (global training mean for tissues unseen in training).
    X is a single column of tissue labels, not gene expression."""

    def fit(self, X, y):
        tissues = np.asarray(X).ravel()
        s = pd.Series(np.asarray(y, dtype=float), index=tissues)
        self.global_mean_ = float(s.mean())
        self.means_ = s.groupby(level=0).mean().to_dict()
        return self

    def predict(self, X):
        return np.array([self.means_.get(t, self.global_mean_)
                         for t in np.asarray(X).ravel()], dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_target_genes(putative_target_str: str) -> list:
    if pd.isna(putative_target_str):
        return []
    genes = []
    for t in str(putative_target_str).split(","):
        clean = t.strip()
        if not clean or clean.lower() in SKIP_WORDS:
            continue
        genes.extend(TARGET_TO_GENES.get(clean, [clean]))
    return list(set(genes))


def clean_words(text: str) -> set:
    text = re.sub(r"[^\w\s]", " ", text.upper())
    return {w for w in text.split() if w not in STOP_WORDS}


def find_msigdb_matches(pathway_name: str, hallmark: dict) -> list:
    pw_words = clean_words(pathway_name)
    return [h for h in hallmark if pw_words & clean_words(h.replace("HALLMARK_", ""))]


def expand_with_msigdb(pathway_name, existing_genes, hallmark, auto_mapping):
    extra = []
    for s in auto_mapping.get(pathway_name, []):
        extra.extend(hallmark.get(s, []))
    return list(set(existing_genes + extra))


def make_pca_pipeline(n_features: int, n_train: int, cfg: dict) -> Pipeline:
    n_comp = max(cfg["pca_min"], min(cfg["pca_max"], n_features, n_train - 1))
    return Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n_comp))])

def build_estimators(cfg: dict) -> dict:
    """Fresh, unfitted estimators (call again for a second set)."""
    alphas = np.logspace(*cfg["alpha_log_range"])
    return {
        "Linear":       LinearRegression(),
        "Lasso":        LassoCV(alphas=alphas, cv=cfg["n_splits"],
                                max_iter=cfg["max_iter"], precompute=False),
        "ElasticNet":   ElasticNetCV(l1_ratio=cfg["l1_ratios"], alphas=alphas,
                                     cv=cfg["n_splits"], max_iter=cfg["max_iter"],
                                     precompute=False),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                              max_features="sqrt",
                                              random_state=cfg["random_state"],
                                              n_jobs=1),
        "XGBoost":      XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                     subsample=0.8, colsample_bytree=0.8,
                                     reg_alpha=0.1, reg_lambda=1.0,
                                     random_state=cfg["random_state"],
                                     verbosity=0, n_jobs=1),
    }

def build_model_specs(cfg: dict) -> list:
    """[(label, base_name, input_type, estimator), ...] — every model under
    both representations, plus the mean-predicting Dummy baseline."""
    specs = [("Dummy", "Dummy", "Raw", DummyRegressor(strategy="mean"))]
    for inp in ("Raw", "PCA"):
        for base, est in build_estimators(cfg).items():
            specs.append((f"{base}-{inp}", base, inp, est))
    return specs

def select_genes(pathway_name, target_str, available, hallmark, auto_mapping, cfg):
    """Candidate genes offered to the models for one drug.
    Returns (sorted gene list, used_fallback). Sorted => reproducible feature
    order (set order differs between worker processes)."""
    target_genes  = get_target_genes(target_str)
    pathway_genes = PATHWAY_TO_GENES.get(pathway_name, [])
    expanded      = expand_with_msigdb(pathway_name, pathway_genes, hallmark, auto_mapping)
    avail = sorted(g for g in set(target_genes + expanded) if g in available)
    if len(avail) < cfg["min_avail_genes"]:
        return sorted(g for g in FALLBACK_GENES if g in available), True
    return avail, False

def pca_n_components(n_features: int, n_train: int, cfg: dict) -> int:
    fold_n = int(n_train * (1 - 1 / cfg["n_splits"]))   # size of a CV training fold
    return max(2, min(cfg["pca_max"], n_features, fold_n - 1))

def build_pipeline(estimator, use_pca: bool, n_components: int) -> Pipeline:
    """Every model is ONE pipeline that takes the RAW gene matrix.
    Scaling (and PCA) are fitted inside each CV fold -> no leakage, and
    importance can always be computed on gene columns."""
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components)))
    steps.append(("model", estimator))
    return Pipeline(steps)

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(drug_name: str, cfg: dict) -> str:
    safe = re.sub(r"[^\w\-]", "_", drug_name)
    return os.path.join(cfg["output_dir"], "cache", f"{safe}.json")

def _save_cache(drug_name: str, payload: tuple, cfg: dict):
    result_rows, cv_rows, strat_rows, imp_rows, pi_rows, pred_payload=payload
    data = {
        "result_rows": result_rows,
        "cv_rows":     cv_rows,
        "strat_rows":  strat_rows,
        "imp_rows":    imp_rows,
        "pi_rows":     pi_rows,
        "pred_payload": pred_payload,
    }
    with open(_cache_path(drug_name, cfg), "w") as f:
        json.dump(data, f)

def _load_cache(drug_name: str, cfg: dict):
    p = _cache_path(drug_name, cfg)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        data = json.load(f)
    return (
        data["result_rows"],
        data["cv_rows"],
        data.get("strat_rows", []),
        data.get("imp_rows",   []),
        data.get("pi_rows",    []),
        data.get("pred_payload", None),
    )


# ══════════════════════════════════════════════════════════════════════════════
# P1 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def compute_tree_importance(model, gene_names: list) -> pd.DataFrame:
    """Built-in gain-based feature importance for XGBoost / RandomForest.
    Available on every fitted sklearn-compatible tree model — no extra libs."""
    try:
        importances = model.feature_importances_
        return (
            pd.DataFrame({"gene": gene_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        log.warning(f"    Tree importance failed: {e}")
        return pd.DataFrame(columns=["gene", "importance"])


def compute_correlation_importance(
    X_test: np.ndarray, y_test: np.ndarray, gene_names: list
) -> pd.DataFrame:
    """Pearson |r| between each gene's expression and LN_IC50 on the test set.
    Direction (positive / negative correlation) is preserved in the 'r' column."""
    rows = []
    for i, gene in enumerate(gene_names):
        try:
            r, p = pearsonr(X_test[:, i], y_test)
        except Exception:
            r, p = 0.0, 1.0
        rows.append({"gene": gene, "importance": abs(r), "r": r, "p": p})
    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_permutation_imp(
    model, X_test: np.ndarray, y_test: np.ndarray,
    gene_names: list, cfg: dict,
) -> pd.DataFrame:
    """Permutation importance for linear models in PCA space."""
    try:
        pi = permutation_importance(
            model, X_test, y_test,
            n_repeats=cfg["perm_n_repeats"],
            random_state=cfg["random_state"],
            n_jobs=1,
        )
        labels = (gene_names if len(gene_names) == X_test.shape[1]
                  else [f"PC{i+1}" for i in range(X_test.shape[1])])
        return (
            pd.DataFrame({
                "gene":       labels,
                "importance": pi.importances_mean,
                "std":        pi.importances_std,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        log.warning(f"    Permutation importance failed: {e}")
        return pd.DataFrame(columns=["gene", "importance", "std"])


# ══════════════════════════════════════════════════════════════════════════════
# P2 — STRATIFIED ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def stratified_r2(
    preds: np.ndarray,
    y_test: np.ndarray,
    test_ids: list,
    cell_tissue: pd.Series,
    drug_name: str,
    cfg: dict,
) -> list:
    rows = []
    ct        = cell_tissue[~cell_tissue.index.duplicated(keep="first")]
    id_series = pd.to_numeric(pd.Series(test_ids), errors="coerce")
    tissues   = ct.reindex(id_series).values

    if pd.Series(tissues).notna().sum() == 0:
        log.warning(f"  stratified_r2: 0/{len(test_ids)} IDs matched for {drug_name}")
        return []

    for tissue in pd.Series(tissues).dropna().unique():
        mask = tissues == tissue
        if mask.sum() < cfg["min_tissue_samples"]:
            continue
        rows.append({
            "Drug":   drug_name,
            "Tissue": tissue,
            "R2":     r2_score(y_test[mask], preds[mask]),
            "N":      int(mask.sum()),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# P4 — PREDICTION INTERVALS
# ══════════════════════════════════════════════════════════════════════════════

def fit_prediction_intervals(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    drug_name: str, cfg: dict,
) -> list:
    rows = []
    try:
        kw = dict(
            n_estimators=cfg["pi_n_estimators"],
            max_depth=cfg["pi_max_depth"],
            learning_rate=cfg["pi_learning_rate"],
            random_state=cfg["random_state"],
        )
        lower_m = GradientBoostingRegressor(loss="quantile", alpha=0.10, **kw)
        upper_m = GradientBoostingRegressor(loss="quantile", alpha=0.90, **kw)
        lower_m.fit(X_train, y_train)
        upper_m.fit(X_train, y_train)
        lo  = lower_m.predict(X_test)
        hi  = upper_m.predict(X_test)
        rows.append({
            "Drug":         drug_name,
            "PI_coverage":  float(np.mean((y_test >= lo) & (y_test <= hi))),
            "PI_avg_width": float(np.mean(hi - lo)),
        })
    except Exception as e:
        log.warning(f"    Prediction interval fitting failed: {e}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# PER-DRUG WORKER
# ══════════════════════════════════════════════════════════════════════════════

def run_drug(
    drug_name, drug_response,
    expr_values, expr_genes, expr_cells,
    drug_to_pathway, drug_to_target,
    hallmark, auto_msig_mapping, cell_tissue, cfg,
):
    expr = pd.DataFrame(expr_values, index=expr_genes,
                        columns=expr_cells).astype(np.float32)

    # ── Filter samples ────────────────────────────────────────────────────────
    drug_df    = drug_response[drug_response["DRUG_NAME"] == drug_name]
    drug_df    = drug_df[["COSMIC_ID", "LN_IC50"]].dropna()
    common_ids = list(set(expr.columns).intersection(set(drug_df["COSMIC_ID"])))
    if len(common_ids) < cfg["min_samples"]:
        log.info(f"[SKIP] {drug_name}: only {len(common_ids)} samples.")
        return None

    X_full = expr[common_ids].T.copy()
    X_full.columns = X_full.columns.astype(str)
    X_full = X_full.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    drug_indexed = drug_df.set_index("COSMIC_ID")
    drug_indexed = drug_indexed[~drug_indexed.index.duplicated(keep="first")]
    y_series     = drug_indexed.reindex(X_full.index.astype(int))["LN_IC50"]
    valid_mask   = y_series.notna().values
    X_full       = X_full.iloc[valid_mask]
    y            = y_series.values[valid_mask]
    common_ids   = list(X_full.index.astype(int))
    if len(common_ids) < cfg["min_samples"]:
        log.info(f"[SKIP] {drug_name}: only {len(common_ids)} samples after alignment.")
        return None
    assert len(X_full) == len(y)

    X_full = X_full.loc[:, ~X_full.columns.duplicated()]
    X_full = X_full.loc[:, X_full.columns != "nan"]

    # ── Gene set ──────────────────────────────────────────────────────────────
    pathway_name = drug_to_pathway.get(drug_name, "Other")
    avail_genes, using_fallback = select_genes(
        pathway_name, drug_to_target.get(drug_name, ""),
        set(X_full.columns), hallmark, auto_msig_mapping, cfg,
    )
    log.info(f"{drug_name} | {pathway_name} | genes={len(avail_genes)} "
             f"{'(fallback)' if using_fallback else ''}")

    X = X_full[avail_genes].values.astype(np.float32)

    # Tissue label per sample, for the TissueMean baseline (None if no annotation)
    X_tissue = None
    if cell_tissue is not None:
        ct = cell_tissue[~cell_tissue.index.duplicated(keep="first")]
        tis = ct.reindex(pd.Series(common_ids)).fillna("Unknown").astype(str).values
        X_tissue = tis.reshape(-1, 1)

    # ── Split once; test set used only for final reporting ────────────────────
    idx = np.arange(len(common_ids))
    idx_train, idx_test = train_test_split(
        idx, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    test_ids = [common_ids[i] for i in idx_test]

    n_comp = pca_n_components(X_train.shape[1], X_train.shape[0], cfg)
    kf     = KFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["random_state"])

    result_rows, cv_rows, imp_rows, strat_rows, pi_rows = [], [], [], [], []
    best_cv, best = -np.inf, None          # best = (label, base, pipe, preds)

    def _score_and_log(label, base, inp, est_or_pipe, Xtr, Xte, n_genes):
        """CV on the training split, then one fit + test evaluation."""
        cv = cross_validate(
            est_or_pipe, Xtr, y_train, cv=kf,
            scoring={"r2": "r2", "mse": "neg_mean_squared_error"},
        )
        cv_r2, cv_rmse = cv["test_r2"], np.sqrt(-cv["test_mse"])
        est_or_pipe.fit(Xtr, y_train)
        preds = est_or_pipe.predict(Xte)
        r2    = float(r2_score(y_test, preds))
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        common = {"Drug": drug_name, "Pathway": pathway_name, "Model": label,
                  "Base_Model": base, "Input": inp, "N_Genes": n_genes}
        cv_rows.append({**common,
                        "CV_R2_mean": float(cv_r2.mean()), "CV_R2_std": float(cv_r2.std()),
                        "CV_RMSE_mean": float(cv_rmse.mean()), "CV_RMSE_std": float(cv_rmse.std())})
        result_rows.append({**common, "R2": r2, "RMSE": rmse, "Selected": False})
        log.info(f"  {label:18s} | CV R²={cv_r2.mean():.3f} ± {cv_r2.std():.3f} | Test R²={r2:.3f}")
        return float(cv_r2.mean()), preds

    # ── Gene-based models, both representations ───────────────────────────────
    for label, base, inp, estimator in build_model_specs(cfg):
        pipe = build_pipeline(estimator, inp == "PCA", n_comp)
        cv_mean, preds = _score_and_log(label, base, inp, pipe, X_train, X_test,
                                        len(avail_genes))
        # choose by CV R²; baselines are never eligible
        if base not in BASELINES and cv_mean > best_cv:
            best_cv, best = cv_mean, (label, base, pipe, preds)

    # ── Tissue-mean baseline (reported, never selected) ───────────────────────
    if X_tissue is not None:
        _score_and_log("TissueMean", "TissueMean", "Tissue", TissueMeanRegressor(),
                       X_tissue[idx_train], X_tissue[idx_test], 0)

    best_label = best[0] if best else None
    for row in result_rows:
        row["Selected"] = bool(row["Model"] == best_label)

    # ── Feature importance — always in gene space ─────────────────────────────
    best_pipe = best[2] if best else None
    best_base = best[1] if best else None
    best_preds = best[3] if best else None
    if best_pipe is not None:
        if best_base in TREE_MODELS:
            imp_local = compute_tree_importance(best_pipe.named_steps["model"], avail_genes)
            method    = "XGB_importance" if best_base == "XGBoost" else "RF_importance"
        else:
            imp_local = compute_permutation_imp(best_pipe, X_test, y_test, avail_genes, cfg)
            method    = "Permutation"
        corr_local = compute_correlation_importance(X, y, avail_genes)

        for rank, row in enumerate(imp_local.itertuples(index=False)):
            imp_rows.append({
                "Drug": drug_name, "Pathway": pathway_name, "Model": best_label,
                "Method": method, "Gene": row.gene,
                "Importance": float(row.importance), "Rank": rank + 1,
            })
        for rank, row in enumerate(corr_local.itertuples(index=False)):
            imp_rows.append({
                "Drug": drug_name, "Pathway": pathway_name, "Model": best_label,
                "Method": "Pearson", "Gene": row.gene,
                "Importance": float(row.importance), "Rank": rank + 1,
                "r": float(row.r),
            })

    if best_preds is not None and cell_tissue is not None:
        strat_rows = stratified_r2(best_preds, y_test, test_ids, cell_tissue, drug_name, cfg)
    if best_pipe is not None:
        pi_rows = fit_prediction_intervals(X_train, y_train, X_test, y_test, drug_name, cfg)

    return result_rows, cv_rows, strat_rows, imp_rows, pi_rows, {
        "Drug": drug_name,
        "y_test": y_test.tolist(),
        "preds": best_preds.tolist() if best_preds is not None else None,
        "test_ids": test_ids,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_hallmark_direct() -> dict:
    url = (
        "https://maayanlab.cloud/Enrichr/geneSetLibrary"
        "?mode=text&libraryName=MSigDB_Hallmark_2020"
    )
    log.info("Fetching Hallmark gene sets from Enrichr API...")
    hallmark = {}
    with urllib.request.urlopen(url, timeout=30) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            parts = line.split("\t")
            hallmark[parts[0]] = [g for g in parts[2:] if g]
    log.info(f"Fetched {len(hallmark)} Hallmark gene sets.")
    return hallmark


def load_data(cfg: dict):
    data_dir = cfg["data_dir"]
    pkl_expr = f"{data_dir}/expr.pkl"

    if not os.path.exists(pkl_expr):
        expr = pd.read_csv(
            f"{data_dir}/Cell_line_RMA_proc_basalExp.txt", sep="\t", index_col=0
        )
        expr.columns = expr.columns.str.replace("DATA.", "", regex=False)
        expr = expr.loc[:, expr.columns != "GENE_title"]
        expr.columns = pd.to_numeric(expr.columns, errors="coerce")
        expr = expr.loc[:, expr.columns.notna()]
        expr.columns = expr.columns.astype(int)

        log.info("Cleaning expression matrix values...")
        expr = (
            expr.astype(str)
            .apply(lambda col: col.str.strip().str.strip("[]"))
            .apply(pd.to_numeric, errors="coerce")
        )
        expr = expr.replace([np.inf, -np.inf], np.nan)
        expr = expr.dropna(how="all")
        expr.index = expr.index.astype(str).str.strip()
        expr = expr.loc[expr.index.notna()]
        expr = expr.loc[~expr.index.isin(["", "nan"])]
        expr = expr.loc[~expr.index.duplicated(keep="first")]
        expr.to_pickle(pkl_expr)
        log.info(f"Expression matrix: {expr.shape[0]} genes × {expr.shape[1]} cell lines")
    else:
        expr = pd.read_pickle(pkl_expr)
        expr.columns = pd.to_numeric(expr.columns, errors="coerce")
        expr = expr.loc[:, expr.columns.notna()]
        expr.columns = expr.columns.astype(int)
        log.info(f"Loaded expression matrix from pickle: {expr.shape}")

    for key, xl, csv in [
        ("gdsc1", f"{data_dir}/GDSC1_fitted_dose_response_27Oct23.xlsx",
                  f"{data_dir}/GDSC1_dose_response.csv"),
        ("gdsc2", f"{data_dir}/GDSC2_fitted_dose_response_27Oct23.xlsx",
                  f"{data_dir}/GDSC2_dose_response.csv"),
    ]:
        if not os.path.exists(csv):
            pd.read_excel(xl).to_csv(csv, index=False)

    gdsc1 = pd.read_csv(f"{data_dir}/GDSC1_dose_response.csv")
    gdsc2 = pd.read_csv(f"{data_dir}/GDSC2_dose_response.csv")

    cell_tissue = None
    for fname in ["Model.csv", "GDSC_cell_lines_details.csv", "cell_line_details.csv"]:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            ann = pd.read_csv(fpath)
            ann.columns = ann.columns.str.strip()
            ann = ann.rename(columns={"COSMICID": "COSMIC_ID", "OncotreeLineage": "TISSUE"})
            cell_tissue = ann.set_index("COSMIC_ID")["TISSUE"]
            log.info(f"Loaded tissue annotations from {fname} ({len(cell_tissue)} lines)")
            break

    if cell_tissue is None:
        log.warning(
            "No tissue annotation file found. "
            "Stratified analysis (P2, Fig 2E, 2G) will be skipped."
        )

    return expr, gdsc1, gdsc2, cell_tissue

def build_drug_response(gdsc1: pd.DataFrame, gdsc2: pd.DataFrame) -> pd.DataFrame:
    """Keep GDSC1 and GDSC2 as separate analysis units.

    They use different assays/drug panels, so averaging LN_IC50 across them
    mixes two measurement processes. Each (drug, screen) pair becomes its own
    'drug' named e.g. 'Erlotinib [GDSC2]', so everything downstream
    (caching, CV, figures) works unchanged. DRUG_BASE keeps the plain name for
    later cross-dataset matching (CTRP / PRISM)."""
    cols = ["COSMIC_ID", "DRUG_NAME", "LN_IC50", "PATHWAY_NAME", "PUTATIVE_TARGET"]
    dr = pd.concat(
        [gdsc1[cols].assign(SCREEN="GDSC1"), gdsc2[cols].assign(SCREEN="GDSC2")],
        ignore_index=True,
    ).dropna(subset=["COSMIC_ID", "DRUG_NAME", "LN_IC50"])
    dr["DRUG_BASE"] = dr["DRUG_NAME"].astype(str)
    dr["DRUG_NAME"] = dr["DRUG_BASE"] + " [" + dr["SCREEN"] + "]"
    return (
        dr.groupby(["COSMIC_ID", "DRUG_NAME"], as_index=False)
          .agg(LN_IC50=("LN_IC50", "mean"),          # only replicate rows within ONE screen
               PATHWAY_NAME=("PATHWAY_NAME", "first"),
               PUTATIVE_TARGET=("PUTATIVE_TARGET", "first"),
               DRUG_BASE=("DRUG_BASE", "first"),
               SCREEN=("SCREEN", "first"))
    )


# ══════════════════════════════════════════════════════════════════════════════
# BIOMARKER EXPORT
# ══════════════════════════════════════════════════════════════════════════════

"""
Export biomarker summary tables for downstream biological interpretation.
Generates:
    1. biomarker_summary.csv
        -> one row per biomarker gene
    2. pathway_biomarker_summary.csv
        -> grouped pathway-level biomarkers
    3. tissue_biomarker_summary.csv
        -> tissue-specific predictive performance
"""

def export_biomarker_summary(
    imp_df: pd.DataFrame,
    strat_df: pd.DataFrame,
    drug_to_pathway: dict,
    cfg: dict,
):
    out_dir = cfg["output_dir"]
    if imp_df.empty:
        log.warning("No feature importance data available.")
        return

    biomarker_rows = []
    for _, row in imp_df.iterrows():
        drug     = row["Drug"]
        pathway  = row.get("Pathway", drug_to_pathway.get(drug, "Unknown"))
        gene     = row["Gene"]
        method   = row["Method"]
        rank     = row["Rank"]
        imp      = row["Importance"]
        biomarker_type = "Unknown"

        # Pearson correlations preserve direction
        if method == "Pearson" and "r" in row:
            r = row["r"]
            if r > 0:
                biomarker_type = "Resistance"
            elif r < 0:
                biomarker_type = "Sensitivity"

        biomarker_rows.append({
            "Drug": drug,
            "Pathway": pathway,
            "Gene": gene,
            "Importance_Method": method,
            "Importance_Score": imp,
            "Rank": rank,
            "Biomarker_Type": biomarker_type,
            "Correlation_r": row.get("r", np.nan),
        })

    biomarker_df = pd.DataFrame(biomarker_rows)
    biomarker_df = biomarker_df.sort_values(
        ["Drug", "Importance_Score"],
        ascending=[True, False]
    )
    biomarker_df.to_csv(
        os.path.join(out_dir, "biomarker_summary.csv"),
        index=False
    )
    log.info("Saved biomarker_summary.csv")

    # Gene-level biomarker aggregation
    gene_summary = (
        biomarker_df
        .groupby("Gene")
        .agg(
            Num_Drugs=("Drug", "nunique"),
            Mean_Importance=("Importance_Score", "mean"),
            Max_Importance=("Importance_Score", "max")
        )
        .reset_index()
        .sort_values(
            ["Num_Drugs", "Mean_Importance"],
            ascending=False
        )
    )
    gene_summary.to_csv(
        os.path.join(out_dir, "gene_biomarker_summary.csv"),
        index=False
    )
    log.info("Saved gene_biomarker_summary.csv")

    # Pathway-level biomarker aggregation
    # Pathway-level summary
    pathway_level_summary = (
        biomarker_df
        .groupby("Pathway")
        .agg(
            Num_Genes=("Gene", "nunique"),
            Total_Drug_Associations=("Drug", "count"),
            Mean_Importance=("Importance_Score", "mean")
        )
        .reset_index()
        .sort_values(
            "Total_Drug_Associations",
            ascending=False
        )
    )

    pathway_level_summary.to_csv(
        os.path.join(out_dir, "pathway_level_summary.csv"),
        index=False
    )

    log.info("Saved pathway_level_summary.csv")

    # Tissue-specific summary
    if not strat_df.empty:
        tissue_summary = (
            strat_df
            .groupby(["Tissue", "Drug"])
            .agg(
                Mean_R2=("R2", "mean"),
                Samples=("N", "sum"),
            )
            .reset_index()
            .sort_values(
                ["Tissue", "Mean_R2"],
                ascending=[True, False]
            )
        )
        tissue_summary.to_csv(
            os.path.join(out_dir, "tissue_biomarker_summary.csv"),
            index=False
        )
        log.info("Saved tissue_biomarker_summary.csv")


"""
Compare discovered biomarker genes against known cancer genes (OncoKB-style dataset).
Outputs overlap statistics + annotated gene table.
"""

def validate_known_biomarkers(
    biomarker_csv_path: str,
    oncokb_csv_path: str,
    output_dir: str = CFG["output_dir"]
):

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------
    biomarkers = pd.read_csv(biomarker_csv_path)
    oncokb = pd.read_csv(oncokb_csv_path, sep="\t")
    biomarkers["Gene"] = biomarkers["Gene"].astype(str).str.upper().str.strip()
    oncokb["Hugo Symbol"] = oncokb["Hugo Symbol"].astype(str).str.upper().str.strip()
    oncokb_genes = set(oncokb["Hugo Symbol"])

    # annotate overlap
    biomarkers["In_OncoKB"] = biomarkers["Gene"].isin(oncokb_genes)

    # optional extra annotation if column exists
    if "Gene Type" in oncokb.columns:
        gene_type_map = oncokb.set_index("Hugo Symbol")["Gene Type"].to_dict()
        biomarkers["Gene_Type"] = biomarkers["Gene"].map(gene_type_map).fillna("Unknown")
    else:
        biomarkers["Gene_Type"] = "Unknown"

    # summary stats
    total_genes = biomarkers["Gene"].nunique()
    known_genes = biomarkers.loc[biomarkers["In_OncoKB"], "Gene"].nunique()
    novel_genes = total_genes - known_genes
    overlap_pct = (known_genes / total_genes * 100) if total_genes > 0 else 0

    top_known = (
        biomarkers[biomarkers["In_OncoKB"]]
        .groupby(["Importance_Method", "Gene"])
        .agg(
            Mean_Importance=("Importance_Score", "mean"),
            Num_Drugs=("Drug", "nunique"),
            Num_Pathways=("Pathway", "nunique")
        )
        .sort_values(["Importance_Method", "Mean_Importance"], ascending=[True, False])
        .reset_index()
    )

    # save outputs
    overlap_file = os.path.join(output_dir, "known_cancer_gene_overlap.csv")
    summary_file = os.path.join(output_dir, "known_cancer_gene_summary.csv")

    biomarkers.to_csv(overlap_file, index=False)
    top_known.to_csv(summary_file, index=False)

    print("\n" + "=" * 70)
    print("GENERAL CANCER GENE VALIDATION")
    print("=" * 70)

    print(f"Total unique genes discovered: {total_genes}")
    print(f"Known cancer genes recovered:   {known_genes}")
    print(f"Not in OncoKB:                 {novel_genes}")
    print(f"In OncoKB (descriptive only):  {overlap_pct:.2f}%")

    print("\nTop recovered known cancer genes (Pearson):")
    if len(top_known) > 0:
        print(top_known[top_known["Importance_Method"] == "Pearson"].head(15))

    print("\nSaved files:")
    print(f"- {overlap_file}")
    print(f"- {summary_file}")

    print("\nValidation completed successfully.")
    print("=" * 70)

def compute_oncokb_enrichment(universe_genes, biomarker_genes, oncokb_genes) -> dict:
    """One-sided hypergeometric test. `universe_genes` must be the pool the
    biomarkers could actually have come from (the candidate genes offered to
    the models), NOT every gene on the array."""
    onc      = {g.upper() for g in oncokb_genes}
    universe = {g.upper() for g in universe_genes}
    bio      = {g.upper() for g in biomarker_genes} & universe

    N, K = len(universe), len(universe & onc)
    n, k = len(bio), len(bio & onc)
    p = hypergeom(N, K, n).sf(k - 1)
    expected = n * K / N if N else 0.0
    return {
        "N_total_genes": N, "K_oncokb_genes": K, "n_biomarkers": n, "k_overlap": k,
        "expected_overlap": expected,
        "enrichment_fold": (k / expected) if expected > 0 else 0.0,
        "p_value": float(p),
    }

def oncokb_permutation_null(candidate_pools: dict, biomarkers: pd.DataFrame,
                            oncokb_genes, n_perm: int = 2000, seed: int = 42) -> dict:
    """Drug-aware null. For each drug, redraw as many genes as were selected
    from THAT drug's own candidate pool; take the union across drugs and count
    OncoKB genes. Preserves pool composition and the number of picks per drug."""
    rng = np.random.default_rng(seed)
    onc = {g.upper() for g in oncokb_genes}

    bm = biomarkers.assign(Gene=biomarkers["Gene"].astype(str).str.upper())
    bm = bm[bm["Drug"].isin(candidate_pools)]
    n_per_drug = bm.groupby("Drug")["Gene"].nunique()
    k_obs = len(set(bm["Gene"]) & onc)

    pools, picks = [], []
    for d, n in n_per_drug.items():
        pool = np.array(sorted({g.upper() for g in candidate_pools[d]}))
        pools.append(pool); picks.append(min(int(n), len(pool)))

    onc_arr = np.array(sorted(onc))
    null = np.empty(n_perm, dtype=int)
    for b in range(n_perm):
        drawn = np.unique(np.concatenate(
            [rng.choice(p, size=m, replace=False) for p, m in zip(pools, picks)]
        ))
        null[b] = np.isin(drawn, onc_arr).sum()

    mean = float(null.mean())
    return {
        "k_observed": k_obs, "null_counts": null.tolist(), "null_mean": mean, "null_sd": float(null.std()),
        "fold_vs_null": (k_obs / mean) if mean > 0 else float("nan"),
        "empirical_p": float((1 + (null >= k_obs).sum()) / (1 + n_perm)),
        "n_perm": n_perm,
    }

def print_enrichment_results(label: str, res: dict, null: dict = None):
    print("\n" + "=" * 70)
    print(f"ONCOKB ENRICHMENT — {label}")
    print("=" * 70)
    print(f"Universe (candidate genes offered, N):  {res['N_total_genes']}")
    print(f"OncoKB genes in universe (K):           {res['K_oncokb_genes']}")
    print(f"Biomarker genes (n):                    {res['n_biomarkers']}")
    print(f"Observed overlap (k):                   {res['k_overlap']}")
    print(f"Expected overlap:                       {res['expected_overlap']:.1f}")
    print(f"Enrichment fold (hypergeometric):       {res['enrichment_fold']:.2f}x")
    print(f"p-value (hypergeometric):               {res['p_value']:.2e}")
    if null:
        print(f"Drug-aware permutation null:            mean {null['null_mean']:.1f} "
              f"± {null['null_sd']:.1f}  ->  {null['fold_vs_null']:.2f}x, "
              f"empirical p = {null['empirical_p']:.4f} ({null['n_perm']} perms)")
    print("Note: descriptive annotation; candidates are cancer-curated by design.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Loading data...")
    expr, gdsc1, gdsc2, cell_tissue = load_data(CFG)

    # Safety net: re-clean if any object columns survived pickling
    if expr.dtypes.eq(object).any():
        log.info("Stripping bracket notation from expression matrix...")
        expr = (
            expr.astype(str)
            .apply(lambda col: col.str.strip().str.strip("[]"))
            .apply(pd.to_numeric, errors="coerce")
        )
        expr = expr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        expr.to_pickle(os.path.join(CFG["data_dir"], "expr.pkl"))
        log.info(f"Expression matrix re-saved: {expr.shape}")

    # ── Hallmark ──────────────────────────────────────────────────────────────
    hallmark_cache = os.path.join(CFG["data_dir"], "hallmark.pkl")
    if os.path.exists(hallmark_cache):
        hallmark = pd.read_pickle(hallmark_cache)
        log.info("Loaded Hallmark from local cache.")
    else:
        hallmark = fetch_hallmark_direct()
        pd.to_pickle(hallmark, hallmark_cache)

    # ── Drug response ─────────────────────────────────────────────────────────
    drug_response = build_drug_response(gdsc1, gdsc2)

    auto_msig_mapping = {
        p: find_msigdb_matches(p, hallmark)
        for p in PATHWAY_TO_GENES
        if find_msigdb_matches(p, hallmark)
    }

    drug_meta = (
        drug_response.groupby("DRUG_NAME")
        .agg(pathway=("PATHWAY_NAME", "first"), target=("PUTATIVE_TARGET", "first"),
             n_samples=("COSMIC_ID", "nunique"))
        .reset_index()
    )
    drug_meta = drug_meta[
        (drug_meta["n_samples"] >= CFG["min_samples"]) &
        (~drug_meta["pathway"].isin(CFG["skip_pathways"])) &
        (drug_meta["pathway"].notna())
    ].copy()

    drug_list       = drug_meta["DRUG_NAME"].tolist()
    drug_to_pathway = dict(zip(drug_meta["DRUG_NAME"], drug_meta["pathway"]))
    drug_to_target  = dict(zip(drug_meta["DRUG_NAME"], drug_meta["target"]))

    log.info(f"Total drugs: {len(drug_list)}")

    # ── Cache-aware dispatch ──────────────────────────────────────────────────
    to_run, cached = [], {}
    for d in drug_list:
        hit = _load_cache(d, CFG)
        if hit is not None:
            log.info(f"[CACHE HIT] {d}")
            cached[d] = hit
        else:
            to_run.append(d)

    log.info(f"Cache hits: {len(cached)} | To run: {len(to_run)}")

    expr_values = expr.values
    expr_genes  = expr.index.tolist()
    expr_cells  = expr.columns.tolist()

    fresh_outputs = Parallel(n_jobs=-1, verbose=5, backend="loky")(
        delayed(run_drug)(
            d, drug_response, expr_values, expr_genes, expr_cells,
            drug_to_pathway, drug_to_target,
            hallmark, auto_msig_mapping, cell_tissue, CFG,
        )
        for d in to_run
    )

    all_results, all_cv, all_strat, all_imp, all_pi = [], [], [], [], []
    all_predictions = {}
    for d, out in zip(to_run, fresh_outputs):
        if out is None:
            continue
        _save_cache(d, out, CFG)
        r, c, s, i, p, pred_payload = out
        all_results.extend(r); all_cv.extend(c)
        all_strat.extend(s);   all_imp.extend(i); all_pi.extend(p)
        all_predictions[d] = pred_payload

    for r, c, s, i, p, pred_payload in cached.values():
        all_results.extend(r); all_cv.extend(c)
        all_strat.extend(s);   all_imp.extend(i); all_pi.extend(p)

        if pred_payload is not None:
            all_predictions[pred_payload["Drug"]] = pred_payload

    if not all_results:
        log.error("No results collected.")
        return

    log.info(f"Total result rows: {len(all_results)} | CV rows: {len(all_cv)}")

    # ── DataFrames & CSVs ─────────────────────────────────────────────────────
    out_dir       = CFG["output_dir"]
    results_df    = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
                by="R2",
                ascending=False
    )
    results_df.to_csv(
        f"{out_dir}/results_test.tsv",
        sep="\t",
        index=False
    )

    cv_results_df = pd.DataFrame(all_cv)
    imp_df        = pd.DataFrame(all_imp)
    imp_df = imp_df[imp_df["Rank"] <= 20]
    strat_df      = pd.DataFrame(all_strat)
    pi_df         = pd.DataFrame(all_pi)

    results_df.to_csv(   f"{out_dir}/results_pca.csv",           index=False)
    cv_results_df = cv_results_df.sort_values(
        by="CV_R2_mean",
        ascending=False
    )
    cv_results_df.to_csv(
        f"{out_dir}/results_cv.csv",
        index=False
)
    imp_df.to_csv(       f"{out_dir}/feature_importance.csv",    index=False)
    strat_df.to_csv(     f"{out_dir}/stratified_r2.csv",         index=False)
    pi_df.to_csv(        f"{out_dir}/prediction_intervals.csv",  index=False)

    export_biomarker_summary(
        imp_df,
        strat_df,
        drug_to_pathway,
        CFG
    )

    validate_known_biomarkers(
        biomarker_csv_path=os.path.join(CFG["output_dir"], "biomarker_summary.csv"),
        oncokb_csv_path=os.path.join(CFG["data_dir"], "cancerGeneList.tsv"),
        output_dir=CFG["output_dir"]
)
    oncokb = pd.read_csv(os.path.join(CFG["data_dir"], "cancerGeneList.tsv"), sep="\t")
    bm = pd.read_csv(os.path.join(CFG["output_dir"], "biomarker_summary.csv"))
    bm["Gene"] = bm["Gene"].astype(str).str.upper()

    # Candidate pool each drug's models were actually offered
    expr_set = set(expr.index.astype(str))
    pools = {
        d: select_genes(drug_to_pathway[d], drug_to_target.get(d, ""),
                        expr_set, hallmark, auto_msig_mapping, CFG)[0]
        for d in bm["Drug"].unique()
    }
    universe = set().union(*pools.values())

    groups = {
        "Model-based": bm[bm["Importance_Method"] != "Pearson"],
        "Pearson":     bm[bm["Importance_Method"] == "Pearson"],
    }
    for label, sub in groups.items():
        res  = compute_oncokb_enrichment(universe, sub["Gene"], oncokb["Hugo Symbol"])
        null = oncokb_permutation_null(pools, sub, oncokb["Hugo Symbol"])
        print_enrichment_results(label, res, null)
        with open(os.path.join(CFG["output_dir"], f"oncokb_null_{label}.json"), "w") as f:
            json.dump({"label": label, **null}, f)

    non_dummy_cv = cv_results_df[~cv_results_df["Model"].isin(BASELINES)]
    best_per_drug = (
        non_dummy_cv
        .loc[non_dummy_cv.groupby("Drug")["CV_R2_mean"].idxmax()]
        [["Drug", "Pathway", "Model", "Input", "N_Genes", "CV_R2_mean", "CV_R2_std"]]
        .reset_index(drop=True)
        .sort_values("CV_R2_mean", ascending=False)
    )
    best_per_drug.to_csv(f"{out_dir}/best_per_drug_all.csv",      index=False)
    best_pos = best_per_drug[best_per_drug["CV_R2_mean"] > 0].copy()
    best_pos.to_csv(f"{out_dir}/best_per_drug_positive.csv",      index=False)

    log.info(f"Drugs with positive signal: {len(best_pos)} / {len(best_per_drug)}")
    #log.info(f"\n{best_pos.to_string(index=False)}")

    log.info("Done. All outputs saved.")


if __name__ == "__main__":
    main()