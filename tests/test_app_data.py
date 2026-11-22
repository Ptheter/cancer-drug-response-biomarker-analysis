"""
Data helpers for the Streamlit app.

Pure pandas/numpy (no streamlit import) so they can be unit-tested in CI.
Every function takes the tables the pipeline already writes to results/tables/.
"""
import re

import numpy as np
import pandas as pd

BASELINES = ("Dummy", "TissueMean")


def safe_name(drug: str) -> str:
    """File-name-safe version of a drug label (matches make_figures.py)."""
    return re.sub(r"[^\w\-]", "_", drug)


def build_unit_table(results: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    """One row per drug-screen unit: the CV-selected model next to its baselines."""
    sel = results[results["Selected"]].set_index("Drug")
    out = pd.DataFrame({"Pathway": sel["Pathway"], "Model": sel["Model"],
                        "Test_R2": sel["R2"]})
    cv_mean = cv.set_index(["Drug", "Model"])["CV_R2_mean"]
    out["CV_R2"] = [cv_mean.get((d, m), np.nan) for d, m in zip(out.index, out["Model"])]
    for model_name, col in [("TissueMean", "Tissue_R2"), ("Dummy", "Dummy_R2")]:
        s = results[results["Model"] == model_name].set_index("Drug")["R2"]
        out[col] = s.reindex(out.index)
    out["Delta_vs_tissue"] = out["Test_R2"] - out["Tissue_R2"]
    out["Beats_tissue"] = (out["Delta_vs_tissue"] > 0) & out["Tissue_R2"].notna()
    return out.reset_index().rename(columns={"index": "Drug"})


def model_table(results: pd.DataFrame, cv: pd.DataFrame, drug: str) -> pd.DataFrame:
    """All models for one drug, with CV and test R² side by side."""
    r = results[results["Drug"] == drug][["Model", "Base_Model", "Input", "R2", "Selected"]]
    r = r.rename(columns={"R2": "Test_R2"})
    c = cv[cv["Drug"] == drug][["Model", "CV_R2_mean", "CV_R2_std"]]
    m = r.merge(c, on="Model", how="left")
    m["Kind"] = np.where(m["Base_Model"].isin(BASELINES), "Baseline", "Gene model")
    m["Label"] = np.where(m["Selected"], "★ " + m["Model"], m["Model"])
    return m.sort_values("Test_R2", ascending=False).reset_index(drop=True)


def headline(units: pd.DataFrame) -> dict:
    """Headline numbers shown on the overview page."""
    has_tissue = units["Tissue_R2"].notna()
    return {
        "n_units": int(len(units)),
        "median_test_r2": float(units["Test_R2"].median()),
        "median_cv_r2": float(units["CV_R2"].median()),
        "pct_test_positive": float(100 * (units["Test_R2"] > 0).mean()),
        "median_tissue_r2": float(units["Tissue_R2"].median()) if has_tissue.any() else np.nan,
        "pct_beats_tissue": float(100 * units.loc[has_tissue, "Beats_tissue"].mean())
        if has_tissue.any() else np.nan,
    }


def raw_vs_pca(cv: pd.DataFrame) -> pd.DataFrame:
    """Median CV R² per base model, by feature representation."""
    g = cv[~cv["Base_Model"].isin(BASELINES)]
    return g.groupby(["Base_Model", "Input"])["CV_R2_mean"].median().unstack()


def selected_counts(results: pd.DataFrame) -> pd.Series:
    return results[results["Selected"]]["Model"].value_counts()


def top_genes(imp: pd.DataFrame, drug: str, n: int = 10):
    """(model-importance table, Pearson table with direction) for one drug."""
    d = imp[imp["Drug"] == drug]
    model = (d[d["Method"] != "Pearson"].sort_values("Rank").head(n)
             [["Rank", "Gene", "Importance", "Method"]])
    pearson = d[d["Method"] == "Pearson"].sort_values("Rank").head(n)
    cols = [c for c in ["Rank", "Gene", "Importance", "r"] if c in pearson.columns]
    pearson = pearson[cols].copy()
    if "r" in pearson.columns:
        pearson["Direction"] = np.where(pearson["r"] > 0,
                                        "higher expression → more resistant",
                                        "higher expression → more sensitive")
    return model, pearson


def tissue_summary(strat: pd.DataFrame) -> pd.DataFrame:
    """Within-tissue R² of the selected models, summarised per tissue."""
    g = strat.groupby("Tissue")
    out = pd.DataFrame({
        "Drugs": g["Drug"].nunique(),
        "Median_R2": g["R2"].median(),
        "Pct_positive": g["R2"].apply(lambda x: 100 * (x > 0).mean()),
        "Test_samples": g["N"].sum(),
    })
    return out.sort_values("Median_R2", ascending=False).reset_index()


def external_summary(ev: pd.DataFrame) -> pd.DataFrame:
    """PRISM external-validation medians per screen."""
    rows = []
    for screen, d in ev.groupby("screen"):
        both = d.dropna(subset=["rho_prism", "rho_tissue_prism"])
        rows.append({
            "Screen": screen,
            "Units": int(d["unit"].nunique()),
            "Model vs GDSC": d["rho_gdsc"].median(),
            "Model vs PRISM": d["rho_prism"].median(),
            "Tissue baseline vs PRISM": d["rho_tissue_prism"].median(),
            "GDSC vs PRISM (ceiling)": d["rho_assay"].median(),
            "% model > tissue": 100 * (both["rho_prism"] > both["rho_tissue_prism"]).mean()
            if len(both) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("Units", ascending=False).reset_index(drop=True)


def verdict(row: pd.Series) -> str:
    """Plain-language reading of one unit's result (always shows the baseline)."""
    if pd.isna(row["Tissue_R2"]):
        return "No tissue baseline is available for this unit."
    delta = row["Delta_vs_tissue"]
    if row["Test_R2"] <= 0:
        return ("The selected model is no better than predicting the average response "
                "(test R² ≤ 0).")
    if delta > 0.02:
        return (f"Expression adds information beyond tissue: the model beats the "
                f"tissue-only baseline by {delta:.2f} R².")
    if delta > -0.02:
        return "The model performs about as well as knowing only the tissue of origin."
    return (f"A tissue-only baseline does better than the model here "
            f"(by {-delta:.2f} R²); the signal is mostly lineage.")