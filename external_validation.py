"""
external_validation.py — test GDSC-trained models on the PRISM Repurposing screen.

Design (per GDSC drug-screen unit that has a PRISM counterpart):
  * shared cell lines = lines with BOTH a GDSC response and a PRISM AUC
  * HOLD THEM OUT: train only on the remaining GDSC cell lines
  * model = the configuration selected for that unit in the main run
            (results_pca.csv, 'Selected'), refit on the non-shared lines
  * predict the held-out lines, then compare predictions with
        (a) the GDSC measurement of the same lines      -> rho_gdsc
        (b) the PRISM measurement of the same lines     -> rho_prism   (headline)
    and report two references:
        rho_assay        GDSC vs PRISM measurements themselves (ceiling)
        rho_tissue_prism tissue-mean baseline vs PRISM
  * Spearman within each drug (scales differ: LN_IC50 vs viability AUC)

Expression is GDSC microarray for all lines, so no platform shift in this test.
PRISM screens are never pooled; each (unit, screen) is its own row.

    python external_validation.py --n-units 5     # quick test
    python external_validation.py                 # full run
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon

import train as T          # reuses the project's feature selection and pipelines

warnings.filterwarnings("ignore")

# PRISM names that are a different compound from the GDSC drug of the same name
EXCLUDE_PRISM_NAMES = {"crizotinib-(S)"}


def norm(name) -> str:
    s = re.sub(r"\(.*?\)", "", str(name)).lower()
    return re.sub(r"[^a-z0-9]", "", s)


def find_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 10 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


def load_prism(path, data_dir):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    c_name = find_col(header, "name", "pert_name", "compound")
    c_id = find_col(header, "depmap_id", "modelid", "model_id")
    c_auc = find_col(header, "auc")
    c_scr = find_col(header, "screen_id")
    if not (c_name and c_id and c_auc):
        raise SystemExit(f"Missing name/cell-line/AUC columns in {header}")
    use = [c for c in (c_name, c_id, c_auc, c_scr) if c]
    pr = pd.read_csv(path, usecols=use).rename(
        columns={c_name: "name", c_id: "depmap_id", c_auc: "auc",
                 **({c_scr: "screen"} if c_scr else {})})
    if not c_scr:
        pr["screen"] = "all"
    pr = pr.dropna(subset=["name", "depmap_id", "auc"])
    pr = pr[~pr.name.isin(EXCLUDE_PRISM_NAMES)]

    ann = pd.read_csv(os.path.join(data_dir, "Model.csv"), usecols=["ModelID", "COSMICID"])
    ann["COSMIC_ID"] = pd.to_numeric(ann["COSMICID"], errors="coerce")
    ann = ann.dropna(subset=["COSMIC_ID"]).drop_duplicates("ModelID")
    pr["COSMIC_ID"] = pr["depmap_id"].map(dict(zip(ann.ModelID, ann.COSMIC_ID.astype(int))))
    pr = pr.dropna(subset=["COSMIC_ID"])
    pr["COSMIC_ID"] = pr["COSMIC_ID"].astype(int)
    pr["key"] = pr["name"].map(norm)
    pr = pr.groupby(["key", "screen", "COSMIC_ID"], as_index=False).auc.mean()
    return {(k, s): d.set_index("COSMIC_ID").auc for (k, s), d in pr.groupby(["key", "screen"])}


def validate_unit(unit, screen, label, gdsc_y, prism_auc,
                  expr_values, expr_genes, expr_cells,
                  pathway, target, hallmark, auto_map, cell_tissue, min_shared, min_train):
    expr = pd.DataFrame(expr_values, index=expr_genes, columns=expr_cells).astype(np.float32)
    cells = sorted(set(gdsc_y.index) & set(expr_cells))
    X_full = expr[cells].T.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    genes, _ = T.select_genes(pathway, target, set(X_full.columns.astype(str)),
                              hallmark, auto_map, T.CFG)
    X = X_full[genes].values.astype(np.float32)
    y = gdsc_y.loc[cells].values.astype(float)

    mask = np.array([c in prism_auc.index for c in cells])
    n_te, n_tr = int(mask.sum()), int((~mask).sum())
    if n_te < min_shared or n_tr < min_train:
        return None
    y_prism = prism_auc.loc[np.array(cells)[mask]].values.astype(float)

    base, inp = label.rsplit("-", 1)
    n_comp = T.pca_n_components(X[~mask].shape[1], n_tr, T.CFG)
    pipe = T.build_pipeline(T.build_estimators(T.CFG)[base], inp == "PCA", n_comp)
    pipe.fit(X[~mask], y[~mask])
    pred = pipe.predict(X[mask])

    rho_tissue = np.nan
    if cell_tissue is not None:
        ct = cell_tissue[~cell_tissue.index.duplicated(keep="first")]
        tis = ct.reindex(pd.Series(cells)).fillna("Unknown").astype(str).values.reshape(-1, 1)
        tm = T.TissueMeanRegressor().fit(tis[~mask], y[~mask])
        rho_tissue = rho(tm.predict(tis[mask]), y_prism)

    return {"unit": unit, "screen": screen, "model": label, "n_genes": len(genes),
            "n_train": n_tr, "n_test": n_te,
            "rho_prism": rho(pred, y_prism),            # headline
            "rho_gdsc": rho(pred, y[mask]),             # same cell lines, GDSC truth
            "rho_assay": rho(y[mask], y_prism),         # ceiling
            "rho_tissue_prism": rho_tissue}


def summarize(df, title):
    print(f"\n{title}  (n = {len(df)} units)")
    if df.empty:
        return
    for c, nm in [("rho_gdsc", "model vs GDSC (same held-out lines)"),
                  ("rho_prism", "model vs PRISM  <- external"),
                  ("rho_tissue_prism", "tissue-mean baseline vs PRISM"),
                  ("rho_assay", "GDSC vs PRISM measurements (ceiling)")]:
        x = df[c].dropna()
        print(f"  {nm:40s} median {x.median():+.3f} | >0 in {100 * (x > 0).mean():.0f}%")
    both = df[["rho_prism", "rho_tissue_prism"]].dropna()
    if len(both) > 10:
        d = both.rho_prism - both.rho_tissue_prism
        print(f"  model beats tissue baseline on PRISM in {100 * (d > 0).mean():.0f}% "
              f"(paired Wilcoxon p = {wilcoxon(d).pvalue:.2g})")
    both = df[["rho_gdsc", "rho_prism"]].dropna()
    if len(both) > 10:
        print(f"  do drugs that predict well in GDSC also do in PRISM? "
              f"Spearman across drugs = {spearmanr(both.rho_gdsc, both.rho_prism)[0]:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prism", default="data/secondary-screen-dose-response-curve-parameters.csv")
    ap.add_argument("--screens", nargs="+", default=["HTS002", "MTS010", "MTS006"])
    ap.add_argument("--min-shared", type=int, default=50)
    ap.add_argument("--min-train", type=int, default=200)
    ap.add_argument("--n-units", type=int, default=None, help="limit (quick test)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = T.CFG["output_dir"]

    expr, gdsc1, gdsc2, cell_tissue = T.load_data(T.CFG)
    expr = expr.loc[:, ~expr.columns.duplicated(keep="first")]      # duplicate cell lines
    expr = expr.loc[~expr.index.duplicated(keep="first")]

    dr = T.build_drug_response(gdsc1, gdsc2)
    meta = dr.groupby("DRUG_NAME").agg(pathway=("PATHWAY_NAME", "first"),
                                       target=("PUTATIVE_TARGET", "first"),
                                       base=("DRUG_BASE", "first"))
    hallmark = pd.read_pickle(os.path.join(T.CFG["data_dir"], "hallmark.pkl"))
    auto_map = {p: T.find_msigdb_matches(p, hallmark) for p in T.PATHWAY_TO_GENES
                if T.find_msigdb_matches(p, hallmark)}

    res = pd.read_csv(os.path.join(out_dir, "results_pca.csv"))
    selected = res[res.Selected].set_index("Drug").Model.to_dict()
    prism = load_prism(a.prism, T.CFG["data_dir"])

    tasks = []
    for unit, label in selected.items():
        if unit not in meta.index:
            continue
        key = norm(meta.loc[unit, "base"])
        for sc in a.screens:
            if (key, sc) in prism:
                tasks.append((unit, sc, label, key))
    if a.n_units:
        tasks = tasks[:a.n_units]
    print(f"{len(tasks)} (unit, screen) pairs to evaluate")

    gy = {u: d.set_index("COSMIC_ID").LN_IC50 for u, d in dr.groupby("DRUG_NAME")}
    # build the big array ONCE: a fresh copy per task would be written to joblib's
    # temp folder 294 times and fill the disk
    ev, eg, ec = np.ascontiguousarray(expr.values), expr.index.tolist(), expr.columns.tolist()
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(validate_unit)(
            u, sc, lab, gy[u][~gy[u].index.duplicated()], prism[(k, sc)],
            ev, eg, ec,
            meta.loc[u, "pathway"], meta.loc[u, "target"] if pd.notna(meta.loc[u, "target"]) else "",
            hallmark, auto_map, cell_tissue, a.min_shared, a.min_train)
        for u, sc, lab, k in tasks)
    df = pd.DataFrame([r for r in rows if r])
    if df.empty:
        raise SystemExit("No unit passed the sample-size thresholds.")

    path = a.out or os.path.join(out_dir, "external_validation.csv")
    df.to_csv(path, index=False)
    print("\nsaved", path)

    print("=" * 72)
    for sc in a.screens:
        summarize(df[df.screen == sc], f"PRISM screen {sc}")
    sub = df[(df.screen == a.screens[0]) & (df.rho_assay >= 0.3)]
    summarize(sub, f"{a.screens[0]}, subset where the assays agree (rho_assay >= 0.3)")
    print("=" * 72)


if __name__ == "__main__":
    main()