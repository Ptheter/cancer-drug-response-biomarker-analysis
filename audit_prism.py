"""
audit_prism.py — how much do GDSC and the PRISM Repurposing secondary screen overlap?

Answers, before any modeling:
  1. How many of your modeled drug-screen units have a PRISM counterpart (by name)?
  2. For those, how many cell lines are shared (and how many GDSC cell lines
     would remain for training if the shared ones are held out)?
  3. Is the sign convention what we expect? (GDSC LN_IC50 vs PRISM AUC should
     correlate POSITIVELY: higher = more resistant in both.)
  4. How well do the two assays agree on the same cell lines? (a ceiling for
     any model trained on one and tested on the other)

    python audit_prism.py --prism data/secondary-screen-dose-response-curve-parameters.csv
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def norm(name: str) -> str:
    """'Nutlin-3a (-)' -> 'nutlin3a'; lowercase alphanumerics only, drop '(…)' tags."""
    s = re.sub(r"\(.*?\)", "", str(name)).lower()
    return re.sub(r"[^a-z0-9]", "", s)


def find_col(cols, *candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prism", default="data/secondary-screen-dose-response-curve-parameters.csv")
    ap.add_argument("--data", default="data")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--min-shared", type=int, default=50,
                    help="min shared cell lines to count a drug as testable")
    ap.add_argument("--min-train", type=int, default=200,
                    help="min GDSC cell lines left for training after holding out shared ones")
    a = ap.parse_args()

    # ── PRISM ────────────────────────────────────────────────────────────────
    header = pd.read_csv(a.prism, nrows=0).columns.tolist()
    c_name = find_col(header, "name", "pert_name", "compound", "drug_name")
    c_id   = find_col(header, "depmap_id", "modelid", "model_id")
    c_auc  = find_col(header, "auc")
    c_scr  = find_col(header, "screen_id")
    print("PRISM columns used:", {"name": c_name, "cell line": c_id, "auc": c_auc,
                                  "screen": c_scr})
    if not (c_name and c_id and c_auc):
        raise SystemExit(f"Could not find name/cell-line/AUC columns in: {header}")
    use = [c for c in (c_name, c_id, c_auc, c_scr) if c]
    pr = pd.read_csv(a.prism, usecols=use).rename(
        columns={c_name: "name", c_id: "depmap_id", c_auc: "auc"})
    if c_scr:
        print("PRISM screens:", pr[c_scr].value_counts().to_dict())
    if c_scr:
        pr = pr.rename(columns={c_scr: "screen"})
    else:
        pr["screen"] = "all"
    pr = pr.dropna(subset=["name", "depmap_id", "auc"])
    pr["key"] = pr["name"].map(norm)
    print(f"PRISM: {pr.name.nunique():,} compounds, {pr.depmap_id.nunique():,} cell lines, "
          f"{len(pr):,} rows")

    # ── cell-line ID bridge: ACH-... -> COSMIC_ID ────────────────────────────
    ann = pd.read_csv(os.path.join(a.data, "Model.csv"), usecols=["ModelID", "COSMICID"])
    ann = ann.dropna()
    ann["COSMIC_ID"] = pd.to_numeric(ann["COSMICID"], errors="coerce")
    ann = ann.dropna(subset=["COSMIC_ID"]).drop_duplicates("ModelID")
    ach2cosmic = dict(zip(ann.ModelID, ann.COSMIC_ID.astype(int)))
    pr["COSMIC_ID"] = pr["depmap_id"].map(ach2cosmic)
    print(f"PRISM cell lines with a COSMIC ID: "
          f"{pr.dropna(subset=['COSMIC_ID']).depmap_id.nunique()} / {pr.depmap_id.nunique()}")
    pr = pr.dropna(subset=["COSMIC_ID"]).copy()
    pr["COSMIC_ID"] = pr["COSMIC_ID"].astype(int)
    names_per_key = pr.groupby("key")["name"].agg(lambda x: " | ".join(sorted(set(x))))
    n_names_per_key = pr.groupby("key")["name"].nunique()
    # one AUC per (compound, screen, cell line): average only replicate rows
    # WITHIN a screen; different PRISM screens are kept apart
    pr = pr.groupby(["key", "screen", "COSMIC_ID"], as_index=False).agg(
        auc=("auc", "mean"), prism_name=("name", "first"))

    # ── GDSC: expression cell lines + response ───────────────────────────────
    expr = pd.read_pickle(os.path.join(a.data, "expr.pkl"))
    cols = pd.to_numeric(expr.columns, errors="coerce")
    expr_cells = set(int(c) for c in cols[~np.isnan(cols)])
    print(f"GDSC expression cell lines: {len(expr_cells)}")

    def load(screen):
        d = pd.read_csv(os.path.join(a.data, f"{screen}_dose_response.csv"),
                        usecols=["COSMIC_ID", "DRUG_NAME", "LN_IC50"])
        return d.assign(SCREEN=screen)
    g = pd.concat([load("GDSC1"), load("GDSC2")], ignore_index=True).dropna()
    g["COSMIC_ID"] = g["COSMIC_ID"].astype(int)
    g["unit"] = g["DRUG_NAME"] + " [" + g["SCREEN"] + "]"
    g["key"] = g["DRUG_NAME"].map(norm)
    g = g[g.COSMIC_ID.isin(expr_cells)]
    g = g.groupby(["unit", "key", "COSMIC_ID"], as_index=False).LN_IC50.mean()

    # restrict to units the pipeline actually modeled
    rp = os.path.join(a.outputs, "results_pca.csv")
    sel_r2 = {}
    if os.path.exists(rp):
        res = pd.read_csv(rp)
        sel_r2 = res[res.Selected].set_index("Drug").R2.to_dict()
        g = g[g.unit.isin(sel_r2)]
    print(f"GDSC units considered: {g.unit.nunique()}")

    # ── per-(unit, screen) overlap and agreement ─────────────────────────────
    pr_by = {(k, sc): d.set_index("COSMIC_ID").auc for (k, sc), d in pr.groupby(["key", "screen"])}
    screens_for = pr.groupby("key").screen.unique().to_dict()
    name_of = pr.groupby("key").prism_name.first().to_dict()
    strict = lambda x: re.sub(r"[\s\-_]", "", str(x).lower())   # keeps (S), (+), (-) tags
    rows = []
    for unit, d in g.groupby("unit"):
        key = d.key.iloc[0]
        if key not in screens_for:
            continue
        base = re.sub(r"\s*\[GDSC[12]\]$", "", unit)
        d = d.set_index("COSMIC_ID").LN_IC50
        for sc in screens_for[key]:
            p = pr_by[(key, sc)]
            shared = d.index.intersection(p.index)
            rho = np.nan
            if len(shared) >= 10:
                rho = spearmanr(d.loc[shared].to_numpy(float), p.loc[shared].to_numpy(float))[0]
            rows.append({"unit": unit, "prism_name": name_of[key], "screen": sc,
                         "all_prism_names": names_per_key[key],
                         "name_differs": strict(base) != strict(name_of[key]),
                         "ambiguous": bool(n_names_per_key[key] > 1),
                         "n_gdsc": len(d), "n_prism": len(p), "n_shared": len(shared),
                         "n_train_after_holdout": len(d) - len(shared),
                         "assay_spearman": rho,
                         "gdsc_selected_test_R2": sel_r2.get(unit, np.nan)})
    out = pd.DataFrame(rows).sort_values(["unit", "n_shared"], ascending=[True, False])
    os.makedirs(a.outputs, exist_ok=True)
    path = os.path.join(a.outputs, "prism_overlap_audit.csv")
    out.to_csv(path, index=False)

    # ── summary ──────────────────────────────────────────────────────────────
    ok_rows = out[(out.n_shared >= a.min_shared) & (out.n_train_after_holdout >= a.min_train)]
    print("\n" + "=" * 70)
    print(f"GDSC units modeled:                              {g.unit.nunique()}")
    print(f"... with a name match in PRISM:                  {out.unit.nunique()}")
    print(f"... testable in >= 1 screen (>= {a.min_shared} shared, >= {a.min_train} to train): "
          f"{ok_rows.unit.nunique()}")
    print("\nPer PRISM screen (testable unit-screen pairs only):")
    summ = (ok_rows.groupby("screen")
            .agg(units=("unit", "nunique"), median_shared=("n_shared", "median"),
                 median_assay_rho=("assay_spearman", "median"),
                 pct_positive=("assay_spearman", lambda x: 100 * (x.dropna() > 0).mean()))
            .round(2))
    print(summ.to_string())
    nd = out[out.name_differs | out.ambiguous].drop_duplicates("unit")
    print(f"\nMatches to REVIEW (name differs beyond case/hyphens, or PRISM has several "
          f"names for it): {len(nd)} units")
    print(nd[["unit", "all_prism_names"]].head(30).to_string(index=False))
    print("=" * 70)
    print("saved", path)


if __name__ == "__main__":
    main()