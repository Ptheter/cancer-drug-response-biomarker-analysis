"""
make_figures.py — README figures, built from the pipeline's saved tables.

Reads  outputs/*.csv (+ outputs/oncokb_null_*.json, and data/ for the biomarker
example) and writes PNGs to results/figures/. No models are retrained, so it
runs in seconds and can be rerun while tweaking the plots.

    python make_figures.py
    python make_figures.py --outputs outputs --data data --out results/figures
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, wilcoxon

BASELINES   = ["Dummy", "TissueMean"]
BASE_MODELS = ["Linear", "Lasso", "ElasticNet", "RandomForest", "XGBoost"]
PAL = sns.color_palette("colorblind")
C_MODEL, C_TISSUE, C_DUMMY = PAL[0], PAL[1], PAL[7]

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
plt.rcParams.update({"axes.titlesize": 12, "axes.titleweight": "bold",
                     "figure.dpi": 100, "savefig.dpi": 200})


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def short(drug: str) -> str:
    """'Venetoclax [GDSC2]' -> 'Venetoclax (G2)' for compact labels."""
    return re.sub(r"\s*\[GDSC([12])\]", r" (G\1)", drug)


# ── 1. Performance distribution vs baselines ─────────────────────────────────
def fig_performance(res, out):
    sel = res[res.Selected].R2.rename("Selected model")
    tm  = res[res.Model == "TissueMean"].R2.rename("Tissue-mean baseline")
    du  = res[res.Model == "Dummy"].R2.rename("Dummy (global mean)")
    data = pd.concat([sel, tm, du], axis=1)
    long = data.melt(var_name="Model", value_name="Test R²").dropna()

    fig, ax = plt.subplots(figsize=(7, 4.8))
    order = list(data.columns)
    sns.boxplot(data=long, x="Model", y="Test R²", hue="Model", order=order,
                hue_order=order, showfliers=False, legend=False,
                palette=[C_MODEL, C_TISSUE, C_DUMMY], width=0.55, ax=ax)
    sns.stripplot(data=long, x="Model", y="Test R²", order=order, color="black",
                  alpha=0.15, size=2, jitter=0.25, ax=ax)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    lo = max(long["Test R²"].quantile(0.01), -1.0)
    ax.set_ylim(lo - 0.05, long["Test R²"].max() + 0.05)
    for i, c in enumerate(order):
        ax.text(i, ax.get_ylim()[1], f"median {data[c].median():.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("")
    ax.set_title(f"Held-out R² across {len(sel)} drug–screen units", pad=22)
    fig.tight_layout()
    save(fig, out, "fig1_performance_vs_baselines.png")


# ── 2. Raw vs PCA, per base model ────────────────────────────────────────────
def fig_raw_vs_pca(cv, out):
    g = cv[cv.Base_Model.isin(BASE_MODELS)]
    fig, axes = plt.subplots(1, len(BASE_MODELS), figsize=(16, 3.6), sharex=True, sharey=True)
    lim = (-0.4, 0.8)
    for ax, base in zip(axes, BASE_MODELS):
        p = (g[g.Base_Model == base].pivot(index="Drug", columns="Input",
                                           values="CV_R2_mean").dropna())
        ax.scatter(p["PCA"], p["Raw"], s=10, alpha=0.4, color=C_MODEL, linewidths=0)
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_box_aspect(1)
        better_raw = (p["Raw"] > p["PCA"]).mean() * 100
        ax.set_title(f"{base}\nRaw better for {better_raw:.0f}%", fontsize=10)
        ax.set_xlabel("PCA features: CV R²")
    axes[0].set_ylabel("Raw features: CV R²")
    fig.suptitle("Raw vs PCA features, one point per drug (above the line = Raw better)",
                 y=1.04, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig2_raw_vs_pca.png")


# ── 3. Selected model vs tissue-mean baseline ────────────────────────────────
def fig_vs_tissue(res, out):
    s  = res[res.Selected].set_index("Drug").R2.rename("sel")
    t  = res[res.Model == "TissueMean"].set_index("Drug").R2.rename("tm")
    j  = pd.concat([s, t], axis=1).dropna()
    win = (j.sel > j.tm).mean() * 100
    try:
        p = wilcoxon(j.sel, j.tm).pvalue
        ptxt = f"paired Wilcoxon p = {p:.1e}"
    except Exception:
        ptxt = ""
    print(f"selected beats tissue-mean in {win:.1f}% of {len(j)} units; {ptxt}")

    lim = (-0.4, 0.8)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    beats = j.sel > j.tm
    ax.scatter(j.tm[~beats], j.sel[~beats], s=12, alpha=0.5, color=C_TISSUE,
               linewidths=0, label="tissue-mean better")
    ax.scatter(j.tm[beats], j.sel[beats], s=12, alpha=0.5, color=C_MODEL,
               linewidths=0, label="expression model better")
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Tissue-mean baseline: test R²")
    ax.set_ylabel("Selected expression model: test R²")
    ax.set_title("Does expression add anything beyond tissue?")
    ax.text(0.03, 0.97, f"model better in {win:.0f}% of {len(j)} units\n{ptxt}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig3_model_vs_tissue_baseline.png")


# ── 4. Top drugs, model vs tissue baseline ───────────────────────────────────
def fig_top_drugs(res, out, n=20):
    s = res[res.Selected].set_index("Drug")[["R2", "Model"]].rename(columns={"R2": "sel"})
    t = res[res.Model == "TissueMean"].set_index("Drug").R2.rename("tm")
    j = s.join(t).dropna().sort_values("sel", ascending=False).head(n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 0.3 * n + 1.5))
    y = np.arange(len(j))
    ax.hlines(y, j.tm, j.sel, color="#bbbbbb", lw=1.5, zorder=1)
    ax.scatter(j.tm, y, color=C_TISSUE, s=40, zorder=2, label="tissue-mean baseline")
    ax.scatter(j.sel, y, color=C_MODEL, s=40, zorder=3, label="selected expression model")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{short(d)}  [{m}]" for d, m in zip(j.index, j.Model)], fontsize=8)
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Held-out test R²")
    ax.set_title(f"Top {n} drug–screen units by test R²")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8,
              frameon=False)
    fig.tight_layout()
    save(fig, out, "fig4_top_drugs.png")


# ── 5. Within-tissue performance ─────────────────────────────────────────────
def fig_tissue(strat, out):
    if strat.empty:
        print("skip tissue figure: no stratified_r2.csv rows"); return
    order = strat.groupby("Tissue").R2.median().sort_values(ascending=False).index
    n = strat.groupby("Tissue").Drug.nunique()
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(order) + 1.5))
    sns.boxplot(data=strat, y="Tissue", x="R2", order=order, showfliers=False,
                color=C_MODEL, width=0.6, ax=ax)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlim(max(strat.R2.quantile(0.02), -1.5), min(strat.R2.quantile(0.995), 1.0))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{t}  (n={n[t]})" for t in order])
    ax.set_xlabel("Within-tissue test R² of the selected model (one value per drug)")
    ax.set_ylabel("")
    ax.set_title("Predictive performance within each tissue")
    fig.tight_layout()
    save(fig, out, "fig5_within_tissue_r2.png")


# ── 6. OncoKB permutation null ───────────────────────────────────────────────
def fig_null(outputs, out):
    files = sorted(glob.glob(os.path.join(outputs, "oncokb_null_*.json")))
    if not files:
        print("skip null figure: no oncokb_null_*.json (rerun train.py with the null-saving patch)")
        return
    fig, axes = plt.subplots(1, len(files), figsize=(5.2 * len(files), 3.8), squeeze=False)
    for ax, f in zip(axes[0], files):
        d = json.load(open(f))
        null = np.array(d["null_counts"])
        ax.hist(null, bins=30, color=C_DUMMY, edgecolor="white")
        ax.axvline(d["k_observed"], color=C_TISSUE, lw=2,
                   label=f"observed = {d['k_observed']}")
        ax.set_title(f"{d['label']} biomarkers", fontsize=11)
        ax.set_xlabel("OncoKB genes among chosen genes")
        ax.text(0.03, 0.95, f"null mean {d['null_mean']:.0f}\n"
                            f"fold vs null {d['fold_vs_null']:.2f}×\n"
                            f"empirical p = {d['empirical_p']:.2f}",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
    axes[0][0].set_ylabel("Permutations")
    fig.suptitle("Observed OncoKB overlap vs drug-aware random null", y=1.03,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig6_oncokb_null.png")


# ── data helpers for the biomarker figures ───────────────────────────────────
def load_expression(data_dir):
    path = os.path.join(data_dir, "expr.pkl")
    if not os.path.exists(path):
        print("skip biomarker figures: data/expr.pkl not found"); return None
    expr = pd.read_pickle(path)
    expr.columns = pd.to_numeric(expr.columns, errors="coerce")
    expr = expr.loc[:, expr.columns.notna()]
    expr = expr.loc[:, ~expr.columns.duplicated(keep="first")]
    expr = expr.loc[~expr.index.duplicated(keep="first")]
    expr.columns = expr.columns.astype(int)
    return expr


def load_tissue(data_dir):
    path = os.path.join(data_dir, "Model.csv")
    if not os.path.exists(path):
        return None
    ann = pd.read_csv(path).rename(columns={"COSMICID": "COSMIC_ID",
                                            "OncotreeLineage": "TISSUE"})
    tis = ann.drop_duplicates("COSMIC_ID").set_index("COSMIC_ID").TISSUE
    tis.index = pd.to_numeric(tis.index, errors="coerce")
    tis = tis[tis.index.notna()]
    tis.index = tis.index.astype(int)
    return tis


def drug_frame(drug, expr, tis, data_dir, min_group=5):
    """Cell lines x genes matrix, LN_IC50 vector and tissue groups for one unit.
    Tissues with < min_group cell lines are merged into 'Other'."""
    m = re.match(r"(.*) \[(GDSC[12])\]$", drug)
    if not m:
        return None
    base, screen = m.groups()
    path = os.path.join(data_dir, f"{screen}_dose_response.csv")
    if not os.path.exists(path):
        print(f"skip {drug}: {path} not found"); return None
    dr = pd.read_csv(path)
    dr = dr[dr.DRUG_NAME == base].groupby("COSMIC_ID").LN_IC50.mean().dropna()
    ids = [c for c in dr.index if c in expr.columns]
    X = expr[ids].T.astype(float)
    X = X.loc[:, X.notna().all(axis=0)]
    y = dr.loc[ids].values
    if tis is not None:
        groups = tis.reindex(ids).fillna("Unknown")
    else:
        groups = pd.Series("All", index=ids)
    groups = groups.where(groups.map(groups.value_counts()) >= min_group, "Other")
    return X, y, groups.values


def bh_fdr(p):
    """Benjamini–Hochberg q-values."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def gene_associations(X, y, groups=None):
    """Pearson r between every gene and y, with t-test p-values and BH q-values.
    If `groups` is given, genes and y are demeaned within each group first, so
    r is the correlation *within* tissues (tissue-adjusted)."""
    from scipy.stats import t as student_t
    Xv = X.values.astype(float)
    yv = np.asarray(y, float)
    n = len(yv)
    if groups is None:
        df = n - 2
    else:
        codes, uniq = pd.factorize(np.asarray(groups))
        Xv = Xv - pd.DataFrame(Xv).groupby(codes).transform("mean").values
        yv = yv - pd.Series(yv).groupby(codes).transform("mean").values
        df = n - len(uniq) - 1
    Xc = Xv - Xv.mean(0)
    yc = yv - yv.mean()
    den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    ok = den > 1e-12
    r = np.zeros(Xv.shape[1])
    r[ok] = (Xc[:, ok].T @ yc) / den[ok]
    r = np.clip(r, -1, 1)
    t = r * np.sqrt(df / np.maximum(1 - r ** 2, 1e-12))
    p = 2 * student_t.sf(np.abs(t), df)
    out = pd.DataFrame({"gene": X.columns, "r": r, "p": p})[ok].copy()
    out["q"] = bh_fdr(out.p.values)
    return out


# ── 7. Biomarker panels: volcano (pooled), volcano (tissue-adjusted), top genes ──
def _spread_labels(ax, pts, toward_right):
    """Label (x, y, name) points without overlap: stack the labels at least
    `gap` apart vertically, offset sideways, and join each to its point."""
    if not pts:
        return
    pts = sorted(pts, key=lambda t: -t[1])
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    gap = 0.06 * (y1 - y0)
    dx = (0.10 if toward_right else -0.10) * (x1 - x0)
    prev = None
    for x, y, name in pts:
        ty = y if prev is None else min(y, prev - gap)
        prev = ty
        ax.annotate(name, (x, y), xytext=(x + dx, ty), textcoords="data",
                    fontsize=8, va="center", ha="left" if toward_right else "right",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888",
                                    shrinkA=0, shrinkB=2))


def fig_biomarker_panels(drug, X, y, groups, out, q_thr=0.05, r_thr=0.2,
                         n_label=3, n_bars=12, n_table=20):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    C_WITHIN = PAL[2]
    naive = gene_associations(X, y)
    adj = gene_associations(X, y, groups)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, df, title in [(axes[0], naive, "All cell lines pooled"),
                          (axes[1], adj, "Within tissue (tissue-adjusted)")]:
        ylog = -np.log10(df.q.clip(lower=1e-300))
        hit = (df.q < q_thr) & (df.r.abs() >= r_thr)
        pos, neg = hit & (df.r > 0), hit & (df.r < 0)
        ax.scatter(df.r[~hit], ylog[~hit], s=6, alpha=0.3, color=C_DUMMY, linewidths=0)
        ax.scatter(df.r[pos], ylog[pos], s=10, alpha=0.8, color=C_TISSUE, linewidths=0)
        ax.scatter(df.r[neg], ylog[neg], s=10, alpha=0.8, color=C_MODEL, linewidths=0)
        ax.axhline(-np.log10(q_thr), color="grey", ls="--", lw=0.8)
        for v in (-r_thr, r_thr):
            ax.axvline(v, color="grey", ls=":", lw=0.8)
        ax.margins(x=0.18)
        d = df.assign(ylog=ylog)
        left = d[neg].reindex(d[neg].r.abs().sort_values(ascending=False).index).head(n_label)
        right = d[pos].reindex(d[pos].r.abs().sort_values(ascending=False).index).head(n_label)
        _spread_labels(ax, list(zip(left.r, left.ylog, left.gene)), toward_right=True)
        _spread_labels(ax, list(zip(right.r, right.ylog, right.gene)), toward_right=False)
        ax.set_title(f"{title}\n{int(hit.sum()):,} of {len(df):,} genes: "
                     f"FDR q<{q_thr}, |r|≥{r_thr}", fontsize=10)
        ax.set_xlabel("Pearson r with LN(IC50)")
    axes[0].set_ylabel("−log10 FDR q-value")

    m = naive.merge(adj, on="gene", suffixes=("_pooled", "_within"))
    top = m.reindex(m.r_within.abs().sort_values(ascending=False).index).head(n_bars).iloc[::-1]
    y_pos = np.arange(len(top))
    axes[2].barh(y_pos + 0.2, top.r_pooled, height=0.38, color=C_DUMMY)
    axes[2].barh(y_pos - 0.2, top.r_within, height=0.38, color=C_WITHIN)
    axes[2].set_yticks(y_pos); axes[2].set_yticklabels(top.gene, fontsize=8)
    axes[2].axvline(0, color="black", lw=0.7)
    axes[2].set_xlabel("Pearson r  (← sensitive | resistant →)")
    axes[2].set_title(f"Top {n_bars} genes by within-tissue |r|", fontsize=10)

    handles = [
        Line2D([], [], marker="o", ls="", color=C_TISSUE, label="higher expression = more resistant"),
        Line2D([], [], marker="o", ls="", color=C_MODEL, label="higher expression = more sensitive"),
        Patch(color=C_DUMMY, label="r, all cell lines pooled"),
        Patch(color=C_WITHIN, label="r, within tissue"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Gene–response associations: {short(drug)}  (n = {len(y)} cell lines)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    safe = re.sub(r"[^\w\-]", "_", drug)
    save(fig, os.path.join(out, "biomarker"), f"biomarker_{safe}.png")

    # tables for the README / downstream use
    rows = []
    sig = m[m.q_within < q_thr]
    for direction, sub in [("resistance", sig[sig.r_within > 0]),
                           ("sensitivity", sig[sig.r_within < 0])]:
        sub = sub.reindex(sub.r_within.abs().sort_values(ascending=False).index).head(n_table)
        for rank, row in enumerate(sub.itertuples(index=False), start=1):
            rows.append({"Drug": drug, "Direction": direction, "Rank": rank,
                         "Gene": row.gene, "r_within_tissue": row.r_within,
                         "q_within_tissue": row.q_within,
                         "r_pooled": row.r_pooled, "q_pooled": row.q_pooled})
    counts = {"Drug": drug, "n_cell_lines": len(y), "n_genes_tested": len(naive),
              "hits_pooled": int(((naive.q < q_thr) & (naive.r.abs() >= r_thr)).sum()),
              "hits_within_tissue": int(((adj.q < q_thr) & (adj.r.abs() >= r_thr)).sum())}
    return rows, counts


# ── 8. One concrete biomarker, expression vs response ────────────────────────
def fig_biomarker_example(drug, gene, X, y, groups, out, top_tissues=6):
    if gene not in X.columns:
        print(f"skip biomarker example: {gene} not in expression matrix"); return
    df = pd.DataFrame({"expr": X[gene].values, "ic50": y, "tissue": groups})
    keep = df.tissue.value_counts().head(top_tissues).index
    df["group"] = np.where(df.tissue.isin(keep), df.tissue, "Other")
    r, _ = pearsonr(df.expr, df.ic50)

    fig, ax = plt.subplots(figsize=(6.4, 5))
    sns.scatterplot(data=df, x="expr", y="ic50", hue="group", s=22, alpha=0.8,
                    linewidth=0, ax=ax)
    sns.regplot(data=df, x="expr", y="ic50", scatter=False, color="black",
                line_kws={"lw": 1.2}, ax=ax)
    ax.set_xlabel(f"{gene} expression (RMA)")
    ax.set_ylabel(f"LN(IC50), {short(drug)}")
    ax.set_title(f"{gene} expression vs sensitivity to {short(drug)}")
    ax.text(0.03, 0.03, f"Pearson r = {r:.2f}, n = {len(df)}", transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.legend(title="Tissue", fontsize=8, title_fontsize=9, loc="upper left",
              bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    save(fig, out, "fig7_biomarker_example.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--drug", default=None, help="drug for the biomarker example")
    ap.add_argument("--gene", default=None, help="gene for the biomarker example")
    ap.add_argument("--tables", default="results/tables",
                    help="where to write biomarker_top_genes.csv and biomarker_hit_counts.csv")
    ap.add_argument("--n-biomarker-drugs", type=int, default=6,
                    help="number of top drugs to get gene-association panels")
    a = ap.parse_args()

    res = pd.read_csv(os.path.join(a.outputs, "results_pca.csv"))
    cv  = pd.read_csv(os.path.join(a.outputs, "results_cv.csv"))
    imp = pd.read_csv(os.path.join(a.outputs, "feature_importance.csv"))
    sp  = os.path.join(a.outputs, "stratified_r2.csv")
    strat = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame()

    fig_performance(res, a.out)
    fig_raw_vs_pca(cv, a.out)
    fig_vs_tissue(res, a.out)
    fig_top_drugs(res, a.out)
    fig_tissue(strat, a.out)
    fig_null(a.outputs, a.out)

    expr = load_expression(a.data)
    if expr is None:
        return
    tis = load_tissue(a.data)
    ranked = res[res.Selected].sort_values("R2", ascending=False).Drug.tolist()

    top_rows, count_rows = [], []
    for d in ranked[:a.n_biomarker_drugs]:
        fr = drug_frame(d, expr, tis, a.data)
        if fr is not None:
            rows, counts = fig_biomarker_panels(d, *fr, a.out)
            top_rows += rows
            count_rows.append(counts)
    if top_rows:
        os.makedirs(a.tables, exist_ok=True)
        pd.DataFrame(top_rows).to_csv(os.path.join(a.tables, "biomarker_top_genes.csv"),
                                      index=False)
        pd.DataFrame(count_rows).to_csv(os.path.join(a.tables, "biomarker_hit_counts.csv"),
                                        index=False)
        print("saved", os.path.join(a.tables, "biomarker_top_genes.csv"),
              "and biomarker_hit_counts.csv")

    drug = a.drug or ranked[0]
    pe = imp[(imp.Drug == drug) & (imp.Method == "Pearson")].sort_values("Rank")
    gene = a.gene or (pe.Gene.iloc[0] if not pe.empty else None)
    fr = drug_frame(drug, expr, tis, a.data)
    if fr is not None and gene:
        fig_biomarker_example(drug, gene, *fr, a.out)


if __name__ == "__main__":
    main()