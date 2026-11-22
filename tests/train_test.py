"""
Tests for the core pipeline (train.py).

Each test pins down something that was a real bug or design decision:
  * biomarkers are always labelled with genes (never principal components)
  * the best model is chosen by CV R², not test R², and never a baseline
  * GDSC1 and GDSC2 are kept as separate analysis units
  * the OncoKB enrichment uses the candidate pool and a drug-aware null
All data is tiny and synthetic; no GDSC/PRISM files are needed.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

import train
from reference_data import FALLBACK_GENES, PATHWAY_TO_GENES

P53 = PATHWAY_TO_GENES["p53 pathway"]


# ── gene selection ───────────────────────────────────────────────────────────
def test_select_genes_is_sorted_deterministic_and_within_available():
    avail = set(P53) | {"NOT_A_CANDIDATE"}
    g1, fallback1 = train.select_genes("p53 pathway", "MDM2", avail, {}, {}, train.CFG)
    g2, _ = train.select_genes("p53 pathway", "MDM2", avail, {}, {}, train.CFG)
    assert not fallback1
    assert g1 == g2 == sorted(g1)
    assert set(g1) <= avail
    assert "NOT_A_CANDIDATE" not in g1
    assert "MDM2" in g1


def test_select_genes_falls_back_when_too_few_candidates():
    avail = {"TP53", "MYC", "KRAS"}            # < min_avail_genes candidates present
    genes, used_fallback = train.select_genes("p53 pathway", "", avail, {}, {}, train.CFG)
    assert used_fallback
    assert set(genes) == {"TP53", "MYC", "KRAS"}
    assert set(genes) <= set(FALLBACK_GENES)


# ── GDSC1 / GDSC2 stay separate ──────────────────────────────────────────────
def test_build_drug_response_keeps_screens_separate():
    cols = dict(PATHWAY_NAME="P", PUTATIVE_TARGET="T")
    g1 = pd.DataFrame({"COSMIC_ID": [1, 1, 2], "DRUG_NAME": "A",
                       "LN_IC50": [1.0, 3.0, 5.0], **cols})
    g2 = pd.DataFrame({"COSMIC_ID": [1], "DRUG_NAME": "A", "LN_IC50": [10.0], **cols})
    dr = train.build_drug_response(g1, g2)

    assert set(dr.DRUG_NAME) == {"A [GDSC1]", "A [GDSC2]"}
    v = dr.set_index(["COSMIC_ID", "DRUG_NAME"]).LN_IC50
    assert np.isclose(v[(1, "A [GDSC1]")], 2.0)      # replicates inside one screen are averaged
    assert np.isclose(v[(1, "A [GDSC2]")], 10.0)     # but never across screens
    assert set(dr.DRUG_BASE) == {"A"}


# ── pipeline building blocks ─────────────────────────────────────────────────
def test_pca_n_components_is_bounded():
    cfg = train.CFG
    assert train.pca_n_components(1000, 500, cfg) == cfg["pca_max"]       # capped
    assert train.pca_n_components(10, 100, cfg) == 10                      # <= n_features
    assert train.pca_n_components(100, 20, cfg) == 15                      # < CV-fold size


def test_build_pipeline_has_pca_only_when_requested():
    est = train.build_estimators(train.CFG)["Linear"]
    assert [n for n, _ in train.build_pipeline(est, False, 5).steps] == ["scaler", "model"]
    est = train.build_estimators(train.CFG)["Linear"]
    assert [n for n, _ in train.build_pipeline(est, True, 5).steps] == ["scaler", "pca", "model"]


def test_every_model_runs_under_both_representations():
    labels = [spec[0] for spec in train.build_model_specs(train.CFG)]
    assert labels[0] == "Dummy"
    for base in train.BASE_MODELS:
        assert f"{base}-Raw" in labels and f"{base}-PCA" in labels
    assert len(labels) == 11


def test_tissue_mean_regressor():
    X = np.array([["Lung"], ["Lung"], ["Skin"]])
    m = train.TissueMeanRegressor().fit(X, np.array([1.0, 3.0, 10.0]))
    pred = m.predict(np.array([["Lung"], ["Skin"], ["Bone"]]))
    assert np.allclose(pred, [2.0, 10.0, 14.0 / 3])                        # unseen tissue -> global mean


# ── OncoKB enrichment ────────────────────────────────────────────────────────
def test_enrichment_matches_hand_computed_hypergeometric():
    # universe of 10 genes, 4 known; pick 3 biomarkers, 2 of them known.
    # P(X >= 2) = (C(4,2)*C(6,1) + C(4,3)*C(6,0)) / C(10,3) = 40/120 = 1/3
    universe = [f"G{i}" for i in range(10)]
    known = ["G0", "G1", "G2", "G3"]
    res = train.compute_oncokb_enrichment(universe, ["G0", "G1", "G9"], known)
    assert (res["N_total_genes"], res["K_oncokb_genes"]) == (10, 4)
    assert (res["n_biomarkers"], res["k_overlap"]) == (3, 2)
    assert np.isclose(res["expected_overlap"], 1.2)
    assert np.isclose(res["enrichment_fold"], 2 / 1.2)
    assert np.isclose(res["p_value"], 1 / 3)


def test_enrichment_ignores_genes_outside_the_universe():
    res = train.compute_oncokb_enrichment(["A", "B", "C"], ["A", "OUTSIDE"], ["A"])
    assert res["n_biomarkers"] == 1


def test_permutation_null_is_exactly_observed_when_all_pool_genes_are_picked():
    pools = {"D": ["A", "B", "C", "D"]}
    bm = pd.DataFrame({"Drug": "D", "Gene": ["A", "B", "C", "D"]})
    null = train.oncokb_permutation_null(pools, bm, ["A", "B"], n_perm=40)
    assert null["k_observed"] == 2
    assert set(null["null_counts"]) == {2}
    assert np.isclose(null["fold_vs_null"], 1.0)
    assert np.isclose(null["empirical_p"], 1.0)


def test_permutation_null_is_reproducible_and_json_serialisable():
    pools = {"D1": [f"G{i}" for i in range(30)], "D2": [f"G{i}" for i in range(10, 50)]}
    bm = pd.DataFrame({"Drug": ["D1"] * 5 + ["D2"] * 5,
                       "Gene": [f"G{i}" for i in range(5)] + [f"G{i}" for i in range(20, 25)]})
    a = train.oncokb_permutation_null(pools, bm, [f"G{i}" for i in range(0, 50, 3)], n_perm=60, seed=1)
    b = train.oncokb_permutation_null(pools, bm, [f"G{i}" for i in range(0, 50, 3)], n_perm=60, seed=1)
    assert a["null_counts"] == b["null_counts"]
    json.dumps(a)


# ── end-to-end on a tiny synthetic drug ──────────────────────────────────────
@lru_cache(maxsize=1)
def _synthetic_run():
    """One drug whose response = tissue effect - 1.2 * MDM2 expression + noise."""
    rng = np.random.default_rng(0)
    n = 160
    cells = list(range(100, 100 + n))
    genes = P53 + [f"N{i}" for i in range(20)]
    X = rng.normal(size=(len(genes), n))
    tissues = rng.choice(["Lung", "Breast", "Skin"], n)
    tissue_effect = pd.Series(tissues).map({"Lung": -1.0, "Breast": 0.5, "Skin": 1.5}).values
    y = tissue_effect - 1.2 * X[genes.index("MDM2")] + rng.normal(size=n) * 0.5
    dr = pd.DataFrame({"COSMIC_ID": cells, "DRUG_NAME": "DrugX [GDSC1]", "LN_IC50": y})
    cell_tissue = pd.Series(tissues, index=cells)

    cfg = dict(train.CFG)
    cfg.update(min_samples=100, n_splits=3, alpha_log_range=(-2, 1, 8), l1_ratios=[0.5, 1.0],
               max_iter=2000, perm_n_repeats=3, pi_n_estimators=20, min_tissue_samples=5)
    out = train.run_drug("DrugX [GDSC1]", dr, X, genes, cells,
                         {"DrugX [GDSC1]": "p53 pathway"}, {"DrugX [GDSC1]": "MDM2"},
                         {}, {}, cell_tissue, cfg)
    return out, set(genes)


def test_run_drug_returns_all_models_and_one_selected_non_baseline():
    (results, cv, strat, imp, pi, payload), _ = _synthetic_run()
    res = pd.DataFrame(results)
    assert len(res) == 12                                   # Dummy + 10 gene models + TissueMean
    assert res.Selected.sum() == 1
    assert res[res.Selected].Base_Model.iloc[0] not in train.BASELINES


def test_selected_model_is_the_cv_winner_not_the_test_winner():
    (results, cv, *_), _ = _synthetic_run()
    cvdf = pd.DataFrame(cv)
    cvdf = cvdf[~cvdf.Base_Model.isin(train.BASELINES)]
    best_by_cv = cvdf.loc[cvdf.CV_R2_mean.idxmax(), "Model"]
    selected = pd.DataFrame(results).query("Selected").Model.iloc[0]
    assert selected == best_by_cv


def test_model_beats_both_baselines_on_planted_signal():
    (results, *_), _ = _synthetic_run()
    res = pd.DataFrame(results).set_index("Model")
    selected_r2 = res[res.Selected].R2.iloc[0]
    assert selected_r2 > 0.3
    assert selected_r2 > res.loc["TissueMean", "R2"] > res.loc["Dummy", "R2"]


def test_importance_rows_are_always_gene_names_and_find_the_planted_gene():
    (_, _, _, imp, _, _), genes = _synthetic_run()
    imp = pd.DataFrame(imp)
    assert set(imp.Gene) <= genes                            # never "PC1", "PC2", ...
    pearson = imp[imp.Method == "Pearson"].sort_values("Rank")
    assert "MDM2" in pearson.Gene.head(3).tolist()
    assert pearson[pearson.Gene == "MDM2"].r.iloc[0] < 0      # higher MDM2 -> more sensitive


def test_run_drug_output_is_cacheable_as_json():
    (out, _) = _synthetic_run()
    json.dumps(out)