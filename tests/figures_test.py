"""
Tests for the statistics in make_figures.py and the matching helpers in
external_validation.py. Synthetic data only.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import external_validation as ev
from make_figures import bh_fdr, gene_associations


# ── Benjamini–Hochberg ───────────────────────────────────────────────────────
def test_bh_fdr_known_values():
    # sorted p: .005 .01 .03 .04 -> raw q: .02 .02 .04 .04 (already monotone)
    q = bh_fdr([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(q, [0.02, 0.04, 0.04, 0.02])


def test_bh_fdr_is_monotone_in_p_and_capped_at_one():
    p = np.random.default_rng(0).uniform(size=200)
    q = bh_fdr(p)
    assert q.max() <= 1.0
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)        # q never decreases as p increases
    assert np.all(q >= p - 1e-12)                      # q >= p


# ── gene–response associations ───────────────────────────────────────────────
def _toy(n=400, seed=1):
    rng = np.random.default_rng(seed)
    tissues = rng.choice(["Lung", "Breast", "Skin", "Bowel"], n)
    tissue_effect = pd.Series(tissues).map({"Lung": -1.5, "Breast": 1.0, "Skin": 2.0, "Bowel": 0.0}).values
    driver = rng.normal(size=n)                                   # within-tissue driver
    X = pd.DataFrame({
        "DRIVER": -driver + rng.normal(size=n) * 0.5,
        "LINEAGE_MARKER": tissue_effect + rng.normal(size=n) * 0.6,
        "NOISE": rng.normal(size=n),
    })
    y = tissue_effect + driver + rng.normal(size=n) * 0.5
    return X, y, tissues


def test_pooled_correlation_matches_scipy():
    X, y, _ = _toy()
    out = gene_associations(X, y).set_index("gene")
    for g in X.columns:
        r, p = pearsonr(X[g], y)
        assert np.isclose(out.loc[g, "r"], r)
        assert np.isclose(out.loc[g, "p"], p)


def test_tissue_adjustment_removes_a_pure_lineage_marker_but_keeps_the_driver():
    X, y, tissues = _toy()
    pooled = gene_associations(X, y).set_index("gene")
    adjusted = gene_associations(X, y, tissues).set_index("gene")
    # the lineage marker looks highly significant when tissues are pooled ...
    assert pooled.loc["LINEAGE_MARKER", "q"] < 1e-10
    # ... but carries no within-tissue signal
    assert adjusted.loc["LINEAGE_MARKER", "q"] > 0.05
    # the real driver stays strongly significant either way
    assert adjusted.loc["DRIVER", "q"] < 1e-10
    assert adjusted.loc["DRIVER", "r"] < 0                       # higher expression = more sensitive


def test_adjusted_correlation_equals_partial_correlation_by_regression():
    X, y, tissues = _toy()
    dummies = pd.get_dummies(tissues).values.astype(float)
    resid = lambda v: v - dummies @ np.linalg.lstsq(dummies, v, rcond=None)[0]
    expected, _ = pearsonr(resid(X["DRIVER"].values), resid(y))
    got = gene_associations(X, y, tissues).set_index("gene").loc["DRIVER", "r"]
    assert np.isclose(got, expected)


# ── PRISM matching helpers ───────────────────────────────────────────────────
def test_norm_ignores_case_hyphens_and_parenthesised_tags():
    assert ev.norm("Nutlin-3a (-)") == "nutlin3a"
    assert ev.norm("MK-1775") == ev.norm("mk1775") == "mk1775"


def test_excluded_stereoisomer_would_otherwise_collide_with_the_drug():
    # norm() drops the "(S)" tag, so without the exclusion list this PRISM entry
    # would be merged with plain crizotinib; this test documents why it exists
    assert ev.norm("crizotinib-(S)") == ev.norm("crizotinib")
    assert "crizotinib-(S)" in ev.EXCLUDE_PRISM_NAMES


def test_spearman_helper_edge_cases():
    x = np.arange(50.0)
    assert np.isclose(ev.rho(x, 2 * x + 1), 1.0)
    assert np.isclose(ev.rho(x, -x), -1.0)
    assert np.isnan(ev.rho(x, np.ones(50)))              # constant -> NaN, not an error
    assert np.isnan(ev.rho(x[:5], x[:5]))                # too few points