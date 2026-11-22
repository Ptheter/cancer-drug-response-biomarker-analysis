"""
Streamlit app: explore drug-response predictions, biomarkers and honest
performance metrics from the GDSC / PRISM analysis.

Reads the committed tables and figures in ../results, so it needs no raw data.

    streamlit run app/streamlit_app.py
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import app_data as ad

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"

BLUE, ORANGE, GREEN, GREY = "#0173B2", "#DE8F05", "#029E73", "#949494"   # colour-blind safe

st.set_page_config(page_title="Drug response from gene expression", layout="wide")


# ── loading ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(name: str):
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_null_summaries():
    out = []
    for path in sorted(TABLES.glob("oncokb_null_*.json")):
        d = json.loads(path.read_text())
        d.pop("null_counts", None)
        out.append(d)
    return out


def show_image(name: str, caption: str = None):
    path = FIGS / name
    if not path.exists():
        st.info(f"Figure not found: {name}")
        return
    try:                                   # the sizing argument has changed between versions
        st.image(str(path), caption=caption, width="stretch")
    except Exception:
        try:
            st.image(str(path), caption=caption, use_container_width=True)
        except Exception:
            st.image(str(path), caption=caption)


def download(df: pd.DataFrame, filename: str):
    st.download_button("Download table as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")


results = load_csv("results_pca.csv")
cv = load_csv("results_cv.csv")
if results is None or cv is None:
    st.error("results/tables/results_pca.csv and results_cv.csv are required. "
             "Run train.py and copy the tables into results/tables/.")
    st.stop()

strat = load_csv("stratified_r2.csv")
imp = load_csv("feature_importance.csv")
ev = load_csv("external_validation.csv")
gene_sum = load_csv("gene_biomarker_summary.csv")
path_sum = load_csv("pathway_level_summary.csv")
top_tbl = load_csv("biomarker_top_genes.csv")
hit_counts = load_csv("biomarker_hit_counts.csv")


@st.cache_data
def get_units():
    return ad.build_unit_table(results, cv)


units = get_units()
head = ad.headline(units)


# ── pages ────────────────────────────────────────────────────────────────────
def page_overview():
    st.title("Predicting cancer-drug response from gene expression")
    st.write(
        "Machine-learning models trained on GDSC cell-line data predict how sensitive a "
        "cell line is to a drug (LN IC50) from the expression of a drug-specific set of "
        "genes. This app shows the results honestly: every number is shown next to simple "
        "baselines, and the models are tested on an independent dataset (PRISM)."
    )
    st.caption("Research project, not a clinical tool. Not for treatment decisions.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drug-screen units", head["n_units"])
    c2.metric("Median held-out R²", f"{head['median_test_r2']:.3f}")
    c3.metric("Median R², tissue-only baseline", f"{head['median_tissue_r2']:.3f}")
    c4.metric("Units where model beats tissue baseline", f"{head['pct_beats_tissue']:.0f}%")

    st.info(
        f"A typical drug is only weakly predictable (median held-out R² "
        f"{head['median_test_r2']:.2f}). A baseline that knows only the cell line's tissue "
        f"reaches {head['median_tissue_r2']:.2f}, so most of the signal reflects tissue of "
        f"origin and expression adds a modest amount on top."
    )

    if ev is not None:
        s = ad.external_summary(ev)
        if not s.empty:
            top = s.iloc[0]
            st.write(
                f"**External check (PRISM, screen {top['Screen']}, {int(top['Units'])} units):** "
                f"model vs PRISM median Spearman {top['Model vs PRISM']:.2f}, against "
                f"{top['Model vs GDSC']:.2f} for the same held-out cell lines in GDSC and "
                f"{top['GDSC vs PRISM (ceiling)']:.2f} for GDSC-vs-PRISM measurement agreement."
            )

    left, right = st.columns(2)
    with left:
        show_image("fig1_performance_vs_baselines.png")
    with right:
        show_image("fig3_model_vs_tissue_baseline.png")


def page_drug():
    st.header("Drug explorer")
    pathways = sorted(units["Pathway"].dropna().unique())
    chosen = st.multiselect("Filter by pathway (optional)", pathways)
    pool = units[units["Pathway"].isin(chosen)] if chosen else units
    pool = pool.sort_values("Test_R2", ascending=False)
    drug = st.selectbox("Drug–screen unit (type to search)", pool["Drug"].tolist())
    row = units[units["Drug"] == drug].iloc[0]

    st.subheader(drug)
    st.caption(f"Pathway: {row['Pathway']}  |  selected model: {row['Model']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model, test R²", f"{row['Test_R2']:.3f}")
    c2.metric("Selected model, CV R²", f"{row['CV_R2']:.3f}")
    c3.metric("Tissue-only baseline, test R²",
              "n/a" if pd.isna(row["Tissue_R2"]) else f"{row['Tissue_R2']:.3f}")
    c4.metric("Mean-only baseline, test R²",
              "n/a" if pd.isna(row["Dummy_R2"]) else f"{row['Dummy_R2']:.3f}")
    st.write(ad.verdict(row))

    t_models, t_genes, t_tissue, t_prism, t_adj = st.tabs(
        ["Models", "Top genes", "Within tissue", "PRISM validation", "Tissue-adjusted genes"])

    with t_models:
        m = ad.model_table(results, cv, drug)
        metric = st.radio("Metric", ["Test R²", "CV R²"], horizontal=True)
        col = "Test_R2" if metric == "Test R²" else "CV_R2_mean"
        chart = (
            alt.Chart(m).mark_bar().encode(
                x=alt.X(f"{col}:Q", title=metric),
                y=alt.Y("Label:N", sort="-x", title=None),
                color=alt.Color("Kind:N", title=None,
                                scale=alt.Scale(domain=["Gene model", "Baseline"],
                                                range=[BLUE, ORANGE])),
                tooltip=["Model", "Kind",
                         alt.Tooltip("Test_R2:Q", format=".3f"),
                         alt.Tooltip("CV_R2_mean:Q", format=".3f")],
            ).properties(height=28 * len(m) + 30)
        )
        st.altair_chart(chart)
        st.caption("★ = model selected by cross-validated R² on the training split. "
                   "Test R² is computed once, on held-out cell lines. Orange bars are baselines.")
        shown = m[["Model", "Input", "Test_R2", "CV_R2_mean", "CV_R2_std"]].round(3)
        st.dataframe(shown, hide_index=True)

    with t_genes:
        if imp is None:
            st.info("feature_importance.csv not found.")
        else:
            model_df, pearson_df = ad.top_genes(imp, drug, 10)
            a, b = st.columns(2)
            with a:
                st.markdown("**Selected model's top genes**")
                st.dataframe(model_df.round(4), hide_index=True)
            with b:
                st.markdown("**Strongest correlations with LN IC50 (all cell lines pooled)**")
                st.dataframe(pearson_df.round(3), hide_index=True)
            st.caption("Only genes in this drug's curated candidate set were considered. "
                       "Correlations are pooled across tissues, so lineage markers can rank "
                       "highly; see the tissue-adjusted tab. These are associations, not causes.")

    with t_tissue:
        if strat is None:
            st.info("stratified_r2.csv not found.")
        else:
            s = strat[strat["Drug"] == drug].sort_values("R2")
            if s.empty:
                st.info("No tissue had enough held-out cell lines for this drug.")
            else:
                chart = (
                    alt.Chart(s).mark_bar(color=BLUE).encode(
                        x=alt.X("R2:Q", title="Within-tissue test R²"),
                        y=alt.Y("Tissue:N", sort="-x", title=None),
                        tooltip=["Tissue", alt.Tooltip("R2:Q", format=".3f"), "N"],
                    ).properties(height=26 * len(s) + 30)
                )
                st.altair_chart(chart)
                st.caption("R² computed inside each tissue on its held-out cell lines (N in the "
                           "tooltip). Small groups give noisy, downward-biased values.")

    with t_prism:
        if ev is None:
            st.info("external_validation.csv not found.")
        else:
            e = ev[ev["unit"] == drug]
            if e.empty:
                st.info("This drug has no name match in the PRISM screens, "
                        "or too few shared cell lines.")
            else:
                e = e.rename(columns={
                    "screen": "PRISM screen", "n_train": "Train lines", "n_test": "Held-out lines",
                    "rho_gdsc": "Model vs GDSC", "rho_prism": "Model vs PRISM",
                    "rho_tissue_prism": "Tissue baseline vs PRISM",
                    "rho_assay": "GDSC vs PRISM (ceiling)"})
                cols = ["PRISM screen", "Train lines", "Held-out lines", "Model vs GDSC",
                        "Model vs PRISM", "Tissue baseline vs PRISM", "GDSC vs PRISM (ceiling)"]
                st.dataframe(e[cols].round(3), hide_index=True)
                st.caption("Within-drug Spearman correlations on cell lines held out of training. "
                           "Compare 'Model vs PRISM' with the tissue baseline and the ceiling.")

    with t_adj:
        shown = False
        if top_tbl is not None and drug in set(top_tbl["Drug"]):
            shown = True
            show_image(f"biomarker/biomarker_{ad.safe_name(drug)}.png")
            t = top_tbl[top_tbl["Drug"] == drug]
            a, b = st.columns(2)
            for col, direction in [(a, "sensitivity"), (b, "resistance")]:
                with col:
                    st.markdown(f"**Top {direction} genes (within tissue)**")
                    d = t[t["Direction"] == direction].sort_values("Rank").head(10)
                    st.dataframe(d[["Rank", "Gene", "r_within_tissue", "q_within_tissue",
                                    "r_pooled"]].round(4), hide_index=True)
        if not shown:
            st.info("Genome-wide, tissue-adjusted associations were computed only for the "
                    "best-predicted drugs. Pick one of those to see them.")


def page_models():
    st.header("Model comparison")
    st.write("Each model was evaluated on raw standardized genes and on PCA components, "
             "with scaling and PCA fitted inside the cross-validation folds.")
    a, b = st.columns(2)
    with a:
        st.markdown("**Median CV R² across drugs**")
        st.dataframe(ad.raw_vs_pca(cv).round(3))
    with b:
        st.markdown("**How often each model was selected**")
        st.bar_chart(ad.selected_counts(results))
    show_image("fig2_raw_vs_pca.png")

    st.subheader("All drug–screen units")
    f1, f2, f3 = st.columns(3)
    pathways = sorted(units["Pathway"].dropna().unique())
    chosen = f1.multiselect("Pathway", pathways)
    min_r2 = f2.slider("Minimum test R²", -0.5, 0.8, 0.0, 0.05)
    only_beats = f3.checkbox("Only units where the model beats the tissue baseline")
    t = units.copy()
    if chosen:
        t = t[t["Pathway"].isin(chosen)]
    t = t[t["Test_R2"] >= min_r2]
    if only_beats:
        t = t[t["Beats_tissue"]]
    t = t.sort_values("Test_R2", ascending=False)
    cols = ["Drug", "Pathway", "Model", "Test_R2", "CV_R2", "Tissue_R2", "Delta_vs_tissue"]
    st.caption(f"{len(t)} of {len(units)} units")
    st.dataframe(t[cols].round(3), hide_index=True)
    download(t[cols].round(4), "drug_response_units.csv")
    show_image("fig4_top_drugs.png")


def page_biomarkers():
    st.header("Biomarkers")
    st.warning(
        "Gene rankings are associations in cell lines, not causal effects, and only genes in "
        "each drug's curated candidate set were considered. Genes that appear for many drugs "
        "mostly reflect how many candidate lists contain them."
    )
    t1, t2, t3, t4 = st.tabs(["Frequent genes", "Pathways", "OncoKB overlap", "Examples"])

    with t1:
        if gene_sum is None:
            st.info("gene_biomarker_summary.csv not found.")
        else:
            st.dataframe(gene_sum.head(30)[["Gene", "Num_Drugs"]], hide_index=True)
            st.caption("Number of drug–screen units in which the gene was among the top 20 "
                       "features. 'Mean importance' is omitted because it mixes methods on "
                       "different scales.")
    with t2:
        if path_sum is None:
            st.info("pathway_level_summary.csv not found.")
        else:
            st.dataframe(path_sum[["Pathway", "Num_Genes", "Total_Drug_Associations"]],
                         hide_index=True)
            st.caption("Totals scale with the number of drugs mapped to each pathway.")
    with t3:
        st.write("Candidate genes are cancer-curated by design, so overlap with OncoKB is "
                 "compared with a drug-aware random null, not with all genes.")
        for d in load_null_summaries():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{d['label']}: observed overlap", d["k_observed"])
            c2.metric("Null mean", f"{d['null_mean']:.0f}")
            c3.metric("Fold vs null", f"{d['fold_vs_null']:.2f}×")
            c4.metric("Empirical p", f"{d['empirical_p']:.2f}")
        show_image("fig6_oncokb_null.png")
        st.caption("The identified biomarkers are not enriched for known cancer genes beyond "
                   "what the candidate lists already contain.")
    with t4:
        if top_tbl is None or hit_counts is None:
            st.info("biomarker_top_genes.csv / biomarker_hit_counts.csv not found.")
        else:
            drugs = sorted(top_tbl["Drug"].unique())
            drug = st.selectbox("Drug", drugs, key="bm_drug")
            h = hit_counts[hit_counts["Drug"] == drug]
            if not h.empty:
                r = h.iloc[0]
                st.write(f"Genes passing FDR q < 0.05 and |r| ≥ 0.2: "
                         f"**{int(r['hits_pooled']):,}** pooled across tissues, "
                         f"**{int(r['hits_within_tissue']):,}** within tissue.")
            show_image(f"biomarker/biomarker_{ad.safe_name(drug)}.png")
        show_image("fig7_biomarker_example.png")


def page_tissue():
    st.header("Performance within each tissue")
    st.write("Models are trained across all tissues, then scored within each tissue on "
             "held-out cell lines. Within-tissue performance is near zero or negative for "
             "most tissues, so predictions may not transfer within a cancer type.")
    show_image("fig5_within_tissue_r2.png")
    if strat is not None:
        st.dataframe(ad.tissue_summary(strat).round(3), hide_index=True)
        st.caption("Median within-tissue R² across drugs. Tissues with few drugs or "
                   "held-out samples are not interpretable.")


def page_external():
    st.header("External validation on PRISM")
    with st.expander("How this was done", expanded=True):
        st.markdown(
            "- Drugs were matched by name between GDSC and the PRISM Repurposing secondary screen.\n"
            "- For each matched drug, **all cell lines measured in both datasets were held out**; "
            "the model type selected in the main run was refit on the remaining GDSC lines.\n"
            "- Predictions for the held-out lines were compared with PRISM AUC using "
            "**within-drug Spearman correlation** (response scales differ, so R² is not comparable).\n"
            "- The PRISM screens were analyzed separately. Expression is GDSC microarray throughout, "
            "so this tests transfer of the drug-response measurement, not of the expression platform."
        )
    if ev is None:
        st.info("external_validation.csv not found.")
        return
    summary = ad.external_summary(ev)
    st.dataframe(summary.round(3), hide_index=True)
    st.caption("Median across units. 'GDSC vs PRISM (ceiling)' is how well the two assays "
               "agree on the same cell lines, which bounds any model trained on one and tested "
               "on the other. Units are not independent (same cell lines for every drug), so "
               "any p-values are descriptive.")
    show_image("fig8_external_validation.png")

    st.subheader("Per-unit results")
    c1, c2 = st.columns(2)
    screens = sorted(ev["screen"].unique())
    screen = c1.selectbox("PRISM screen", screens)
    min_assay = c2.slider("Minimum assay agreement (GDSC vs PRISM)", -0.2, 0.6, -0.2, 0.05)
    t = ev[(ev["screen"] == screen) & (ev["rho_assay"].fillna(-1) >= min_assay)]
    cols = ["unit", "model", "n_train", "n_test", "rho_gdsc", "rho_prism",
            "rho_tissue_prism", "rho_assay"]
    t = t.sort_values("rho_prism", ascending=False)[cols]
    st.caption(f"{len(t)} units")
    st.dataframe(t.round(3), hide_index=True)
    download(t.round(4), "prism_external_validation.csv")


def page_about():
    st.header("About")
    st.markdown(
        """
**Data.** GDSC1 and GDSC2 drug-response data (analyzed separately), GDSC basal gene expression,
tissue annotations, the OncoKB cancer gene list, and the PRISM Repurposing secondary screen
(external validation). Raw data are not included in the repository.

**Method in brief.** For each drug, models see only a curated candidate gene set (drug targets,
pathway genes, MSigDB Hallmark sets). Five model types are evaluated on raw and PCA features
with 5-fold cross-validation on the training split; the best model is chosen by CV R² and
scored once on held-out cell lines. A tissue-mean baseline shows how much is lineage alone.

**Limitations**
- Performance is modest for most drugs, and most of it is explained by tissue of origin.
- Each estimate comes from a single train/test split.
- Candidate gene sets are hand-curated, which limits discovery and builds cancer-relevant
  genes into the biomarker lists.
- Biomarker rankings are associations in cell lines, not causal or clinical evidence.
- Within-tissue performance is poor.
- External validation is limited by how well the two assays agree with each other.

**Not for clinical use.** This is a research and portfolio project.
        """
    )


PAGES = {
    "Overview": page_overview,
    "Drug explorer": page_drug,
    "Model comparison": page_models,
    "Biomarkers": page_biomarkers,
    "Tissue": page_tissue,
    "External validation": page_external,
    "About": page_about,
}

choice = st.sidebar.radio("Page", list(PAGES))
PAGES[choice]()
st.sidebar.caption("Research project. Not for treatment decisions.")