"""Biomarker figures: ROC curves, AUC comparisons, panel performance and cross-material transfer."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .biomarkers import (
    PanelEvaluation,
    _labels,
    _safe_auc,
    cross_sample_type_transfer,
    evaluate_panel,
    nested_cv,
    out_of_fold_scores,
    panel_size_curve,
    permutation_test,
    rank_panel,
    select_candidates,
    signature_score,
    univariate_performance,
)
from .config import Config
from .de import DEResult
from .plotting import (
    FigureBook,
    condition_palette,
    discrete_palette,
    empty_figure,
    legend_from_map,
)
from .utils import log, sanitize, significance_stars, zscore_rows

SECTION = "biomarker"


def auc_ranking(performance: pd.DataFrame, cfg: Config, book: FigureBook, n: int = 25) -> None:
    table = performance.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, max(5.0, 0.32 * len(table) + 1.6)))
    colors = ["#c0392b" if d == "up" else "#2874a6" for d in table["direction"]]
    ax.barh(range(len(table)), table["auc"], color=colors, height=0.7)
    errors = np.vstack([
        (table["auc"] - table["auc_ci_low"]).fillna(0).clip(lower=0),
        (table["auc_ci_high"] - table["auc"]).fillna(0).clip(lower=0),
    ])
    ax.errorbar(table["auc"], range(len(table)), xerr=errors, fmt="none", ecolor="#2c3e50",
                elinewidth=1.0, capsize=2.5)
    ax.axvline(0.5, color="#7f8c8d", ls="--", lw=1.0, label="no discrimination")
    ax.axvline(cfg.biomarkers.min_auc, color="#27ae60", ls=":", lw=1.4,
               label=f"target AUC = {cfg.biomarkers.min_auc}")
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["gene_name"], fontsize=8)
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("AUC (all samples, with bootstrap 95% CI)")
    ax.set_title(f"Top {len(table)} single-gene biomarkers")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#c0392b", "#2874a6"]]
    ax.legend(handles + ax.get_legend_handles_labels()[0],
              [f"up in {cfg.input.case_label}", f"down in {cfg.input.case_label}",
               "no discrimination", f"target AUC = {cfg.biomarkers.min_auc}"], fontsize=8, loc="lower right")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "auc_ranking", "Single-gene AUC ranking",
              "Discriminative power of each candidate on its own. Bootstrap intervals that cross 0.5 mean the "
              "marker is not reliable in this cohort, whatever its differential-expression p-value.")


def roc_top_genes(expr: pd.DataFrame, meta: pd.DataFrame, performance: pd.DataFrame, cfg: Config,
                  book: FigureBook, n: int = 8) -> None:
    y = _labels(meta, cfg).to_numpy()
    genes = performance.head(n).index
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    palette = discrete_palette([str(g) for g in genes], "tab10")
    for gene in genes:
        values = expr.loc[gene, meta.index].to_numpy(dtype=float)
        if performance.loc[gene, "direction"] == "down":
            values = -values
        fpr, tpr, _ = roc_curve(y, values)
        ax.plot(fpr, tpr, lw=1.9, color=palette[str(gene)],
                label=f"{performance.loc[gene, 'gene_name']} (AUC {performance.loc[gene, 'auc']:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#7f8c8d", lw=1.0)
    ax.set_xlabel("1 - specificity (false positive rate)")
    ax.set_ylabel("Sensitivity (true positive rate)")
    ax.set_title("ROC curves of the leading single-gene markers")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "roc_single_genes", "Single-gene ROC curves",
              "Full sensitivity/specificity trade-off for the best individual genes, which a single AUC number "
              "hides. A curve that rises steeply at the left is what a rule-in test needs.")


def auc_by_sample_type(performance: pd.DataFrame, cfg: Config, book: FigureBook, n: int = 30) -> None:
    columns = [c for c in performance.columns if c.startswith("auc_")
               and c not in {"auc_ci_low", "auc_ci_high", "auc_min_across_types", "auc_range_across_types"}]
    if not columns:
        return
    table = performance.head(n)
    matrix = table[columns].copy()
    matrix.columns = [c.replace("auc_", "") for c in columns]
    matrix.index = table["gene_name"]
    fig, axes = plt.subplots(1, 2, figsize=(6.0 + 1.3 * matrix.shape[1] + 5.0,
                                            max(6.0, 0.28 * len(matrix) + 2.0)),
                             gridspec_kw={"width_ratios": [1.5, 1.0]})
    sns.heatmap(matrix, ax=axes[0], cmap="RdYlGn", vmin=0.4, vmax=1.0, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "AUC within sample type"})
    axes[0].set_title("Marker performance per sample type")
    axes[0].set_ylabel("")
    plt.setp(axes[0].get_yticklabels(), fontsize=7.5, rotation=0)
    plt.setp(axes[0].get_xticklabels(), rotation=25, ha="right")

    consistency = matrix.max(axis=1) - matrix.min(axis=1)
    order = consistency.sort_values().index
    axes[1].barh(range(len(order)), consistency.loc[order], color="#8e44ad", height=0.7)
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels(order, fontsize=7.5)
    axes[1].set_xlabel("AUC spread across sample types (max - min)")
    axes[1].set_title("Transferability between materials")
    axes[1].axvline(0.1, color="#27ae60", ls="--", lw=1.2, label="spread < 0.10")
    axes[1].legend(fontsize=8)
    sns.despine(ax=axes[1])
    fig.tight_layout()
    book.save(fig, SECTION, "auc_by_sample_type", "AUC per sample type",
              "The central comparison: how each candidate performs in every sample type. A small spread "
              "(right panel) means one assay and one cut-off can be used across materials; a large spread "
              "means the marker must be validated separately per material.")


def lfc_vs_auc(performance: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    size = 30 + 120 * performance["youden_j"].clip(lower=0)
    scatter = ax.scatter(performance["log2FC"], performance["auc"],
                         c=-np.log10(performance["mannwhitney_padj"].clip(lower=1e-300)),
                         s=size, cmap="viridis", alpha=0.85, edgecolor="white", lw=0.5)
    ax.axhline(0.5, color="#7f8c8d", ls="--", lw=1.0)
    ax.axhline(cfg.biomarkers.min_auc, color="#27ae60", ls=":", lw=1.4)
    ax.axvline(0, color="#7f8c8d", lw=0.8)
    fig.colorbar(scatter, ax=ax, label="-log10 adjusted p-value")
    top = performance.nlargest(12, "auc")
    for gene, row in top.iterrows():
        ax.annotate(row["gene_name"], (row["log2FC"], row["auc"]), fontsize=7.5,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("log2 fold change (cancer / control)")
    ax.set_ylabel("AUC")
    ax.set_title("Effect size versus diagnostic performance")
    ax.text(0.02, 0.02, "point size = Youden's J", transform=ax.transAxes, fontsize=8, style="italic")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "lfc_vs_auc", "Effect size vs classification performance",
              "Large fold changes do not automatically make good markers: what matters for diagnostics is "
              "separation relative to within-group variability, which is what the AUC captures.")


def biomarker_boxplots(expr: pd.DataFrame, meta: pd.DataFrame, performance: pd.DataFrame,
                       cfg: Config, book: FigureBook, n: int = 8) -> None:
    genes = performance.head(n).index
    cond_col = cfg.input.condition_col
    stype_col = cfg.input.sample_type_col
    use_type = bool(stype_col and stype_col in meta.columns and meta[stype_col].nunique() > 1)
    palette = condition_palette(cfg)
    order = [cfg.input.control_label, cfg.input.case_label]
    ncols = 4
    nrows = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.5 * nrows), squeeze=False)
    for ax, gene in zip(axes.ravel(), genes):
        frame = pd.DataFrame({
            "expression": expr.loc[gene, meta.index].values,
            cond_col: meta[cond_col].astype(str).values,
        }, index=meta.index)
        if use_type:
            frame[stype_col] = meta[stype_col].astype(str).values
            sns.violinplot(data=frame, x=stype_col, y="expression", hue=cond_col, hue_order=order,
                           palette=palette, ax=ax, split=len(order) == 2, inner="quart",
                           linewidth=0.8, cut=0, legend=False)
            sns.stripplot(data=frame, x=stype_col, y="expression", hue=cond_col, hue_order=order,
                          dodge=True, ax=ax, size=2.8, color="#2c3e50", alpha=0.55, legend=False)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7.5)
        else:
            sns.violinplot(data=frame, x=cond_col, y="expression", order=order, hue=cond_col,
                           hue_order=order, palette=palette, ax=ax, inner="quart", linewidth=0.8,
                           cut=0, legend=False)
            sns.stripplot(data=frame, x=cond_col, y="expression", order=order, ax=ax, size=3.2,
                          color="#2c3e50", alpha=0.6)
        row = performance.loc[gene]
        case = frame.loc[frame[cond_col] == cfg.input.case_label, "expression"]
        control = frame.loc[frame[cond_col] == cfg.input.control_label, "expression"]
        pvalue = stats.mannwhitneyu(case, control, alternative="two-sided").pvalue
        ax.set_title(f"{row['gene_name']}  ({significance_stars(pvalue)})\n"
                     f"AUC {row['auc']:.2f} | sens {row['sensitivity']:.2f} | spec {row['specificity']:.2f}",
                     fontsize=8.5)
        ax.set_xlabel("")
        ax.set_ylabel("log2 CPM")
    for ax in axes.ravel()[len(genes):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[l]) for l in order]
    fig.legend(handles, order, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Expression of the leading biomarker candidates", fontweight="bold", y=1.0)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "biomarker_boxplots", "Biomarker expression distributions",
              "Distributions per condition and sample type with sensitivity/specificity at the optimal "
              "cut-off. Look for separation that is present in every material, not only on average.")


def panel_roc(evaluations: List[PanelEvaluation], cfg: Config, book: FigureBook) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    palette = discrete_palette([e.model for e in evaluations], "Dark2")
    best = max(evaluations, key=lambda e: e.mean_auc if np.isfinite(e.mean_auc) else -1)
    for evaluation in evaluations:
        if evaluation.tprs.size == 0:
            continue
        mean_tpr = evaluation.tprs.mean(axis=0)
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
        axes[0].plot(evaluation.mean_fpr, mean_tpr, lw=2.2, color=palette[evaluation.model],
                     label=f"{evaluation.label}: AUC {evaluation.mean_auc:.3f} ± {evaluation.sd_auc:.3f}")
        if evaluation.model == best.model:
            sd = evaluation.tprs.std(axis=0)
            axes[0].fill_between(evaluation.mean_fpr, np.clip(mean_tpr - sd, 0, 1),
                                 np.clip(mean_tpr + sd, 0, 1), color=palette[evaluation.model], alpha=0.18)
    axes[0].plot([0, 1], [0, 1], ls="--", color="#7f8c8d", lw=1.0)
    axes[0].set_xlabel("1 - specificity")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_title(f"Cross-validated ROC of the {len(best.genes)}-gene panel")
    axes[0].legend(fontsize=8.5, loc="lower right")

    frame = pd.DataFrame({e.label: pd.Series(e.auc_folds) for e in evaluations})
    melted = frame.melt(var_name="model", value_name="auc").dropna()
    sns.boxplot(data=melted, x="model", y="auc", hue="model", ax=axes[1], width=0.55,
                palette=[palette[e.model] for e in evaluations], legend=False, showfliers=False)
    sns.stripplot(data=melted, x="model", y="auc", ax=axes[1], size=2.6, color="#2c3e50", alpha=0.45)
    axes[1].axhline(0.5, color="#c0392b", ls="--", lw=1.0)
    axes[1].set_ylabel("AUC per cross-validation fold")
    axes[1].set_xlabel("")
    axes[1].set_title("Fold-to-fold stability")
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "panel_roc", "Multivariate panel ROC",
              f"Repeated {cfg.biomarkers.cv_folds}-fold cross-validation of the combined gene panel for each "
              "classifier. The spread on the right shows how much the estimate depends on which samples land "
              "in the test fold - wide boxes mean the cohort is too small for a firm claim.")


def panel_pr_and_confusion(evaluations: List[PanelEvaluation], cfg: Config, book: FigureBook) -> None:
    best = max(evaluations, key=lambda e: e.mean_auc if np.isfinite(e.mean_auc) else -1)
    palette = discrete_palette([e.model for e in evaluations], "Dark2")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    prevalence = float(np.mean(best.y_true))
    for evaluation in evaluations:
        precision, recall, _ = precision_recall_curve(evaluation.y_true, evaluation.y_score)
        axes[0].plot(recall, precision, lw=2.0, color=palette[evaluation.model],
                     label=f"{evaluation.label}: AP {evaluation.metrics['average_precision']:.3f}")
    axes[0].axhline(prevalence, ls="--", color="#7f8c8d", lw=1.0, label=f"prevalence {prevalence:.2f}")
    axes[0].set_xlabel("Recall (sensitivity)")
    axes[0].set_ylabel("Precision (PPV)")
    axes[0].set_title("Precision-recall curves")
    axes[0].legend(fontsize=8, loc="lower left")

    threshold = best.metrics["youden_threshold"]
    predicted = (best.y_score >= threshold).astype(int)
    matrix = confusion_matrix(best.y_true, predicted)
    labels = [cfg.input.control_label, cfg.input.case_label]
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[1], cbar=False,
                xticklabels=labels, yticklabels=labels, linewidths=1.0, linecolor="white",
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"Confusion matrix at Youden's cut-off\n({best.label}, pooled CV predictions)")

    try:
        fraction, mean_predicted = calibration_curve(best.y_true, np.clip(best.y_score, 0, 1),
                                                     n_bins=min(8, max(3, len(best.y_true) // 10)))
        axes[2].plot(mean_predicted, fraction, "o-", color="#c0392b", lw=2.0, label=best.label)
    except Exception:
        axes[2].text(0.5, 0.5, "calibration not estimable", ha="center", va="center")
    axes[2].plot([0, 1], [0, 1], ls="--", color="#7f8c8d", lw=1.0, label="perfect calibration")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed fraction of cancer samples")
    axes[2].set_title("Calibration")
    axes[2].legend(fontsize=8)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "panel_pr_confusion", "Precision-recall, confusion matrix and calibration",
              "Precision-recall is the honest view when the classes are unbalanced. The confusion matrix "
              "translates the model into counts of missed cancers and false alarms, and the calibration curve "
              "shows whether the predicted probabilities can be read as risks.")


def feature_importance_plot(evaluations: List[PanelEvaluation], performance: pd.DataFrame,
                            book: FigureBook) -> None:
    frames = []
    for evaluation in evaluations:
        importance = evaluation.importances
        if importance.abs().sum() > 0:
            importance = importance / importance.abs().sum()
        frames.append(importance.rename(evaluation.label))
    matrix = pd.concat(frames, axis=1).fillna(0.0)
    matrix.index = [performance["gene_name"].get(g, g) for g in matrix.index]
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8.6, max(5.0, 0.34 * len(matrix) + 1.6)))
    matrix.iloc[::-1].plot(kind="barh", ax=ax, width=0.8,
                           color=list(discrete_palette(list(matrix.columns), "Dark2").values()))
    ax.set_xlabel("Relative contribution to the model (normalised)")
    ax.set_title("Panel gene importance per classifier")
    ax.legend(fontsize=8, title="model")
    plt.setp(ax.get_yticklabels(), fontsize=8)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "feature_importance", "Panel gene importance",
              "How much each panel gene contributes in each classifier. Genes ranked highly by several very "
              "different models are the safest choices for a reduced assay.")


def validation_plot(evaluations: List[PanelEvaluation], best: PanelEvaluation, cfg: Config,
                    book: FigureBook) -> None:
    """Optimism check: panel-on-all-data CV versus nested CV versus the permuted null."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    labels, values, errors, colors = [], [], [], []
    for evaluation in evaluations:
        labels.append(f"{evaluation.label}\n(fixed panel)")
        values.append(evaluation.mean_auc)
        errors.append(evaluation.sd_auc)
        colors.append("#5dade2")
        if evaluation.nested_auc:
            labels.append(f"{evaluation.label}\n(nested CV)")
            values.append(float(np.mean(evaluation.nested_auc)))
            errors.append(float(np.std(evaluation.nested_auc)))
            colors.append("#27ae60")
    axes[0].bar(range(len(values)), values, yerr=errors, color=colors, width=0.62, capsize=4)
    axes[0].axhline(0.5, color="#c0392b", ls="--", lw=1.0)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.3, 1.12)
    axes[0].set_title("Selection bias check")
    for i, (value, error) in enumerate(zip(values, errors)):
        if np.isfinite(value):
            axes[0].text(i, value + (error if np.isfinite(error) else 0) + 0.015,
                         f"{value:.2f}", ha="center", fontsize=8)
    axes[0].text(0.5, 0.03, "blue: genes chosen on all samples (optimistic)\n"
                            "green: genes re-chosen inside each training fold (honest)",
                 transform=axes[0].transAxes, ha="center", fontsize=8, style="italic")

    if best.permutation_auc is not None and best.permutation_auc.size:
        axes[1].hist(best.permutation_auc, bins=30, color="#b3b6b7", edgecolor="white")
        axes[1].axvline(best.mean_auc, color="#c0392b", lw=2.4,
                        label=f"observed AUC = {best.mean_auc:.3f}")
        axes[1].set_xlabel("AUC under permuted labels")
        axes[1].set_ylabel("Permutations")
        axes[1].set_title(f"Label-permutation test ({best.label})\n"
                          f"empirical p = {best.permutation_p:.4f}")
        axes[1].legend(fontsize=9)
    else:
        axes[1].axis("off")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "validation_checks", "Overfitting and permutation checks",
              "Two safeguards against an over-optimistic panel. The nested estimate repeats gene selection "
              "inside each fold, and the permutation test shows what AUC this pipeline would report on data "
              "with no real signal.")


def panel_size_plot(curve: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    if curve.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(curve["panel_size"], curve["mean_auc"], "o-", color="#2874a6", lw=2.0, ms=5)
    ax.fill_between(curve["panel_size"], curve["mean_auc"] - curve["sd_auc"],
                    curve["mean_auc"] + curve["sd_auc"], color="#2874a6", alpha=0.15)
    best = curve.loc[curve["mean_auc"].idxmax()]
    ax.axvline(best["panel_size"], color="#27ae60", ls="--", lw=1.2,
               label=f"best at {int(best['panel_size'])} genes (AUC {best['mean_auc']:.3f})")
    ax.axhline(0.5, color="#c0392b", ls=":", lw=1.0)
    ax.set_xlabel("Number of genes in the panel")
    ax.set_ylabel("Cross-validated AUC")
    ax.set_title("Performance versus panel size")
    ax.legend(fontsize=9)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "panel_size_curve", "Panel size versus performance",
              "Genes are added in rank order. The curve usually plateaus after a handful of markers - the "
              "plateau is the panel size worth taking to a targeted assay.")


def transfer_heatmap(calibrated: Optional[pd.DataFrame], raw: Optional[pd.DataFrame],
                     cfg: Config, book: FigureBook) -> None:
    panels = [(raw, "Raw expression\n(no recalibration between materials)"),
              (calibrated, "After within-material standardisation\n(each gene z-scored inside its material)")]
    panels = [(matrix, title) for matrix, title in panels
              if matrix is not None and not matrix.dropna(how="all").empty]
    if not panels:
        return
    n_types = len(panels[0][0].columns)
    fig, axes = plt.subplots(1, len(panels),
                             figsize=((1.6 * n_types + 3.4) * len(panels), 1.25 * n_types + 4.0),
                             squeeze=False)
    for ax, (matrix, title) in zip(axes[0], panels):
        sns.heatmap(matrix.astype(float), ax=ax, cmap="RdYlGn", vmin=0.2, vmax=1.0, annot=True,
                    fmt=".2f", linewidths=1.0, linecolor="white", cbar_kws={"label": "AUC"},
                    annot_kws={"fontsize": 11, "fontweight": "bold"})
        ax.set_xlabel("Tested on")
        ax.set_ylabel("Trained on")
        ax.set_title(title, fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
    fig.suptitle("Cross-sample-type transfer of the biomarker panel "
                 "(diagonal = cross-validated within the same material)", fontweight="bold", y=1.0)
    fig.tight_layout()
    book.save(fig, SECTION, "cross_sample_type_transfer", "Panel transfer between sample types",
              "Off-diagonal cells answer the practical question: trained on one material, does the classifier "
              "still work on another? The left panel uses expression as-is and mostly measures the large "
              "baseline difference between materials - values far below 0.5 mean the decision boundary lands "
              "in the wrong place entirely. The right panel z-scores each gene within its material first "
              "(no labels used, but it does need a batch of samples to calibrate against), which isolates "
              "whether the marker behaviour itself carries over. Diagonal cells are cross-validated on a "
              "smaller training set than the off-diagonal cells, so compare columns rather than rows.")


def signature_figures(scores: pd.Series, meta: pd.DataFrame, cfg: Config, book: FigureBook) -> pd.DataFrame:
    cond_col = cfg.input.condition_col
    stype_col = cfg.input.sample_type_col
    use_type = bool(stype_col and stype_col in meta.columns and meta[stype_col].nunique() > 1)
    palette = condition_palette(cfg)
    order = [cfg.input.control_label, cfg.input.case_label]
    y = _labels(meta, cfg).to_numpy()
    frame = pd.DataFrame({"score": scores, cond_col: meta[cond_col].astype(str)})
    if use_type:
        frame[stype_col] = meta[stype_col].astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4))
    if use_type:
        sns.boxplot(data=frame, x=stype_col, y="score", hue=cond_col, hue_order=order, palette=palette,
                    ax=axes[0], showfliers=False, width=0.65, linewidth=0.9)
        sns.stripplot(data=frame, x=stype_col, y="score", hue=cond_col, hue_order=order, dodge=True,
                      ax=axes[0], size=3.4, color="#2c3e50", alpha=0.6, legend=False)
        axes[0].legend(title=cond_col, fontsize=8)
        plt.setp(axes[0].get_xticklabels(), rotation=15, ha="right")
    else:
        sns.boxplot(data=frame, x=cond_col, y="score", order=order, hue=cond_col, hue_order=order,
                    palette=palette, ax=axes[0], showfliers=False, width=0.5, legend=False)
        sns.stripplot(data=frame, x=cond_col, y="score", order=order, ax=axes[0], size=4,
                      color="#2c3e50", alpha=0.65)
    axes[0].set_ylabel("Signature score (up markers - down markers)")
    axes[0].set_xlabel("")
    axes[0].set_title("Composite signature score")

    rows = []
    axes[1].plot([0, 1], [0, 1], ls="--", color="#7f8c8d", lw=1.0)
    overall_auc = _safe_auc(y, scores.to_numpy())
    fpr, tpr, _ = roc_curve(y, scores.to_numpy())
    axes[1].plot(fpr, tpr, lw=2.6, color="#2c3e50", label=f"all samples (AUC {overall_auc:.3f})")
    rows.append({"sample_type": "all samples", "n": len(scores), "auc": overall_auc})
    if use_type:
        type_palette = discrete_palette(sorted(frame[stype_col].unique()), "Set2")
        for sample_type, subset in frame.groupby(stype_col, observed=True):
            sub_y = (subset[cond_col] == cfg.input.case_label).astype(int).to_numpy()
            if len(np.unique(sub_y)) < 2:
                continue
            auc = _safe_auc(sub_y, subset["score"].to_numpy())
            fpr, tpr, _ = roc_curve(sub_y, subset["score"].to_numpy())
            axes[1].plot(fpr, tpr, lw=2.0, color=type_palette[sample_type],
                         label=f"{sample_type} (AUC {auc:.3f}, n={len(subset)})")
            rows.append({"sample_type": sample_type, "n": len(subset), "auc": auc})
    axes[1].set_xlabel("1 - specificity")
    axes[1].set_ylabel("Sensitivity")
    axes[1].set_title("Signature score ROC per sample type")
    axes[1].legend(fontsize=8, loc="lower right")

    ordered = scores.sort_values()
    colors = meta.loc[ordered.index, cond_col].astype(str).map(palette)
    axes[2].bar(range(len(ordered)), ordered.values, color=colors.values, width=1.0)
    axes[2].axhline(0, color="#2c3e50", lw=0.9)
    axes[2].set_xlabel("Samples ordered by signature score")
    axes[2].set_ylabel("Signature score")
    axes[2].set_title("Waterfall of individual samples")
    legend_from_map(axes[2], palette, cond_col, loc="upper left")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "signature_score", "Composite signature score",
              "One number per sample built from the panel. The per-sample-type ROC curves show whether the "
              "same score threshold works in every material, and the waterfall exposes exactly which samples "
              "are misclassified.")
    return pd.DataFrame(rows)


def panel_sample_heatmap(expr: pd.DataFrame, meta: pd.DataFrame, genes: Sequence[str],
                         scores: pd.Series, probabilities: pd.Series, performance: pd.DataFrame,
                         cfg: Config, book: FigureBook) -> None:
    order = probabilities.sort_values().index
    matrix = zscore_rows(expr.loc[genes, order]).clip(-2.5, 2.5)
    matrix.index = [performance["gene_name"].get(g, g) for g in genes]
    cond_col = cfg.input.condition_col
    stype_col = cfg.input.sample_type_col
    palette = condition_palette(cfg)
    use_type = bool(stype_col and stype_col in meta.columns and meta[stype_col].nunique() > 1)

    height = max(6.5, 0.28 * len(matrix) + 3.2)
    fig = plt.figure(figsize=(max(9.0, 0.2 * len(order) + 5.0), height))
    grid = GridSpec(4, 2, height_ratios=[1.5, 0.28, 0.28 if use_type else 0.02, 6.0],
                    width_ratios=[40, 1], hspace=0.06, wspace=0.02, figure=fig)
    ax_prob = fig.add_subplot(grid[0, 0])
    ax_cond = fig.add_subplot(grid[1, 0])
    ax_type = fig.add_subplot(grid[2, 0])
    ax_heat = fig.add_subplot(grid[3, 0])
    ax_cbar = fig.add_subplot(grid[3, 1])

    ax_prob.bar(range(len(order)), probabilities.loc[order].values,
                color=meta.loc[order, cond_col].astype(str).map(palette).values, width=0.9)
    ax_prob.axhline(0.5, color="#2c3e50", ls="--", lw=1.0)
    ax_prob.set_xlim(-0.5, len(order) - 0.5)
    ax_prob.set_ylabel("P(cancer)", fontsize=9)
    ax_prob.set_xticks([])
    ax_prob.set_title("Panel expression, samples ordered by held-out predicted probability",
                      fontweight="bold")
    ax_prob.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax_prob)

    ax_cond.set_xlim(-0.5, len(order) - 0.5)
    for i, sample in enumerate(order):
        ax_cond.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1,
                                        color=palette[str(meta.loc[sample, cond_col])]))
    ax_cond.set_ylim(0, 1)
    ax_cond.set_xticks([])
    ax_cond.set_yticks([])
    ax_cond.set_ylabel(cond_col, rotation=0, ha="right", va="center", fontsize=8)
    for spine in ax_cond.spines.values():
        spine.set_visible(False)

    if use_type:
        type_palette = discrete_palette(sorted(meta[stype_col].astype(str).unique()), "Set2")
        ax_type.set_xlim(-0.5, len(order) - 0.5)
        for i, sample in enumerate(order):
            ax_type.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1,
                                            color=type_palette[str(meta.loc[sample, stype_col])]))
        ax_type.set_ylim(0, 1)
        ax_type.set_ylabel(stype_col, rotation=0, ha="right", va="center", fontsize=8)
    ax_type.set_xticks([])
    ax_type.set_yticks([])
    for spine in ax_type.spines.values():
        spine.set_visible(False)
    if not use_type:
        ax_type.axis("off")

    image = ax_heat.imshow(matrix.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                           interpolation="nearest")
    ax_heat.set_yticks(range(len(matrix)))
    ax_heat.set_yticklabels(matrix.index, fontsize=8)
    ax_heat.set_xticks(range(len(order)))
    ax_heat.set_xticklabels(order, rotation=90, fontsize=5.5)
    ax_heat.grid(False)
    fig.colorbar(image, cax=ax_cbar, label="row z-score")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    labels = list(palette)
    if use_type:
        handles += [plt.Rectangle((0, 0), 1, 1, color=c) for c in type_palette.values()]
        labels += list(type_palette)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False, fontsize=8)
    book.save(fig, SECTION, "panel_sample_heatmap", "Panel expression per sample",
              "Every sample ranked by its held-out predicted probability, with the panel expression below. "
              "Misclassified samples appear as colour blocks that break the gradient and are worth "
              "inspecting individually.")
    _ = scores


def panel_correlation(expr: pd.DataFrame, meta: pd.DataFrame, genes: Sequence[str],
                      performance: pd.DataFrame, book: FigureBook) -> None:
    matrix = expr.loc[genes, meta.index]
    matrix.index = [performance["gene_name"].get(g, g) for g in genes]
    corr = matrix.T.corr(method="spearman")
    grid = sns.clustermap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                          figsize=(max(7.0, 0.34 * len(corr) + 3.5), max(6.5, 0.34 * len(corr) + 3.0)),
                          linewidths=0.3, cbar_kws={"label": "Spearman rho"},
                          annot=len(corr) <= 18, fmt=".2f", annot_kws={"fontsize": 6})
    grid.figure.suptitle("Co-expression of the panel genes", fontweight="bold", y=1.01)
    plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7.5)
    plt.setp(grid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7.5)
    book.save(grid.figure, SECTION, "panel_correlation", "Panel co-expression",
              "Highly correlated markers carry redundant information: they inflate apparent panel size "
              "without adding accuracy, and one of each block can usually be dropped.")


def run_all(expr: pd.DataFrame, meta: pd.DataFrame, de_results: Dict[str, DEResult],
            meta_table: Optional[pd.DataFrame], gene_names: pd.Series, cfg: Config,
            book: FigureBook) -> Dict[str, pd.DataFrame]:
    """Full biomarker workflow plus every figure; returns tables for export."""
    candidates = select_candidates(de_results, meta_table, cfg)
    candidates = [g for g in candidates if g in expr.index]
    if len(candidates) < 2:
        book.save(empty_figure("No candidate genes reached the significance thresholds,\n"
                               "so no biomarker evaluation could be run."),
                  SECTION, "not_available", "Biomarker analysis unavailable",
                  "No candidates passed the differential-expression filters.")
        return {}

    performance = univariate_performance(expr, meta, candidates, cfg, gene_names)
    panel = rank_panel(performance, expr, meta, cfg)
    log.info("biomarker panel: %d genes selected from %d candidates", len(panel), len(candidates))

    auc_ranking(performance, cfg, book, n=min(cfg.biomarkers.top_n, len(performance)))
    roc_top_genes(expr, meta, performance, cfg, book)
    auc_by_sample_type(performance, cfg, book, n=min(cfg.biomarkers.top_n, len(performance)))
    lfc_vs_auc(performance, cfg, book)
    biomarker_boxplots(expr, meta, performance, cfg, book)

    evaluations: List[PanelEvaluation] = []
    for model_name in cfg.biomarkers.models:
        try:
            evaluation = evaluate_panel(expr, meta, panel, model_name, cfg)
            evaluation.nested_auc = nested_cv(expr, meta, candidates, model_name, cfg)
            evaluations.append(evaluation)
            log.info("panel %-16s CV AUC %.3f ± %.3f | nested AUC %.3f",
                     model_name, evaluation.mean_auc, evaluation.sd_auc,
                     float(np.mean(evaluation.nested_auc)) if evaluation.nested_auc else np.nan)
        except Exception as exc:
            log.warning("model '%s' failed: %s", model_name, exc)
    if not evaluations:
        return {"biomarker_univariate_performance": performance}

    best = max(evaluations, key=lambda e: e.mean_auc if np.isfinite(e.mean_auc) else -1)
    null, pvalue = permutation_test(expr, meta, panel, best.model, cfg, best.mean_auc)
    best.permutation_auc, best.permutation_p = null, pvalue

    panel_roc(evaluations, cfg, book)
    panel_pr_and_confusion(evaluations, cfg, book)
    feature_importance_plot(evaluations, performance, book)
    validation_plot(evaluations, best, cfg, book)
    panel_size_plot(panel_size_curve(expr, meta, performance.index.tolist(), best.model, cfg), cfg, book)

    transfer = transfer_raw = None
    if cfg.biomarkers.cross_sample_type_validation:
        transfer = cross_sample_type_transfer(expr, meta, panel, best.model, cfg, standardize=True)
        transfer_raw = cross_sample_type_transfer(expr, meta, panel, best.model, cfg, standardize=False)
        transfer_heatmap(transfer, transfer_raw, cfg, book)

    up_genes = [g for g in panel if performance.loc[g, "direction"] == "up"]
    down_genes = [g for g in panel if performance.loc[g, "direction"] == "down"]
    scores = signature_score(expr, meta, up_genes, down_genes, cfg)
    signature_auc = signature_figures(scores, meta, cfg, book)

    probabilities = out_of_fold_scores(expr, meta, panel, best.model, cfg)
    panel_sample_heatmap(expr, meta, panel, scores, probabilities, performance, cfg, book)
    panel_correlation(expr, meta, panel, performance, book)

    panel_summary = pd.DataFrame([{
        "model": evaluation.label,
        "n_genes": len(evaluation.genes),
        "cv_auc_mean": evaluation.mean_auc,
        "cv_auc_sd": evaluation.sd_auc,
        "nested_cv_auc_mean": float(np.mean(evaluation.nested_auc)) if evaluation.nested_auc else np.nan,
        "permutation_p": evaluation.permutation_p,
        **evaluation.metrics,
    } for evaluation in evaluations])

    final = performance.copy()
    final["in_panel"] = final.index.isin(panel)
    final["panel_rank"] = [panel.index(g) + 1 if g in panel else np.nan for g in final.index]

    per_sample = pd.DataFrame({
        "signature_score": scores,
        "predicted_probability": probabilities,
    }).join(meta.astype(str))

    tables = {
        "biomarker_univariate_performance": final,
        "biomarker_panel_summary": panel_summary,
        "biomarker_signature_auc_by_sample_type": signature_auc,
        "biomarker_sample_scores": per_sample,
    }
    if transfer is not None:
        tables["biomarker_cross_sample_type_transfer_auc"] = transfer
    if transfer_raw is not None:
        tables["biomarker_cross_sample_type_transfer_auc_uncalibrated"] = transfer_raw
    return tables
