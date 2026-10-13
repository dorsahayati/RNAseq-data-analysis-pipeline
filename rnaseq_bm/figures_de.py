"""Differential-expression figures: volcano, MA, dispersion, heatmaps, gene-level boxplots."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import Config
from .de import DEResult
from .plotting import (
    FigureBook,
    annotate_points,
    condition_palette,
    discrete_palette,
    empty_figure,
    heatmap_with_annotations,
    significance_bracket,
)
from .utils import sanitize, significance_stars, zscore_rows

SECTION = "de"
UP_COLOR = "#c0392b"
DOWN_COLOR = "#2874a6"
NS_COLOR = "#b3b6b7"


def _direction_colors(table: pd.DataFrame) -> np.ndarray:
    return np.where(table["direction"] == "up", UP_COLOR,
                    np.where(table["direction"] == "down", DOWN_COLOR, NS_COLOR))


def volcano(result: DEResult, cfg: Config, book: FigureBook, section: str = SECTION) -> None:
    table = result.table.dropna(subset=["pvalue"]).copy()
    y = table["neg_log10_padj"].replace([np.inf], np.nan)
    if y.notna().sum() == 0:
        y = -np.log10(table["pvalue"].clip(lower=1e-300))
        y_label = "-log10 p-value"
    else:
        y = y.fillna(0.0)
        y_label = "-log10 adjusted p-value"
    x = table["log2FoldChange"]

    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    colors = _direction_colors(table)
    ax.scatter(x, y, c=colors, s=12, alpha=0.65, edgecolor="none", rasterized=True)
    ax.axhline(-np.log10(cfg.analysis.alpha), color="#2c3e50", ls="--", lw=0.9)
    if cfg.analysis.lfc_threshold:
        for sign in (-1, 1):
            ax.axvline(sign * cfg.analysis.lfc_threshold, color="#2c3e50", ls="--", lw=0.9)
    ax.set_xlabel(f"log2 fold change ({cfg.input.case_label} / {cfg.input.control_label})")
    ax.set_ylabel(y_label)
    ax.set_title(f"Volcano plot - {result.label}")

    significant = table[table["significant"]]
    score = significant["neg_log10_padj"].fillna(0) * significant["log2FoldChange"].abs()
    annotate_points(ax, significant["log2FoldChange"], y.reindex(significant.index),
                    significant["gene_name"], n=cfg.plots.label_top_n, score=score)
    n_up, n_down = len(result.up), len(result.down)
    ax.text(0.02, 0.98, f"{n_down:,} down", transform=ax.transAxes, va="top", ha="left",
            color=DOWN_COLOR, fontweight="bold")
    ax.text(0.98, 0.98, f"{n_up:,} up", transform=ax.transAxes, va="top", ha="right",
            color=UP_COLOR, fontweight="bold")
    limit = np.nanpercentile(np.abs(x), 99.8) if len(x) else 1
    ax.set_xlim(-max(limit, cfg.analysis.lfc_threshold * 1.5) * 1.15,
                max(limit, cfg.analysis.lfc_threshold * 1.5) * 1.15)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, section, f"volcano_{sanitize(result.name)}", f"Volcano plot ({result.label})",
              f"Effect size against statistical significance for {result.label}. Genes passing "
              f"FDR < {cfg.analysis.alpha} and |log2FC| >= {cfg.analysis.lfc_threshold} are coloured; the "
              f"{cfg.plots.label_top_n} strongest are labelled.")


def ma_plot(result: DEResult, cfg: Config, book: FigureBook) -> None:
    table = result.table.dropna(subset=["baseMean", "log2FoldChange"]).copy()
    table = table[table["baseMean"] > 0]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.scatter(table["baseMean"], table["log2FoldChange"], c=_direction_colors(table),
               s=11, alpha=0.6, edgecolor="none", rasterized=True)
    ax.set_xscale("log")
    ax.axhline(0, color="#2c3e50", lw=1.0)
    for sign in (-1, 1):
        ax.axhline(sign * cfg.analysis.lfc_threshold, color="#2c3e50", ls="--", lw=0.8)
    ax.set_xlabel("Mean normalised expression (baseMean)")
    ax.set_ylabel("log2 fold change")
    ax.set_title(f"MA plot - {result.label}")
    significant = table[table["significant"]]
    annotate_points(ax, significant["baseMean"], significant["log2FoldChange"],
                    significant["gene_name"], n=12,
                    score=significant["log2FoldChange"].abs() * np.log10(significant["baseMean"] + 10))
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
               for l, c in [("up in " + cfg.input.case_label, UP_COLOR),
                            ("down in " + cfg.input.case_label, DOWN_COLOR), ("not significant", NS_COLOR)]]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, f"ma_plot_{sanitize(result.name)}", f"MA plot ({result.label})",
              "Log fold change against mean expression. Shrunken estimates pull low-count genes towards "
              "zero, so the significant hits should not be concentrated at the low-expression edge.")


def pvalue_histogram(results: Dict[str, DEResult], cfg: Config, book: FigureBook) -> None:
    items = list(results.values())
    ncols = min(3, len(items))
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows), squeeze=False)
    for ax, result in zip(axes.ravel(), items):
        pvals = result.table["pvalue"].dropna()
        ax.hist(pvals, bins=40, range=(0, 1), color="#5d6d7e", edgecolor="white", linewidth=0.4)
        expected = len(pvals) / 40
        ax.axhline(expected, color="#c0392b", ls="--", lw=1.1, label="uniform expectation")
        # Storey's pi0: fraction of tests consistent with the null.
        pi0 = min(1.0, float((pvals > 0.5).sum()) / (0.5 * max(len(pvals), 1)))
        ax.set_title(f"{result.label}\n(estimated null fraction {pi0:.0%})", fontsize=10)
        ax.set_xlabel("p-value")
        ax.set_ylabel("Genes")
        ax.legend(fontsize=7.5)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "pvalue_histograms", "P-value distributions",
              "A well-behaved test gives a flat histogram with a spike near zero. A hump in the middle or at "
              "one means the model is misspecified (unmodelled batch, wrong dispersion) and the FDR is unreliable.")


def dispersion_plot(result: DEResult, book: FigureBook) -> None:
    if result.dispersion is None or result.dispersion.empty:
        return
    data = result.dispersion.replace([np.inf, -np.inf], np.nan).dropna(subset=["baseMean", "final"])
    data = data[data["baseMean"] > 0]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.scatter(data["baseMean"], data["genewise"], s=7, alpha=0.35, color="#7f8c8d",
               label="gene-wise estimate", rasterized=True)
    order = data["baseMean"].sort_values().index
    ax.plot(data.loc[order, "baseMean"], data.loc[order, "fitted"], color="#c0392b", lw=2.0,
            label="fitted trend")
    ax.scatter(data["baseMean"], data["final"], s=7, alpha=0.35, color="#2874a6",
               label="final (shrunk)", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean of normalised counts")
    ax.set_ylabel("Dispersion")
    ax.set_title(f"DESeq2 dispersion estimates - {result.label}")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, f"dispersion_{sanitize(result.name)}", "Dispersion estimates",
              "Gene-wise dispersions are shrunk towards a fitted trend, which is what makes small-n RNA-seq "
              "testing stable. A cloud that does not follow the red curve indicates a poor model fit.")


def de_counts_barplot(results: Dict[str, DEResult], cfg: Config, book: FigureBook) -> None:
    rows = [{"contrast": r.label, "up": len(r.up), "down": -len(r.down), "n": r.n_case + r.n_control}
            for r in results.values()]
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(frame) + 3), 5.0))
    ax.bar(frame["contrast"], frame["up"], color=UP_COLOR, width=0.62, label=f"up in {cfg.input.case_label}")
    ax.bar(frame["contrast"], frame["down"], color=DOWN_COLOR, width=0.62, label=f"down in {cfg.input.case_label}")
    for i, row in frame.iterrows():
        ax.text(i, row["up"], f"{row['up']:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i, row["down"], f"{abs(row['down']):,}", ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(i, 0, f"n={row['n']}", ha="center", va="center", fontsize=7.5, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="#2c3e50", ec="none"))
    ax.axhline(0, color="#2c3e50", lw=1.0)
    ax.set_ylabel("Differentially expressed genes")
    ax.set_title(f"DE genes per contrast (FDR < {cfg.analysis.alpha}, |log2FC| >= {cfg.analysis.lfc_threshold})")
    ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "de_gene_counts", "DE gene counts per contrast",
              "Number of significant genes for the overall contrast and within each sample type. Differences "
              "are driven by both biology and sample size, so read this together with the group sizes shown.")


def top_gene_heatmap(result: DEResult, expr: pd.DataFrame, meta: pd.DataFrame, cfg: Config,
                     book: FigureBook, n: Optional[int] = None) -> None:
    n = n or cfg.plots.heatmap_top_n
    significant = result.significant
    if significant.empty:
        book.save(empty_figure(f"No significant genes for {result.label}"), SECTION,
                  f"heatmap_{sanitize(result.name)}", f"Top DE genes ({result.label})",
                  "No gene passed the significance thresholds for this contrast.")
        return
    ranked = significant.reindex(
        significant["neg_log10_padj"].fillna(0).mul(significant["log2FoldChange"].abs()).sort_values(
            ascending=False).index
    )
    genes = ranked.head(n).index
    samples = [s for s in result.samples if s in expr.columns]
    matrix = zscore_rows(expr.loc[genes, samples])
    matrix.index = [result.table.loc[g, "gene_name"] for g in genes]
    palettes = {cfg.input.condition_col: condition_palette(cfg)}
    stype = cfg.input.sample_type_col
    if stype and stype in meta.columns and meta.loc[samples, stype].nunique() > 1:
        palettes[stype] = discrete_palette(sorted(meta.loc[samples, stype].astype(str).unique()), "Set2")
    fig = heatmap_with_annotations(
        matrix, meta.loc[samples], palettes,
        f"Top {len(genes)} differentially expressed genes - {result.label}",
        cbar_label="row z-score", vmin=-2.5, vmax=2.5,
    )
    book.save(fig, SECTION, f"heatmap_{sanitize(result.name)}", f"Top DE gene heatmap ({result.label})",
              "Row-scaled expression of the strongest DE genes. Because genes were selected using the "
              "condition labels, the block structure is expected - judge instead how consistent it is "
              "within each group.")


def lfc_barplot(result: DEResult, cfg: Config, book: FigureBook, n: int = 20) -> None:
    significant = result.significant
    if significant.empty:
        return
    up = significant[significant["direction"] == "up"].nlargest(n, "log2FoldChange")
    down = significant[significant["direction"] == "down"].nsmallest(n, "log2FoldChange")
    combined = pd.concat([down, up]).sort_values("log2FoldChange")
    fig, ax = plt.subplots(figsize=(7.6, max(5.0, 0.28 * len(combined) + 1.5)))
    colors = [UP_COLOR if v > 0 else DOWN_COLOR for v in combined["log2FoldChange"]]
    bars = ax.barh(range(len(combined)), combined["log2FoldChange"], color=colors, height=0.72)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined["gene_name"], fontsize=8)
    ax.axvline(0, color="#2c3e50", lw=1.0)
    ax.set_xlabel(f"log2 fold change ({cfg.input.case_label} / {cfg.input.control_label})")
    ax.set_title(f"Strongest up- and down-regulated genes - {result.label}")
    for bar, (_, row) in zip(bars, combined.iterrows()):
        offset = 0.06 * np.sign(row["log2FoldChange"]) * ax.get_xlim()[1]
        ax.text(row["log2FoldChange"] + offset * 0.2, bar.get_y() + bar.get_height() / 2,
                f"FDR={row['padj']:.1e}", va="center",
                ha="left" if row["log2FoldChange"] > 0 else "right", fontsize=6.5, color="#555555")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, f"top_lfc_{sanitize(result.name)}", f"Largest fold changes ({result.label})",
              "The most strongly regulated significant genes in each direction, annotated with their FDR.")


def top_gene_boxplots(result: DEResult, expr: pd.DataFrame, meta: pd.DataFrame, cfg: Config,
                      book: FigureBook, n: int = 12) -> None:
    significant = result.significant
    if significant.empty:
        return
    genes = significant.head(n).index
    cond_col = cfg.input.condition_col
    stype_col = cfg.input.sample_type_col
    use_type = bool(stype_col and stype_col in meta.columns and meta[stype_col].nunique() > 1)
    ncols = 4
    nrows = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False)
    palette = condition_palette(cfg)
    order = [cfg.input.control_label, cfg.input.case_label]
    for ax, gene in zip(axes.ravel(), genes):
        frame = pd.DataFrame({
            "expression": expr.loc[gene, meta.index].values,
            cond_col: meta[cond_col].astype(str).values,
        }, index=meta.index)
        if use_type:
            frame[stype_col] = meta[stype_col].astype(str).values
            sns.boxplot(data=frame, x=stype_col, y="expression", hue=cond_col, hue_order=order,
                        palette=palette, ax=ax, showfliers=False, width=0.68, linewidth=0.9)
            sns.stripplot(data=frame, x=stype_col, y="expression", hue=cond_col, hue_order=order,
                          dodge=True, ax=ax, size=3.0, color="#2c3e50", alpha=0.6, legend=False)
            ax.legend_.remove() if ax.get_legend() else None
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7.5)
        else:
            sns.boxplot(data=frame, x=cond_col, y="expression", order=order, hue=cond_col,
                        hue_order=order, palette=palette, ax=ax, showfliers=False, width=0.55,
                        linewidth=0.9, legend=False)
            sns.stripplot(data=frame, x=cond_col, y="expression", order=order, ax=ax, size=3.5,
                          color="#2c3e50", alpha=0.65)
        case = frame.loc[frame[cond_col] == cfg.input.case_label, "expression"]
        control = frame.loc[frame[cond_col] == cfg.input.control_label, "expression"]
        pvalue = stats.mannwhitneyu(case, control, alternative="two-sided").pvalue if min(
            len(case), len(control)) > 2 else np.nan
        top = frame["expression"].max()
        span = (frame["expression"].max() - frame["expression"].min()) or 1
        if not use_type:
            significance_bracket(ax, 0, 1, top + 0.05 * span, 0.05 * span, significance_stars(pvalue))
        row = result.table.loc[gene]
        ax.set_title(f"{row['gene_name']}\nlog2FC={row['log2FoldChange']:.2f}, FDR={row['padj']:.1e}",
                     fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("log2 CPM")
        ax.set_ylim(top=top + 0.22 * span)
    for ax in axes.ravel()[len(genes):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[l]) for l in order]
    fig.legend(handles, order, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(f"Expression of the top DE genes - {result.label}", fontweight="bold", y=1.0)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, f"top_gene_boxplots_{sanitize(result.name)}",
              f"Top gene expression ({result.label})",
              "Per-sample expression of the leading DE genes with individual points shown, split by sample "
              "type when available. Overlapping distributions warn that a low FDR does not imply a usable marker.")


def rank_plot(result: DEResult, cfg: Config, book: FigureBook) -> None:
    table = result.table.dropna(subset=["log2FoldChange"]).sort_values("log2FoldChange")
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ranks = np.arange(len(table))
    ax.plot(ranks, table["log2FoldChange"], color="#5d6d7e", lw=1.2)
    significant = table["significant"].values
    ax.scatter(ranks[significant], table["log2FoldChange"].values[significant],
               c=_direction_colors(table)[significant], s=8, zorder=3)
    ax.axhline(0, color="#2c3e50", lw=0.9)
    for sign in (-1, 1):
        ax.axhline(sign * cfg.analysis.lfc_threshold, color="#2c3e50", ls="--", lw=0.7)
    ax.set_xlabel("Genes ranked by fold change")
    ax.set_ylabel("log2 fold change")
    ax.set_title(f"Fold-change rank curve - {result.label}")
    ax.text(0.02, 0.95, f"{len(result.down):,} significant down", transform=ax.transAxes,
            color=DOWN_COLOR, fontsize=9, va="top")
    ax.text(0.98, 0.08, f"{len(result.up):,} significant up", transform=ax.transAxes,
            color=UP_COLOR, fontsize=9, ha="right")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, f"rank_curve_{sanitize(result.name)}", f"Fold-change rank curve ({result.label})",
              "All genes ordered by fold change. A steep S-curve with long tails indicates a strong, "
              "asymmetric transcriptional response rather than diffuse noise.")


def interaction_figure(table: Optional[pd.DataFrame], expr: pd.DataFrame, meta: pd.DataFrame,
                       cfg: Config, book: FigureBook, n: int = 8) -> None:
    """Genes whose cancer-vs-control effect differs between sample types."""
    if table is None or table.empty:
        return
    stype_col = cfg.input.sample_type_col
    cond_col = cfg.input.condition_col
    genes = table.head(n).index
    ncols = 4
    nrows = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False)
    palette = condition_palette(cfg)
    order = [cfg.input.control_label, cfg.input.case_label]
    for ax, gene in zip(axes.ravel(), genes):
        frame = pd.DataFrame({
            "expression": expr.loc[gene, meta.index].values,
            cond_col: meta[cond_col].astype(str).values,
            stype_col: meta[stype_col].astype(str).values,
        })
        means = frame.groupby([stype_col, cond_col], observed=True)["expression"].mean().unstack()
        for level in order:
            if level in means.columns:
                ax.plot(range(len(means)), means[level], "o-", color=palette[level], lw=2.0, ms=6, label=level)
        sns.stripplot(data=frame, x=stype_col, y="expression", hue=cond_col, hue_order=order,
                      palette=palette, dodge=True, ax=ax, size=3.0, alpha=0.55, legend=False)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=20, ha="right", fontsize=7.5)
        ax.set_title(f"{table.loc[gene, 'gene_name']}\ninteraction FDR={table.loc[gene, 'padj']:.1e}",
                     fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("log2 CPM")
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=7.5)
    for ax in axes.ravel()[len(genes):]:
        ax.axis("off")
    fig.suptitle("Genes with a sample-type-dependent cancer effect (interaction test)",
                 fontweight="bold", y=1.0)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "interaction_genes", "Condition x sample-type interaction",
              "Interaction profiles: non-parallel lines mean the cancer effect differs between sample types, "
              "so a single marker threshold will not transfer across materials.")


def run_all(results: Dict[str, DEResult], expr: pd.DataFrame, meta: pd.DataFrame,
            interaction: Optional[pd.DataFrame], cfg: Config, book: FigureBook) -> None:
    overall = results["all_samples"]
    de_counts_barplot(results, cfg, book)
    for result in results.values():
        volcano(result, cfg, book)
    ma_plot(overall, cfg, book)
    for name, result in results.items():
        if name != "all_samples":
            ma_plot(result, cfg, book)
    pvalue_histogram(results, cfg, book)
    for result in results.values():
        dispersion_plot(result, book)
    top_gene_heatmap(overall, expr, meta, cfg, book)
    for name, result in results.items():
        if name != "all_samples":
            top_gene_heatmap(result, expr, meta, cfg, book, n=min(30, cfg.plots.heatmap_top_n))
    lfc_barplot(overall, cfg, book)
    top_gene_boxplots(overall, expr, meta, cfg, book, n=cfg.biomarkers.n_boxplot_genes)
    rank_plot(overall, cfg, book)
    interaction_figure(interaction, expr, meta, cfg, book)
