"""Quality-control and exploratory figures (library size, RLE, PCA, clustering, ...)."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

from .config import Config
from .plotting import (
    FigureBook,
    condition_palette,
    discrete_palette,
    empty_figure,
    heatmap_with_annotations,
    legend_from_map,
)
from .utils import log, zscore_rows

SECTION = "qc"


def _sample_colors(meta: pd.DataFrame, cfg: Config):
    cond_pal = condition_palette(cfg)
    conditions = meta[cfg.input.condition_col].astype(str)
    colors = conditions.map(cond_pal).fillna("#95a5a6")
    return colors, cond_pal


def _sample_type(meta: pd.DataFrame, cfg: Config) -> Optional[pd.Series]:
    col = cfg.input.sample_type_col
    if col and col in meta.columns and meta[col].nunique() > 1:
        return meta[col].astype(str)
    return None


def _order_samples(meta: pd.DataFrame, cfg: Config) -> list:
    keys = [cfg.input.condition_col]
    stype = cfg.input.sample_type_col
    if stype and stype in meta.columns:
        keys = [stype] + keys
    return meta.sort_values(keys).index.tolist()


def design_overview(meta: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    cond_col = cfg.input.condition_col
    stype = _sample_type(meta, cfg)
    cond_pal = condition_palette(cfg)
    if stype is None:
        table = meta[cond_col].astype(str).value_counts().to_frame("n")
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.bar(table.index, table["n"], color=[cond_pal.get(i, "#95a5a6") for i in table.index], width=0.6)
        for i, value in enumerate(table["n"]):
            ax.text(i, value, f"{value}", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Samples")
    else:
        table = pd.crosstab(stype, meta[cond_col].astype(str))
        fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * len(table) + 2.5), 4.2))
        table.plot(kind="bar", ax=ax, color=[cond_pal.get(c, "#95a5a6") for c in table.columns],
                   width=0.72, edgecolor="white")
        for container in ax.containers:
            ax.bar_label(container, fontsize=9)
        ax.set_xlabel("Sample type")
        ax.set_ylabel("Samples")
        ax.legend(title=cond_col)
        plt.setp(ax.get_xticklabels(), rotation=0)
    ax.set_title("Study design: samples per group")
    ax.grid(axis="y", alpha=0.25)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "design_overview", "Study design",
              "Number of samples per condition (and sample type). Unbalanced or very small groups "
              "limit the power of every downstream test.")


def library_size(counts: pd.DataFrame, meta: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    order = _order_samples(meta, cfg)
    lib = counts[order].sum(axis=0) / 1e6
    colors, cond_pal = _sample_colors(meta.loc[order], cfg)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.24 * len(order) + 3), 4.4))
    ax.bar(range(len(order)), lib.values, color=colors.loc[order].values, width=0.75)
    median = float(np.median(lib.values))
    ax.axhline(median, color="#2c3e50", ls="--", lw=1.0)
    ax.text(len(order) - 0.5, median, f" median {median:.1f}M", va="bottom", ha="right", fontsize=8)
    low = lib[lib < 0.5 * median]
    for sample in low.index:
        ax.text(order.index(sample), lib[sample], "!", ha="center", va="bottom", color="#c0392b", fontweight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6.5)
    ax.set_ylabel("Library size (million assigned reads)")
    ax.set_title("Sequencing depth per sample")
    legend_from_map(ax, cond_pal, cfg.input.condition_col)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "library_size", "Library size per sample",
              "Total assigned counts per sample. Samples flagged with '!' have less than half the median "
              "depth and are candidates for exclusion; the dashed line is the median.")


def detected_genes(counts: pd.DataFrame, meta: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    order = _order_samples(meta, cfg)
    detected = (counts[order] > 0).sum(axis=0)
    lib = counts[order].sum(axis=0) / 1e6
    colors, cond_pal = _sample_colors(meta.loc[order], cfg)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    axes[0].bar(range(len(order)), detected.values, color=colors.loc[order].values, width=0.75)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=90, fontsize=6.5)
    axes[0].set_ylabel("Genes with >0 counts")
    axes[0].set_title("Detected genes per sample")
    legend_from_map(axes[0], cond_pal, cfg.input.condition_col)

    axes[1].scatter(lib.values, detected.values, c=colors.loc[order].values, s=45, edgecolor="white", lw=0.6)
    axes[1].set_xlabel("Library size (millions)")
    axes[1].set_ylabel("Genes detected")
    axes[1].set_title("Detection saturation")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "detected_genes", "Gene detection",
              "Left: number of genes with at least one count. Right: detection versus depth - points far "
              "below the trend indicate degraded libraries or low complexity.")


def filtering_summary(summary: pd.DataFrame, counts_raw: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    axes[0].barh(range(len(summary)), summary["genes_retained"], color="#5d6d7e", height=0.6)
    axes[0].set_yticks(range(len(summary)))
    axes[0].set_yticklabels(summary["step"], fontsize=8.5)
    axes[0].invert_yaxis()
    for i, value in enumerate(summary["genes_retained"]):
        axes[0].text(value, i, f" {value:,}", va="center", fontsize=8.5)
    axes[0].set_xlabel("Genes retained")
    axes[0].set_title("Gene filtering cascade")

    mean_counts = counts_raw.mean(axis=1)
    axes[1].hist(np.log10(mean_counts + 1), bins=60, color="#5d6d7e")
    axes[1].axvline(np.log10(cfg.filtering.min_count + 1), color="#c0392b", ls="--",
                    label=f"min_count = {cfg.filtering.min_count}")
    axes[1].set_xlabel("log10(mean count + 1)")
    axes[1].set_ylabel("Genes")
    axes[1].set_title("Mean expression distribution")
    axes[1].legend()
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "gene_filtering", "Low-expression filtering",
              "Genes removed at each filtering step and the raw mean-count distribution. Filtering removes "
              "the noise-dominated left mode and increases detection power after multiple-testing correction.")


def count_distributions(raw_logcpm: pd.DataFrame, norm_logcpm: pd.DataFrame, meta: pd.DataFrame,
                        cfg: Config, book: FigureBook) -> None:
    order = _order_samples(meta, cfg)
    colors, cond_pal = _sample_colors(meta.loc[order], cfg)
    fig, axes = plt.subplots(2, 1, figsize=(max(7.0, 0.24 * len(order) + 3), 8.4), sharex=True)
    for ax, data, title in zip(
        axes, [raw_logcpm[order], norm_logcpm[order]],
        ["Before normalisation (log2 CPM)", f"After {cfg.analysis.normalization.upper()} normalisation (log2)"],
    ):
        bp = ax.boxplot([data[s].values for s in order], patch_artist=True, showfliers=False, widths=0.65)
        for patch, sample in zip(bp["boxes"], order):
            patch.set_facecolor(colors.loc[sample])
            patch.set_alpha(0.8)
            patch.set_edgecolor("#2c3e50")
            patch.set_linewidth(0.6)
        for element in ("medians", "whiskers", "caps"):
            plt.setp(bp[element], color="#2c3e50", linewidth=0.8)
        ax.set_ylabel("log2 expression")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[1].set_xticks(range(1, len(order) + 1))
    axes[1].set_xticklabels(order, rotation=90, fontsize=6.5)
    legend_from_map(axes[0], cond_pal, cfg.input.condition_col)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "count_distributions", "Expression distributions before/after normalisation",
              "Per-sample log2 expression distributions. After normalisation the medians should line up; "
              "residual shifts indicate composition bias that normalisation could not absorb.")


def density_and_zero_fraction(logcpm: pd.DataFrame, counts: pd.DataFrame, meta: pd.DataFrame,
                              cfg: Config, book: FigureBook) -> None:
    order = _order_samples(meta, cfg)
    colors, cond_pal = _sample_colors(meta.loc[order], cfg)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for sample in order:
        sns.kdeplot(logcpm[sample], ax=axes[0], color=colors.loc[sample], lw=0.9, alpha=0.6)
    axes[0].set_xlabel("log2 CPM")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Expression density per sample")
    legend_from_map(axes[0], cond_pal, cfg.input.condition_col)

    zero_fraction = (counts[order] == 0).mean(axis=0)
    axes[1].bar(range(len(order)), zero_fraction.values, color=colors.loc[order].values, width=0.75)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=90, fontsize=6.5)
    axes[1].set_ylabel("Fraction of zero counts")
    axes[1].set_title("Sparsity per sample")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "density_sparsity", "Density and sparsity",
              "Overlaid expression densities (multi-modal or shifted curves flag problem samples) and the "
              "fraction of zero counts per sample.")


def rle_plot(norm_log: pd.DataFrame, meta: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    order = _order_samples(meta, cfg)
    deviations = norm_log[order].sub(norm_log[order].median(axis=1), axis=0)
    colors, cond_pal = _sample_colors(meta.loc[order], cfg)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.24 * len(order) + 3), 4.8))
    bp = ax.boxplot([deviations[s].values for s in order], patch_artist=True, showfliers=False, widths=0.7)
    for patch, sample in zip(bp["boxes"], order):
        patch.set_facecolor(colors.loc[sample])
        patch.set_alpha(0.8)
        patch.set_edgecolor("#2c3e50")
        patch.set_linewidth(0.6)
    for element in ("medians", "whiskers", "caps"):
        plt.setp(bp[element], color="#2c3e50", linewidth=0.8)
    ax.axhline(0, color="#c0392b", lw=1.0)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=90, fontsize=6.5)
    ax.set_ylabel("Relative log expression")
    ax.set_title("RLE plot (normalised data)")
    legend_from_map(ax, cond_pal, cfg.input.condition_col)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "rle", "Relative log expression",
              "Each box shows a sample's deviation from the gene-wise median. Well-normalised samples are "
              "centred on zero with similar spread; shifted or wide boxes mark technical outliers.")


def size_factor_plot(size_factors: pd.Series, counts: pd.DataFrame, meta: pd.DataFrame,
                     cfg: Config, book: FigureBook) -> None:
    lib = counts.sum(axis=0) / 1e6
    colors, cond_pal = _sample_colors(meta, cfg)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(lib.reindex(size_factors.index), size_factors.values,
               c=colors.reindex(size_factors.index).values, s=55, edgecolor="white", lw=0.7)
    for sample in size_factors.index:
        ax.annotate(sample, (lib[sample], size_factors[sample]), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Library size (millions)")
    ax.set_ylabel(f"{cfg.analysis.normalization} scaling factor")
    ax.set_title("Normalisation factors vs sequencing depth")
    legend_from_map(ax, cond_pal, cfg.input.condition_col)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "size_factors", "Normalisation factors",
              "Scaling factors should increase roughly linearly with library size. Strong departures mean "
              "a few highly expressed genes dominate a library (composition bias).")


def top_expressed_genes(counts: pd.DataFrame, gene_names: pd.Series, book: FigureBook, n: int = 20) -> None:
    fraction = counts.divide(counts.sum(axis=0), axis=1) * 100
    top = fraction.mean(axis=1).sort_values(ascending=False).head(n).index[::-1]
    data = fraction.loc[top]
    fig, ax = plt.subplots(figsize=(7.6, max(4.5, 0.32 * n + 1.2)))
    bp = ax.boxplot([data.loc[g].values for g in top], vert=False, patch_artist=True, widths=0.62,
                    showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#5dade2")
        patch.set_alpha(0.85)
        patch.set_edgecolor("#2c3e50")
        patch.set_linewidth(0.6)
    for element in ("medians", "whiskers", "caps"):
        plt.setp(bp[element], color="#2c3e50", linewidth=0.8)
    ax.set_yticks(range(1, len(top) + 1))
    ax.set_yticklabels([gene_names.get(g, g) for g in top], fontsize=8)
    ax.set_xlabel("% of total counts in the sample")
    ax.set_title(f"Top {n} most highly expressed genes")
    total = fraction.loc[top].sum(axis=0).mean()
    ax.text(0.98, 0.02, f"these genes carry {total:.1f}% of all reads on average",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, style="italic")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "top_expressed_genes", "Most highly expressed genes",
              "Share of the library taken by the most abundant genes. A handful of genes taking >30-50% of "
              "reads (rRNA, haemoglobin, mitochondrial) indicates a depletion problem.")


def correlation_heatmap(expr: pd.DataFrame, meta: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    corr = expr.corr(method="spearman")
    annotations = meta.copy()
    palettes = {cfg.input.condition_col: condition_palette(cfg)}
    stype = _sample_type(meta, cfg)
    if stype is not None:
        palettes[cfg.input.sample_type_col] = discrete_palette(sorted(stype.unique()), "Set2")
    if cfg.input.batch_col and cfg.input.batch_col in meta.columns:
        palettes[cfg.input.batch_col] = discrete_palette(sorted(meta[cfg.input.batch_col].astype(str).unique()), "Pastel1")
    fig = heatmap_with_annotations(
        corr, annotations, palettes, "Sample-sample Spearman correlation",
        cbar_label="rho", cmap="viridis", vmin=float(np.nanpercentile(corr.values, 1)), vmax=1.0,
    )
    book.save(fig, SECTION, "sample_correlation", "Sample correlation heatmap",
              "Spearman correlation between samples on normalised expression. Samples should cluster by "
              "biology (condition / sample type); a sample correlating poorly with everything is an outlier.")


def sample_distance_heatmap(expr: pd.DataFrame, meta: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    distance = pd.DataFrame(squareform(pdist(expr.T.values, metric="euclidean")),
                            index=expr.columns, columns=expr.columns)
    palettes = {cfg.input.condition_col: condition_palette(cfg)}
    stype = _sample_type(meta, cfg)
    if stype is not None:
        palettes[cfg.input.sample_type_col] = discrete_palette(sorted(stype.unique()), "Set2")
    fig = heatmap_with_annotations(distance, meta, palettes, "Euclidean sample distances (VST)",
                                   cbar_label="distance", cmap="mako_r", vmin=0)
    book.save(fig, SECTION, "sample_distance", "Sample distance heatmap",
              "Euclidean distances on variance-stabilised data, the standard DESeq2 sample-clustering view.")


def dendrogram(expr: pd.DataFrame, meta: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    linkage = hierarchy.linkage(expr.T.values, method="ward", metric="euclidean")
    colors, cond_pal = _sample_colors(meta, cfg)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.26 * expr.shape[1] + 3), 5.0))
    dend = hierarchy.dendrogram(linkage, labels=expr.columns.tolist(), ax=ax, color_threshold=None,
                                above_threshold_color="#7f8c8d", leaf_font_size=7)
    for label in ax.get_xmajorticklabels():
        label.set_color(colors.get(label.get_text(), "#2c3e50"))
        label.set_rotation(90)
    ax.set_ylabel("Ward linkage distance")
    ax.set_title("Hierarchical clustering of samples")
    legend_from_map(ax, cond_pal, cfg.input.condition_col)
    ax.grid(False)
    sns.despine(fig=fig, left=False)
    fig.tight_layout()
    book.save(fig, SECTION, "dendrogram", "Hierarchical clustering",
              "Ward clustering of samples; leaf labels are coloured by condition. Perfect separation is not "
              "expected, but systematic mixing suggests weak signal or a dominant technical factor.")
    _ = dend


def pca_figures(expr: pd.DataFrame, meta: pd.DataFrame, cfg: Config, book: FigureBook) -> pd.DataFrame:
    values = expr.T.values
    values = values - values.mean(axis=0, keepdims=True)
    n_components = int(min(10, min(values.shape) - 1))
    pca = PCA(n_components=n_components, random_state=cfg.project.seed)
    scores = pca.fit_transform(values)
    variance = pca.explained_variance_ratio_ * 100
    pcs = pd.DataFrame(scores, index=expr.columns, columns=[f"PC{i+1}" for i in range(n_components)])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    axes[0].bar(range(1, n_components + 1), variance, color="#5d6d7e", width=0.65)
    axes[0].plot(range(1, n_components + 1), np.cumsum(variance), "o-", color="#c0392b", lw=1.4, ms=4)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("% variance explained")
    axes[0].set_title("Scree plot (bars) and cumulative variance (line)")
    axes[0].set_xticks(range(1, n_components + 1))

    factors = {cfg.input.condition_col: meta[cfg.input.condition_col].astype(str)}
    stype = _sample_type(meta, cfg)
    if stype is not None:
        factors[cfg.input.sample_type_col] = stype
    if cfg.input.batch_col and cfg.input.batch_col in meta.columns:
        factors[cfg.input.batch_col] = meta[cfg.input.batch_col].astype(str)
    correlations = []
    for name, series in factors.items():
        codes = pd.factorize(series.reindex(pcs.index))[0]
        for pc in pcs.columns[:min(6, n_components)]:
            groups = [pcs[pc].values[codes == c] for c in np.unique(codes)]
            if len(groups) > 1 and all(len(g) > 1 for g in groups):
                from scipy.stats import f_oneway
                stat, pval = f_oneway(*groups)
                correlations.append({"factor": name, "PC": pc, "neglog10p": -np.log10(max(pval, 1e-300))})
    if correlations:
        matrix = pd.DataFrame(correlations).pivot(index="factor", columns="PC", values="neglog10p")
        sns.heatmap(matrix, ax=axes[1], cmap="Reds", annot=True, fmt=".1f", cbar_kws={"label": "-log10 p"},
                    linewidths=0.5, linecolor="white")
        axes[1].set_title("Association of metadata factors with PCs (ANOVA)")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")
    else:
        axes[1].axis("off")
    sns.despine(ax=axes[0])
    fig.tight_layout()
    book.save(fig, SECTION, "pca_variance", "PCA variance and factor association",
              "How much variance each component explains, and which experimental factor drives it. If a "
              "batch factor dominates PC1 while condition appears only in PC4, correct for batch in the model.")

    n_panels = len(factors)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 5.0), squeeze=False)
    for ax, (name, series) in zip(axes[0], factors.items()):
        levels = sorted(series.unique())
        palette = condition_palette(cfg) if name == cfg.input.condition_col else discrete_palette(levels, "Set2")
        for level in levels:
            mask = series.reindex(pcs.index) == level
            ax.scatter(pcs.loc[mask, "PC1"], pcs.loc[mask, "PC2"], s=75, alpha=0.9,
                       color=palette.get(level, "#95a5a6"), edgecolor="white", lw=0.8, label=str(level))
            if mask.sum() > 2:
                _confidence_ellipse(ax, pcs.loc[mask, "PC1"], pcs.loc[mask, "PC2"], palette.get(level, "#95a5a6"))
        ax.set_xlabel(f"PC1 ({variance[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({variance[1]:.1f}%)")
        ax.set_title(f"PCA coloured by {name}")
        ax.legend(title=name, fontsize=8)
        ax.axhline(0, color="#bdc3c7", lw=0.6)
        ax.axvline(0, color="#bdc3c7", lw=0.6)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "pca_scatter", "Principal component analysis",
              "PC1 vs PC2 with 95% confidence ellipses per group. Separation along an axis means the factor "
              "is a major source of transcriptome-wide variation.")

    n_show = min(4, n_components)
    fig, axes = plt.subplots(n_show, n_show, figsize=(2.7 * n_show, 2.7 * n_show))
    condition = meta[cfg.input.condition_col].astype(str).reindex(pcs.index)
    cond_pal = condition_palette(cfg)
    for i in range(n_show):
        for j in range(n_show):
            ax = axes[i, j]
            if i == j:
                for level, color in cond_pal.items():
                    subset = pcs.loc[condition == level, f"PC{i+1}"]
                    if len(subset) > 1:
                        sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.35, lw=1.0)
                ax.set_ylabel("")
            else:
                ax.scatter(pcs[f"PC{j+1}"], pcs[f"PC{i+1}"], s=26,
                           c=condition.map(cond_pal).values, edgecolor="white", lw=0.4)
            if i == n_show - 1:
                ax.set_xlabel(f"PC{j+1} ({variance[j]:.0f}%)", fontsize=8.5)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"PC{i+1} ({variance[i]:.0f}%)", fontsize=8.5)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle("Pairwise principal components (coloured by condition)", fontweight="bold")
    fig.tight_layout()
    book.save(fig, SECTION, "pca_pairs", "PCA component pairs",
              "The condition signal is sometimes carried by a later component; this grid shows PC1-PC4 so a "
              "biological axis hidden behind technical variation is not missed.")
    return pcs


def _confidence_ellipse(ax, x, y, color, n_std: float = 1.96) -> None:
    from matplotlib.patches import Ellipse
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or np.linalg.det(cov) <= 0:
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 0))
    ax.add_patch(Ellipse((x.mean(), y.mean()), width, height, angle=angle,
                         facecolor=color, alpha=0.10, edgecolor=color, lw=1.0, ls="--"))


def embeddings(expr: pd.DataFrame, meta: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    """t-SNE and MDS (plus UMAP when the package is installed)."""
    n_samples = expr.shape[1]
    values = expr.T.values
    condition = meta[cfg.input.condition_col].astype(str)
    cond_pal = condition_palette(cfg)
    stype = _sample_type(meta, cfg)
    type_pal = discrete_palette(sorted(stype.unique()), "Set2") if stype is not None else None

    layouts = {}
    if n_samples >= 8:
        perplexity = float(max(2, min(30, (n_samples - 1) // 3)))
        layouts["t-SNE"] = TSNE(n_components=2, perplexity=perplexity, init="pca",
                                random_state=cfg.project.seed, max_iter=1000).fit_transform(values)
    layouts["MDS"] = MDS(n_components=2, random_state=cfg.project.seed, normalized_stress="auto",
                         n_init=4).fit_transform(values)
    try:
        import umap  # noqa: F401  (optional dependency)
        if n_samples >= 10:
            layouts["UMAP"] = umap.UMAP(n_neighbors=min(15, n_samples - 1), min_dist=0.3,
                                        random_state=cfg.project.seed).fit_transform(values)
    except Exception:
        log.info("UMAP not installed - skipping UMAP embedding (pip install umap-learn to enable)")

    fig, axes = plt.subplots(1, len(layouts), figsize=(5.5 * len(layouts), 5.0), squeeze=False)
    for ax, (name, coords) in zip(axes[0], layouts.items()):
        for level in condition.unique():
            mask = (condition == level).values
            marker_groups = [(None, mask)] if stype is None else [
                (t, mask & (stype == t).values) for t in sorted(stype.unique())
            ]
            for type_name, sel in marker_groups:
                if sel.sum() == 0:
                    continue
                ax.scatter(coords[sel, 0], coords[sel, 1], s=80, alpha=0.9,
                           color=cond_pal.get(level, "#95a5a6"),
                           edgecolor=type_pal[type_name] if type_pal and type_name else "white",
                           linewidth=2.0 if type_pal and type_name else 0.8)
        ax.set_title(f"{name} embedding")
        ax.set_xlabel(f"{name} 1")
        ax.set_ylabel(f"{name} 2")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in cond_pal.items()]
    if type_pal:
        handles += [plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=c,
                               markeredgewidth=2, label=l) for l, c in type_pal.items()]
    axes[0][0].legend(handles=handles, fontsize=8, title="fill = condition / edge = sample type")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "embeddings", "Non-linear sample embeddings",
              "t-SNE / MDS (and UMAP if installed) views of the same expression matrix. Useful for spotting "
              "sub-structure that PCA's linear axes miss.")


def outlier_scores(expr: pd.DataFrame, meta: pd.DataFrame, cfg: Config, book: FigureBook) -> pd.DataFrame:
    corr = expr.corr(method="spearman")
    np.fill_diagonal(corr.values, np.nan)
    median_corr = corr.median(axis=1)
    centroid = expr.mean(axis=1)
    distance = ((expr.sub(centroid, axis=0)) ** 2).sum(axis=0) ** 0.5
    robust_z = (distance - distance.median()) / (1.4826 * (distance - distance.median()).abs().median() or 1)

    order = median_corr.sort_values().index.tolist()
    colors, cond_pal = _sample_colors(meta, cfg)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].barh(range(len(order)), median_corr.loc[order].values, color=colors.loc[order].values, height=0.7)
    threshold = median_corr.median() - 3 * (median_corr - median_corr.median()).abs().median() * 1.4826
    axes[0].axvline(threshold, color="#c0392b", ls="--", lw=1.0, label="outlier threshold")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=6.5)
    axes[0].set_xlabel("Median Spearman correlation to other samples")
    axes[0].set_xlim(min(median_corr.min() - 0.02, threshold - 0.02), 1.0)
    axes[0].set_title("Sample connectivity")
    axes[0].legend(fontsize=8)

    axes[1].scatter(median_corr, robust_z, c=colors.reindex(median_corr.index).values, s=60,
                    edgecolor="white", lw=0.7)
    axes[1].axhline(3, color="#c0392b", ls="--", lw=1.0)
    for sample in robust_z[robust_z > 3].index:
        axes[1].annotate(sample, (median_corr[sample], robust_z[sample]), fontsize=7,
                         xytext=(4, 2), textcoords="offset points")
    axes[1].set_xlabel("Median correlation")
    axes[1].set_ylabel("Robust z of distance to centroid")
    axes[1].set_title("Outlier detection")
    legend_from_map(axes[1], cond_pal, cfg.input.condition_col)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "outlier_detection", "Outlier detection",
              "Two complementary outlier scores. Samples with low connectivity and a robust z above 3 should "
              "be reviewed and possibly removed before differential testing.")
    return pd.DataFrame({"median_correlation": median_corr, "distance_robust_z": robust_z})


def mean_variance_trend(counts: pd.DataFrame, norm_counts: pd.DataFrame, book: FigureBook) -> None:
    mean = norm_counts.mean(axis=1)
    variance = norm_counts.var(axis=1, ddof=1)
    keep = (mean > 0) & (variance > 0)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    axes[0].scatter(np.log10(mean[keep]), np.log10(variance[keep]), s=5, alpha=0.25, color="#34495e",
                    rasterized=True)
    lims = [np.log10(mean[keep]).min(), np.log10(mean[keep]).max()]
    axes[0].plot(lims, lims, color="#c0392b", lw=1.4, label="Poisson (variance = mean)")
    axes[0].set_xlabel("log10 mean normalised count")
    axes[0].set_ylabel("log10 variance")
    axes[0].set_title("Mean-variance relationship")
    axes[0].legend(fontsize=8)

    dispersion = ((variance[keep] - mean[keep]) / mean[keep] ** 2).clip(lower=1e-4)
    axes[1].scatter(np.log10(mean[keep]), np.log10(dispersion), s=5, alpha=0.25, color="#34495e",
                    rasterized=True)
    binned = pd.DataFrame({"m": np.log10(mean[keep]), "d": np.log10(dispersion)})
    bins = pd.qcut(binned["m"], q=min(30, max(3, len(binned) // 50)), duplicates="drop")
    trend = binned.groupby(bins, observed=True)["d"].median()
    centers = [interval.mid for interval in trend.index]
    axes[1].plot(centers, trend.values, color="#e67e22", lw=2.2, label="median trend")
    axes[1].set_xlabel("log10 mean normalised count")
    axes[1].set_ylabel("log10 dispersion")
    axes[1].set_title("Overdispersion trend")
    axes[1].legend(fontsize=8)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "mean_variance", "Mean-variance and dispersion trend",
              "RNA-seq counts are overdispersed relative to Poisson, which is why a negative-binomial model "
              "(DESeq2) is used. Dispersion should decrease with expression.")
    _ = counts


def most_variable_heatmap(expr: pd.DataFrame, meta: pd.DataFrame, cfg: Config, book: FigureBook,
                          n: int = 50) -> None:
    top = expr.var(axis=1).sort_values(ascending=False).head(n).index
    matrix = zscore_rows(expr.loc[top])
    palettes = {cfg.input.condition_col: condition_palette(cfg)}
    stype = _sample_type(meta, cfg)
    if stype is not None:
        palettes[cfg.input.sample_type_col] = discrete_palette(sorted(stype.unique()), "Set2")
    fig = heatmap_with_annotations(matrix, meta, palettes,
                                   f"Top {n} most variable genes (unsupervised)",
                                   cbar_label="row z-score", vmin=-3, vmax=3)
    book.save(fig, SECTION, "most_variable_genes", "Most variable genes",
              "Unsupervised heatmap of the most variable genes. Because no group information is used, "
              "clustering that mirrors the condition annotation is genuine evidence of a strong signal.")


def run_all(counts_raw: pd.DataFrame, counts: pd.DataFrame, norm_counts: pd.DataFrame,
            vst_expr: pd.DataFrame, meta: pd.DataFrame, size_factors: pd.Series,
            filter_summary: pd.DataFrame, gene_names: pd.Series, cfg: Config,
            book: FigureBook) -> Dict[str, pd.DataFrame]:
    """Produce the whole QC figure set; returns tables worth writing to disk."""
    from .normalize import cpm, top_variable_genes

    raw_logcpm = cpm(counts, log2=True)
    norm_log = np.log2(norm_counts + 1)

    design_overview(meta, cfg, book)
    library_size(counts_raw, meta, cfg, book)
    detected_genes(counts_raw, meta, cfg, book)
    filtering_summary(filter_summary, counts_raw, cfg, book)
    count_distributions(raw_logcpm, norm_log, meta, cfg, book)
    density_and_zero_fraction(raw_logcpm, counts_raw, meta, cfg, book)
    rle_plot(norm_log, meta, cfg, book)
    size_factor_plot(size_factors, counts_raw, meta, cfg, book)
    top_expressed_genes(counts_raw, gene_names, book)
    mean_variance_trend(counts, norm_counts, book)

    variable = top_variable_genes(vst_expr, cfg.filtering.top_variable_genes)
    subset = vst_expr.loc[variable]
    correlation_heatmap(subset, meta, cfg, book)
    sample_distance_heatmap(subset, meta, cfg, book)
    dendrogram(subset, meta, cfg, book)
    pcs = pca_figures(subset, meta, cfg, book)
    embeddings(subset, meta, cfg, book)
    outliers = outlier_scores(subset, meta, cfg, book)
    most_variable_heatmap(vst_expr, meta, cfg, book, n=cfg.plots.heatmap_top_n)

    qc_table = pd.DataFrame({
        "library_size": counts_raw.sum(axis=0),
        "genes_detected": (counts_raw > 0).sum(axis=0),
        "zero_fraction": (counts_raw == 0).mean(axis=0),
        "size_factor": size_factors,
    }).join(outliers)
    qc_table = qc_table.join(meta.astype(str))
    return {"sample_qc_metrics": qc_table, "pca_coordinates": pcs}
