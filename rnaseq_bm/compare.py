"""Compare the cancer-vs-control signal between sample types: overlap, concordance, meta-analysis."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

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
    discrete_palette,
    draw_upset,
    draw_venn,
    empty_figure,
)
from .utils import bh_fdr, jaccard, log

SECTION = "compare"


def sample_type_results(results: Dict[str, DEResult]) -> Dict[str, DEResult]:
    return {name: result for name, result in results.items() if name != "all_samples"}


def build_matrices(results: Dict[str, DEResult]) -> Dict[str, pd.DataFrame]:
    """Gene x sample-type matrices of log2FC, standard error, p-value and FDR."""
    matrices = {}
    for field, column in [("lfc", "log2FoldChange"), ("se", "lfcSE"),
                          ("pvalue", "pvalue"), ("padj", "padj"), ("baseMean", "baseMean")]:
        matrices[field] = pd.DataFrame({name: r.table[column] for name, r in results.items()})
    matrices["gene_name"] = next(iter(results.values())).table["gene_name"]
    return matrices


def signature_sets(results: Dict[str, DEResult]) -> Dict[str, Dict[str, set]]:
    return {
        "all": {name: set(r.significant.index) for name, r in results.items()},
        "up": {name: set(r.up) for name, r in results.items()},
        "down": {name: set(r.down) for name, r in results.items()},
    }


def meta_analysis(results: Dict[str, DEResult], cfg: Config) -> pd.DataFrame:
    """Stouffer's weighted-Z combination of the per-sample-type contrasts."""
    matrices = build_matrices(results)
    lfc, pval = matrices["lfc"], matrices["pvalue"]
    weights = pd.Series({name: np.sqrt(r.n_case + r.n_control) for name, r in results.items()})

    z = pd.DataFrame(
        stats.norm.isf(np.clip(pval.to_numpy(dtype=float) / 2.0, 1e-300, 0.5)),
        index=pval.index, columns=pval.columns,
    ) * np.sign(lfc)
    usable = z.notna() & lfc.notna()
    w = pd.DataFrame(np.tile(weights[z.columns].to_numpy(), (len(z), 1)), index=z.index, columns=z.columns)
    w = w.where(usable)
    combined_z = (z * w).sum(axis=1, min_count=1) / np.sqrt((w ** 2).sum(axis=1, min_count=1))
    combined_p = 2 * stats.norm.sf(np.abs(combined_z))

    same_direction = np.sign(lfc).where(lfc.notna())
    n_types = usable.sum(axis=1)
    concordant = same_direction.eq(np.sign(combined_z), axis=0).where(usable).sum(axis=1)

    table = pd.DataFrame({
        "gene_name": matrices["gene_name"].reindex(lfc.index),
        "mean_log2FC": lfc.mean(axis=1),
        "min_abs_log2FC": lfc.abs().min(axis=1),
        "sd_log2FC": lfc.std(axis=1),
        "n_sample_types": n_types,
        "n_concordant": concordant,
        "combined_z": combined_z,
        "combined_pvalue": combined_p,
        "combined_padj": bh_fdr(combined_p),
    }, index=lfc.index)
    table["consistent"] = (table["n_concordant"] == table["n_sample_types"]) & (table["n_sample_types"] > 1)
    significant_in = pd.DataFrame({name: r.table["significant"] for name, r in results.items()})
    table["n_types_significant"] = significant_in.sum(axis=1)
    table["significant_in"] = significant_in.apply(lambda row: ",".join(row.index[row.fillna(False)]), axis=1)
    table["core_marker"] = (
        (table["n_types_significant"] == len(results)) & table["consistent"]
        & (table["combined_padj"] < cfg.analysis.alpha)
    )
    return table.sort_values("combined_pvalue")


def concordance_stats(results: Dict[str, DEResult]) -> pd.DataFrame:
    """Pairwise agreement between sample types."""
    matrices = build_matrices(results)
    lfc = matrices["lfc"]
    sets = signature_sets(results)["all"]
    rows = []
    for a, b in combinations(lfc.columns, 2):
        pair = lfc[[a, b]].dropna()
        union_significant = sorted(sets[a] | sets[b])
        subset = lfc.loc[[g for g in union_significant if g in lfc.index], [a, b]].dropna()
        same_sign = float((np.sign(subset[a]) == np.sign(subset[b])).mean()) if len(subset) else np.nan
        rows.append({
            "sample_type_a": a,
            "sample_type_b": b,
            "pearson_r_all_genes": pair[a].corr(pair[b]) if len(pair) > 2 else np.nan,
            "spearman_r_all_genes": pair[a].corr(pair[b], method="spearman") if len(pair) > 2 else np.nan,
            "pearson_r_de_genes": subset[a].corr(subset[b]) if len(subset) > 2 else np.nan,
            "sign_concordance_de_genes": same_sign,
            "n_de_a": len(sets[a]),
            "n_de_b": len(sets[b]),
            "n_shared_de": len(sets[a] & sets[b]),
            "jaccard": jaccard(sets[a], sets[b]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

def overlap_figures(results: Dict[str, DEResult], cfg: Config, book: FigureBook) -> None:
    sets = signature_sets(results)
    names = list(results)
    if 2 <= len(names) <= 3:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
        for ax, key, title in zip(axes, ["all", "up", "down"],
                                  ["All significant genes",
                                   f"Up in {cfg.input.case_label}",
                                   f"Down in {cfg.input.case_label}"]):
            draw_venn(ax, sets[key])
            ax.set_title(title)
        fig.suptitle("Overlap of differentially expressed genes between sample types",
                     fontweight="bold", y=1.02)
        fig.tight_layout()
        book.save(fig, SECTION, "venn_overlap", "DE gene overlap between sample types",
                  "Genes reaching significance in each sample type. The central region is the shared core "
                  "response - the most promising source of markers that transfer between materials.")
    for key, title in [("all", "all significant genes"), ("up", "up-regulated"), ("down", "down-regulated")]:
        if len(names) < 2 or not any(sets[key].values()):
            continue
        fig = plt.figure(figsize=(max(8.0, 0.55 * 20 + 3), 5.6))
        draw_upset(fig, sets[key], max_intersections=20)
        fig.suptitle(f"UpSet: {title} across sample types", fontweight="bold", y=1.0)
        book.save(fig, SECTION, f"upset_{key}", f"UpSet plot ({title})",
                  "Exclusive intersections of the per-sample-type gene sets, ordered by size. Reads more "
                  "reliably than a Venn diagram when there are more than three groups.")


def lfc_scatter_matrix(results: Dict[str, DEResult], cfg: Config, book: FigureBook) -> None:
    matrices = build_matrices(results)
    lfc = matrices["lfc"]
    names = list(lfc.columns)
    if len(names) < 2:
        return
    sets = signature_sets(results)["all"]
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.2 * n), squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(lfc[names[i]].dropna(), bins=60, color="#5d6d7e")
                ax.set_title(names[i], fontsize=10, fontweight="bold")
                ax.set_yticks([])
                continue
            if j > i:
                ax.axis("off")
                pair = lfc[[names[j], names[i]]].dropna()
                if len(pair) > 2:
                    r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                    rho = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
                    ax.text(0.5, 0.5, f"r = {r:.2f}\nrho = {rho:.2f}", ha="center", va="center",
                            fontsize=14, color="#c0392b" if r > 0.5 else "#2c3e50")
                continue
            x, y = lfc[names[j]], lfc[names[i]]
            shared = sets[names[i]] & sets[names[j]]
            either = (sets[names[i]] | sets[names[j]]) - shared
            ax.axhline(0, color="#bdc3c7", lw=0.7)
            ax.axvline(0, color="#bdc3c7", lw=0.7)
            ax.scatter(x, y, s=5, color="#d5d8dc", alpha=0.5, rasterized=True)
            idx = [g for g in either if g in x.index]
            ax.scatter(x.loc[idx], y.loc[idx], s=9, color="#f39c12", alpha=0.75, label="DE in one")
            idx = [g for g in shared if g in x.index]
            ax.scatter(x.loc[idx], y.loc[idx], s=12, color="#c0392b", alpha=0.85, label="DE in both")
            limit = np.nanpercentile(np.abs(pd.concat([x, y])), 99.5)
            ax.plot([-limit, limit], [-limit, limit], ls="--", color="#2c3e50", lw=0.8)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            if j == 0:
                ax.set_ylabel(f"log2FC {names[i]}", fontsize=9)
            if i == n - 1:
                ax.set_xlabel(f"log2FC {names[j]}", fontsize=9)
            if i == 1 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Fold-change concordance between sample types", fontweight="bold", y=1.0)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "lfc_scatter_matrix", "Fold-change concordance",
              "Pairwise log2 fold changes with the identity line. Points on the diagonal behave the same way "
              "in both materials; points in the off-diagonal quadrants change direction and are unreliable markers.")


def correlation_and_jaccard(results: Dict[str, DEResult], concordance: pd.DataFrame,
                            book: FigureBook) -> None:
    lfc = build_matrices(results)["lfc"]
    names = list(lfc.columns)
    if len(names) < 2:
        return
    corr = lfc.corr(method="spearman")
    sets = signature_sets(results)["all"]
    jac = pd.DataFrame(
        [[1.0 if a == b else jaccard(sets[a], sets[b]) for b in names] for a in names],
        index=names, columns=names,
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    sns.heatmap(corr, ax=axes[0], cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f",
                linewidths=0.6, linecolor="white", cbar_kws={"label": "Spearman rho"})
    axes[0].set_title("Genome-wide fold-change correlation")
    sns.heatmap(jac, ax=axes[1], cmap="Purples", vmin=0, vmax=1, annot=True, fmt=".2f",
                linewidths=0.6, linecolor="white", cbar_kws={"label": "Jaccard index"})
    axes[1].set_title("DE gene-set overlap (Jaccard)")
    if not concordance.empty:
        labels = concordance["sample_type_a"] + "\nvs " + concordance["sample_type_b"]
        axes[2].bar(range(len(concordance)), concordance["sign_concordance_de_genes"] * 100,
                    color="#16a085", width=0.6)
        axes[2].axhline(50, color="#c0392b", ls="--", lw=1.0, label="chance level")
        axes[2].set_xticks(range(len(concordance)))
        axes[2].set_xticklabels(labels, fontsize=8)
        axes[2].set_ylabel("% of DE genes with the same direction")
        axes[2].set_ylim(0, 105)
        axes[2].set_title("Directional agreement")
        axes[2].legend(fontsize=8)
        for i, value in enumerate(concordance["sign_concordance_de_genes"] * 100):
            if np.isfinite(value):
                axes[2].text(i, value, f"{value:.0f}%", ha="center", va="bottom", fontsize=9)
    else:
        axes[2].axis("off")
    fig.tight_layout()
    book.save(fig, SECTION, "concordance_summary", "Sample-type concordance summary",
              "Three views of how similar the cancer signal is between sample types: genome-wide correlation "
              "of effects, overlap of the significant gene lists, and the share of DE genes that move in the "
              "same direction.")


def rank_concordance_curve(results: Dict[str, DEResult], book: FigureBook, max_k: int = 2000) -> None:
    matrices = build_matrices(results)
    pval = matrices["pvalue"]
    names = list(pval.columns)
    if len(names) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ks = np.unique(np.linspace(10, min(max_k, len(pval)), 60).astype(int))
    for a, b in combinations(names, 2):
        rank_a = pval[a].rank(na_option="bottom")
        rank_b = pval[b].rank(na_option="bottom")
        overlaps = [len(set(rank_a.nsmallest(k).index) & set(rank_b.nsmallest(k).index)) / k for k in ks]
        ax.plot(ks, overlaps, lw=2.0, label=f"{a} vs {b}")
    ax.plot(ks, ks / len(pval), ls="--", color="#2c3e50", lw=1.2, label="expected by chance")
    ax.set_xscale("log")
    ax.set_xlabel("Top k genes (by p-value)")
    ax.set_ylabel("Fraction of the top-k lists shared")
    ax.set_ylim(0, 1)
    ax.set_title("Rank concordance between sample types")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "rank_concordance", "Rank concordance",
              "Agreement of the ranked gene lists at increasing list depth, independent of any significance "
              "cutoff. Curves far above the chance line mean the same biology is detected in both materials.")


def lfc_heatmap(meta_table: pd.DataFrame, results: Dict[str, DEResult], cfg: Config,
                book: FigureBook, n: int = 40) -> None:
    lfc = build_matrices(results)["lfc"]
    if lfc.shape[1] < 2:
        return
    ranked = meta_table.sort_values("combined_pvalue").head(n)
    matrix = lfc.reindex(ranked.index)
    matrix.index = ranked["gene_name"].values
    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * matrix.shape[1] + 3.2), max(6.0, 0.24 * len(matrix) + 1.8)))
    limit = float(np.nanpercentile(np.abs(matrix.values), 99)) or 1.0
    sns.heatmap(matrix, ax=ax, cmap="RdBu_r", center=0, vmin=-limit, vmax=limit,
                annot=matrix.shape[0] <= 45, fmt=".1f", annot_kws={"fontsize": 6.5},
                linewidths=0.4, linecolor="white", cbar_kws={"label": "log2 fold change"})
    ax.set_title(f"Fold change of the top {len(matrix)} meta-analysis genes across sample types")
    ax.set_xlabel("")
    plt.setp(ax.get_yticklabels(), fontsize=7.5, rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    book.save(fig, SECTION, "lfc_heatmap", "Fold change across sample types",
              "Effect sizes of the strongest combined hits in each material. Rows with the same colour "
              "everywhere are material-independent; rows that switch colour are context-specific.")


def meta_volcano(meta_table: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    table = meta_table.dropna(subset=["combined_pvalue", "mean_log2FC"])
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    y = -np.log10(table["combined_padj"].clip(lower=1e-300))
    colors = np.where(table["core_marker"], "#8e44ad",
                      np.where(table["combined_padj"] < cfg.analysis.alpha, "#f39c12", "#b3b6b7"))
    ax.scatter(table["mean_log2FC"], y, c=colors, s=13, alpha=0.7, edgecolor="none", rasterized=True)
    ax.axhline(-np.log10(cfg.analysis.alpha), color="#2c3e50", ls="--", lw=0.9)
    ax.set_xlabel("Mean log2 fold change across sample types")
    ax.set_ylabel("-log10 combined FDR (Stouffer)")
    ax.set_title("Meta-analysis across sample types")
    core = table[table["core_marker"]]
    annotate_points(ax, core["mean_log2FC"], y.reindex(core.index), core["gene_name"],
                    n=cfg.plots.label_top_n,
                    score=y.reindex(core.index) * core["min_abs_log2FC"].abs())
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in
               [("core marker (significant and concordant everywhere)", "#8e44ad"),
                ("significant in the meta-analysis", "#f39c12"), ("not significant", "#b3b6b7")]]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "meta_volcano", "Cross-sample-type meta-analysis",
              "Sample types combined with Stouffer's weighted-Z method. Purple genes are significant in "
              "every material with a consistent direction and are the primary biomarker candidates.")


def forest_plot(meta_table: pd.DataFrame, results: Dict[str, DEResult], cfg: Config,
                book: FigureBook, n: int = 15) -> None:
    matrices = build_matrices(results)
    lfc, se = matrices["lfc"], matrices["se"]
    if lfc.shape[1] < 2:
        return
    candidates = meta_table[meta_table["consistent"]].sort_values("combined_pvalue").head(n)
    if candidates.empty:
        candidates = meta_table.sort_values("combined_pvalue").head(n)
    names = list(lfc.columns)
    palette = discrete_palette(names, "Dark2")
    fig, ax = plt.subplots(figsize=(8.0, max(5.5, 0.55 * len(candidates) + 1.6)))
    offsets = np.linspace(-0.28, 0.28, len(names))
    for gene_pos, gene in enumerate(candidates.index):
        for offset, name in zip(offsets, names):
            estimate = lfc.loc[gene, name]
            error = se.loc[gene, name]
            if not np.isfinite(estimate):
                continue
            error = 1.96 * error if np.isfinite(error) else 0.0
            ax.errorbar(estimate, gene_pos + offset, xerr=error, fmt="o", ms=5, lw=1.4,
                        color=palette[name], capsize=2.5,
                        label=name if gene_pos == 0 else None)
        ax.axhline(gene_pos + 0.5, color="#ecf0f1", lw=0.8)
    ax.axvline(0, color="#2c3e50", lw=1.0)
    ax.set_yticks(range(len(candidates)))
    ax.set_yticklabels(candidates["gene_name"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"log2 fold change ({cfg.input.case_label} / {cfg.input.control_label}) with 95% CI")
    ax.set_title("Effect size per sample type for the top consistent genes")
    ax.legend(fontsize=8, title="sample type")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "forest_plot", "Per-sample-type effect sizes",
              "Forest plot of the leading consistent genes. Overlapping confidence intervals across materials "
              "support a marker that behaves the same regardless of sample type.")


def specificity_barplot(results: Dict[str, DEResult], meta_table: pd.DataFrame,
                        book: FigureBook) -> None:
    if len(results) < 2:
        return
    sets = signature_sets(results)["all"]
    names = list(results)
    rows = []
    for name in names:
        others = set().union(*(sets[o] for o in names if o != name))
        rows.append({"sample_type": name, "shared": len(sets[name] & others),
                     "specific": len(sets[name] - others)})
    frame = pd.DataFrame(rows).set_index("sample_type")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    frame.plot(kind="bar", stacked=True, ax=axes[0], color=["#16a085", "#e67e22"], width=0.66,
               edgecolor="white")
    for container in axes[0].containers:
        axes[0].bar_label(container, label_type="center", fontsize=8, color="white")
    axes[0].set_ylabel("Significant genes")
    axes[0].set_xlabel("")
    axes[0].set_title("Shared versus sample-type-specific DE genes")
    axes[0].legend(["shared with another sample type", "specific to this sample type"], fontsize=8)
    plt.setp(axes[0].get_xticklabels(), rotation=0)

    counts = meta_table["n_types_significant"].value_counts().sort_index()
    counts = counts[counts.index > 0]
    axes[1].bar(counts.index.astype(int), counts.values, color="#8e44ad", width=0.6)
    for x, value in zip(counts.index.astype(int), counts.values):
        axes[1].text(x, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xlabel("Number of sample types where the gene is significant")
    axes[1].set_ylabel("Genes")
    axes[1].set_xticks(counts.index.astype(int))
    axes[1].set_title("Reproducibility of the DE calls")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "shared_vs_specific", "Shared vs specific signal",
              "How much of each sample type's signature is reproduced elsewhere. Genes detected in every "
              "material are the robust core; single-material genes may be real biology or noise.")


def run_all(results: Dict[str, DEResult], cfg: Config, book: FigureBook) -> Dict[str, pd.DataFrame]:
    """Run the whole cross-sample-type comparison; returns tables for export."""
    per_type = sample_type_results(results)
    if len(per_type) < 2:
        book.save(
            empty_figure(
                "Cross-sample-type comparison needs at least two sample types with\n"
                "enough cancer and control samples in each. Add a 'sample_type' column\n"
                "to the metadata (for example tissue / plasma / whole_blood) to enable it."
            ),
            SECTION, "not_available", "Cross-sample-type comparison unavailable",
            "Only one sample type could be analysed, so this section is empty.",
        )
        log.warning("fewer than two sample types available - skipping the comparison section")
        return {}

    meta_table = meta_analysis(per_type, cfg)
    concordance = concordance_stats(per_type)

    overlap_figures(per_type, cfg, book)
    lfc_scatter_matrix(per_type, cfg, book)
    correlation_and_jaccard(per_type, concordance, book)
    rank_concordance_curve(per_type, book)
    lfc_heatmap(meta_table, per_type, cfg, book)
    meta_volcano(meta_table, cfg, book)
    forest_plot(meta_table, per_type, cfg, book)
    specificity_barplot(per_type, meta_table, book)

    log.info("cross-sample-type: %d core markers significant and concordant in all %d sample types",
             int(meta_table["core_marker"].sum()), len(per_type))
    lfc = build_matrices(per_type)["lfc"]
    lfc.columns = [f"log2FC_{c}" for c in lfc.columns]
    return {
        "cross_sample_type_meta_analysis": meta_table.join(lfc),
        "sample_type_concordance": concordance,
    }
