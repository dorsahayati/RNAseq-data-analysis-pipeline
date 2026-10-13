"""Functional interpretation: over-representation analysis and pre-ranked GSEA (offline by default)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import Config
from .de import DEResult
from .plotting import FigureBook, discrete_palette, empty_figure
from .utils import bh_fdr, log, sanitize

SECTION = "enrichment"


def read_gmt(path: Path) -> Dict[str, List[str]]:
    gene_sets: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = [p.strip() for p in line.rstrip("\n").split("\t") if p.strip()]
            if len(parts) < 3:
                continue
            gene_sets[parts[0]] = sorted(set(parts[2:]))
    return gene_sets


def over_representation(query: Sequence[str], background: Sequence[str],
                        gene_sets: Dict[str, List[str]], min_size: int = 5,
                        max_size: int = 2000) -> pd.DataFrame:
    """Hypergeometric over-representation of a gene list within a background."""
    background = set(background)
    query = set(query) & background
    rows = []
    for name, members in gene_sets.items():
        members_in_background = set(members) & background
        if not (min_size <= len(members_in_background) <= max_size):
            continue
        overlap = query & members_in_background
        if not overlap:
            continue
        pvalue = stats.hypergeom.sf(len(overlap) - 1, len(background),
                                    len(members_in_background), len(query))
        expected = len(query) * len(members_in_background) / len(background)
        rows.append({
            "term": name,
            "n_genes_in_set": len(members_in_background),
            "n_overlap": len(overlap),
            "expected_overlap": expected,
            "fold_enrichment": len(overlap) / expected if expected else np.nan,
            "pvalue": pvalue,
            "genes": ";".join(sorted(overlap)),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["padj"] = bh_fdr(table["pvalue"])
    return table.sort_values("pvalue").reset_index(drop=True)


def _running_enrichment(ranked_genes: np.ndarray, hits: np.ndarray, weights: np.ndarray):
    hit_weights = np.abs(weights) * hits
    total_hit = hit_weights.sum()
    if total_hit <= 0:
        return np.zeros(len(hits)), 0.0
    miss_step = 1.0 / max((~hits.astype(bool)).sum(), 1)
    running = np.cumsum(np.where(hits.astype(bool), hit_weights / total_hit, -miss_step))
    peak = running[np.argmax(np.abs(running))]
    _ = ranked_genes
    return running, float(peak)


def gsea_prerank(ranking: pd.Series, gene_sets: Dict[str, List[str]], n_perm: int = 1000,
                 min_size: int = 10, max_size: int = 500, seed: int = 0) -> Tuple[pd.DataFrame, Dict]:
    """Pre-ranked GSEA (weighted Kolmogorov-Smirnov statistic with a permutation null).

    Gene labels are permuted, which is the appropriate null when the number of samples is
    too small for phenotype permutation.
    """
    ranking = ranking.dropna().sort_values(ascending=False)
    genes = ranking.index.to_numpy()
    weights = ranking.to_numpy(dtype=float)
    index = {gene: i for i, gene in enumerate(genes)}
    rng = np.random.default_rng(seed)

    rows, curves = [], {}
    for name, members in gene_sets.items():
        positions = np.array([index[g] for g in members if g in index], dtype=int)
        if not (min_size <= len(positions) <= max_size):
            continue
        hits = np.zeros(len(genes), dtype=float)
        hits[positions] = 1.0
        running, observed = _running_enrichment(genes, hits, weights)

        null = np.empty(n_perm)
        for i in range(n_perm):
            permuted = np.zeros(len(genes), dtype=float)
            permuted[rng.choice(len(genes), size=len(positions), replace=False)] = 1.0
            null[i] = _running_enrichment(genes, permuted, weights)[1]
        same_sign = null[null >= 0] if observed >= 0 else null[null < 0]
        scale = np.abs(same_sign).mean() if same_sign.size else np.nan
        nes = observed / scale if scale and np.isfinite(scale) and scale > 0 else np.nan
        pvalue = ((np.sum(same_sign >= observed) + 1) / (same_sign.size + 1)) if observed >= 0 else \
                 ((np.sum(same_sign <= observed) + 1) / (same_sign.size + 1))
        leading = genes[positions[positions <= np.argmax(np.abs(running))]] if observed >= 0 else \
            genes[positions[positions >= np.argmax(np.abs(running))]]
        rows.append({
            "term": name,
            "set_size": len(positions),
            "ES": observed,
            "NES": nes,
            "pvalue": pvalue,
            "leading_edge_size": len(leading),
            "leading_edge": ";".join(map(str, leading[:60])),
        })
        curves[name] = (running, positions)
    table = pd.DataFrame(rows)
    if table.empty:
        return table, curves
    table["padj"] = bh_fdr(table["pvalue"])
    return table.sort_values("pvalue").reset_index(drop=True), curves


def enrichr_online(query: Sequence[str], gene_sets: Sequence[str], organism: str) -> Optional[pd.DataFrame]:
    try:
        import gseapy
    except ImportError:
        log.warning("gseapy is not installed - skipping the online Enrichr query")
        return None
    try:
        result = gseapy.enrichr(gene_list=list(query), gene_sets=list(gene_sets),
                                organism=organism, outdir=None, no_plot=True)
        table = result.results.rename(columns={
            "Term": "term", "P-value": "pvalue", "Adjusted P-value": "padj",
            "Odds Ratio": "fold_enrichment", "Genes": "genes", "Gene_set": "library",
        })
        table["n_overlap"] = table["Overlap"].str.split("/").str[0].astype(int)
        return table.sort_values("pvalue").reset_index(drop=True)
    except Exception as exc:
        log.warning("Enrichr query failed (%s) - falling back to the offline gene sets", exc)
        return None


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

def ora_dotplot(tables: Dict[str, pd.DataFrame], cfg: Config, book: FigureBook) -> None:
    frames = []
    for direction, table in tables.items():
        if table is None or table.empty:
            continue
        subset = table.head(cfg.enrichment.top_terms).copy()
        subset["direction"] = direction
        frames.append(subset)
    if not frames:
        return
    data = pd.concat(frames)
    directions = list(dict.fromkeys(data["direction"]))
    fig, axes = plt.subplots(1, len(directions), figsize=(7.2 * len(directions), 
                                                          max(4.6, 0.34 * cfg.enrichment.top_terms + 1.6)),
                             squeeze=False)
    for ax, direction in zip(axes[0], directions):
        subset = data[data["direction"] == direction].sort_values("pvalue", ascending=False)
        y = range(len(subset))
        scatter = ax.scatter(-np.log10(subset["pvalue"].clip(lower=1e-300)), y,
                             s=30 + 14 * subset["n_overlap"], c=subset["fold_enrichment"],
                             cmap="plasma", edgecolor="#2c3e50", lw=0.5)
        ax.set_yticks(list(y))
        ax.set_yticklabels([t.replace("_", " ").title()[:52] for t in subset["term"]], fontsize=8)
        ax.axvline(-np.log10(0.05), color="#c0392b", ls="--", lw=1.0)
        ax.set_xlabel("-log10 p-value")
        ax.set_title(f"Over-representation: {direction}")
        fig.colorbar(scatter, ax=ax, label="fold enrichment", pad=0.02)
    fig.suptitle("Pathway over-representation of the differentially expressed genes",
                 fontweight="bold", y=1.0)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "ora_dotplot", "Over-representation analysis",
              "Pathways containing more DE genes than expected by chance. Point size is the number of DE "
              "genes in the pathway and colour the enrichment over background; the dashed line is p = 0.05.")


def gsea_barplot(table: pd.DataFrame, cfg: Config, book: FigureBook) -> None:
    if table is None or table.empty:
        return
    data = table.dropna(subset=["NES"]).copy()
    data = data.reindex(data["NES"].abs().sort_values(ascending=False).index).head(cfg.enrichment.top_terms)
    data = data.sort_values("NES")
    fig, ax = plt.subplots(figsize=(8.6, max(4.8, 0.36 * len(data) + 1.6)))
    colors = ["#c0392b" if v > 0 else "#2874a6" for v in data["NES"]]
    ax.barh(range(len(data)), data["NES"], color=colors, height=0.72)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([t.replace("_", " ").title()[:52] for t in data["term"]], fontsize=8)
    ax.axvline(0, color="#2c3e50", lw=1.0)
    for i, (_, row) in enumerate(data.iterrows()):
        star = "*" if row["padj"] < 0.05 else ""
        ax.text(row["NES"] + 0.05 * np.sign(row["NES"]), i,
                f"FDR {row['padj']:.3f}{star}", va="center", fontsize=6.8,
                ha="left" if row["NES"] > 0 else "right", color="#555555")
    ax.set_xlabel("Normalised enrichment score")
    ax.set_title(f"Pre-ranked GSEA (positive = enriched in {cfg.input.case_label})")
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "gsea_nes", "Pre-ranked GSEA",
              "GSEA uses the whole ranked gene list, so it detects pathways where many genes shift slightly - "
              "coordinated changes that a hard significance cut-off would discard.")


def gsea_running_plots(table: pd.DataFrame, curves: Dict, ranking: pd.Series, book: FigureBook,
                       n: int = 4) -> None:
    if table is None or table.empty or not curves:
        return
    selected = table.dropna(subset=["NES"])
    selected = selected.reindex(selected["NES"].abs().sort_values(ascending=False).index).head(n)
    fig, axes = plt.subplots(2, len(selected), figsize=(4.6 * len(selected), 6.0), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    ordered = ranking.sort_values(ascending=False)
    for column, (_, row) in enumerate(selected.iterrows()):
        running, positions = curves[row["term"]]
        color = "#c0392b" if row["NES"] > 0 else "#2874a6"
        axes[0][column].plot(running, color=color, lw=1.8)
        axes[0][column].axhline(0, color="#2c3e50", lw=0.8)
        peak = int(np.argmax(np.abs(running)))
        axes[0][column].axvline(peak, color="#7f8c8d", ls=":", lw=1.0)
        axes[0][column].set_title(f"{row['term'].replace('_', ' ').title()[:40]}\n"
                                  f"NES {row['NES']:.2f}, FDR {row['padj']:.3f}", fontsize=9)
        axes[0][column].set_ylabel("Running enrichment score" if column == 0 else "")
        axes[1][column].vlines(positions, 0, 1, color=color, lw=0.6)
        axes[1][column].plot(np.arange(len(ordered)), (ordered.values - ordered.values.min()) /
                             (np.ptp(ordered.values) or 1), color="#bdc3c7", lw=0.8)
        axes[1][column].set_ylim(0, 1)
        axes[1][column].set_yticks([])
        axes[1][column].set_xlabel("Gene rank")
        axes[1][column].grid(False)
    fig.suptitle("GSEA enrichment profiles for the leading pathways", fontweight="bold", y=1.0)
    sns.despine(fig=fig)
    fig.tight_layout()
    book.save(fig, SECTION, "gsea_running_score", "GSEA enrichment profiles",
              "The running enrichment score along the ranked gene list; ticks mark the pathway members. "
              "An early, sharp peak means the pathway sits at the very top of the ranking.")


def pathway_across_sample_types(results: Dict[str, DEResult], gene_sets: Dict[str, List[str]],
                                cfg: Config, book: FigureBook, n_perm: int = 300) -> Optional[pd.DataFrame]:
    """Compare pathway-level activity between sample types."""
    per_type = {name: result for name, result in results.items() if name != "all_samples"}
    if len(per_type) < 2:
        return None
    frames = {}
    for name, result in per_type.items():
        ranking = _ranking_from_result(result)
        table, _ = gsea_prerank(ranking, gene_sets, n_perm=n_perm, seed=cfg.project.seed)
        if not table.empty:
            frames[name] = table.set_index("term")["NES"]
    if len(frames) < 2:
        return None
    matrix = pd.DataFrame(frames).dropna(how="all")
    matrix = matrix.reindex(matrix.abs().max(axis=1).sort_values(ascending=False).index).head(30)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * matrix.shape[1] + 4.5),
                                    max(6.0, 0.32 * len(matrix) + 2.0)))
    limit = float(np.nanmax(np.abs(matrix.values))) or 1.0
    sns.heatmap(matrix, ax=ax, cmap="RdBu_r", center=0, vmin=-limit, vmax=limit, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Normalised enrichment score"})
    ax.set_yticklabels([t.get_text().replace("_", " ").title()[:48] for t in ax.get_yticklabels()],
                       fontsize=8, rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_title("Pathway activity per sample type")
    ax.set_xlabel("")
    fig.tight_layout()
    book.save(fig, SECTION, "pathway_by_sample_type", "Pathway activity across sample types",
              "Pathway-level view of the sample-type comparison. Pathways with the same sign everywhere "
              "describe biology that is detectable regardless of the material sampled.")
    return matrix


def _ranking_from_result(result: DEResult) -> pd.Series:
    table = result.table
    stat = table["stat"] if table["stat"].notna().any() else \
        np.sign(table["log2FoldChange"]) * -np.log10(table["pvalue"].clip(lower=1e-300))
    ranking = pd.Series(np.asarray(stat, dtype=float), index=table["gene_name"].astype(str).values)
    ranking = ranking.groupby(level=0).max()
    return ranking.dropna()


def run_all(results: Dict[str, DEResult], cfg: Config, book: FigureBook) -> Dict[str, pd.DataFrame]:
    if not cfg.enrichment.enable:
        return {}
    gmt_path = cfg.resolve(cfg.enrichment.offline_gmt)
    gene_sets: Dict[str, List[str]] = {}
    if gmt_path and Path(gmt_path).exists():
        gene_sets = read_gmt(Path(gmt_path))
        log.info("loaded %d gene sets from %s", len(gene_sets), Path(gmt_path).name)

    overall = results["all_samples"]
    background = overall.table["gene_name"].astype(str).tolist()
    up = overall.table.loc[overall.up, "gene_name"].astype(str).tolist()
    down = overall.table.loc[overall.down, "gene_name"].astype(str).tolist()

    tables: Dict[str, pd.DataFrame] = {}
    if not gene_sets and not cfg.enrichment.online:
        book.save(empty_figure("No gene sets available for enrichment.\n"
                               "Point enrichment.offline_gmt at an MSigDB .gmt file,\n"
                               "or set enrichment.online: true to query Enrichr."),
                  SECTION, "not_available", "Enrichment unavailable",
                  "No gene-set resource was configured.")
        return tables

    if gene_sets:
        ora = {
            f"up in {cfg.input.case_label}": over_representation(up, background, gene_sets),
            f"down in {cfg.input.case_label}": over_representation(down, background, gene_sets),
        }
        ora_dotplot(ora, cfg, book)
        for direction, table in ora.items():
            if table is not None and not table.empty:
                tables[f"enrichment_ora_{sanitize(direction)}"] = table

        ranking = _ranking_from_result(overall)
        gsea, curves = gsea_prerank(ranking, gene_sets, n_perm=1000, seed=cfg.project.seed)
        if not gsea.empty:
            gsea_barplot(gsea, cfg, book)
            gsea_running_plots(gsea, curves, ranking, book)
            tables["enrichment_gsea_all_samples"] = gsea
        matrix = pathway_across_sample_types(results, gene_sets, cfg, book)
        if matrix is not None:
            tables["enrichment_pathway_nes_by_sample_type"] = matrix

    if cfg.enrichment.online:
        online = enrichr_online(up, cfg.enrichment.gene_sets, cfg.enrichment.organism)
        if online is not None and not online.empty:
            tables["enrichment_enrichr_up"] = online
            ora_dotplot({"Enrichr (up-regulated)": online}, cfg, book)
    return tables
