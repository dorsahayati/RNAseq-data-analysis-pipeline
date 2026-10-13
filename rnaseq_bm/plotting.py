"""Figure styling, a registry that saves/catalogues every figure, and custom plot primitives."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from .config import Config
from .utils import ensure_dir, log, sanitize

SECTIONS = {
    "qc": "01_quality_control",
    "de": "02_differential_expression",
    "compare": "03_sample_type_comparison",
    "biomarker": "04_biomarkers",
    "enrichment": "05_enrichment",
}

SECTION_TITLES = {
    "qc": "Quality control and exploratory analysis",
    "de": "Differential expression: cancer vs control",
    "compare": "Cross sample-type comparison",
    "biomarker": "Biomarker discovery and evaluation",
    "enrichment": "Functional enrichment",
}


def setup_style(cfg: Config) -> None:
    sns.set_theme(context=cfg.plots.context, style="ticks", font_scale=0.95)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "legend.frameon": False,
        "figure.autolayout": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


@dataclass
class FigureRecord:
    section: str
    index: int
    name: str
    title: str
    caption: str
    paths: Dict[str, Path]

    @property
    def primary(self) -> Path:
        return self.paths.get("png", next(iter(self.paths.values())))


class FigureBook:
    """Saves figures in section folders with a running index and collects captions."""

    def __init__(self, outdir: Path, cfg: Config):
        self.root = ensure_dir(Path(outdir) / "figures")
        self.cfg = cfg
        self.records: List[FigureRecord] = []
        self._counters: Dict[str, int] = {key: 0 for key in SECTIONS}

    def save(self, fig, section: str, name: str, title: str, caption: str = "", close: bool = True) -> FigureRecord:
        if section not in SECTIONS:
            raise KeyError(f"unknown figure section '{section}'")
        self._counters[section] += 1
        index = self._counters[section]
        folder = ensure_dir(self.root / SECTIONS[section])
        stem = f"{index:02d}_{sanitize(name)}"
        paths: Dict[str, Path] = {}
        for fmt in self.cfg.plots.formats:
            path = folder / f"{stem}.{fmt}"
            fig.savefig(path, dpi=self.cfg.plots.dpi, bbox_inches="tight", format=fmt)
            paths[fmt] = path
        if close:
            plt.close(fig)
        record = FigureRecord(section, index, stem, title, caption, paths)
        self.records.append(record)
        log.info("figure %-12s %s", f"[{section}]", stem)
        return record

    def by_section(self) -> Dict[str, List[FigureRecord]]:
        grouped: Dict[str, List[FigureRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.section, []).append(record)
        return grouped

    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "section": SECTIONS[r.section],
                "figure": r.name,
                "title": r.title,
                "caption": r.caption,
                "file": str(r.primary.relative_to(self.root.parent)),
            }
            for r in self.records
        ])


# --------------------------------------------------------------------------------------
# palettes
# --------------------------------------------------------------------------------------

def condition_palette(cfg: Config) -> Dict[str, str]:
    return {cfg.input.control_label: cfg.plots.control_color, cfg.input.case_label: cfg.plots.case_color}


def discrete_palette(levels: Sequence[str], name: str = "tab10") -> Dict[str, tuple]:
    levels = list(dict.fromkeys(levels))
    cmap = plt.get_cmap(name)
    n = max(getattr(cmap, "N", 10), 1)
    return {level: cmap(i % n) for i, level in enumerate(levels)}


def legend_from_map(ax, mapping: Dict[str, object], title: str, loc: str = "best", **kwargs) -> None:
    handles = [Patch(facecolor=color, edgecolor="none", label=str(label)) for label, color in mapping.items()]
    ax.legend(handles=handles, title=title, loc=loc, **kwargs)


# --------------------------------------------------------------------------------------
# custom primitives
# --------------------------------------------------------------------------------------

def draw_venn(ax, sets: Dict[str, set], colors: Optional[Sequence] = None) -> None:
    """Area-agnostic 2- or 3-circle Venn diagram with region counts."""
    names = list(sets)
    if len(names) not in (2, 3):
        raise ValueError("draw_venn supports 2 or 3 sets")
    colors = colors or ["#2874a6", "#c0392b", "#27ae60"]
    ax.set_aspect("equal")
    ax.axis("off")

    if len(names) == 2:
        centers = [(-0.35, 0.0), (0.35, 0.0)]
        radius = 0.6
        label_pos = [(-0.62, 0.0), (0.62, 0.0), (0.0, 0.0)]
        a, b = (sets[names[0]], sets[names[1]])
        counts = [len(a - b), len(b - a), len(a & b)]
        name_pos = [(-0.62, 0.72), (0.62, 0.72)]
    else:
        centers = [(-0.35, 0.22), (0.35, 0.22), (0.0, -0.38)]
        radius = 0.6
        a, b, c = (sets[n] for n in names)
        counts = [
            len(a - b - c), len(b - a - c), len(c - a - b),
            len((a & b) - c), len((a & c) - b), len((b & c) - a), len(a & b & c),
        ]
        label_pos = [
            (-0.65, 0.45), (0.65, 0.45), (0.0, -0.68),
            (0.0, 0.52), (-0.42, -0.22), (0.42, -0.22), (0.0, 0.05),
        ]
        name_pos = [(-0.72, 0.95), (0.72, 0.95), (0.0, -1.05)]

    for (cx, cy), color in zip(centers, colors):
        ax.add_patch(Circle((cx, cy), radius, alpha=0.35, facecolor=color, edgecolor=color, lw=1.6))
    for (x, y), value in zip(label_pos, counts):
        ax.text(x, y, f"{value:,}", ha="center", va="center", fontsize=11, fontweight="bold")
    for (x, y), name, color in zip(name_pos, names, colors):
        ax.text(x, y, name, ha="center", va="center", fontsize=11, color=color, fontweight="bold")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.25)


def exclusive_intersections(sets: Dict[str, set]) -> List[Tuple[Tuple[str, ...], int]]:
    """Sizes of every non-empty exclusive intersection, largest first."""
    names = list(sets)
    universe = set().union(*sets.values()) if sets else set()
    results = []
    for size in range(len(names), 0, -1):
        for combo in itertools.combinations(names, size):
            inside = set.intersection(*(sets[n] for n in combo)) if combo else set()
            outside = set().union(*(sets[n] for n in names if n not in combo)) if size < len(names) else set()
            exclusive = inside - outside
            if exclusive:
                results.append((combo, len(exclusive)))
    results.sort(key=lambda item: item[1], reverse=True)
    _ = universe
    return results


def draw_upset(fig, sets: Dict[str, set], max_intersections: int = 20, color: str = "#34495e"):
    """Minimal UpSet plot: intersection bars on top, membership matrix below, set sizes at left."""
    names = list(sets)
    combos = exclusive_intersections(sets)[:max_intersections]
    if not combos:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "no intersections to display", ha="center", va="center")
        ax.axis("off")
        return

    n_sets = len(names)
    # wspace leaves room for the set names, which sit between the size bars and the matrix.
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 3.4], height_ratios=[2.6, max(1.0, 0.32 * n_sets)],
        wspace=0.34, hspace=0.06,
    )
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_bar)
    ax_sets = fig.add_subplot(gs[1, 0], sharey=ax_matrix)
    fig.add_subplot(gs[0, 0]).axis("off")

    xs = np.arange(len(combos))
    heights = [size for _, size in combos]
    ax_bar.bar(xs, heights, color=color, width=0.62)
    for x, h in zip(xs, heights):
        ax_bar.text(x, h, f"{h:,}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_ylabel("Genes in intersection")
    ax_bar.set_ylim(0, max(heights) * 1.18)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(labelbottom=False)
    ax_bar.grid(axis="y", alpha=0.25)

    y_positions = {name: n_sets - 1 - i for i, name in enumerate(names)}
    for i in range(n_sets):
        if i % 2 == 0:
            ax_matrix.axhspan(i - 0.5, i + 0.5, color="#00000008")
    for x, (combo, _) in enumerate(combos):
        ys = [y_positions[n] for n in combo]
        ax_matrix.scatter([x] * n_sets, list(range(n_sets)), s=55, color="#d5d8dc", zorder=1)
        ax_matrix.scatter([x] * len(ys), ys, s=55, color=color, zorder=2)
        if len(ys) > 1:
            ax_matrix.plot([x, x], [min(ys), max(ys)], color=color, lw=1.8, zorder=2)
    ax_matrix.set_yticks(range(n_sets))
    ax_matrix.set_yticklabels([names[n_sets - 1 - i] for i in range(n_sets)])
    ax_matrix.set_xticks([])
    ax_matrix.set_ylim(-0.6, n_sets - 0.4)
    ax_matrix.set_xlim(-0.7, len(combos) - 0.3)
    ax_matrix.grid(False)
    ax_matrix.spines[["top", "right", "bottom", "left"]].set_visible(False)

    sizes = [len(sets[names[n_sets - 1 - i]]) for i in range(n_sets)]
    ax_sets.barh(range(n_sets), sizes, color="#7f8c8d", height=0.55)
    ax_sets.invert_xaxis()
    ax_sets.set_xlabel("Set size")
    ax_sets.tick_params(labelleft=False)
    ax_sets.grid(axis="x", alpha=0.25)
    ax_sets.spines[["top", "left", "right"]].set_visible(False)


def annotate_points(ax, x, y, labels, n: int = 15, score=None, fontsize: int = 7.5,
                    min_gap: float = 0.042) -> None:
    """Label the n most extreme points.

    Labels are pushed outwards from the centre of the plot and stacked vertically with a
    guaranteed minimum gap, so dense regions of a volcano plot stay readable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels, dtype=object)
    if score is None:
        score = np.abs(x) * np.abs(y)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return
    chosen = np.argsort(np.where(valid, score, -np.inf))[::-1][:n]

    log_x = ax.get_xscale() == "log"
    span_y = (np.nanmax(y[valid]) - np.nanmin(y[valid])) or 1.0
    span_x = (np.nanmax(x[valid]) - np.nanmin(x[valid])) or 1.0
    centre = np.nanmedian(x[valid]) if log_x else 0.0
    if log_x and centre <= 0:
        centre = np.nanmedian(x[valid][x[valid] > 0]) if (x[valid] > 0).any() else 1.0

    highest = -np.inf
    for side in (1, -1):
        on_side = [i for i in chosen if (1 if x[i] >= centre else -1) == side]
        on_side.sort(key=lambda i: y[i])
        previous = -np.inf
        for i in on_side:
            target_y = max(y[i], previous + min_gap * span_y)
            previous = target_y
            highest = max(highest, target_y)
            target_x = x[i] * (1.35 ** side) if log_x else x[i] + side * 0.025 * span_x
            ax.annotate(
                str(labels[i]), (x[i], y[i]), xytext=(target_x, target_y), fontsize=fontsize,
                ha="left" if side == 1 else "right", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#95a5a6", shrinkA=0, shrinkB=2.5),
            )
    if np.isfinite(highest):
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, max(top, highest + 0.04 * span_y))


def significance_bracket(ax, x1: float, x2: float, y: float, height: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=1.0, color="#2c3e50")
    ax.text((x1 + x2) / 2, y + height, text, ha="center", va="bottom", fontsize=9)


def heatmap_with_annotations(
    matrix: pd.DataFrame,
    col_annotations: pd.DataFrame,
    palettes: Dict[str, Dict[str, object]],
    title: str,
    cbar_label: str = "z-score",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    row_cluster: bool = True,
    col_cluster: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Clustered heatmap with categorical colour bars above the columns."""
    col_colors = pd.DataFrame(index=matrix.columns)
    for column, palette in palettes.items():
        if column in col_annotations.columns:
            col_colors[column] = col_annotations[column].astype(str).map(palette)
    if col_colors.empty:
        col_colors = None

    height = figsize[1] if figsize else min(14, max(5.0, 0.19 * matrix.shape[0] + 2.4))
    width = figsize[0] if figsize else min(18, max(7.0, 0.24 * matrix.shape[1] + 4.0))
    grid = sns.clustermap(
        matrix, cmap=cmap, center=0 if vmin is None else None, vmin=vmin, vmax=vmax,
        col_colors=col_colors, row_cluster=row_cluster and matrix.shape[0] > 2,
        col_cluster=col_cluster and matrix.shape[1] > 2,
        xticklabels=matrix.shape[1] <= 60, yticklabels=matrix.shape[0] <= 80,
        figsize=(width, height), linewidths=0,
        cbar_kws={"label": cbar_label},
        dendrogram_ratio=(0.12, 0.10),
    )
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(grid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7)
    handles, labels = [], []
    for column, palette in palettes.items():
        if col_colors is not None and column in col_colors.columns:
            handles.append(Patch(facecolor="white", edgecolor="none"))
            labels.append(f"$\\bf{{{column.replace('_', chr(92) + '_')}}}$")
            for level, color in palette.items():
                handles.append(Patch(facecolor=color, edgecolor="none"))
                labels.append(str(level))
    if handles:
        grid.figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 0.98),
                           frameon=False, fontsize=8)
    grid.figure.suptitle(title, y=1.01, fontsize=12, fontweight="bold")
    return grid.figure


def empty_figure(message: str, figsize: Tuple[float, float] = (7, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#7f8c8d", wrap=True)
    ax.axis("off")
    return fig
