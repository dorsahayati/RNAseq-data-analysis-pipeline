"""End-to-end orchestration: counts in, figures / tables / report out."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from . import compare, enrichment, figures_biomarker, figures_de, figures_qc
from .config import Config, load_config
from .de import expression_for_plots, interaction_test, run_all_contrasts
from .io_counts import load_counts, load_gene_annotation, load_metadata
from .normalize import filter_genes, normalized_counts, tpm, vst
from .plotting import FigureBook, setup_style
from .report import build_report
from .utils import ensure_dir, get_logger, set_seed, step, write_table


def run(config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> Path:
    cfg = load_config(config_path)
    for key, value in (overrides or {}).items():
        section, _, option = key.partition(".")
        if option and hasattr(cfg, section) and hasattr(getattr(cfg, section), option):
            setattr(getattr(cfg, section), option, value)

    outdir = ensure_dir(cfg.outdir)
    ensure_dir(outdir / "tables")
    log = get_logger(outdir / "pipeline.log")
    set_seed(cfg.project.seed)
    setup_style(cfg)
    book = FigureBook(outdir, cfg)
    tables: Dict[str, pd.DataFrame] = {}

    log.info("=" * 78)
    log.info("%s | %s vs %s", cfg.project.name, cfg.input.case_label, cfg.input.control_label)
    log.info("results -> %s", outdir.resolve())
    log.info("=" * 78)

    with step("loading counts and metadata"):
        counts_raw, gene_lengths, embedded_names = load_counts(cfg)
        meta, counts_raw = load_metadata(cfg, counts_raw)
        gene_names = load_gene_annotation(cfg, counts_raw.index, embedded_names)

    with step("filtering and normalising"):
        counts, filter_summary = filter_genes(counts_raw, meta, cfg)
        norm_counts, size_factors = normalized_counts(counts, cfg.analysis.normalization)
        vst_expr = vst(norm_counts)
        logcpm = expression_for_plots(counts, cfg)
        tables["normalised_counts"] = norm_counts.round(2)
        tables["vst_expression"] = vst_expr.round(3)
        tpm_values = tpm(counts, gene_lengths)
        if tpm_values is not None:
            tables["tpm_expression"] = tpm_values.round(2)
            log.info("gene lengths available - TPM exported alongside the normalised counts")

    with step("quality control figures"):
        tables.update(figures_qc.run_all(counts_raw, counts, norm_counts, vst_expr, meta,
                                         size_factors, filter_summary, gene_names, cfg, book))

    with step("differential expression"):
        de_results = run_all_contrasts(counts, meta, cfg, gene_names)
        interaction = interaction_test(counts, meta, cfg, gene_names)
        for name, result in de_results.items():
            tables[f"de_{name}"] = result.table
        tables["de_summary"] = pd.DataFrame([r.summary_row() for r in de_results.values()])
        if interaction is not None:
            tables["de_condition_by_sample_type_interaction"] = interaction

    with step("differential expression figures"):
        figures_de.run_all(de_results, logcpm, meta, interaction, cfg, book)

    with step("cross sample-type comparison"):
        tables.update(compare.run_all(de_results, cfg, book))

    with step("biomarker discovery and evaluation"):
        tables.update(figures_biomarker.run_all(
            logcpm, meta, de_results, tables.get("cross_sample_type_meta_analysis"),
            gene_names, cfg, book,
        ))

    with step("functional enrichment"):
        tables.update(enrichment.run_all(de_results, cfg, book))

    with step("writing tables and report"):
        for name, frame in tables.items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                write_table(frame, outdir / "tables" / f"{name}.csv")
        write_table(book.manifest(), outdir / "figures" / "figure_index.csv", index=False)
        report_path = None
        if cfg.report.enable:
            report_path = build_report(cfg, book, de_results, tables, meta, counts.shape, outdir)

    log.info("=" * 78)
    log.info("done: %d figures, %d tables", len(book.records),
             sum(1 for f in tables.values() if isinstance(f, pd.DataFrame) and not f.empty))
    if report_path:
        log.info("open the report: %s", report_path.resolve())
    log.info("=" * 78)
    return outdir
