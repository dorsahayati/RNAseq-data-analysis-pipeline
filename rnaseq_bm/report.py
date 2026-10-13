"""Self-contained HTML report collecting every figure, table and headline number."""

from __future__ import annotations

import base64
import datetime as dt
import html
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import Config
from .de import DEResult
from .plotting import SECTION_TITLES, SECTIONS, FigureBook
from .utils import log

CSS = """
:root { --bg:#f5f6f8; --card:#ffffff; --ink:#1f2a37; --muted:#6b7280; --line:#e5e7eb;
        --case:#c0392b; --control:#2874a6; --accent:#8e44ad; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       line-height:1.55; }
header { background:linear-gradient(120deg,#1f2a37,#374151); color:#fff; padding:36px 40px 30px; }
header h1 { margin:0 0 6px; font-size:26px; letter-spacing:-0.2px; }
header p { margin:2px 0; color:#cbd5e1; font-size:13.5px; }
main { max-width:1180px; margin:0 auto; padding:26px 24px 80px; }
nav { position:sticky; top:0; background:rgba(245,246,248,.94); backdrop-filter:blur(6px);
      border-bottom:1px solid var(--line); padding:10px 24px; z-index:10; }
nav a { color:var(--ink); text-decoration:none; font-size:13px; margin-right:18px; font-weight:600; }
nav a:hover { color:var(--case); }
section { margin:34px 0 0; }
h2 { font-size:20px; margin:0 0 4px; padding-bottom:8px; border-bottom:2px solid var(--line); }
h3 { font-size:15px; margin:0 0 6px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:18px 0 6px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.card .value { font-size:25px; font-weight:700; letter-spacing:-0.5px; }
.card .label { font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.6px; }
.card .sub { font-size:11.5px; color:var(--muted); margin-top:3px; }
figure { background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:16px 18px 12px; margin:18px 0; }
figure img { width:100%; height:auto; border-radius:4px; }
figcaption { font-size:12.5px; color:var(--muted); margin-top:10px; }
figcaption b { color:var(--ink); }
table { border-collapse:collapse; width:100%; font-size:12.5px; background:var(--card); }
th, td { border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; }
th { background:#eef1f5; font-weight:600; }
tr:hover td { background:#fafbfc; }
.note { background:#fff8e6; border-left:4px solid #f0b429; padding:12px 16px; border-radius:6px;
        font-size:13px; margin:16px 0; }
.good { background:#eefaf1; border-left:4px solid #27ae60; padding:12px 16px; border-radius:6px;
        font-size:13px; margin:16px 0; }
.pill { display:inline-block; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:600;
        background:#eef1f5; color:var(--muted); margin-right:6px; }
code { background:#eef1f5; padding:1px 5px; border-radius:4px; font-size:12px; }
footer { color:var(--muted); font-size:12px; text-align:center; padding:26px; }
"""


def _img_tag(path: Path, embed: bool) -> str:
    if embed and path.exists():
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.stem)}">'
    return f'<img src="{path.as_posix()}" alt="{html.escape(path.stem)}">'


def _table_html(frame: pd.DataFrame, max_rows: int = 15, float_format: str = "{:.3g}") -> str:
    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda v: float_format.format(v) if pd.notna(v) else "")
    return display.to_html(index=True, escape=True, border=0)


def _card(value: str, label: str, sub: str = "", color: str = "") -> str:
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="card"><div class="value"{style}>{html.escape(str(value))}</div>'
            f'<div class="label">{html.escape(label)}</div>'
            f'<div class="sub">{html.escape(sub)}</div></div>')


def build_report(cfg: Config, book: FigureBook, de_results: Dict[str, DEResult],
                 tables: Dict[str, pd.DataFrame], meta: pd.DataFrame, counts_shape,
                 outdir: Path) -> Path:
    overall = de_results["all_samples"]
    stype_col = cfg.input.sample_type_col
    sample_types = (sorted(meta[stype_col].astype(str).unique())
                    if stype_col and stype_col in meta.columns else [])

    parts: List[str] = []
    parts.append(f"<!doctype html><html><head><meta charset='utf-8'>"
                 f"<title>{html.escape(cfg.project.name)}</title><style>{CSS}</style></head><body>")
    parts.append(
        "<header>"
        f"<h1>{html.escape(cfg.project.name)}</h1>"
        f"<p>RNA-seq differential expression and biomarker report &middot; "
        f"{cfg.input.case_label} vs {cfg.input.control_label}</p>"
        f"<p>Generated {dt.datetime.now():%Y-%m-%d %H:%M} &middot; {len(meta)} samples &middot; "
        f"{counts_shape[0]:,} genes tested"
        + (f" &middot; sample types: {', '.join(sample_types)}" if sample_types else "") +
        "</p></header>"
    )

    grouped = book.by_section()
    nav_links = "".join(
        f'<a href="#{key}">{html.escape(SECTION_TITLES[key])}</a>'
        for key in SECTIONS if key in grouped
    )
    parts.append(f"<nav>{nav_links}</nav><main>")

    # ---- headline numbers -------------------------------------------------------------
    condition_counts = meta[cfg.input.condition_col].astype(str).value_counts()
    cards = [
        _card(f"{condition_counts.get(cfg.input.case_label, 0)} / "
              f"{condition_counts.get(cfg.input.control_label, 0)}",
              f"{cfg.input.case_label} / {cfg.input.control_label}", "samples analysed"),
        _card(f"{len(overall.up):,}", "up-regulated", f"FDR < {cfg.analysis.alpha}", "#c0392b"),
        _card(f"{len(overall.down):,}", "down-regulated",
              f"|log2FC| >= {cfg.analysis.lfc_threshold}", "#2874a6"),
    ]
    meta_table = tables.get("cross_sample_type_meta_analysis")
    if meta_table is not None and "core_marker" in meta_table:
        cards.append(_card(f"{int(meta_table['core_marker'].sum()):,}", "core markers",
                           "significant in every sample type", "#8e44ad"))
    panel_summary = tables.get("biomarker_panel_summary")
    if panel_summary is not None and not panel_summary.empty:
        best = panel_summary.loc[panel_summary["cv_auc_mean"].idxmax()]
        cards.append(_card(f"{best['cv_auc_mean']:.3f}", "panel AUC",
                           f"{best['model']}, {int(best['n_genes'])} genes, cross-validated", "#27ae60"))
        if pd.notna(best.get("nested_cv_auc_mean")):
            cards.append(_card(f"{best['nested_cv_auc_mean']:.3f}", "nested AUC",
                               "selection repeated inside each fold"))
    parts.append(f'<div class="cards">{"".join(cards)}</div>')

    summary = pd.DataFrame([r.summary_row() for r in de_results.values()])
    parts.append("<section><h2>Contrast summary</h2>" + _table_html(summary.set_index("contrast"), 20)
                 + "</section>")

    if panel_summary is not None and not panel_summary.empty:
        best = panel_summary.loc[panel_summary["cv_auc_mean"].idxmax()]
        gap = (best["cv_auc_mean"] - best["nested_cv_auc_mean"]) if pd.notna(
            best.get("nested_cv_auc_mean")) else 0
        if gap > 0.08:
            parts.append(
                f'<div class="note"><b>Selection bias warning.</b> The panel scores '
                f'{best["cv_auc_mean"]:.3f} when the genes are chosen on all samples but only '
                f'{best["nested_cv_auc_mean"]:.3f} when selection is repeated inside every training fold. '
                f'The nested number is the one to quote and to power a validation cohort on.</div>'
            )
        else:
            parts.append(
                f'<div class="good"><b>The panel survives nested validation.</b> Cross-validated AUC '
                f'{best["cv_auc_mean"]:.3f} versus {best["nested_cv_auc_mean"]:.3f} with gene selection '
                f'repeated inside each fold, so the performance is not an artefact of choosing genes on '
                f'the full cohort.</div>'
            )

    # ---- figure sections --------------------------------------------------------------
    for key in SECTIONS:
        if key not in grouped:
            continue
        parts.append(f'<section id="{key}"><h2>{html.escape(SECTION_TITLES[key])}</h2>')
        for record in grouped[key]:
            parts.append(
                "<figure>"
                f"<h3>{record.index}. {html.escape(record.title)}</h3>"
                f"{_img_tag(record.primary, cfg.report.embed_images)}"
                f"<figcaption>{html.escape(record.caption)}</figcaption>"
                "</figure>"
            )
        parts.append("</section>")

    # ---- key tables -------------------------------------------------------------------
    parts.append('<section id="tables"><h2>Key result tables</h2>')
    highlight = [
        ("Top differentially expressed genes (all samples)",
         overall.table.head(15)[["gene_name", "baseMean", "log2FoldChange", "pvalue", "padj", "direction"]]),
    ]
    performance = tables.get("biomarker_univariate_performance")
    if performance is not None and not performance.empty:
        columns = [c for c in ["gene_name", "direction", "log2FC", "auc", "auc_ci_low", "auc_ci_high",
                               "sensitivity", "specificity", "auc_min_across_types", "in_panel"]
                   if c in performance.columns]
        highlight.append(("Biomarker candidates ranked by cross-material performance",
                          performance.head(20)[columns]))
    transfer = tables.get("biomarker_cross_sample_type_transfer_auc")
    if transfer is not None:
        highlight.append(("Panel AUC when trained on one sample type and tested on another",
                          transfer.astype(float)))
    if meta_table is not None:
        columns = [c for c in ["gene_name", "mean_log2FC", "n_types_significant", "combined_padj",
                               "core_marker", "significant_in"] if c in meta_table.columns]
        highlight.append(("Cross-sample-type meta-analysis", meta_table.head(15)[columns]))
    for title, frame in highlight:
        parts.append(f"<h3>{html.escape(title)}</h3>{_table_html(frame, 20)}")
    written = sorted(p.name for p in (outdir / "tables").glob("*.csv"))
    parts.append("<h3>All exported tables</h3><p>"
                 + " ".join(f'<span class="pill">{html.escape(name)}</span>' for name in written)
                 + f"</p><p>Files are in <code>{(outdir / 'tables').as_posix()}</code>.</p>")
    parts.append("</section>")

    # ---- methods ----------------------------------------------------------------------
    parts.append(
        "<section id='methods'><h2>Methods</h2><p style='font-size:13.5px'>"
        f"Counts were filtered to genes with at least {cfg.filtering.min_count} reads in a minimum "
        f"number of samples equal to the smallest group, normalised with "
        f"{cfg.analysis.normalization.upper()} scaling factors and variance-stabilised for all "
        f"unsupervised analyses. Differential expression used <b>{html.escape(overall.method)}</b> "
        f"with the contrast {cfg.input.case_label} versus {cfg.input.control_label}"
        + (f" and {', '.join(overall.covariates)} as covariate(s)" if overall.covariates else "")
        + f". Genes were called differentially expressed at FDR &lt; {cfg.analysis.alpha} "
        f"(Benjamini-Hochberg) and |log2 fold change| &ge; {cfg.analysis.lfc_threshold}. "
        "Sample types were analysed separately and combined with Stouffer's weighted-Z method. "
        "Biomarker candidates were scored by the area under the ROC curve with bootstrap confidence "
        f"intervals, and a {cfg.biomarkers.panel_size}-gene panel was evaluated by repeated stratified "
        f"{cfg.biomarkers.cv_folds}-fold cross-validation, a nested cross-validation in which gene "
        f"selection is repeated inside every training fold, and a {cfg.biomarkers.permutations}-round "
        "label-permutation test."
        "</p></section>"
    )
    parts.append(
        "<footer>Generated by the RNA-seq biomarker pipeline. "
        "Figures are also available as PDF in the figures folder.</footer></main></body></html>"
    )

    path = outdir / cfg.report.filename
    path.write_text("".join(parts), encoding="utf-8")
    log.info("report written to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path
