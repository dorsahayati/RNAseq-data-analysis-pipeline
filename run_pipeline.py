#!/usr/bin/env python
"""Command-line entry point for the RNA-seq cancer-vs-control biomarker pipeline.

Examples
--------
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --counts data/counts_matrix.tsv --outdir results_matrix
    python run_pipeline.py --config config.yaml --counts-dir data/counts --outdir results_run2
    python run_pipeline.py --config config.yaml --de-method ttest --no-enrichment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rnaseq_bm.pipeline import run


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RNA-seq differential expression and biomarker pipeline (cancer vs control).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="YAML configuration file")
    parser.add_argument("--counts",
                        help="count input: a genes x samples matrix file, or a folder of per-sample files")
    parser.add_argument("--counts-dir",
                        help="folder of per-sample count files (switches off --counts-matrix)")
    parser.add_argument("--counts-matrix",
                        help="single genes x samples matrix, genes in rows and samples in columns")
    parser.add_argument("--metadata", "-m", help="override input.metadata")
    parser.add_argument("--outdir", "-o", help="override project.outdir")
    parser.add_argument("--de-method", choices=["auto", "pydeseq2", "ttest"],
                        help="override analysis.de_method")
    parser.add_argument("--alpha", type=float, help="FDR threshold")
    parser.add_argument("--lfc", type=float, help="minimum absolute log2 fold change")
    parser.add_argument("--panel-size", type=int, help="number of genes in the biomarker panel")
    parser.add_argument("--permutations", type=int, help="label permutations for the panel null")
    parser.add_argument("--no-enrichment", action="store_true", help="skip functional enrichment")
    parser.add_argument("--online-enrichment", action="store_true",
                        help="query Enrichr in addition to the offline gene sets")
    parser.add_argument("--seed", type=int, help="random seed")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = args.config if Path(args.config).exists() else None
    if config is None and args.config != "config.yaml":
        print(f"config file not found: {args.config}", file=sys.stderr)
        return 2

    overrides = {}
    mapping = {
        "counts": "input.counts",
        "counts_dir": "input.counts_dir",
        "counts_matrix": "input.counts_matrix",
        "metadata": "input.metadata",
        "outdir": "project.outdir",
        "de_method": "analysis.de_method",
        "alpha": "analysis.alpha",
        "lfc": "analysis.lfc_threshold",
        "panel_size": "biomarkers.panel_size",
        "permutations": "biomarkers.permutations",
        "seed": "project.seed",
    }
    for arg_name, option in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[option] = value
    # The two input modes are mutually exclusive: naming one on the command line clears
    # whatever the config file had set for the other.
    if args.counts:
        overrides["input.counts_dir"] = None
        overrides["input.counts_matrix"] = None
    if args.counts_matrix and not args.counts_dir:
        overrides["input.counts"] = None
        overrides["input.counts_dir"] = None
    if args.counts_dir and not args.counts_matrix:
        overrides["input.counts"] = None
        overrides["input.counts_matrix"] = None
    if args.no_enrichment:
        overrides["enrichment.enable"] = False
    if args.online_enrichment:
        overrides["enrichment.online"] = True

    try:
        outdir = run(config, overrides)
    except Exception as exc:  # surface a clean message instead of a raw traceback
        print(f"\nPipeline failed: {exc}\n", file=sys.stderr)
        raise
    print(f"\nResults written to {Path(outdir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
