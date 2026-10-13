#!/usr/bin/env python
"""Simulate a realistic multi-material RNA-seq dataset so the pipeline can be run end-to-end.

Three sample types (tumour tissue, plasma cell-free RNA, whole blood) are generated with a
shared "core" cancer signature whose effect is progressively diluted in the liquid biopsies,
plus material-specific responses. Counts are negative-binomial with an expression-dependent
dispersion, varying library sizes and a batch effect, and are written as featureCounts-style
per-sample files.

    python make_example_data.py --outdir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rnaseq_bm.enrichment import read_gmt

# Effect of the core cancer signature relative to solid tissue: liquid biopsies see a diluted
# tumour signal mixed with a large background of normal transcripts.
SAMPLE_TYPES = {
    "tumor_tissue": {"n_case": 12, "n_control": 12, "dilution": 1.00,
                     "dispersion": 0.12, "subject_noise": 0.45},
    "plasma_cfRNA": {"n_case": 14, "n_control": 14, "dilution": 0.62,
                     "dispersion": 0.40, "subject_noise": 0.60},
    "whole_blood": {"n_case": 14, "n_control": 14, "dilution": 0.45,
                    "dispersion": 0.32, "subject_noise": 0.50},
}

UP_PATHWAYS = ["HALLMARK_G2M_CHECKPOINT", "HALLMARK_E2F_TARGETS", "HALLMARK_HYPOXIA",
               "HALLMARK_GLYCOLYSIS"]
DOWN_PATHWAYS = ["HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HALLMARK_P53_PATHWAY"]


def build_gene_universe(gmt_path: Path, n_total: int, rng) -> pd.DataFrame:
    """Real symbols from the offline gene sets plus filler genes, keyed by Ensembl-style ids."""
    gene_sets = read_gmt(gmt_path) if gmt_path.exists() else {}
    symbols = sorted({gene for members in gene_sets.values() for gene in members})
    n_filler = max(0, n_total - len(symbols))
    filler = [f"LOC{100000 + i}" for i in range(n_filler)]
    all_symbols = symbols + filler
    ids = [f"ENSG{i:011d}.{rng.integers(1, 20)}" for i in range(len(all_symbols))]
    return pd.DataFrame({"gene_id": ids, "gene_name": all_symbols,
                         "length": rng.integers(400, 12000, len(all_symbols))})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="data")
    parser.add_argument("--n-genes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(__file__).resolve().parent
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    counts_dir = outdir / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)

    genes = build_gene_universe(root / "resources" / "hallmark_lite.gmt", args.n_genes, rng)
    gene_sets = read_gmt(root / "resources" / "hallmark_lite.gmt")
    n_genes = len(genes)
    symbol_to_row = {name: i for i, name in enumerate(genes["gene_name"])}

    # ---- baseline expression --------------------------------------------------------
    baseline = rng.lognormal(mean=4.2, sigma=1.5, size=n_genes) + 1.0
    # A real annotation contains many genes that are silent in the tissue being sequenced;
    # these are what the low-expression filter is meant to remove.
    silent = rng.random(n_genes) < 0.15
    baseline[silent] *= rng.uniform(0.0005, 0.02, silent.sum())

    # Each material has its own expression profile (different cell composition).
    type_profile = {
        name: rng.normal(0, 0.55, n_genes) + (rng.random(n_genes) < 0.04) * rng.normal(0, 2.2, n_genes)
        for name in SAMPLE_TYPES
    }

    # ---- cancer effects -------------------------------------------------------------
    effect = {name: np.zeros(n_genes) for name in SAMPLE_TYPES}

    core_up = [symbol_to_row[g] for p in UP_PATHWAYS for g in gene_sets.get(p, [])
               if g in symbol_to_row]
    core_down = [symbol_to_row[g] for p in DOWN_PATHWAYS for g in gene_sets.get(p, [])
                 if g in symbol_to_row]
    core_up = np.unique(core_up)
    core_down = np.unique(np.setdiff1d(core_down, core_up))
    extra = rng.choice(np.setdiff1d(np.arange(n_genes), np.concatenate([core_up, core_down])),
                       size=120, replace=False)
    core_extra_up, core_extra_down = extra[:60], extra[60:]

    base_effect = np.zeros(n_genes)
    base_effect[core_up] = rng.uniform(0.5, 2.0, len(core_up))
    base_effect[core_extra_up] = rng.uniform(0.5, 1.8, len(core_extra_up))
    base_effect[core_down] = -rng.uniform(0.5, 1.7, len(core_down))
    base_effect[core_extra_down] = -rng.uniform(0.5, 1.6, len(core_extra_down))

    for name, spec in SAMPLE_TYPES.items():
        effect[name] = base_effect * spec["dilution"]

    # Material-specific responses: stromal remodelling only in tissue, an immune/inflammatory
    # signature only in blood - these create the genuinely sample-type-specific hits.
    tissue_specific = [symbol_to_row[g] for g in
                       gene_sets.get("HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", [])
                       + gene_sets.get("HALLMARK_ANGIOGENESIS", []) if g in symbol_to_row]
    blood_specific = [symbol_to_row[g] for g in
                      gene_sets.get("HALLMARK_INFLAMMATORY_RESPONSE", [])
                      + gene_sets.get("HALLMARK_IL6_JAK_STAT3_SIGNALING", []) if g in symbol_to_row]
    effect["tumor_tissue"][np.unique(tissue_specific)] += rng.uniform(0.7, 2.0, len(np.unique(tissue_specific)))
    effect["whole_blood"][np.unique(blood_specific)] += rng.uniform(0.6, 1.7, len(np.unique(blood_specific)))

    # ---- simulate samples -----------------------------------------------------------
    metadata_rows = []
    counts_by_sample = {}
    for sample_type, spec in SAMPLE_TYPES.items():
        for group, label in [("case", "cancer"), ("control", "control")]:
            for replicate in range(spec[f"n_{group}"]):
                sample_id = f"{sample_type[:4].upper()}_{label[:3].upper()}_{replicate + 1:02d}"
                is_case = label == "cancer"
                batch = f"B{(replicate % 2) + 1}"
                library_size = float(rng.uniform(12e6, 38e6))

                log_mu = (np.log(baseline) + type_profile[sample_type]
                          + (effect[sample_type] if is_case else 0.0) * np.log(2)
                          + (0.22 if batch == "B2" else 0.0) * rng.normal(0, 1, n_genes)
                          # between-subject biological variation, the dominant noise source
                          + rng.normal(0, spec["subject_noise"], n_genes))
                mu = np.exp(log_mu)
                mu = mu / mu.sum() * library_size

                dispersion = spec["dispersion"] + 4.0 / (mu + 8.0)
                size = 1.0 / dispersion
                counts = rng.negative_binomial(size, size / (size + mu)).astype(int)
                counts_by_sample[sample_id] = counts

                frame = pd.DataFrame({
                    "Geneid": genes["gene_id"],
                    "Chr": "chr1",
                    "Start": 1000,
                    "End": 1000 + genes["length"],
                    "Strand": "+",
                    "Length": genes["length"],
                    f"aligned/{sample_id}.bam": counts,
                })
                path = counts_dir / f"{sample_id}.counts.txt"
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("# Program:featureCounts v2.0.3; Command:simulated\n")
                    frame.to_csv(handle, sep="\t", index=False, lineterminator="\n")

                metadata_rows.append({
                    "sample_id": sample_id,
                    "condition": label,
                    "sample_type": sample_type,
                    "batch": batch,
                    "age": int(rng.normal(63 if is_case else 58, 9)),
                    "sex": rng.choice(["F", "M"]),
                    "library_size_target": int(library_size),
                })

    # The same data is also written as a single genes x samples matrix, so both supported
    # input modes can be exercised. Gene symbols and lengths ride along as annotation columns.
    matrix = pd.DataFrame(counts_by_sample, index=genes["gene_id"].values)
    matrix.index.name = "gene_id"
    matrix.insert(0, "Length", genes["length"].values)
    matrix.insert(0, "gene_name", genes["gene_name"].values)
    matrix.to_csv(outdir / "counts_matrix.tsv", sep="\t")

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(outdir / "metadata.csv", index=False)
    genes[["gene_id", "gene_name"]].to_csv(outdir / "gene_annotation.csv", index=False)
    genes[["gene_id", "length"]].to_csv(outdir / "gene_lengths.csv", index=False)

    truth = pd.DataFrame({
        "gene_id": genes["gene_id"],
        "gene_name": genes["gene_name"],
        **{f"true_log2FC_{name}": effect[name] for name in SAMPLE_TYPES},
    })
    truth["is_core_signature"] = np.isin(np.arange(n_genes),
                                         np.concatenate([core_up, core_down, extra]))
    truth.to_csv(outdir / "simulation_truth.csv", index=False)

    print(f"wrote {len(metadata)} count files to {counts_dir}")
    print(f"  matrix:     {outdir / 'counts_matrix.tsv'}  (same data, genes x samples)")
    print(f"  metadata:   {outdir / 'metadata.csv'}")
    print(f"  annotation: {outdir / 'gene_annotation.csv'}")
    print(f"  ground truth (for benchmarking): {outdir / 'simulation_truth.csv'}")
    print(metadata.groupby(["sample_type", "condition"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
