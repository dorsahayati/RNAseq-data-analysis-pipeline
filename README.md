# RNA-seq cancer vs control biomarker pipeline

A complete, reproducible transcriptomics pipeline that starts from **one count file per sample**
and ends with a **ranked biomarker panel compared across sample types**. It produces ~70 figures,
~20 result tables and a single self-contained HTML report.

Everything runs in Python. `PyDESeq2` provides the DESeq2 negative-binomial statistics; if it is
not installed the pipeline falls back to a Welch t-test on log2 CPM so it never hard-fails.

---

## What it does

| Stage | Output |
|---|---|
| **1. Input** | Accepts **either** one count file per sample (featureCounts, htseq-count, STAR `ReadsPerGene`, RSEM/salmon, generic gene/count tables) **or** a single genes × samples count matrix. Merges, annotates and validates against the sample sheet. |
| **2. QC** | Library size, gene detection, sparsity, RLE, normalisation factors, mean-variance trend, correlation and distance heatmaps, dendrogram, PCA (+ factor-association test), t-SNE/MDS/UMAP, outlier scores. |
| **3. Differential expression** | DESeq2-equivalent Wald test with LFC shrinkage, run on all samples **and separately within each sample type**, plus a condition × sample-type interaction test. Volcano, MA, dispersion, p-value histograms, heatmaps, per-gene boxplots. |
| **4. Sample-type comparison** | Venn / UpSet overlaps, pairwise fold-change concordance, Jaccard and rank-concordance curves, Stouffer meta-analysis, forest plots, shared-vs-specific breakdown. |
| **5. Biomarkers** | Per-gene ROC/AUC with bootstrap CIs and AUC **per sample type**, redundancy-aware panel selection, three classifiers with repeated cross-validation, nested CV, label-permutation test, panel-size curve, **train-on-one-material / test-on-another transfer matrix**, composite signature score. |
| **6. Enrichment** | Offline over-representation and pre-ranked GSEA (permutation NES + running-score plots), pathway activity compared across sample types, optional Enrichr. |
| **7. Report** | `results/report.html` with every figure, caption, headline metric and key table embedded. |

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.9 or newer. Optional extras: `pip install gseapy umap-learn`.

---

## Quick start with the bundled example

```bash
python make_example_data.py          # simulates 80 samples across 3 sample types
python run_pipeline.py --config config.yaml
```

The simulator writes the same dataset in both input formats — `data/counts/*.counts.txt`
(featureCounts style, one per sample) and `data/counts_matrix.tsv` (genes × samples with
`gene_name` and `Length` columns) — so you can try either mode:

```bash
python run_pipeline.py -c config.yaml --counts-matrix data/counts_matrix.tsv -o results_matrix
```

Then open `results/report.html`. The simulation writes `data/simulation_truth.csv` containing the
true fold changes, so you can benchmark the pipeline's calls against the ground truth.

---

## Using your own data

### 1. Counts — pick one of two input modes

**Mode A — one count file per sample.** Put the files in a folder and set `input.counts_dir`.
Any of these are auto-detected without configuration:

```
featureCounts   # Geneid  Chr  Start  End  Strand  Length  <bam>
htseq-count     # ENSG00000141510	1523        (no header, __no_feature rows dropped)
STAR            # <sample>ReadsPerGene.out.tab   (unstranded column used, N_* rows dropped)
RSEM / salmon   # gene_id ... expected_count / NumReads
generic         # gene_id,count
```

The sample id comes from the file name with the extension stripped (`TUMOR_01.counts.txt` →
`TUMOR_01`) and must match the sample sheet.

**Mode B — one merged matrix, genes in rows and samples in columns.** Set
`input.counts_matrix` and leave `input.counts_dir: null`:

```
gene_id           gene_name  Length  TUMOR_01  TUMOR_02  PLASMA_01
ENSG00000141510   TP53       2591    1420      1655      310
ENSG00000012048   BRCA1      5589    880       931       142
```

Only the gene column and the sample columns are required. `.csv`, `.tsv`, `.txt` (optionally
`.gz`) and `.xlsx` all work — the delimiter is sniffed, leading `#` comment lines are skipped,
and annotation columns are separated from sample columns automatically:

- `gene_name` / `symbol` / `hgnc_symbol` → used to label genes in every figure, so you don't
  need a separate annotation file
- `Length` → enables TPM export
- `Chr`, `Start`, `End`, `Strand`, `biotype`, and any other non-numeric column → ignored with a
  warning

Duplicate gene ids are summed, and Ensembl version suffixes are stripped by default.

Switch modes from the command line without editing the config. `--counts` auto-detects
a file (matrix) versus a folder (per-sample files):

```bash
python run_pipeline.py -c config.yaml --counts data/counts
python run_pipeline.py -c config.yaml --counts data/counts_matrix.tsv
python run_pipeline.py -c config.yaml --counts-matrix data/counts_matrix.tsv
python run_pipeline.py -c config.yaml --counts-dir data/counts
```

Naming one clears the other. Setting both in the config is not an error — the matrix wins and
the pipeline says so in the log.

> Use **raw integer counts** in either mode, not TPM/FPKM/normalised values — the
> negative-binomial model needs them. The pipeline warns if most values are not whole numbers.

### 2. Sample sheet

```csv
sample_id,condition,sample_type,batch,age,sex
TUMOR_01,cancer,tumor_tissue,B1,64,F
TUMOR_02,control,tumor_tissue,B1,59,M
PLASMA_01,cancer,plasma_cfRNA,B2,71,M
```

- `condition` — must contain exactly the two labels named in `input.case_label` / `control_label`.
- `sample_type` — the column driving the final comparison. Anything works: specimen type
  (tissue / plasma / blood), tumour subtype, stage, or site. Set it to `null` to skip that section.
- `batch` — added to the DE design automatically, unless it is confounded with condition
  (in which case it is dropped with a warning rather than producing a meaningless model).

### 3. Configure and run

Edit `config.yaml` (every option is commented), then:

```bash
python run_pipeline.py --config config.yaml
```

Common overrides without editing the file:

```bash
python run_pipeline.py -c config.yaml --counts-dir /data/counts -o results_v2
python run_pipeline.py -c config.yaml --alpha 0.01 --lfc 1.5 --panel-size 10
python run_pipeline.py -c config.yaml --de-method ttest --no-enrichment
```

---

## Output layout

```
results/
├── report.html                          <- start here (self-contained)
├── pipeline.log
├── figures/
│   ├── figure_index.csv                 <- every figure with its caption
│   ├── 01_quality_control/              (19 figures, png + pdf)
│   ├── 02_differential_expression/      (22)
│   ├── 03_sample_type_comparison/       (11)
│   ├── 04_biomarkers/                   (14)
│   └── 05_enrichment/                   (4)
└── tables/
    ├── de_all_samples.csv               full DE result, every gene
    ├── de_<sample_type>.csv             one per sample type
    ├── cross_sample_type_meta_analysis.csv
    ├── biomarker_univariate_performance.csv   AUC per gene per sample type
    ├── biomarker_panel_summary.csv
    ├── biomarker_cross_sample_type_transfer_auc.csv
    ├── biomarker_sample_scores.csv
    └── ...
```

---

## Statistical choices worth knowing

**Differential expression.** DESeq2 median-of-ratios normalisation, gene-wise dispersion shrunk
towards a fitted trend, Wald test, Benjamini-Hochberg FDR, and apeglm-style shrinkage of fold
changes so low-count genes do not dominate the rankings.

**Per-sample-type contrasts.** Each sample type is tested separately rather than pooled, because
pooling assumes the cancer effect has the same size in every material. The interaction test
reports the genes where that assumption fails.

**Combining sample types.** Stouffer's weighted-Z, weighted by √n. A "core marker" is significant
in *every* sample type, moves in the same direction in all of them, and survives the combined FDR.

**Biomarker performance is reported three ways**, because the first is optimistic on its own:

1. Cross-validated AUC of a panel chosen on all samples — what most papers report.
2. **Nested CV**, where gene selection is repeated inside every training fold — the honest estimate.
3. A **label-permutation null**, showing the AUC this pipeline reports on data with no signal.

The report raises a warning when 1 and 2 differ by more than 0.08.

**Cross-material transfer.** Expression is z-scored within each sample type before the transfer
experiment, so the matrix measures whether marker *behaviour* carries over rather than the
(large) baseline differences between tissue, plasma and blood.

---

## Limitations

- Designed for two-group comparisons. Multi-level factors need extra contrasts.
- With fewer than ~5 samples per group the cross-validated AUCs carry very wide intervals; treat
  them as exploratory and read the fold-to-fold spread in the panel ROC figure.
- Cross-sample-type transfer needs both classes present in each material.
- The bundled `resources/hallmark_lite.gmt` is a small demonstration subset. For real analyses
  download the full MSigDB collection and point `enrichment.offline_gmt` at it.
- Any biomarker panel found here still requires an independent validation cohort.
