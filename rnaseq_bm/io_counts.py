"""Read per-sample count files of various flavours and assemble the count matrix."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import Config
from .utils import log, strip_version

# Suffixes stripped from a file name to recover the sample identifier.
_SAMPLE_SUFFIXES = [
    ".ReadsPerGene.out.tab",
    "ReadsPerGene.out.tab",
    ".counts.txt",
    ".count.txt",
    ".featureCounts.txt",
    ".htseq.txt",
    ".genes.results",
    ".quant.sf",
    ".counts.tsv",
    ".counts.csv",
    ".counts",
    ".count",
    ".tab",
    ".tsv",
    ".txt",
    ".csv",
]
_TRAILING_TOKENS = ["_counts", "_count", "_readcounts", "_gene_counts", "_featurecounts", "_htseq"]

# Rows emitted by htseq-count / STAR that are summaries rather than genes.
_META_PREFIXES = ("__", "N_", "no_feature", "ambiguous", "too_low_aQual", "not_aligned", "alignment_not_unique")

_COUNT_COLUMN_CANDIDATES = [
    "count", "counts", "readcount", "read_count", "reads", "numreads",
    "expected_count", "est_counts", "unstranded", "value",
]


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def sample_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in _SAMPLE_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    lowered = name.lower()
    for token in _TRAILING_TOKENS:
        if lowered.endswith(token):
            name = name[: -len(token)]
            break
    return name.strip("._-")


def _split_line(line: str, sep: str) -> List[str]:
    return [cell.strip() for cell in line.rstrip("\n").split(sep)]


def _detect_sep(line: str) -> str:
    for sep in ["\t", ",", ";"]:
        if len(line.split(sep)) > 1:
            return sep
    return r"\s+"


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def read_count_file(
    path: Path,
    gene_id_col: Optional[str] = None,
    count_col: Optional[str] = None,
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """Return (counts, gene_lengths) for one sample file.

    Recognises featureCounts, htseq-count, STAR ReadsPerGene, RSEM/salmon and
    generic two-column gene/count tables, with or without a header line.
    """
    path = Path(path)
    with _open_text(path) as handle:
        lines = [ln for ln in handle.read().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path.name}: file is empty")

    # featureCounts writes a '# Program:...' provenance line first.
    start = 1 if lines[0].startswith("#") else 0
    sep = _detect_sep(lines[start])
    first = _split_line(lines[start], sep) if sep != r"\s+" else lines[start].split()
    has_header = not all(_is_number(cell) for cell in first[1:]) or len(first) == 1

    read_kwargs: Dict = dict(sep=sep, engine="python", skiprows=start, comment=None)
    if not has_header:
        read_kwargs["header"] = None
    frame = pd.read_csv(path, **read_kwargs)
    if not has_header:
        frame.columns = ["gene_id"] + [f"col{i}" for i in range(1, frame.shape[1])]

    frame.columns = [str(c).strip() for c in frame.columns]
    lowered = {c.lower(): c for c in frame.columns}

    gene_col = gene_id_col if gene_id_col in frame.columns else None
    if gene_col is None:
        for candidate in ("geneid", "gene_id", "gene", "name", "id", "target_id", "ensembl_gene_id"):
            if candidate in lowered:
                gene_col = lowered[candidate]
                break
    if gene_col is None:
        gene_col = frame.columns[0]

    lengths = None
    if "length" in lowered:
        lengths = pd.to_numeric(frame[lowered["length"]], errors="coerce")

    # STAR ReadsPerGene: gene, unstranded, forward, reverse -> take the unstranded column.
    is_star = frame.shape[1] == 4 and frame[gene_col].astype(str).str.startswith("N_").any()

    value_col = count_col if count_col in frame.columns else None
    if value_col is None and is_star:
        value_col = frame.columns[1]
    if value_col is None:
        for candidate in _COUNT_COLUMN_CANDIDATES:
            if candidate in lowered:
                value_col = lowered[candidate]
                break
    if value_col is None:
        # featureCounts puts the BAM path last; generic tables put counts last too.
        numeric_cols = [
            c for c in frame.columns
            if c != gene_col and pd.to_numeric(frame[c], errors="coerce").notna().mean() > 0.9
        ]
        if not numeric_cols:
            raise ValueError(f"{path.name}: no numeric count column found")
        value_col = numeric_cols[-1]

    counts = pd.to_numeric(frame[value_col], errors="coerce")
    genes = frame[gene_col].astype(str).str.strip()
    keep = ~genes.str.startswith(_META_PREFIXES) & counts.notna()
    counts = counts[keep]
    genes = genes[keep]
    if lengths is not None:
        lengths = lengths[keep]
        lengths.index = genes.values

    series = pd.Series(counts.to_numpy(), index=genes.values)
    if series.index.duplicated().any():
        series = series.groupby(level=0).sum()
        if lengths is not None:
            lengths = lengths.groupby(level=0).max()
    return series, lengths


ANNOTATION_COLUMNS = {
    "gene_name": ("gene_name", "genename", "symbol", "gene_symbol", "hgnc_symbol", "external_gene_name"),
    "length": ("length", "gene_length", "merged_length", "effective_length", "efflength"),
    "biotype": ("gene_biotype", "biotype", "gene_type", "chr", "chromosome", "strand", "start", "end",
                "description", "entrez", "entrezid"),
}


def read_count_matrix(path: Path, cfg: Config) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """Read one genes x samples matrix, separating sample columns from annotation columns.

    Accepts csv / tsv / txt (optionally gzipped) and xlsx, sniffs the delimiter, skips comment
    lines, and tolerates extra annotation columns such as gene_name or Length sitting between
    the gene id and the sample columns.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            frame = pd.read_excel(path)
        except ImportError as exc:
            raise ImportError(
                f"reading {path.name} needs an Excel engine - run 'pip install openpyxl', "
                "or export the matrix as .csv/.tsv"
            ) from exc
    else:
        with _open_text(path) as handle:
            lines = [ln for ln in handle.read().splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"{path.name}: count matrix is empty")
        skip = 0
        while skip < len(lines) and lines[skip].lstrip().startswith("#"):
            skip += 1
        sep = _detect_sep(lines[skip])
        frame = pd.read_csv(path, sep=sep, engine="python", skiprows=skip)

    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.shape[1] < 2:
        raise ValueError(
            f"{path.name}: expected a gene column plus at least one sample column, "
            f"found {frame.shape[1]} column(s). Is the delimiter correct?"
        )

    lowered = {c.lower(): c for c in frame.columns}
    id_column = cfg.input.gene_id_col if cfg.input.gene_id_col in frame.columns else None
    if id_column is None:
        for candidate in ("gene_id", "geneid", "gene", "id", "ensembl_gene_id", "target_id", "feature"):
            if candidate in lowered:
                id_column = lowered[candidate]
                break
    if id_column is None:
        id_column = frame.columns[0]

    genes = frame[id_column].astype(str).str.strip()

    gene_names = None
    lengths = None
    annotation_columns = {id_column}
    for key, candidates in ANNOTATION_COLUMNS.items():
        for candidate in candidates:
            if candidate in lowered and lowered[candidate] != id_column:
                column = lowered[candidate]
                annotation_columns.add(column)
                if key == "gene_name" and gene_names is None:
                    gene_names = pd.Series(frame[column].astype(str).values, index=genes.values)
                elif key == "length" and lengths is None:
                    lengths = pd.Series(pd.to_numeric(frame[column], errors="coerce").values,
                                        index=genes.values)

    sample_columns, dropped = [], []
    for column in frame.columns:
        if column in annotation_columns:
            continue
        numeric_fraction = pd.to_numeric(frame[column], errors="coerce").notna().mean()
        (sample_columns if numeric_fraction > 0.9 else dropped).append(column)
    if dropped:
        log.warning("ignoring %d non-numeric column(s) in %s: %s",
                    len(dropped), path.name, ", ".join(map(str, dropped[:6])))
    if not sample_columns:
        raise ValueError(f"{path.name}: no numeric sample columns found after removing annotation columns")

    counts = frame[sample_columns].apply(pd.to_numeric, errors="coerce")
    counts.index = genes.values
    meta_rows = genes.str.startswith(_META_PREFIXES)
    if meta_rows.any():
        counts = counts[~meta_rows.values]
        if gene_names is not None:
            gene_names = gene_names[~meta_rows.values]
        if lengths is not None:
            lengths = lengths[~meta_rows.values]
    if counts.index.duplicated().any():
        n_dupes = int(counts.index.duplicated().sum())
        log.warning("%d duplicate gene id(s) in %s summed together", n_dupes, path.name)
        counts = counts.groupby(level=0).sum()
        if gene_names is not None:
            gene_names = gene_names[~gene_names.index.duplicated()]
        if lengths is not None:
            lengths = lengths.groupby(level=0).max()
    log.info("loaded count matrix %s: %d genes x %d samples", path.name, *counts.shape)
    return counts, lengths, gene_names


def resolve_count_source(cfg: Config) -> Tuple[str, Path]:
    """Decide whether to read a genes x samples matrix or a folder of per-sample files.

    Resolution order:
      1. ``input.counts``         - auto-detect: file -> matrix, directory -> per-sample files
      2. ``input.counts_matrix``  - explicit matrix
      3. ``input.counts_dir``     - explicit folder of per-sample files

    The path itself wins over the key name: a file is always a matrix and a directory is
    always per-sample files, so putting the matrix path in ``counts_dir`` still works.
    """
    candidates = []
    unified = cfg.resolve(cfg.input.counts)
    if unified is not None:
        candidates.append(("auto", unified))
    matrix = cfg.resolve(cfg.input.counts_matrix)
    if matrix is not None:
        candidates.append(("matrix", matrix))
    folder = cfg.resolve(cfg.input.counts_dir)
    if folder is not None:
        candidates.append(("dir", folder))

    if not candidates:
        raise ValueError(
            "no counts provided: set input.counts (file or folder), input.counts_dir "
            "(one count file per sample) or input.counts_matrix (genes x samples matrix)"
        )

    mode_hint, path = candidates[0]
    if len(candidates) > 1:
        extras = ", ".join(f"{hint}={p}" for hint, p in candidates[1:])
        log.warning("multiple count inputs are set - using %s=%s and ignoring %s",
                    mode_hint, path, extras)

    if not path.exists():
        raise FileNotFoundError(f"count path not found: {path}")
    if path.is_dir():
        log.info("count input: per-sample files in %s", path)
        return "dir", path
    log.info("count input: genes x samples matrix %s", path)
    return "matrix", path


def _count_files_in_dir(folder: Path, pattern: str) -> List[Path]:
    """List count files, including gzipped counterparts of the glob pattern."""
    matches = list(folder.glob(pattern))
    if not pattern.endswith(".gz"):
        matches.extend(folder.glob(pattern + ".gz"))
        if pattern == "*":
            matches.extend(folder.glob("*.gz"))
    files = []
    seen = set()
    for path in sorted(matches):
        if not path.is_file() or path in seen:
            continue
        if path.suffix.lower() in {".bam", ".bai", ".log"} or path.name.endswith(".summary"):
            continue
        seen.add(path)
        files.append(path)
    return files


def load_counts(cfg: Config) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """Build the genes x samples count matrix.

    Two input modes are supported and auto-selected from the configuration:
      * a folder of per-sample count files (featureCounts, htseq-count, STAR ReadsPerGene,
        RSEM/salmon, or generic gene/count tables)
      * a single pre-merged matrix with genes in rows and samples in columns

    Returns the counts, gene lengths and gene symbols; the latter two are filled in whenever
    the input happens to carry them (featureCounts Length column, matrix annotation columns).
    """
    mode, source = resolve_count_source(cfg)
    matrix_path = source if mode == "matrix" else None
    dir_path = source if mode == "dir" else None

    gene_names = None
    if matrix_path is not None:
        if not matrix_path.exists():
            raise FileNotFoundError(f"count matrix not found: {matrix_path}")
        counts, lengths, gene_names = read_count_matrix(matrix_path, cfg)
    else:
        counts_dir = dir_path
        files = _count_files_in_dir(counts_dir, cfg.input.counts_pattern)
        if not files:
            raise FileNotFoundError(f"no files matched {cfg.input.counts_pattern} in {counts_dir}")
        series_by_sample: Dict[str, pd.Series] = {}
        lengths = None
        for path in files:
            sample = sample_id_from_path(path)
            values, gene_len = read_count_file(path, cfg.input.gene_id_col, cfg.input.count_col)
            if sample in series_by_sample:
                raise ValueError(f"duplicate sample id '{sample}' derived from {path.name}")
            series_by_sample[sample] = values
            if gene_len is not None and lengths is None:
                lengths = gene_len
        counts = pd.DataFrame(series_by_sample)
        log.info("loaded %d count files: %d genes x %d samples", len(files), *counts.shape)

    counts = counts.apply(pd.to_numeric, errors="coerce")
    n_missing = int(counts.isna().sum().sum())
    if n_missing:
        log.warning("%d missing count values set to 0 (genes absent from some files)", n_missing)
    counts = counts.fillna(0.0)

    if cfg.input.strip_ensembl_version:
        counts.index = strip_version(counts.index)
        if counts.index.duplicated().any():
            counts = counts.groupby(level=0).sum()
        if lengths is not None:
            lengths.index = strip_version(lengths.index)
            lengths = lengths.groupby(level=0).max()

    counts.index.name = "gene_id"
    external_lengths = cfg.resolve(cfg.input.gene_lengths)
    if external_lengths is not None:
        table = pd.read_csv(external_lengths, sep=None, engine="python", index_col=0)
        lengths = pd.to_numeric(table.iloc[:, 0], errors="coerce")
        if cfg.input.strip_ensembl_version:
            lengths.index = strip_version(lengths.index)
    if lengths is not None:
        lengths = lengths.reindex(counts.index)
    if gene_names is not None:
        if cfg.input.strip_ensembl_version:
            gene_names.index = strip_version(gene_names.index)
            gene_names = gene_names[~gene_names.index.duplicated()]
        gene_names = gene_names.reindex(counts.index)

    fractional = float((counts != counts.round()).to_numpy().mean())
    if fractional > 0.2:
        log.warning(
            "%.0f%% of values are not whole numbers - these look like normalised expression "
            "(TPM/FPKM/RPKM) rather than raw counts. The negative-binomial model expects raw "
            "counts; results will be unreliable.", 100 * fractional,
        )
    return counts.round().astype(float), lengths, gene_names


def load_metadata(cfg: Config, counts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the sample sheet, aligning it with the count matrix columns."""
    meta_path = cfg.resolve(cfg.input.metadata)
    if meta_path is None or not meta_path.exists():
        raise FileNotFoundError(f"metadata file not found: {cfg.input.metadata}")
    meta = pd.read_csv(meta_path, sep=None, engine="python")
    meta.columns = [str(c).strip() for c in meta.columns]

    ic = cfg.input
    for column in [ic.sample_col, ic.condition_col]:
        if column not in meta.columns:
            raise ValueError(f"metadata is missing required column '{column}' (found: {list(meta.columns)})")
    meta[ic.sample_col] = meta[ic.sample_col].astype(str).str.strip()
    meta = meta.set_index(ic.sample_col)
    meta.index.name = "sample_id"

    if meta.index.duplicated().any():
        dupes = meta.index[meta.index.duplicated()].unique().tolist()
        raise ValueError(f"duplicate sample ids in metadata: {dupes}")

    shared = [s for s in counts.columns if s in meta.index]
    missing_meta = sorted(set(counts.columns) - set(meta.index))
    missing_counts = sorted(set(meta.index) - set(counts.columns))
    if missing_meta:
        log.warning("%d count columns have no metadata and are dropped: %s", len(missing_meta), missing_meta[:5])
    if missing_counts:
        log.warning("%d metadata samples have no counts and are dropped: %s", len(missing_counts), missing_counts[:5])
    if not shared:
        raise ValueError(
            "no overlap between count sample names and metadata sample ids.\n"
            f"  counts: {list(counts.columns)[:5]}\n  metadata: {list(meta.index)[:5]}"
        )
    meta = meta.loc[shared].copy()

    condition = meta[ic.condition_col].astype(str).str.strip()
    labels = set(condition.unique())
    if not {ic.case_label, ic.control_label} <= labels:
        raise ValueError(
            f"condition column '{ic.condition_col}' must contain '{ic.case_label}' and '{ic.control_label}'; "
            f"found {sorted(labels)}"
        )
    meta[ic.condition_col] = pd.Categorical(condition, categories=[ic.control_label, ic.case_label])
    if meta[ic.condition_col].isna().any():
        extra = sorted(labels - {ic.case_label, ic.control_label})
        log.warning("dropping %d samples with other condition labels %s",
                    int(meta[ic.condition_col].isna().sum()), extra)
        meta = meta[meta[ic.condition_col].notna()]

    if ic.sample_type_col and ic.sample_type_col in meta.columns:
        meta[ic.sample_type_col] = meta[ic.sample_type_col].astype(str).str.strip()
    elif ic.sample_type_col:
        log.warning("sample type column '%s' not in metadata; per-sample-type analysis disabled", ic.sample_type_col)

    counts_aligned = counts[meta.index.tolist()]
    log.info("analysis set: %d samples (%s)", len(meta),
             ", ".join(f"{k}={v}" for k, v in meta[ic.condition_col].value_counts().items()))
    return meta, counts_aligned


def load_gene_annotation(cfg: Config, genes: pd.Index,
                         embedded: Optional[pd.Series] = None) -> pd.Series:
    """Map gene ids to display symbols.

    Symbols carried inside the count matrix are used first; an external annotation file, when
    configured, takes precedence over them. Anything still unmapped keeps its gene id.
    """
    mapping = pd.Series(list(genes), index=genes, dtype=object)
    if embedded is not None:
        usable = embedded.reindex(genes)
        usable = usable[usable.notna() & (usable.astype(str).str.strip() != "")]
        mapping[usable.index] = usable.astype(str)
        log.info("using %d gene symbols supplied with the count matrix", len(usable))
    path = cfg.resolve(cfg.input.gene_annotation)
    if path is None or not Path(path).exists():
        return mapping
    table = pd.read_csv(path, sep=None, engine="python")
    table.columns = [str(c).strip() for c in table.columns]
    lowered = {c.lower(): c for c in table.columns}
    id_col = next((lowered[c] for c in ("gene_id", "geneid", "ensembl_gene_id", "id") if c in lowered),
                  table.columns[0])
    name_col = next((lowered[c] for c in ("gene_name", "symbol", "gene_symbol", "hgnc_symbol", "name")
                     if c in lowered), table.columns[-1])
    ids = table[id_col].astype(str)
    if cfg.input.strip_ensembl_version:
        ids = pd.Series(strip_version(ids), index=table.index)
    lookup = pd.Series(table[name_col].astype(str).values, index=ids.values)
    lookup = lookup[~lookup.index.duplicated()]
    resolved = lookup.reindex(genes)
    mapping[resolved.notna()] = resolved[resolved.notna()]
    log.info("annotated %d/%d genes with symbols", int(resolved.notna().sum()), len(genes))
    return mapping
