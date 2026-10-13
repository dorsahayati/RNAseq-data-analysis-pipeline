"""Typed configuration loaded from YAML, with defaults for every option."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ProjectConfig:
    name: str = "RNAseq_cancer_vs_control"
    outdir: str = "results"
    seed: int = 42


@dataclass
class InputConfig:
    counts: Optional[str] = None           # file = genes x samples matrix; folder = one file per sample
    counts_dir: Optional[str] = None       # folder of per-sample count files
    counts_pattern: str = "*"
    counts_matrix: Optional[str] = None    # single genes (rows) x samples (columns) matrix
    metadata: str = "data/metadata.csv"
    sample_col: str = "sample_id"
    condition_col: str = "condition"
    sample_type_col: Optional[str] = "sample_type"
    batch_col: Optional[str] = None
    case_label: str = "cancer"
    control_label: str = "control"
    gene_id_col: Optional[str] = None
    count_col: Optional[str] = None
    gene_annotation: Optional[str] = None
    gene_lengths: Optional[str] = None
    strip_ensembl_version: bool = True


@dataclass
class FilterConfig:
    min_count: int = 10
    min_samples: Optional[int] = None  # defaults to size of the smallest group
    min_total_count: int = 15
    drop_zero_variance: bool = True
    top_variable_genes: Optional[int] = 2000  # used for unsupervised plots only


@dataclass
class AnalysisConfig:
    de_method: str = "auto"  # auto | pydeseq2 | ttest
    normalization: str = "deseq"  # deseq | tmm | cpm
    alpha: float = 0.05
    lfc_threshold: float = 1.0
    shrink_lfc: bool = True
    per_sample_type: bool = True
    include_batch: bool = True
    min_group_size: int = 3


@dataclass
class BiomarkerConfig:
    top_n: int = 30
    min_auc: float = 0.70
    panel_size: int = 15
    cv_folds: int = 5
    cv_repeats: int = 10
    models: List[str] = field(default_factory=lambda: ["logistic_l1", "random_forest", "svm_rbf"])
    cross_sample_type_validation: bool = True
    permutations: int = 200
    n_boxplot_genes: int = 12


@dataclass
class EnrichmentConfig:
    enable: bool = True
    online: bool = False
    gene_sets: List[str] = field(
        default_factory=lambda: ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "GO_Biological_Process_2023"]
    )
    organism: str = "Human"
    offline_gmt: Optional[str] = "resources/hallmark_lite.gmt"
    top_terms: int = 20


@dataclass
class PlotConfig:
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 200
    label_top_n: int = 20
    heatmap_top_n: int = 50
    case_color: str = "#c0392b"
    control_color: str = "#2874a6"
    context: str = "notebook"


@dataclass
class ReportConfig:
    enable: bool = True
    embed_images: bool = True
    filename: str = "report.html"


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    input: InputConfig = field(default_factory=InputConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    biomarkers: BiomarkerConfig = field(default_factory=BiomarkerConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    base_dir: str = "."

    @property
    def outdir(self) -> Path:
        out = Path(self.project.outdir)
        return out if out.is_absolute() else Path(self.base_dir) / out

    def resolve(self, value: Optional[str]) -> Optional[Path]:
        """Resolve a config path relative to the directory holding the config file."""
        if value in (None, ""):
            return None
        path = Path(value)
        return path if path.is_absolute() else Path(self.base_dir) / path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build(cls, payload: Any):
    if not is_dataclass(cls) or payload is None:
        return cls()
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping for {cls.__name__}, got {type(payload).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown option(s) for {cls.__name__}: {', '.join(sorted(unknown))}")
    return cls(**{k: v for k, v in payload.items() if k in known})


def load_config(path: Optional[str]) -> Config:
    """Read a YAML config; missing sections fall back to the defaults above."""
    if path is None:
        return Config()
    cfg_path = Path(path).expanduser().resolve()
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = Config(
        project=_build(ProjectConfig, raw.get("project")),
        input=_build(InputConfig, raw.get("input")),
        filtering=_build(FilterConfig, raw.get("filtering")),
        analysis=_build(AnalysisConfig, raw.get("analysis")),
        biomarkers=_build(BiomarkerConfig, raw.get("biomarkers")),
        enrichment=_build(EnrichmentConfig, raw.get("enrichment")),
        plots=_build(PlotConfig, raw.get("plots")),
        report=_build(ReportConfig, raw.get("report")),
        base_dir=str(cfg_path.parent),
    )
    return cfg
