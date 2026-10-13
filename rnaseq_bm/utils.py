"""Shared helpers: logging, RNG control, multiple-testing correction, small stats."""

from __future__ import annotations

import logging
import random
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER_NAME = "rnaseq_bm"


def get_logger(logfile: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Return the pipeline logger, attaching a file handler the first time a path is given."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(stream)
    if logfile is not None and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(fh)
    return logger


log = get_logger()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def step(message: str):
    """Log the start and duration of a pipeline stage."""
    log.info(">>> %s", message)
    start = perf_counter()
    yield
    log.info("<<< %s (%.1fs)", message, perf_counter() - start)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaN inputs stay NaN and are excluded from n."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    pm = p[finite]
    n = pm.size
    if n == 0:
        return out
    order = np.argsort(pm)
    ranked = pm[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(q, 0.0, 1.0)
    out[finite] = adjusted
    return out


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise z-score; rows with zero variance become all zeros instead of NaN."""
    values = df.to_numpy(dtype=float)
    mu = np.nanmean(values, axis=1, keepdims=True)
    sd = np.nanstd(values, axis=1, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return pd.DataFrame((values - mu) / sd, index=df.index, columns=df.columns)


def clip_outliers(df: pd.DataFrame, limit: float = 3.0) -> pd.DataFrame:
    return df.clip(lower=-limit, upper=limit)


def sanitize(name: str) -> str:
    """Make a string safe for use in a file name."""
    keep = []
    for ch in str(name):
        keep.append(ch if (ch.isalnum() or ch in "-_.") else "_")
    return "".join(keep).strip("_") or "unnamed"


def strip_version(ids: Iterable[str]) -> list:
    """Drop Ensembl version suffixes (ENSG00000141510.16 -> ENSG00000141510)."""
    out = []
    for gene in ids:
        text = str(gene)
        if text.startswith("ENS") and "." in text:
            text = text.split(".")[0]
        out.append(text)
    return out


def significance_stars(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "ns"
    if pvalue < 1e-4:
        return "****"
    if pvalue < 1e-3:
        return "***"
    if pvalue < 1e-2:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return np.nan
    union = len(a | b)
    return len(a & b) / union if union else np.nan


def write_table(df: pd.DataFrame, path: Path, index: bool = True) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    log.info("wrote table %s (%d rows)", path.name, len(df))
    return path
