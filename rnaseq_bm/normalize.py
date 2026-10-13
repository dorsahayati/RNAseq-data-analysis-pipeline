"""Gene filtering and count normalisation (CPM, TMM, DESeq size factors, VST, TPM)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .utils import log


def filter_genes(counts: pd.DataFrame, meta: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep genes expressed above `min_count` in at least `min_samples` samples.

    Returns the filtered matrix and a per-step summary used for the QC report.
    """
    fc = cfg.filtering
    condition = meta[cfg.input.condition_col]
    min_samples = fc.min_samples
    if min_samples is None:
        min_samples = max(2, int(condition.value_counts().min()))

    n_start = counts.shape[0]
    steps = [("input", n_start)]

    expressed = (counts >= fc.min_count).sum(axis=1) >= min_samples
    counts = counts[expressed]
    steps.append((f">={fc.min_count} counts in >={min_samples} samples", counts.shape[0]))

    if fc.min_total_count:
        keep_total = counts.sum(axis=1) >= fc.min_total_count
        counts = counts[keep_total]
        steps.append((f"total count >= {fc.min_total_count}", counts.shape[0]))

    if fc.drop_zero_variance:
        keep_var = counts.var(axis=1) > 0
        counts = counts[keep_var]
        steps.append(("non-constant across samples", counts.shape[0]))

    if counts.shape[0] < 50:
        raise ValueError(
            f"only {counts.shape[0]} genes survived filtering - loosen filtering.min_count / min_samples"
        )
    log.info("gene filtering: %d -> %d genes retained (%.1f%%)",
             n_start, counts.shape[0], 100 * counts.shape[0] / max(n_start, 1))
    summary = pd.DataFrame(steps, columns=["step", "genes_retained"])
    return counts, summary


def cpm(counts: pd.DataFrame, log2: bool = False, prior: float = 1.0) -> pd.DataFrame:
    lib = counts.sum(axis=0).replace(0, np.nan)
    values = counts.divide(lib, axis=1) * 1e6
    if log2:
        values = np.log2(values + prior)
    return values.fillna(0.0)


def tpm(counts: pd.DataFrame, lengths: pd.Series, log2: bool = False) -> Optional[pd.DataFrame]:
    """Transcripts per million; requires gene lengths in bases."""
    if lengths is None:
        return None
    kb = (lengths.reindex(counts.index).astype(float) / 1000.0).replace(0, np.nan)
    if kb.isna().all():
        return None
    rate = counts.divide(kb, axis=0)
    values = rate.divide(rate.sum(axis=0).replace(0, np.nan), axis=1) * 1e6
    values = values.fillna(0.0)
    return np.log2(values + 1) if log2 else values


def deseq_size_factors(counts: pd.DataFrame) -> pd.Series:
    """DESeq2 median-of-ratios size factors."""
    values = counts.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_counts = np.log(values)
        log_geo_mean = np.nanmean(np.where(np.isfinite(log_counts), log_counts, np.nan), axis=1)
    usable = np.isfinite(log_geo_mean) & (values > 0).all(axis=1)
    if usable.sum() < 10:
        usable = np.isfinite(log_geo_mean)
    ratios = log_counts[usable] - log_geo_mean[usable, None]
    with np.errstate(invalid="ignore"):
        factors = np.exp(np.nanmedian(np.where(np.isfinite(ratios), ratios, np.nan), axis=0))
    factors = np.where(np.isfinite(factors) & (factors > 0), factors, 1.0)
    factors = factors / np.exp(np.mean(np.log(factors)))
    return pd.Series(factors, index=counts.columns, name="size_factor")


def tmm_factors(counts: pd.DataFrame, ref_index: Optional[int] = None,
                logratio_trim: float = 0.3, sum_trim: float = 0.05) -> pd.Series:
    """edgeR trimmed mean of M-values normalisation factors (library-size scaled)."""
    values = counts.to_numpy(dtype=float)
    lib = values.sum(axis=0)
    lib[lib == 0] = 1.0
    prop = values / lib
    if ref_index is None:
        # Reference = sample whose upper-quartile is closest to the mean upper-quartile.
        upper_q = np.quantile(prop, 0.75, axis=0)
        ref_index = int(np.argmin(np.abs(upper_q - upper_q.mean())))
    ref = values[:, ref_index]
    ref_lib = lib[ref_index]
    factors = np.ones(values.shape[1])
    for j in range(values.shape[1]):
        obs = values[:, j]
        keep = (obs > 0) & (ref > 0)
        if keep.sum() < 10:
            continue
        o, r = obs[keep], ref[keep]
        m = np.log2((o / lib[j]) / (r / ref_lib))
        a = 0.5 * np.log2((o / lib[j]) * (r / ref_lib))
        w = (lib[j] - o) / (lib[j] * o) + (ref_lib - r) / (ref_lib * r)
        finite = np.isfinite(m) & np.isfinite(a) & np.isfinite(w)
        m, a, w = m[finite], a[finite], w[finite]
        if m.size < 10:
            continue
        m_lo, m_hi = np.quantile(m, [logratio_trim, 1 - logratio_trim])
        a_lo, a_hi = np.quantile(a, [sum_trim, 1 - sum_trim])
        sel = (m >= m_lo) & (m <= m_hi) & (a >= a_lo) & (a <= a_hi)
        if sel.sum() < 5:
            sel = np.ones_like(m, dtype=bool)
        weights = 1.0 / np.where(w[sel] > 0, w[sel], np.nan)
        value = np.nansum(weights * m[sel]) / np.nansum(weights)
        factors[j] = 2 ** value if np.isfinite(value) else 1.0
    factors = factors / np.exp(np.mean(np.log(factors)))
    return pd.Series(factors, index=counts.columns, name="tmm_factor")


def normalized_counts(counts: pd.DataFrame, method: str = "deseq") -> Tuple[pd.DataFrame, pd.Series]:
    """Return library-normalised counts on the count scale plus the scaling factors used."""
    if method == "deseq":
        factors = deseq_size_factors(counts)
        return counts.divide(factors, axis=1), factors
    if method == "tmm":
        factors = tmm_factors(counts)
        effective = factors * counts.sum(axis=0)
        effective = effective / effective.mean()
        return counts.divide(effective, axis=1), factors
    if method == "cpm":
        factors = counts.sum(axis=0) / counts.sum(axis=0).mean()
        return counts.divide(factors, axis=1), factors
    raise ValueError(f"unknown normalization method '{method}'")


def vst(norm_counts: pd.DataFrame) -> pd.DataFrame:
    """Variance-stabilising transform.

    Fits the DESeq2 parametric mean-dispersion trend  disp = a0 + a1/mu  by
    regressing squared coefficient of variation on 1/mean, then applies the
    closed-form VST. Falls back to log2(x+1) when the fit is degenerate.
    """
    values = norm_counts.to_numpy(dtype=float)
    mu = values.mean(axis=1)
    var = values.var(axis=1, ddof=1)
    usable = (mu > 0) & np.isfinite(var)
    if usable.sum() < 50:
        return np.log2(norm_counts + 1)
    disp = (var[usable] - mu[usable]) / (mu[usable] ** 2)
    x = 1.0 / mu[usable]
    good = np.isfinite(disp) & (disp > 0)
    if good.sum() < 50:
        return np.log2(norm_counts + 1)
    design = np.column_stack([np.ones(good.sum()), x[good]])
    coef, *_ = np.linalg.lstsq(design, disp[good], rcond=None)
    a0, a1 = float(max(coef[0], 1e-8)), float(max(coef[1], 1e-8))
    with np.errstate(invalid="ignore", divide="ignore"):
        q = values
        inner = 1 + a1 + 2 * a0 * q + 2 * np.sqrt(np.clip(a0 * q * (1 + a1 + a0 * q), 0, None))
        transformed = (np.log(inner) - np.log(4 * a0)) / np.log(2)
    if not np.isfinite(transformed).all():
        return np.log2(norm_counts + 1)
    return pd.DataFrame(transformed, index=norm_counts.index, columns=norm_counts.columns)


def remove_batch_effect(expr: pd.DataFrame, batch: pd.Series, keep: Optional[pd.Series] = None) -> pd.DataFrame:
    """Least-squares removal of batch effects for visualisation only (limma removeBatchEffect)."""
    batch = batch.reindex(expr.columns).astype(str)
    if batch.nunique() < 2:
        return expr
    design_parts = [pd.get_dummies(batch, drop_first=True).astype(float)]
    if keep is not None and keep.nunique() > 1:
        design_parts.append(pd.get_dummies(keep.reindex(expr.columns).astype(str), drop_first=True).astype(float))
    design = pd.concat(design_parts, axis=1)
    n_batch = design_parts[0].shape[1]
    x = np.column_stack([np.ones(expr.shape[1]), design.to_numpy(dtype=float)])
    y = expr.to_numpy(dtype=float).T
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    batch_effect = x[:, 1:1 + n_batch] @ coef[1:1 + n_batch, :]
    return pd.DataFrame((y - batch_effect).T, index=expr.index, columns=expr.columns)


def top_variable_genes(expr: pd.DataFrame, n: Optional[int]) -> pd.Index:
    if not n or n >= expr.shape[0]:
        return expr.index
    return expr.var(axis=1).sort_values(ascending=False).head(int(n)).index
