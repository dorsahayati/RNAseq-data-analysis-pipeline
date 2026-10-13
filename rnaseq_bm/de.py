"""Differential expression testing: PyDESeq2 (negative binomial) with a rank/t-test fallback."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config
from .normalize import cpm, normalized_counts
from .utils import bh_fdr, log

RESULT_COLUMNS = ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]


@dataclass
class DEResult:
    """Differential expression outcome for one contrast (all samples or one sample type)."""

    name: str
    label: str
    table: pd.DataFrame
    samples: List[str]
    method: str
    n_case: int
    n_control: int
    dispersion: Optional[pd.DataFrame] = None
    covariates: List[str] = field(default_factory=list)

    @property
    def significant(self) -> pd.DataFrame:
        return self.table[self.table["significant"]]

    @property
    def up(self) -> pd.Index:
        return self.table.index[self.table["direction"] == "up"]

    @property
    def down(self) -> pd.Index:
        return self.table.index[self.table["direction"] == "down"]

    def top(self, n: int = 50, by: str = "padj") -> pd.DataFrame:
        table = self.table.copy()
        if by == "padj":
            table = table.sort_values(["padj", "pvalue"], na_position="last")
        else:
            table = table.reindex(table[by].abs().sort_values(ascending=False).index)
        return table.head(n)

    def summary_row(self) -> Dict:
        return {
            "contrast": self.label,
            "method": self.method,
            "n_case": self.n_case,
            "n_control": self.n_control,
            "genes_tested": int(self.table["pvalue"].notna().sum()),
            "significant": int(self.table["significant"].sum()),
            "up": int((self.table["direction"] == "up").sum()),
            "down": int((self.table["direction"] == "down").sum()),
            "covariates": ",".join(self.covariates) or "none",
        }


def _finalize(table: pd.DataFrame, cfg: Config, gene_names: pd.Series) -> pd.DataFrame:
    table = table.copy()
    for column in RESULT_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    table["padj"] = table["padj"].astype(float)
    missing = table["padj"].isna() & table["pvalue"].notna()
    if missing.any():
        table.loc[missing, "padj"] = bh_fdr(table.loc[missing, "pvalue"])
    alpha = cfg.analysis.alpha
    lfc = cfg.analysis.lfc_threshold
    table["neg_log10_padj"] = -np.log10(table["padj"].clip(lower=1e-300))
    table["significant"] = (table["padj"] < alpha) & (table["log2FoldChange"].abs() >= lfc)
    table["direction"] = np.where(
        ~table["significant"], "ns", np.where(table["log2FoldChange"] > 0, "up", "down")
    )
    table["fold_change"] = np.sign(table["log2FoldChange"]) * (2 ** table["log2FoldChange"].abs())
    table["gene_name"] = gene_names.reindex(table.index).fillna(pd.Series(table.index, index=table.index))
    ordered = ["gene_name"] + RESULT_COLUMNS + ["fold_change", "neg_log10_padj", "significant", "direction"]
    rest = [c for c in table.columns if c not in ordered]
    return table[ordered + rest].sort_values(["padj", "pvalue"], na_position="last")


def _covariates(meta: pd.DataFrame, cfg: Config) -> List[str]:
    """Metadata columns to include in the design in addition to condition."""
    covariates = []
    batch = cfg.input.batch_col
    if cfg.analysis.include_batch and batch and batch in meta.columns:
        series = meta[batch].astype(str)
        # A covariate that is constant, or perfectly confounded with condition, cannot be fitted.
        confounded = pd.crosstab(series, meta[cfg.input.condition_col].astype(str))
        if series.nunique() > 1 and (confounded > 0).sum(axis=1).min() > 1:
            covariates.append(batch)
        elif series.nunique() > 1:
            log.warning("batch '%s' is confounded with condition - excluded from the design", batch)
    return covariates


def run_pydeseq2(counts: pd.DataFrame, meta: pd.DataFrame, cfg: Config, covariates: List[str]):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    cond_col = cfg.input.condition_col
    design_meta = meta[[cond_col] + covariates].astype(str).copy()
    design_factors = covariates + [cond_col]
    integer_counts = counts.T.round().astype(int)
    integer_counts = integer_counts.loc[design_meta.index]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(
            counts=integer_counts,
            metadata=design_meta,
            design_factors=design_factors,
            ref_level=[cond_col, cfg.input.control_label],
            refit_cooks=True,
            inference=DefaultInference(n_cpus=1),
            quiet=True,
        )
        dds.deseq2()
        stat = DeseqStats(
            dds,
            contrast=[cond_col, cfg.input.case_label, cfg.input.control_label],
            alpha=cfg.analysis.alpha,
            quiet=True,
        )
        stat.summary()
        table = stat.results_df.copy()
        if cfg.analysis.shrink_lfc:
            coefficients = list(getattr(dds, "varm", {}).get("LFC", pd.DataFrame()).columns)
            wanted = f"{cond_col}_{cfg.input.case_label}_vs_{cfg.input.control_label}"
            coeff = wanted if wanted in coefficients else next(
                (c for c in coefficients if cfg.input.case_label in c and cond_col in c), None
            )
            if coeff is not None:
                try:
                    stat.lfc_shrink(coeff=coeff)
                    shrunk = stat.results_df
                    table["log2FoldChange_unshrunk"] = table["log2FoldChange"]
                    table["log2FoldChange"] = shrunk["log2FoldChange"]
                    table["lfcSE"] = shrunk["lfcSE"]
                except Exception as exc:  # shrinkage is a nicety, never fatal
                    log.warning("LFC shrinkage failed (%s) - reporting unshrunk estimates", exc)
            else:
                log.warning("could not locate the LFC coefficient - reporting unshrunk estimates")

    dispersion = None
    try:
        dispersion = pd.DataFrame({
            "baseMean": np.asarray(dds.varm["_normed_means"]).ravel(),
            "genewise": np.asarray(dds.varm["genewise_dispersions"]).ravel(),
            "fitted": np.asarray(dds.varm["fitted_dispersions"]).ravel(),
            "final": np.asarray(dds.varm["dispersions"]).ravel(),
        }, index=counts.index)
    except Exception:
        log.info("dispersion estimates unavailable from the fitted model")
    return table, dispersion


def run_ttest(counts: pd.DataFrame, meta: pd.DataFrame, cfg: Config):
    """Welch t-test on log2 CPM plus a Mann-Whitney U test, used when PyDESeq2 is unavailable."""
    norm, _ = normalized_counts(counts, cfg.analysis.normalization)
    expr = np.log2(norm + 1)
    cond = meta[cfg.input.condition_col].astype(str)
    case = expr.loc[:, (cond == cfg.input.case_label).values]
    control = expr.loc[:, (cond == cfg.input.control_label).values]
    with np.errstate(invalid="ignore"):
        t_stat, t_p = stats.ttest_ind(case.values, control.values, axis=1, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(case.values, control.values, axis=1, alternative="two-sided")
    lfc = case.mean(axis=1) - control.mean(axis=1)
    n1, n2 = case.shape[1], control.shape[1]
    pooled_sd = np.sqrt(((n1 - 1) * case.var(axis=1, ddof=1) + (n2 - 1) * control.var(axis=1, ddof=1))
                        / max(n1 + n2 - 2, 1))
    table = pd.DataFrame({
        "baseMean": norm.mean(axis=1),
        "log2FoldChange": lfc,
        "lfcSE": pooled_sd * np.sqrt(1 / n1 + 1 / n2),
        "stat": t_stat,
        "pvalue": t_p,
        "padj": bh_fdr(t_p),
        "pvalue_mannwhitney": u_p,
        "padj_mannwhitney": bh_fdr(u_p),
        "cohens_d": lfc / pooled_sd.replace(0, np.nan),
    }, index=counts.index)
    _ = u_stat
    return table, None


def run_contrast(counts: pd.DataFrame, meta: pd.DataFrame, cfg: Config, gene_names: pd.Series,
                 name: str, label: str) -> Optional[DEResult]:
    """Run one cancer-vs-control contrast on the given sample subset."""
    cond = meta[cfg.input.condition_col].astype(str)
    n_case = int((cond == cfg.input.case_label).sum())
    n_control = int((cond == cfg.input.control_label).sum())
    if min(n_case, n_control) < cfg.analysis.min_group_size:
        log.warning("skipping contrast '%s': needs >=%d per group, got %d vs %d",
                    label, cfg.analysis.min_group_size, n_case, n_control)
        return None

    covariates = _covariates(meta, cfg)
    method = cfg.analysis.de_method
    table, dispersion = None, None
    if method in ("auto", "pydeseq2"):
        try:
            table, dispersion = run_pydeseq2(counts, meta, cfg, covariates)
            method = "PyDESeq2 (Wald test, negative binomial)"
        except Exception as exc:
            if cfg.analysis.de_method == "pydeseq2":
                raise
            log.warning("PyDESeq2 failed for '%s' (%s); falling back to Welch t-test", label, exc)
            table = None
    if table is None:
        table, dispersion = run_ttest(counts, meta, cfg)
        method = "Welch t-test on log2 CPM"
        covariates = []

    result = DEResult(
        name=name, label=label, table=_finalize(table, cfg, gene_names),
        samples=list(meta.index), method=method, n_case=n_case, n_control=n_control,
        dispersion=dispersion, covariates=covariates,
    )
    log.info("contrast %-22s %s | %d up, %d down (padj<%.2g, |log2FC|>=%.2g)",
             label, method.split(" ")[0], len(result.up), len(result.down),
             cfg.analysis.alpha, cfg.analysis.lfc_threshold)
    return result


def run_all_contrasts(counts: pd.DataFrame, meta: pd.DataFrame, cfg: Config,
                      gene_names: pd.Series) -> Dict[str, DEResult]:
    """Overall cancer-vs-control plus one contrast within each sample type."""
    results: Dict[str, DEResult] = {}
    overall = run_contrast(counts, meta, cfg, gene_names, "all_samples", "All samples")
    if overall is None:
        raise ValueError("the overall cancer vs control contrast could not be run - check group sizes")
    results["all_samples"] = overall

    stype_col = cfg.input.sample_type_col
    if cfg.analysis.per_sample_type and stype_col and stype_col in meta.columns:
        for sample_type, subset in meta.groupby(meta[stype_col].astype(str), observed=True):
            if subset[stype_col].nunique() == 0:
                continue
            result = run_contrast(
                counts[subset.index], subset, cfg, gene_names,
                name=str(sample_type), label=str(sample_type),
            )
            if result is not None:
                results[str(sample_type)] = result
    return results


def interaction_test(counts: pd.DataFrame, meta: pd.DataFrame, cfg: Config,
                     gene_names: pd.Series) -> Optional[pd.DataFrame]:
    """Test whether the cancer effect differs between sample types (condition x sample_type).

    Uses a likelihood-ratio-style comparison of ordinary-least-squares fits on
    variance-stabilised data, which is robust and does not depend on the DESeq2 design API.
    """
    stype_col = cfg.input.sample_type_col
    if not stype_col or stype_col not in meta.columns or meta[stype_col].nunique() < 2:
        return None
    from .normalize import vst

    norm, _ = normalized_counts(counts, cfg.analysis.normalization)
    expr = vst(norm)
    cond = (meta[cfg.input.condition_col].astype(str) == cfg.input.case_label).astype(float)
    types = pd.get_dummies(meta[stype_col].astype(str), drop_first=True).astype(float)
    intercept = np.ones((len(meta), 1))

    reduced = np.column_stack([intercept, cond.values[:, None], types.values])
    interactions = types.values * cond.values[:, None]
    full = np.column_stack([reduced, interactions])
    y = expr[meta.index].to_numpy(dtype=float).T

    def rss(design):
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coef
        return (residual ** 2).sum(axis=0)

    rss_reduced, rss_full = rss(reduced), rss(full)
    df1 = full.shape[1] - reduced.shape[1]
    df2 = len(meta) - full.shape[1]
    if df2 <= 0 or df1 <= 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        f_stat = ((rss_reduced - rss_full) / df1) / (rss_full / df2)
    pvalue = stats.f.sf(np.clip(f_stat, 0, None), df1, df2)
    table = pd.DataFrame({
        "gene_name": gene_names.reindex(expr.index).values,
        "F": f_stat,
        "pvalue": pvalue,
        "padj": bh_fdr(pvalue),
    }, index=expr.index).sort_values("pvalue")
    n_sig = int((table["padj"] < cfg.analysis.alpha).sum())
    log.info("condition x sample-type interaction: %d genes with FDR < %.2g", n_sig, cfg.analysis.alpha)
    return table


def expression_for_plots(counts: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """log2 CPM used for gene-level boxplots and biomarker modelling."""
    return cpm(counts, log2=True)
