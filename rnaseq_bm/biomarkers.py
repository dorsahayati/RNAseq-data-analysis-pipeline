"""Biomarker discovery: single-gene ROC performance, multivariate panels and cross-sample-type transfer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import Config
from .de import DEResult
from .utils import bh_fdr, log

MODEL_LABELS = {
    "logistic_l1": "L1 logistic regression",
    "logistic_l2": "L2 logistic regression",
    "random_forest": "Random forest",
    "svm_rbf": "SVM (RBF kernel)",
}


def build_model(name: str, seed: int):
    if name == "logistic_l1":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                                    max_iter=5000, random_state=seed))])
    if name == "logistic_l2":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=seed))])
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=1)
    if name == "svm_rbf":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                                     random_state=seed))])
    raise ValueError(f"unknown model '{name}'")


@dataclass
class PanelEvaluation:
    """Cross-validated performance of one classifier on one gene panel."""

    model: str
    genes: List[str]
    auc_folds: List[float]
    mean_fpr: np.ndarray
    tprs: np.ndarray
    y_true: np.ndarray
    y_score: np.ndarray
    metrics: Dict[str, float]
    importances: pd.Series
    nested_auc: Optional[List[float]] = None
    permutation_auc: Optional[np.ndarray] = None
    permutation_p: Optional[float] = None

    @property
    def label(self) -> str:
        return MODEL_LABELS.get(self.model, self.model)

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.auc_folds)) if self.auc_folds else np.nan

    @property
    def sd_auc(self) -> float:
        return float(np.std(self.auc_folds)) if self.auc_folds else np.nan


def _labels(meta: pd.DataFrame, cfg: Config) -> pd.Series:
    return (meta[cfg.input.condition_col].astype(str) == cfg.input.case_label).astype(int)


def _safe_auc(y: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def _cv_splitter(y: np.ndarray, cfg: Config, repeats: Optional[int] = None):
    minority = int(min(np.bincount(y))) if len(y) else 0
    folds = int(max(2, min(cfg.biomarkers.cv_folds, minority)))
    repeats = cfg.biomarkers.cv_repeats if repeats is None else repeats
    if repeats <= 1:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.project.seed)
    return RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=cfg.project.seed)


# --------------------------------------------------------------------------------------
# single-gene performance
# --------------------------------------------------------------------------------------

def youden_operating_point(y: np.ndarray, score: np.ndarray) -> Tuple[float, float, float]:
    """Threshold maximising sensitivity + specificity - 1, with the metrics there."""
    fpr, tpr, thresholds = roc_curve(y, score)
    youden = tpr - fpr
    best = int(np.argmax(youden))
    return float(thresholds[best]), float(tpr[best]), float(1 - fpr[best])


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 500,
                     seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], score[idx]))
    if len(values) < 20:
        return (np.nan, np.nan)
    return tuple(np.percentile(values, [2.5, 97.5]))


def univariate_performance(expr: pd.DataFrame, meta: pd.DataFrame, genes: Sequence[str],
                           cfg: Config, gene_names: pd.Series) -> pd.DataFrame:
    """Diagnostic performance of every candidate gene, overall and within each sample type."""
    y = _labels(meta, cfg)
    stype_col = cfg.input.sample_type_col
    has_types = bool(stype_col and stype_col in meta.columns and meta[stype_col].nunique() > 1)
    rows = []
    for gene in genes:
        values = expr.loc[gene, meta.index].to_numpy(dtype=float)
        yv = y.to_numpy()
        raw_auc = _safe_auc(yv, values)
        # Report every marker in its informative direction so all AUCs are >= 0.5.
        higher_in_controls = np.isfinite(raw_auc) and raw_auc < 0.5
        oriented = -values if higher_in_controls else values
        auc = max(raw_auc, 1 - raw_auc) if np.isfinite(raw_auc) else np.nan
        threshold, sensitivity, specificity = youden_operating_point(yv, oriented)
        low, high = bootstrap_auc_ci(yv, oriented, seed=cfg.project.seed)
        u_p = stats.mannwhitneyu(values[yv == 1], values[yv == 0], alternative="two-sided").pvalue
        row = {
            "gene_id": gene,
            "gene_name": gene_names.get(gene, gene),
            "direction": "down" if higher_in_controls else "up",
            "log2FC": float(np.nanmean(values[yv == 1]) - np.nanmean(values[yv == 0])),
            "auc": auc,
            "auc_ci_low": low,
            "auc_ci_high": high,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "youden_j": sensitivity + specificity - 1,
            "mannwhitney_p": u_p,
        }
        if has_types:
            type_aucs = {}
            for sample_type, subset in meta.groupby(meta[stype_col].astype(str), observed=True):
                sub_y = y.loc[subset.index].to_numpy()
                sub_values = expr.loc[gene, subset.index].to_numpy(dtype=float)
                sub_auc = _safe_auc(sub_y, sub_values)
                if np.isfinite(sub_auc):
                    sub_auc = sub_auc if row["direction"] == "up" else 1 - sub_auc
                type_aucs[sample_type] = sub_auc
                row[f"auc_{sample_type}"] = sub_auc
            finite = [v for v in type_aucs.values() if np.isfinite(v)]
            row["auc_min_across_types"] = min(finite) if finite else np.nan
            row["auc_range_across_types"] = (max(finite) - min(finite)) if len(finite) > 1 else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).set_index("gene_id")
    table["mannwhitney_padj"] = bh_fdr(table["mannwhitney_p"])
    sort_key = "auc_min_across_types" if "auc_min_across_types" in table.columns else "auc"
    table["robust_score"] = table[sort_key].fillna(table["auc"]) * table["log2FC"].abs().clip(upper=4) / 4
    return table.sort_values([sort_key, "auc"], ascending=False)


def select_candidates(de_results: Dict[str, DEResult], meta_table: Optional[pd.DataFrame],
                      cfg: Config, max_candidates: int = 400) -> List[str]:
    """Genes taken forward to biomarker evaluation, preferring cross-sample-type consistency."""
    overall = de_results["all_samples"]
    if meta_table is not None and not meta_table.empty:
        core = meta_table[meta_table["core_marker"]].sort_values("combined_pvalue")
        consistent = meta_table[meta_table["consistent"]
                                & (meta_table["combined_padj"] < cfg.analysis.alpha)].sort_values(
                                    "combined_pvalue")
        candidates = list(dict.fromkeys(list(core.index) + list(consistent.index)))
        if len(candidates) >= 10:
            log.info("biomarker candidates: %d genes from the cross-sample-type meta-analysis", len(candidates))
            return candidates[:max_candidates]
    significant = overall.significant
    if significant.empty:
        significant = overall.table.sort_values("pvalue")
    candidates = list(significant.head(max_candidates).index)
    log.info("biomarker candidates: %d genes from the overall contrast", len(candidates))
    return candidates


def rank_panel(performance: pd.DataFrame, expr: pd.DataFrame, meta: pd.DataFrame,
               cfg: Config) -> List[str]:
    """Greedy, redundancy-aware selection of the final panel."""
    size = cfg.biomarkers.panel_size
    ordered = performance.sort_values(
        ["auc_min_across_types" if "auc_min_across_types" in performance.columns else "auc", "auc"],
        ascending=False,
    )
    pool = [g for g in ordered.index if g in expr.index][: max(size * 6, 60)]
    if not pool:
        return []
    matrix = expr.loc[pool, meta.index]
    correlation = matrix.T.corr(method="spearman").abs()
    selected: List[str] = []
    for gene in pool:
        if len(selected) >= size:
            break
        if selected and correlation.loc[gene, selected].max() > 0.9:
            continue  # near-duplicate of a marker already in the panel
        selected.append(gene)
    if len(selected) < min(size, len(pool)):
        selected += [g for g in pool if g not in selected][: size - len(selected)]
    return selected


# --------------------------------------------------------------------------------------
# multivariate panel evaluation
# --------------------------------------------------------------------------------------

def _predict_scores(model, x_train, y_train, x_test) -> np.ndarray:
    fitted = clone(model).fit(x_train, y_train)
    if hasattr(fitted, "predict_proba"):
        return fitted.predict_proba(x_test)[:, 1]
    return fitted.decision_function(x_test)


def evaluate_panel(expr: pd.DataFrame, meta: pd.DataFrame, genes: Sequence[str], model_name: str,
                   cfg: Config) -> PanelEvaluation:
    """Repeated stratified cross-validation of one model on the fixed panel."""
    x = expr.loc[genes, meta.index].T.to_numpy(dtype=float)
    y = _labels(meta, cfg).to_numpy()
    model = build_model(model_name, cfg.project.seed)
    splitter = _cv_splitter(y, cfg)

    mean_fpr = np.linspace(0, 1, 101)
    tprs, aucs, all_true, all_score = [], [], [], []
    for train_idx, test_idx in splitter.split(x, y):
        score = _predict_scores(model, x[train_idx], y[train_idx], x[test_idx])
        auc = _safe_auc(y[test_idx], score)
        if np.isfinite(auc):
            aucs.append(auc)
            fpr, tpr, _ = roc_curve(y[test_idx], score)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        all_true.append(y[test_idx])
        all_score.append(score)
    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score)
    predicted = (y_score >= 0.5).astype(int) if y_score.min() >= 0 and y_score.max() <= 1 else \
        (y_score >= np.median(y_score)).astype(int)
    threshold, sensitivity, specificity = youden_operating_point(y_true, y_score)
    metrics = {
        "auc_pooled": _safe_auc(y_true, y_score),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "sensitivity_at_youden": sensitivity,
        "specificity_at_youden": specificity,
        "youden_threshold": threshold,
    }

    fitted = clone(model).fit(x, y)
    importances = _model_importance(fitted, x, y, genes, cfg)
    return PanelEvaluation(
        model=model_name, genes=list(genes), auc_folds=aucs, mean_fpr=mean_fpr,
        tprs=np.array(tprs) if tprs else np.zeros((0, len(mean_fpr))),
        y_true=y_true, y_score=y_score, metrics=metrics, importances=importances,
    )


def _model_importance(fitted, x, y, genes: Sequence[str], cfg: Config) -> pd.Series:
    estimator = fitted[-1] if isinstance(fitted, Pipeline) else fitted
    if hasattr(estimator, "coef_"):
        return pd.Series(np.abs(np.ravel(estimator.coef_)), index=list(genes)).sort_values(ascending=False)
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=list(genes)).sort_values(ascending=False)
    result = permutation_importance(fitted, x, y, n_repeats=10, random_state=cfg.project.seed,
                                    scoring="roc_auc")
    return pd.Series(result.importances_mean, index=list(genes)).sort_values(ascending=False)


def nested_cv(expr: pd.DataFrame, meta: pd.DataFrame, candidates: Sequence[str], model_name: str,
              cfg: Config) -> List[float]:
    """Honest performance estimate: genes are re-selected inside every training fold.

    Selecting a panel on all samples and then cross-validating it re-uses the labels twice and
    inflates the AUC. This repeats the selection within each fold so the reported number is
    what you would expect on new samples.
    """
    x_all = expr.loc[candidates, meta.index].T
    y = _labels(meta, cfg).to_numpy()
    model = build_model(model_name, cfg.project.seed)
    splitter = _cv_splitter(y, cfg, repeats=max(1, min(5, cfg.biomarkers.cv_repeats)))
    size = cfg.biomarkers.panel_size
    aucs = []
    for train_idx, test_idx in splitter.split(x_all.to_numpy(), y):
        train = x_all.iloc[train_idx]
        y_train = y[train_idx]
        scores = []
        for gene in x_all.columns:
            values = train[gene].to_numpy(dtype=float)
            auc = _safe_auc(y_train, values)
            scores.append(max(auc, 1 - auc) if np.isfinite(auc) else 0.5)
        ranking = pd.Series(scores, index=x_all.columns).sort_values(ascending=False)
        chosen = list(ranking.head(size).index)
        score = _predict_scores(model, train[chosen].to_numpy(), y_train,
                                x_all.iloc[test_idx][chosen].to_numpy())
        auc = _safe_auc(y[test_idx], score)
        if np.isfinite(auc):
            aucs.append(auc)
    return aucs


def permutation_test(expr: pd.DataFrame, meta: pd.DataFrame, genes: Sequence[str], model_name: str,
                     cfg: Config, observed: float) -> Tuple[np.ndarray, float]:
    """Null distribution of the panel AUC under randomly permuted labels."""
    n_perm = int(cfg.biomarkers.permutations)
    if n_perm <= 0:
        return np.array([]), np.nan
    x = expr.loc[genes, meta.index].T.to_numpy(dtype=float)
    y = _labels(meta, cfg).to_numpy()
    model = build_model(model_name, cfg.project.seed)
    rng = np.random.default_rng(cfg.project.seed)
    null = []
    for _ in range(n_perm):
        permuted = rng.permutation(y)
        splitter = StratifiedKFold(n_splits=int(max(2, min(cfg.biomarkers.cv_folds,
                                                           min(np.bincount(permuted))))),
                                   shuffle=True, random_state=int(rng.integers(1e6)))
        fold_scores, fold_truth = [], []
        for train_idx, test_idx in splitter.split(x, permuted):
            fold_scores.append(_predict_scores(model, x[train_idx], permuted[train_idx], x[test_idx]))
            fold_truth.append(permuted[test_idx])
        auc = _safe_auc(np.concatenate(fold_truth), np.concatenate(fold_scores))
        if np.isfinite(auc):
            null.append(auc)
    null = np.array(null)
    if null.size == 0:
        return null, np.nan
    pvalue = float((np.sum(null >= observed) + 1) / (null.size + 1))
    return null, pvalue


def standardize_within_group(expr: pd.DataFrame, meta: pd.DataFrame, genes: Sequence[str],
                             group: pd.Series) -> pd.DataFrame:
    """Z-score each gene inside each group, removing between-material baseline offsets."""
    blocks = []
    for _, samples in meta.groupby(group.astype(str), observed=True):
        block = expr.loc[genes, samples.index]
        blocks.append(block.sub(block.mean(axis=1), axis=0)
                           .div(block.std(axis=1).replace(0, np.nan), axis=0))
    return pd.concat(blocks, axis=1)[meta.index].fillna(0.0)


def cross_sample_type_transfer(expr: pd.DataFrame, meta: pd.DataFrame, genes: Sequence[str],
                               model_name: str, cfg: Config,
                               standardize: bool = True) -> Optional[pd.DataFrame]:
    """Train the panel on one sample type and test it on each of the others.

    Expression is z-scored within each material by default. Absolute expression levels differ
    enormously between tissue, plasma and blood, so without this step the matrix would only
    measure that offset instead of whether the marker behaviour itself transfers.
    """
    stype_col = cfg.input.sample_type_col
    if not stype_col or stype_col not in meta.columns or meta[stype_col].nunique() < 2:
        return None
    types = sorted(meta[stype_col].astype(str).unique())
    y = _labels(meta, cfg)
    model = build_model(model_name, cfg.project.seed)
    matrix = pd.DataFrame(index=types, columns=types, dtype=float)
    if standardize:
        expr = standardize_within_group(expr, meta, list(genes), meta[stype_col])
    for train_type in types:
        train_idx = meta.index[meta[stype_col].astype(str) == train_type]
        y_train = y.loc[train_idx].to_numpy()
        if len(np.unique(y_train)) < 2 or min(np.bincount(y_train)) < 2:
            continue
        x_train = expr.loc[genes, train_idx].T.to_numpy(dtype=float)
        for test_type in types:
            test_idx = meta.index[meta[stype_col].astype(str) == test_type]
            y_test = y.loc[test_idx].to_numpy()
            if len(np.unique(y_test)) < 2:
                continue
            if train_type == test_type:
                splitter = _cv_splitter(y_train, cfg, repeats=3)
                scores, truth = [], []
                for tr, te in splitter.split(x_train, y_train):
                    scores.append(_predict_scores(model, x_train[tr], y_train[tr], x_train[te]))
                    truth.append(y_train[te])
                matrix.loc[train_type, test_type] = _safe_auc(np.concatenate(truth), np.concatenate(scores))
            else:
                x_test = expr.loc[genes, test_idx].T.to_numpy(dtype=float)
                score = _predict_scores(model, x_train, y_train, x_test)
                matrix.loc[train_type, test_type] = _safe_auc(y_test, score)
    return matrix


def signature_score(expr: pd.DataFrame, meta: pd.DataFrame, up_genes: Sequence[str],
                    down_genes: Sequence[str], cfg: Config) -> pd.Series:
    """Mean z-score of up markers minus mean z-score of down markers.

    Z-scoring happens within each sample type so that baseline expression differences between
    materials do not shift the score.
    """
    stype_col = cfg.input.sample_type_col
    genes = [g for g in list(up_genes) + list(down_genes) if g in expr.index]
    matrix = expr.loc[genes, meta.index]
    if stype_col and stype_col in meta.columns and meta[stype_col].nunique() > 1:
        pieces = []
        for _, subset in meta.groupby(meta[stype_col].astype(str), observed=True):
            block = matrix[subset.index]
            centred = block.sub(block.mean(axis=1), axis=0).div(block.std(axis=1).replace(0, np.nan), axis=0)
            pieces.append(centred)
        z = pd.concat(pieces, axis=1)[meta.index].fillna(0.0)
    else:
        z = matrix.sub(matrix.mean(axis=1), axis=0).div(matrix.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    up = [g for g in up_genes if g in z.index]
    down = [g for g in down_genes if g in z.index]
    score = pd.Series(0.0, index=meta.index)
    if up:
        score = score + z.loc[up].mean(axis=0)
    if down:
        score = score - z.loc[down].mean(axis=0)
    return score.rename("signature_score")


def out_of_fold_scores(expr: pd.DataFrame, meta: pd.DataFrame, genes: Sequence[str],
                       model_name: str, cfg: Config) -> pd.Series:
    """One held-out prediction per sample, for per-sample plots."""
    x = expr.loc[genes, meta.index].T.to_numpy(dtype=float)
    y = _labels(meta, cfg).to_numpy()
    model = build_model(model_name, cfg.project.seed)
    folds = int(max(2, min(cfg.biomarkers.cv_folds, min(np.bincount(y)))))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.project.seed)
    scores = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(x, y):
        scores[test_idx] = _predict_scores(model, x[train_idx], y[train_idx], x[test_idx])
    return pd.Series(scores, index=meta.index, name="predicted_probability")


def panel_size_curve(expr: pd.DataFrame, meta: pd.DataFrame, ranked_genes: Sequence[str],
                     model_name: str, cfg: Config, max_size: int = 30) -> pd.DataFrame:
    """Cross-validated AUC as genes are added one at a time, to justify the panel size."""
    y = _labels(meta, cfg).to_numpy()
    model = build_model(model_name, cfg.project.seed)
    sizes = [s for s in range(1, min(max_size, len(ranked_genes)) + 1)]
    rows = []
    for size in sizes:
        genes = list(ranked_genes[:size])
        x = expr.loc[genes, meta.index].T.to_numpy(dtype=float)
        splitter = _cv_splitter(y, cfg, repeats=max(1, min(3, cfg.biomarkers.cv_repeats)))
        aucs = []
        for train_idx, test_idx in splitter.split(x, y):
            score = _predict_scores(model, x[train_idx], y[train_idx], x[test_idx])
            auc = _safe_auc(y[test_idx], score)
            if np.isfinite(auc):
                aucs.append(auc)
        rows.append({"panel_size": size, "mean_auc": np.mean(aucs) if aucs else np.nan,
                     "sd_auc": np.std(aucs) if aucs else np.nan})
    return pd.DataFrame(rows)
