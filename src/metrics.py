"""Custom metric utilities for fraud detection evaluation.

Implements Precision@r recall with linear interpolation and cost-optimal
threshold computation. Standard sklearn metrics are used directly via
sklearn.metrics — this module adds the domain-specific utilities on top.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
)


# ---------------------------------------------------------------------------
# Core domain utilities
# ---------------------------------------------------------------------------

def cost_optimal_threshold(cost_fp: float, cost_fn: float) -> float:
    """Compute the optimal decision threshold under asymmetric costs.

    Derived from minimizing expected cost: predict fraud when
    p(fraud|x) > C_FP / (C_FP + C_FN).

    See notes/phase1-theory.md §1.3 for the full derivation.

    Parameters
    ----------
    cost_fp :
        Cost of blocking a legitimate transaction (false positive).
    cost_fn :
        Cost of approving a fraudulent transaction (false negative).

    Returns
    -------
    tau_star : float in (0, 1)
    """
    return cost_fp / (cost_fp + cost_fn)


def precision_at_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    recall_target: float,
    interpolate: bool = True,
) -> tuple[float, float]:
    """Compute Precision at a specific recall level using linear interpolation.

    Precision@r is the precision achievable when the model is calibrated to
    catch exactly r * 100% of fraud cases. The value r is a business parameter,
    not a fixed metric property.

    Parameters
    ----------
    y_true :
        True binary labels (1 = fraud, 0 = legitimate).
    y_scores :
        Model fraud probability scores.
    recall_target :
        Desired recall level r in (0, 1].
    interpolate :
        If True, linearly interpolate between adjacent PR curve points.

    Returns
    -------
    (precision, threshold) at the point closest to recall_target.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # precision_recall_curve returns one extra point (threshold=0 sentinel).
    # Align precisions and recalls to the thresholds array.
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if len(thresholds) == 0:
        return 0.0, 0.0

    if not interpolate:
        idx = int(np.argmin(np.abs(recalls - recall_target)))
        return float(precisions[idx]), float(thresholds[idx])

    # Find bracketing indices: one point with recall >= target, one below
    above = np.where(recalls >= recall_target)[0]
    below = np.where(recalls < recall_target)[0]

    if len(above) == 0:
        # recall_target is below the minimum achievable recall — return last point
        return float(precisions[-1]), float(thresholds[-1])
    if len(below) == 0:
        # recall_target is above the maximum achievable recall — return first point
        return float(precisions[0]), float(thresholds[0])

    i_above = above[-1]   # largest index with recall >= target
    i_below = below[0]    # smallest index with recall < target

    r1, p1, t1 = recalls[i_above], precisions[i_above], thresholds[i_above]
    r2, p2, t2 = recalls[i_below], precisions[i_below], thresholds[i_below]

    if r1 == r2:
        return float(p1), float(t1)

    alpha = (recall_target - r1) / (r2 - r1)
    return float(p1 + alpha * (p2 - p1)), float(t1 + alpha * (t2 - t1))


# ---------------------------------------------------------------------------
# Per-threshold computations
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> dict:
    """Compute Precision, Recall, F1, and confusion matrix counts at one threshold.

    Returns
    -------
    dict with keys: threshold, precision, recall, f1, tp, fp, fn, tn.
    """
    y_pred = (y_scores >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute the full evaluation metric suite for a fraud detection model.

    Parameters
    ----------
    y_true :
        True binary labels.
    y_scores :
        Fraud probability scores from the model.
    threshold :
        Decision threshold for converting scores to binary predictions.

    Returns
    -------
    dict with: threshold, precision, recall, f1, tp, fp, fn, tn,
               accuracy, auc_roc, auc_pr.
    """
    metrics = compute_metrics_at_threshold(y_true, y_scores, threshold)
    y_pred = (y_scores >= threshold).astype(int)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # AUC metrics require score variation; handle degenerate case gracefully
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        metrics["auc_roc"] = float("nan")

    try:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_scores))
    except ValueError:
        metrics["auc_pr"] = float("nan")

    return metrics


def sweep_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict]:
    """Compute Precision, Recall, and F1 at every threshold in a grid.

    Returns a list of metric dicts, one per threshold, in the same order
    as the input thresholds array.
    """
    return [compute_metrics_at_threshold(y_true, y_scores, t) for t in thresholds]
