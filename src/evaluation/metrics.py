"""Evaluation metrics and visualisation for deepfake detection.

Provides functions for:
- AUC-ROC computation
- Equal Error Rate (EER) and optimal threshold
- Confusion-matrix statistics (TP, FP, TN, FN, precision, recall, F1)
- A comprehensive ``full_evaluation`` aggregator
- Matplotlib-based ROC curve and confusion-matrix plotting
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Type aliases for readability
ArrayLike = Union[List[float], List[int], np.ndarray]


# ------------------------------------------------------------------
# Core metrics
# ------------------------------------------------------------------

def compute_auc_roc(y_true: ArrayLike, y_scores: ArrayLike) -> float:
    """Compute the Area Under the ROC Curve.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_scores: Predicted probabilities for the positive class.

    Returns:
        AUC-ROC value in ``[0, 1]``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    try:
        return float(roc_auc_score(y_true, y_scores))
    except ValueError as exc:
        logger.warning("AUC-ROC computation failed: %s", exc)
        return 0.0


def compute_eer(
    y_true: ArrayLike,
    y_scores: ArrayLike,
) -> tuple[float, float]:
    """Compute the Equal Error Rate and optimal threshold.

    The EER is the point on the ROC curve where the false-positive rate
    equals the false-negative rate (1 − TPR).

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Predicted probabilities for the positive class.

    Returns:
        A tuple ``(eer_value, optimal_threshold)``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr

    # Find the threshold where |FPR - FNR| is minimised
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer_value = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    optimal_threshold = float(thresholds[eer_idx])

    return eer_value, optimal_threshold


def compute_confusion_matrix(
    y_true: ArrayLike,
    y_preds: ArrayLike,
) -> Dict[str, float]:
    """Compute confusion-matrix statistics.

    Args:
        y_true: Ground-truth binary labels.
        y_preds: Predicted binary labels (already thresholded).

    Returns:
        A dict with keys ``tp``, ``fp``, ``tn``, ``fn``,
        ``precision``, ``recall``, ``f1``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_preds = np.asarray(y_preds, dtype=int)

    cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = float(precision_score(y_true, y_preds, zero_division=0))
    rec = float(recall_score(y_true, y_preds, zero_division=0))
    f1 = float(f1_score(y_true, y_preds, zero_division=0))

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def full_evaluation(
    y_true: ArrayLike,
    y_scores: ArrayLike,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run a comprehensive evaluation and return all metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Predicted probabilities for the positive class.
        threshold: Decision threshold for converting probabilities to
            binary predictions.

    Returns:
        A dict containing ``auc_roc``, ``eer``, ``eer_threshold``,
        ``threshold``, and all keys from
        :func:`compute_confusion_matrix`.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    y_preds = (y_scores >= threshold).astype(int)

    auc_val = compute_auc_roc(y_true, y_scores)
    eer_val, eer_thresh = compute_eer(y_true, y_scores)
    cm_stats = compute_confusion_matrix(y_true, y_preds)

    result: Dict[str, Any] = {
        "auc_roc": auc_val,
        "eer": eer_val,
        "eer_threshold": eer_thresh,
        "threshold": threshold,
        **cm_stats,
    }

    logger.info("Full evaluation: %s", result)
    return result


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def plot_roc_curve(
    y_true: ArrayLike,
    y_scores: ArrayLike,
    save_path: Optional[str] = None,
) -> None:
    """Plot the ROC curve and optionally save to disk.

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Predicted probabilities for the positive class.
        save_path: If provided, save the figure to this file path
            (e.g. ``"results/roc.png"``).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Deepfake Detection")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("ROC curve saved to %s.", save_path)

    plt.close(fig)


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_preds: ArrayLike,
    save_path: Optional[str] = None,
) -> None:
    """Plot a confusion-matrix heatmap and optionally save to disk.

    Args:
        y_true: Ground-truth binary labels.
        y_preds: Predicted binary labels (already thresholded).
        save_path: If provided, save the figure to this file path.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_preds = np.asarray(y_preds, dtype=int)

    cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
    labels = ["Real (0)", "Fake (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    fig.tight_layout()

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Confusion matrix saved to %s.", save_path)

    plt.close(fig)
