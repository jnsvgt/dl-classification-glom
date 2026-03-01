"""Metrics computation and confusion matrix plotting."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
)


@dataclass
class MetricsSet:
    """Container for all evaluation metrics."""

    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    cohen_kappa: float = 0.0
    mcc: float = 0.0
    auc_ovr: Optional[float] = None

    val_loss: float = 0.0
    best_epoch: int = 0

    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_support: Dict[str, int] = field(default_factory=dict)
    confusion_mat: Optional[np.ndarray] = None
    report: str = ""

    def to_dict(self):
        d = {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "cohen_kappa": self.cohen_kappa,
            "mcc": self.mcc,
            "val_loss": self.val_loss,
            "best_epoch": self.best_epoch,
            "per_class_f1": self.per_class_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_support": self.per_class_support,
        }
        if self.auc_ovr is not None:
            d["auc_ovr"] = self.auc_ovr
        if self.confusion_mat is not None:
            d["confusion_matrix"] = self.confusion_mat.tolist()
        return d


def compute_all_metrics(y_true, y_pred, y_proba, class_names,
                        val_loss=0.0, best_epoch=0):
    """Compute a full set of classification metrics."""
    m = MetricsSet()
    m.val_loss = val_loss
    m.best_epoch = best_epoch

    m.accuracy = accuracy_score(y_true, y_pred)
    m.balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    m.macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m.weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    m.macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m.macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    m.cohen_kappa = cohen_kappa_score(y_true, y_pred)
    m.mcc = matthews_corrcoef(y_true, y_pred)

    # per-class
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    precs = precision_score(y_true, y_pred, average=None, zero_division=0)
    recs = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        if i < len(f1s):
            m.per_class_f1[name] = float(f1s[i])
            m.per_class_precision[name] = float(precs[i])
            m.per_class_recall[name] = float(recs[i])
    # support
    _, _, _, support = precision_score(y_true, y_pred, average=None,
                                       zero_division=0), None, None, None
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0, output_dict=True)
    for name in class_names:
        if name in report:
            m.per_class_support[name] = int(report[name]["support"])

    # AUC (one-vs-rest)
    try:
        if y_proba is not None and len(class_names) > 2:
            m.auc_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except (ValueError, IndexError):
        pass

    m.confusion_mat = confusion_matrix(y_true, y_pred)
    m.report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    return m


def plot_confusion_matrix(cm, class_names, save_path=None, normalize=True,
                          figsize=(12, 10), title="Confusion Matrix"):
    """Plot a confusion matrix as a heatmap."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, row_sums, where=row_sums != 0,
                            out=np.zeros_like(cm, dtype=float))
        fmt = ".2f"
    else:
        cm_plot = cm.astype(float)
        fmt = "d"

    # short labels (drop pattern prefix)
    short = []
    for name in class_names:
        parts = name.split("_", 1)
        short.append(parts[1] if len(parts) > 1 else name)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=short, yticklabels=short, ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
