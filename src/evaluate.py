"""Evaluation helpers for TobaPulse models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    recall_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Return JSON-serializable classification metrics."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    negative_recall = recall_score(y_true, y_pred, labels=["negative"], average="macro", zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "negative_recall": float(negative_recall),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "labels": labels,
    }


def class_metrics_frame(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def plot_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    output_path: Path,
    normalize: bool = False,
    title: str = "Sentiment Confusion Matrix",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_values = np.array(matrix)
    values = raw_values.astype(float)
    if normalize:
        row_totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_totals, out=np.zeros_like(values), where=row_totals != 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            label = f"{values[row, col]:.1%}\n(n={raw_values[row, col]})" if normalize else str(raw_values[row, col])
            ax.text(col, row, label, ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_class_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = metrics.set_index("label")[["precision", "recall", "f1"]]
    ax = plot_df.plot(kind="bar", figsize=(8, 5), ylim=(0, 1))
    ax.set_title("Sentiment Class Metrics")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    selected_threshold: float | None = None,
) -> None:
    """Plot the precision-recall tradeoff for complaint detection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    average_precision = average_precision_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#1261a0", linewidth=2, label=f"AP={average_precision:.3f}")
    if selected_threshold is not None and len(thresholds):
        index = int(np.argmin(np.abs(thresholds - selected_threshold)))
        ax.scatter(
            recall[index],
            precision[index],
            color="#c43d3d",
            s=55,
            zorder=3,
            label=f"threshold={selected_threshold:.3f}",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall keluhan")
    ax.set_ylabel("Precision keluhan")
    ax.set_title("Complaint Detector Precision-Recall")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Plot observed complaint frequency against predicted probability."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="ideal")
    ax.plot(predicted, observed, marker="o", color="#007f6d", linewidth=2, label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted complaint probability")
    ax.set_ylabel("Observed complaint rate")
    ax.set_title("Complaint Probability Calibration")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
