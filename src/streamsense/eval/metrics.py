"""Metrics, confusion matrix plotting, and ablation table writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class ClassificationReport:
    """Per-class and aggregate classification metrics."""

    class_names: list[str]
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def to_markdown(self) -> str:
        rows = ["| class | precision | recall | f1 | support |", "|---|---|---|---|---|"]
        for i, name in enumerate(self.class_names):
            rows.append(
                f"| {name} | {self.precision[i]:.3f} | {self.recall[i]:.3f} | "
                f"{self.f1[i]:.3f} | {int(self.support[i])} |"
            )
        rows += [
            "",
            f"- accuracy     : {self.accuracy:.4f}",
            f"- macro F1     : {self.macro_f1:.4f}",
            f"- weighted F1  : {self.weighted_f1:.4f}",
        ]
        return "\n".join(rows)


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]
) -> ClassificationReport:
    """Per-class precision/recall/F1 + macro/weighted F1, no sklearn dep."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1

    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    actual_pos = cm.sum(axis=1).astype(np.float64)

    prec = np.where(pred_pos > 0, tp / np.maximum(pred_pos, 1), 0.0)
    rec = np.where(actual_pos > 0, tp / np.maximum(actual_pos, 1), 0.0)
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / np.maximum(denom, 1e-12), 0.0)

    total = cm.sum()
    accuracy = float(tp.sum() / max(total, 1))
    macro_f1 = float(f1.mean())
    weights = actual_pos / max(total, 1)
    weighted_f1 = float((f1 * weights).sum())

    return ClassificationReport(
        class_names=list(class_names),
        precision=prec, recall=rec, f1=f1, support=actual_pos,
        accuracy=accuracy, macro_f1=macro_f1, weighted_f1=weighted_f1,
        confusion=cm,
    )


def save_confusion_png(cm: np.ndarray, class_names: Sequence[str], out_path: Path) -> Path:
    """Render a normalized confusion matrix and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("confusion (row-normalized)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_ablation_table(
    rows: list[dict], out_path: Path
) -> Path:
    """Write an ablation markdown table comparing model variants."""
    header = "| configuration | macro F1 | weighted F1 | accuracy | params |"
    sep = "|---|---|---|---|---|"
    body = []
    for r in rows:
        body.append(
            f"| {r['name']} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | "
            f"{r['accuracy']:.4f} | {r['params']:,d} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join([header, sep, *body, ""]))
    return out_path
