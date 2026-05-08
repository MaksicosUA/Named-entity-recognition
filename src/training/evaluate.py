from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _flatten_valid_tokens(predictions: np.ndarray, labels: np.ndarray) -> tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=-1)

    y_true: List[int] = []
    y_pred: List[int] = []

    for pred_row, label_row in zip(preds, labels):
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            y_true.append(int(l))
            y_pred.append(int(p))

    return y_true, y_pred


def flatten_valid_tokens(predictions: np.ndarray, labels: np.ndarray) -> tuple[List[int], List[int]]:
    """Public helper used by reporting modules."""
    return _flatten_valid_tokens(predictions, labels)


def compute_token_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    y_true, y_pred = _flatten_valid_tokens(predictions, labels)

    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def token_level_report(predictions: np.ndarray, labels: np.ndarray, id2label: dict[int, str]) -> Dict[str, object]:
    y_true, y_pred = _flatten_valid_tokens(predictions, labels)

    if not y_true:
        return {"per_label": {}, "support": 0}

    unique_ids = sorted(set(y_true))
    report: Dict[str, object] = {"per_label": {}, "support": len(y_true)}

    for label_id in unique_ids:
        label_name = id2label[int(label_id)]
        yt = [1 if x == label_id else 0 for x in y_true]
        yp = [1 if x == label_id else 0 for x in y_pred]
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        support = int(sum(yt))
        report["per_label"][label_name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": support,
        }

    return report
