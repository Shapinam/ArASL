"""Compute weighted accuracy/precision/recall/F1 and pretty-print them."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def pretty(self, title: str = "Evaluation") -> str:
        return (
            f"=== {title} ===\n"
            f"Accuracy:  {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\n"
            f"F1-Score:  {self.f1:.4f}"
        )


def predict_labels(
    model: tf.keras.Model, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, argmax labels)."""
    probs = model.predict(X)
    return probs, probs.argmax(axis=1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    try:
        return ClassificationMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average="weighted", zero_division=0),
            recall=recall_score(y_true, y_pred, average="weighted", zero_division=0),
            f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        )
    except Exception as e:
        raise CustomException(e) from e
