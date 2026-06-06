"""Evaluate a trained model on a held-out signer."""

import numpy as np
import tensorflow as tf

from components.evaluation import (
    ClassificationMetrics,
    compute_metrics,
    predict_labels,
)
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "Evaluation",
) -> ClassificationMetrics:
    try:
        _, y_pred = predict_labels(model, X_test)
        metrics = compute_metrics(y_test, y_pred)
        print(metrics.pretty(title))
        logger.info("%s | %s", title, metrics)
        return metrics
    except Exception as e:
        raise CustomException(e) from e
