"""
evaluation/evaluate.py
======================
Evaluate a trained model on test data and report standard classification metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras


def evaluate_model(
    model:    keras.Model,
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    exp_name: str = "",
    verbose:  int = 1,
) -> dict:
    """
    Run inference on test data and compute weighted classification metrics.

    Args:
        model:    Trained Keras model.
        X_test:   Test sequences, shape (N, SEQ_LEN, FEAT_DIM).
        y_test:   True integer labels, shape (N,).
        exp_name: Label printed in the report header.
        verbose:  Keras verbosity for model.predict.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, y_pred, y_pred_prob.
    """
    y_pred_prob = model.predict(X_test, verbose=verbose)
    y_pred      = y_pred_prob.argmax(axis=1)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "y_pred":      y_pred,
        "y_pred_prob": y_pred_prob,
    }

    header = f"=== {exp_name} ===" if exp_name else "=== Evaluation ==="
    print(f"\n{header}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
) -> None:
    """Print a per-class sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the confusion matrix as a numpy array."""
    return confusion_matrix(y_true, y_pred)


def summarise_all_experiments(results: list) -> None:
    """
    Print a summary table of metrics across all experiments.

    Args:
        results: List of dicts, each with keys "name" and the metric keys
                 returned by evaluate_model.
    """
    print(f"\n{'Experiment':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 62)
    for r in results:
        print(
            f"{r['name']:<20}"
            f"{r['accuracy']:>10.4f}"
            f"{r['precision']:>10.4f}"
            f"{r['recall']:>10.4f}"
            f"{r['f1']:>10.4f}"
        )
