"""
Rich evaluation: classification report and confusion-matrix plotting.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def collect_predictions(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate `dataset` and return (y_true, y_pred) arrays in matching order."""
    try:
        y_true_list, y_pred_list = [], []
        for X_batch, y_batch in dataset:
            preds = model.predict(X_batch, verbose=0)
            y_pred_list.extend(np.argmax(preds, axis=1))
            y_true_list.extend(y_batch.numpy())
        return np.array(y_true_list), np.array(y_pred_list)
    except Exception as e:
        raise CustomException(e) from e


def print_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> str:
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    return report


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: str = None
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info("Saved confusion matrix to %s", save_path)
    plt.show()


def plot_training_history(history, save_path: str = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"],     label="Train accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"],     label="Train loss")
    ax2.plot(history.history["val_loss"], label="Val loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info("Saved training-history plot to %s", save_path)
    plt.show()
