"""
Evaluate a trained model on the test set.
"""

from typing import Tuple

import tensorflow as tf

from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: tf.keras.Model, test_ds: tf.data.Dataset
) -> Tuple[float, float]:
    """Return (test_loss, test_accuracy)."""
    try:
        test_loss, test_acc = model.evaluate(test_ds)
        logger.info("Test accuracy: %.4f, Test loss: %.4f", test_acc, test_loss)
        return test_loss, test_acc
    except Exception as e:
        raise CustomException(e) from e
