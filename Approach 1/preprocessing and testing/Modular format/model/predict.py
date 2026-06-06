"""
Inference utilities: load the saved model and predict on new images.
"""

from typing import List, Tuple

import numpy as np
import tensorflow as tf

from data_processing.preprocess import preprocess_image
from utils.config import CLASS_NAMES_PATH, MODEL_PATH, TARGET_SIZE
from utils.exception_handling import CustomException
from utils.helper_functions import load_class_names
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class ArSLPredictor:
    """Wraps a saved Keras model + class-name list for one-shot inference."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        class_names_path: str = CLASS_NAMES_PATH,
    ):
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.class_names: List[str] = load_class_names(class_names_path)
            logger.info(
                "Loaded model from %s with %d classes", model_path, len(self.class_names)
            )
        except Exception as e:
            raise CustomException(e) from e

    def predict_image(self, image_path: str) -> Tuple[str, float]:
        """Return (predicted_class_name, confidence)."""
        try:
            arr = preprocess_image(image_path, target_size=TARGET_SIZE)
            arr = np.expand_dims(arr, axis=0)  # add batch dim
            probs = self.model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(probs))
            return self.class_names[idx], float(probs[idx])
        except Exception as e:
            raise CustomException(e) from e

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices for a pre-batched numpy array of images."""
        try:
            probs = self.model.predict(X, verbose=0)
            return np.argmax(probs, axis=1)
        except Exception as e:
            raise CustomException(e) from e
