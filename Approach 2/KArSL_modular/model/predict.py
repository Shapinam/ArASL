"""
Inference on a raw video file: extract keypoints, normalize, predict label.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf

from data_processing.preprocess import (
    KeypointExtractor,
    normalize_frame,
    read_video,
    standardize_frames,
)
from utils.config import CLASSES, SEQ_LEN
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class ArSLVideoPredictor:
    """Loads a trained Keras model and predicts the sign from a video file."""

    def __init__(self, model_path: str):
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.kp = KeypointExtractor()
            self.class_names = CLASSES
            logger.info("Loaded model from %s (%d classes)", model_path, len(self.class_names))
        except Exception as e:
            raise CustomException(e) from e

    def _sequence_from_video(self, video_path: str) -> np.ndarray:
        frames = read_video(video_path)
        if not frames:
            raise ValueError(f"No frames decoded from {video_path}")
        std = standardize_frames(frames, SEQ_LEN)
        kps = np.stack([self.kp.extract(f) for f in std])
        norm = np.stack([normalize_frame(f) for f in kps])
        return norm

    def predict_video(self, video_path: str) -> Tuple[str, float]:
        """Return (class_name, confidence)."""
        try:
            seq = self._sequence_from_video(video_path)
            probs = self.model.predict(seq[np.newaxis, ...], verbose=0)[0]
            idx = int(np.argmax(probs))
            return self.class_names[idx], float(probs[idx])
        except Exception as e:
            raise CustomException(e) from e
