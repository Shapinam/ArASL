"""
Video-level augmentation: small random rotation + scale applied to every frame
in the same way (so the temporal structure stays consistent).
"""

from typing import List, Optional

import cv2
import numpy as np

from utils.config import AUG_ROTATION_RANGE, AUG_SCALE_RANGE
from utils.exception_handling import CustomException


def augment_video(
    frames: List[np.ndarray],
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Apply the same random affine (rotate + scale) to every frame."""
    try:
        rng = np.random.default_rng(seed)
        angle = rng.uniform(*AUG_ROTATION_RANGE)
        scale = rng.uniform(*AUG_SCALE_RANGE)
        h, w = frames[0].shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        return [cv2.warpAffine(f, M, (w, h)) for f in frames]
    except Exception as e:
        raise CustomException(e) from e
