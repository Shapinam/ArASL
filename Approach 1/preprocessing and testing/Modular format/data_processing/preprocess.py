"""
Per-image preprocessing: grayscale → letterbox → resize → normalize.
"""

import numpy as np
from PIL import Image

from utils.config import TARGET_SIZE
from utils.exception_handling import CustomException


def preprocess_image(image_path: str, target_size: int = TARGET_SIZE) -> np.ndarray:
    """
    Robust preprocessing for grayscale sign-language images.

    Steps:
        1. Open and force grayscale (handles RGB, P, RGBA, L).
        2. Clip pixel range to [0, 255].
        3. Letterbox-pad to square (preserves aspect ratio).
        4. Resize to target_size x target_size with LANCZOS.
        5. Normalize to [0, 1] and add channel dim -> (H, W, 1) float32.
    """
    try:
        img = Image.open(image_path)

        # 1. Force to grayscale regardless of source mode
        img = img.convert("L")

        # 2. Clip out-of-range pixels
        img_array = np.array(img, dtype=np.float32)
        img_array = np.clip(img_array, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))

        # 3. Letterbox to square (black padding, preserves hand shape)
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        square_img = Image.new("L", (max_dim, max_dim), color=0)
        square_img.paste(img, (pad_left, pad_top))

        # 4. Resize
        square_img = square_img.resize((target_size, target_size), Image.LANCZOS)

        # 5. Normalize and add channel dim
        final_array = np.array(square_img, dtype=np.float32) / 255.0
        final_array = np.expand_dims(final_array, axis=-1)
        return final_array
    except Exception as e:
        raise CustomException(e) from e
