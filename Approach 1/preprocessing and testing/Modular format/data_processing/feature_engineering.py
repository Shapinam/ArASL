"""
Build tf.data pipelines and the augmentation function used during training.
"""

from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from utils.config import (
    BATCH_SIZE,
    BRIGHTNESS_DELTA,
    CONTRAST_LOWER,
    CONTRAST_UPPER,
    RANDOM_STATE,
    ROTATION_FACTOR,
    TRANSLATION_FACTOR,
    ZOOM_FACTOR,
)


# Augmentation layers are built once at import time so they don't create new
# tf.Variables every time the map function is traced.
_aug_rotation    = tf.keras.layers.RandomRotation(ROTATION_FACTOR, seed=RANDOM_STATE)
_aug_zoom        = tf.keras.layers.RandomZoom(ZOOM_FACTOR, seed=RANDOM_STATE)
_aug_translation = tf.keras.layers.RandomTranslation(
    TRANSLATION_FACTOR, TRANSLATION_FACTOR, seed=RANDOM_STATE
)


def augment(image: tf.Tensor, label: tf.Tensor):
    """Photometric + geometric augmentation for training images."""
    image = tf.image.random_flip_left_right(image, seed=RANDOM_STATE)
    image = tf.image.random_brightness(image, max_delta=BRIGHTNESS_DELTA, seed=RANDOM_STATE)
    image = tf.image.random_contrast(image, lower=CONTRAST_LOWER, upper=CONTRAST_UPPER, seed=RANDOM_STATE)
    image = _aug_rotation(image, training=True)
    image = _aug_zoom(image, training=True)
    image = _aug_translation(image, training=True)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def make_dataset(
    X: np.ndarray,
    y: np.ndarray,
    augment_fn: Optional[Callable] = None,
    batch_size: int = BATCH_SIZE,
    shuffle_buffer: int = 1000,
) -> tf.data.Dataset:
    """Build a batched, prefetched tf.data.Dataset from numpy arrays."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
