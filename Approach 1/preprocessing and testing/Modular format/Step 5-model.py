# model.py
# CNN architecture definition

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from config import TARGET_SIZE


def build_arsl_cnn(input_shape=(TARGET_SIZE, TARGET_SIZE, 1), num_classes=32):
    """
    Custom CNN for grayscale Arabic Sign Language recognition.

    Architecture:
    - 3 conv blocks with increasing filters (32 → 64 → 128)
    - BatchNorm after each conv (stabilizes training, acts as regularizer)
    - MaxPool halves spatial dims each block (64 → 32 → 16 → 8)
    - Dropout before Dense layers (prevents overfitting)
    - GlobalAveragePooling instead of Flatten (better generalization)
    - Two Dense layers before final softmax
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1 — edge and texture detection
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2 — shape and part detection
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3 — high-level sign features
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.35),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ], name="ArSL_CNN")

    return model


if __name__ == '__main__':
    model = build_arsl_cnn()
    model.summary()

    dummy = np.zeros((1, TARGET_SIZE, TARGET_SIZE, 1), dtype=np.float32)
    out = model(dummy)
    print(f"\nOutput shape: {out.shape}")   # (1, 32)
