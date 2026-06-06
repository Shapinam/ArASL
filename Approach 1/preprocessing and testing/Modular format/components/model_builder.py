"""
CNN architecture for Arabic Sign Language recognition.
"""

from tensorflow.keras import layers, models

from utils.config import TARGET_SIZE
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_arsl_cnn(
    input_shape=(TARGET_SIZE, TARGET_SIZE, 1),
    num_classes: int = 32,
) -> models.Model:
    """
    Custom CNN designed for grayscale Arabic sign language recognition.

    Architecture:
      - 3 conv blocks with increasing filters (32 -> 64 -> 128)
      - BatchNorm after each conv (stabilizes training, mild regularizer)
      - MaxPool halves spatial dims each block (64 -> 32 -> 16 -> 8)
      - Dropout before Dense layers (prevents overfitting)
      - GlobalAveragePooling2D + two Dense layers + softmax
    """
    try:
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),

                # Block 1 — edge and texture detection
                layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                # Block 2 — shape and part detection
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                # Block 3 — high-level sign features
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.35),

                # Classifier head
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ],
            name="ArSL_CNN",
        )
        logger.info("Built ArSL_CNN with %d classes", num_classes)
        return model
    except Exception as e:
        raise CustomException(e) from e
