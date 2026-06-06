"""
Compile and fit the CNN model.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils.config import (
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LR_MIN,
    LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE,
)
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def _default_callbacks():
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=LR_MIN,
            verbose=1,
        ),
    ]


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = EPOCHS,
):
    """Compile (if needed), fit, return history."""
    try:
        if not model.optimizer:
            compile_model(model)
        callbacks = _default_callbacks()
        logger.info("Starting training for up to %d epochs", epochs)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )
        logger.info("Training complete")
        return history
    except Exception as e:
        raise CustomException(e) from e
