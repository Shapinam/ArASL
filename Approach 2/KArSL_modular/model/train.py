"""Compile + fit with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint."""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LR_MIN,
    LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE,
    MODEL_DIR,
)
from utils.exception_handling import CustomException
from utils.helper_functions import ensure_dir
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def compile_model(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _callbacks(checkpoint_path: str):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=LR_MIN,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
    ]


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    experiment_name: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    model_dir: str = MODEL_DIR,
):
    """Fit `model` and persist the best checkpoint as <experiment_name>.keras."""
    try:
        if not model.optimizer:
            compile_model(model)
        ensure_dir(model_dir)
        ckpt = os.path.join(model_dir, f"best_arsl_tgru_{experiment_name}.keras")
        logger.info("Training %s (epochs<=%d, batch=%d)", experiment_name, epochs, batch_size)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=_callbacks(ckpt),
            verbose=1,
        )
        logger.info("Training %s complete. Best checkpoint: %s", experiment_name, ckpt)
        return history, ckpt
    except Exception as e:
        raise CustomException(e) from e
