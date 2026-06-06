"""
training/trainer.py
===================
Compiles the model, defines callbacks, and runs training for a single experiment.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config.config import (
    BATCH_SIZE,
    EPOCHS,
    ES_PATIENCE,
    LR,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    MODEL_DIR,
    RANDOM_SEED,
)
from model.architecture import build_arsl_tgru


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)


def compile_model(model: keras.Model, lr: float = LR) -> keras.Model:
    """
    Compile the model with Adam + sparse categorical cross-entropy.

    Args:
        model: Uncompiled Keras model.
        lr:    Initial learning rate.

    Returns:
        Compiled model.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("Model compiled ✓")
    return model


def get_callbacks(checkpoint_path: str) -> list:
    """
    Build the standard callback list: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.

    Args:
        checkpoint_path: Where to save the best model weights (.keras or .h5).

    Returns:
        List of Keras callbacks.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=ES_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
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


def train_experiment(
    exp_name:   str,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    model_dir:  str = MODEL_DIR,
    epochs:     int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[keras.Model, keras.callbacks.History]:
    """
    Build, compile, and train a fresh ArSL-TGRU model for one experiment.

    Args:
        exp_name:    Experiment label used to name the checkpoint file.
        X_train:     Training sequences, shape (N, 30, 159).
        y_train:     Training labels, shape (N,).
        X_val:       Validation sequences.
        y_val:       Validation labels.
        model_dir:   Directory where the best checkpoint is saved.
        epochs:      Maximum number of training epochs.
        batch_size:  Mini-batch size.

    Returns:
        (model, history) — the trained model (best weights restored) and its History object.
    """
    set_seeds()

    model = build_arsl_tgru()
    model = compile_model(model)
    model.summary()

    checkpoint_path = os.path.join(model_dir, f"best_arsl_tgru_{exp_name}.keras")
    callbacks       = get_callbacks(checkpoint_path)

    print(f"\n{'='*50}")
    print(f"  Training: {exp_name}")
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")
    print(f"{'='*50}\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n✅ Training complete — {exp_name}")
    return model, history
