"""
model/architecture.py
=====================
Defines the ArSL-TGRU model: a hybrid architecture combining four sequential
Transformer blocks with two parallel GRU branches for Arabic Sign Language recognition.

Input:  (batch, SEQ_LEN=30, FEAT_DIM=159)
Output: (batch, N_CLASSES=59)  softmax probabilities
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config.config import (
    DROPOUT,
    FEAT_DIM,
    FF_DIM,
    GRU_DROPOUT,
    KEY_DIM,
    N_CLASSES,
    NUM_HEADS,
    SEQ_LEN,
)


# ── Positional encoding ───────────────────────────────────────────────────────

def positional_encoding(seq_len: int, dim: int) -> tf.Tensor:
    """
    Sinusoidal positional encoding matrix.

    Returns:
        Tensor of shape (seq_len, dim).
    """
    pe  = np.zeros((seq_len, dim))
    pos = np.arange(seq_len)[:, np.newaxis]
    div = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[: pe[:, 1::2].shape[1]])  # handles odd dim
    return tf.cast(pe, dtype=tf.float32)


class PositionalEmbedding(keras.layers.Layer):
    """
    Projects input features to `dim` dimensions and adds positional encoding.

    Args:
        seq_len: Sequence length (number of frames).
        dim:     Embedding dimensionality (= FEAT_DIM by default).
    """

    def __init__(self, seq_len: int, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed = keras.layers.Dense(dim)
        self.pe    = positional_encoding(seq_len, dim)

    def call(self, x):
        return self.embed(x) + self.pe


# ── Building blocks ───────────────────────────────────────────────────────────

def transformer_block(
    x,
    num_heads: int = NUM_HEADS,
    key_dim:   int = KEY_DIM,
    ff_dim:    int = FF_DIM,
    dropout:  float = DROPOUT,
):
    """
    Single Transformer encoder block followed by global average pooling.

    Architecture:
        LayerNorm → MultiHeadAttention → Residual add
        → Feed-forward (Dense + Dense) → Residual add
        → GlobalAveragePooling1D → Dense(64, relu) → Dropout

    Args:
        x:         Input tensor of shape (batch, time, features).
        num_heads: Number of attention heads.
        key_dim:   Dimensionality per attention head.
        ff_dim:    Hidden size of the feed-forward sub-layer.
        dropout:   Dropout rate after the dense layer.

    Returns:
        Tensor of shape (batch, 64).
    """
    # Self-attention sub-layer
    z = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    a = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(z, z)
    z = keras.layers.Add()([x, a])

    # Feed-forward sub-layer
    f = keras.layers.Dense(ff_dim, activation="relu")(z)
    f = keras.layers.Dense(x.shape[-1])(f)
    z = keras.layers.Add()([z, f])

    # Pool and project
    z = keras.layers.GlobalAveragePooling1D()(z)
    z = keras.layers.Dense(64, activation="relu")(z)
    z = keras.layers.Dropout(dropout)(z)
    return z


def gru_branch(x, dropout: float = GRU_DROPOUT):
    """
    Stacked GRU branch: 64 → 128 → 64 units, then Dense(64) + Dropout.

    Args:
        x:       Input tensor of shape (batch, time, features).
        dropout: Dropout rate after the dense layer.

    Returns:
        Tensor of shape (batch, 64).
    """
    g = keras.layers.GRU(64,  return_sequences=True)(x)
    g = keras.layers.GRU(128, return_sequences=True)(g)
    g = keras.layers.GRU(64,  return_sequences=False)(g)
    g = keras.layers.Dense(64, activation="relu")(g)
    g = keras.layers.Dropout(dropout)(g)
    return g


# ── Full model ────────────────────────────────────────────────────────────────

def build_arsl_tgru(
    seq_len:   int   = SEQ_LEN,
    feat_dim:  int   = FEAT_DIM,
    n_classes: int   = N_CLASSES,
) -> keras.Model:
    """
    Build the ArSL-TGRU model.

    Architecture overview:
        Input (30, 159)
            │
            ├─ PositionalEmbedding
            │  └─ 4 × TransformerBlock (sequential, reshaping between each)
            │
            ├─ GRU Branch 1
            │
            └─ GRU Branch 2
            │
        Concatenate([T, G1, G2])  →  (192,)
        Dense(128, relu) → Dropout(0.5) → Dense(n_classes, softmax)

    Args:
        seq_len:   Number of frames per sequence.
        feat_dim:  Feature vector length per frame.
        n_classes: Number of sign classes.

    Returns:
        Compiled-ready Keras Model.
    """
    inputs = keras.Input(shape=(seq_len, feat_dim), name="keypoints")

    # Transformer branch (4 stacked blocks)
    x = PositionalEmbedding(seq_len, feat_dim)(inputs)
    t = transformer_block(x)                     # (batch, 64)
    t = keras.layers.Reshape((1, 64))(t)
    t = transformer_block(t)                     # (batch, 64)
    t = keras.layers.Reshape((1, 64))(t)
    t = transformer_block(t)                     # (batch, 64)
    t = keras.layers.Reshape((1, 64))(t)
    t = transformer_block(t)                     # (batch, 64)

    # Two parallel GRU branches
    g1 = gru_branch(inputs)                      # (batch, 64)
    g2 = gru_branch(inputs)                      # (batch, 64)

    # Concatenate and classify
    merged = keras.layers.Concatenate()([t, g1, g2])   # (batch, 192)
    out    = keras.layers.Dense(128, activation="relu")(merged)
    out    = keras.layers.Dropout(0.5)(out)
    out    = keras.layers.Dense(n_classes, activation="softmax", name="predictions")(out)

    return keras.Model(inputs, out, name="ArSL_TGRU")
