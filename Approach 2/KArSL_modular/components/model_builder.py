"""
ArSL_TGRU architecture: Transformer branch + two parallel GRU branches,
concatenated and classified.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.config import (
    CLASSIFIER_DROPOUT,
    FEAT_DIM,
    GRU_DROPOUT,
    NUM_CLASSES,
    SEQ_LEN,
    TRANSFORMER_BLOCKS,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_KEY_DIM,
)
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def positional_encoding(seq_len: int, dim: int) -> tf.Tensor:
    """Sinusoidal positional encoding with safe handling of odd `dim`."""
    pe = np.zeros((seq_len, dim))
    pos = np.arange(seq_len)[:, np.newaxis]
    div = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[: pe[:, 1::2].shape[1]])
    return tf.cast(pe, dtype=tf.float32)


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, seq_len: int, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed = keras.layers.Dense(dim)
        self.pe = positional_encoding(seq_len, dim)

    def call(self, x):
        return self.embed(x) + self.pe


def transformer_block(
    x,
    num_heads: int = TRANSFORMER_HEADS,
    key_dim: int = TRANSFORMER_KEY_DIM,
    ff_dim: int = TRANSFORMER_FF_DIM,
    dropout: float = TRANSFORMER_DROPOUT,
):
    """LayerNorm -> MHA + residual -> FFN + residual -> pool + dense + dropout."""
    z = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    a = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(z, z)
    z = keras.layers.Add()([x, a])

    f = keras.layers.Dense(ff_dim, activation="relu")(z)
    f = keras.layers.Dense(x.shape[-1])(f)
    z = keras.layers.Add()([z, f])

    z = keras.layers.GlobalAveragePooling1D()(z)
    z = keras.layers.Dense(64, activation="relu")(z)
    z = keras.layers.Dropout(dropout)(z)
    return z


def gru_branch(x, dropout: float = GRU_DROPOUT):
    """3-layer GRU stack -> Dense(64) -> Dropout."""
    g = keras.layers.GRU(64,  return_sequences=True)(x)
    g = keras.layers.GRU(128, return_sequences=True)(g)
    g = keras.layers.GRU(64,  return_sequences=False)(g)
    g = keras.layers.Dense(64, activation="relu")(g)
    g = keras.layers.Dropout(dropout)(g)
    return g


def build_arsl_tgru(
    seq_len: int = SEQ_LEN,
    feat_dim: int = FEAT_DIM,
    n_classes: int = NUM_CLASSES,
) -> keras.Model:
    """Transformer branch (N blocks) + two parallel GRU branches -> classifier."""
    try:
        inputs = keras.Input(shape=(seq_len, feat_dim))

        # Transformer branch — first block consumes the full sequence; subsequent
        # blocks operate on the pooled (1, 64) vector by reshaping.
        x = PositionalEmbedding(seq_len, feat_dim)(inputs)
        t = transformer_block(x)
        for _ in range(TRANSFORMER_BLOCKS - 1):
            t = keras.layers.Reshape((1, 64))(t)
            t = transformer_block(t)

        g1 = gru_branch(inputs)
        g2 = gru_branch(inputs)

        merged = keras.layers.Concatenate()([t, g1, g2])
        out = keras.layers.Dense(128, activation="relu")(merged)
        out = keras.layers.Dropout(CLASSIFIER_DROPOUT)(out)
        out = keras.layers.Dense(n_classes, activation="softmax")(out)

        model = keras.Model(inputs, out, name="ArSL_TGRU")
        logger.info("Built ArSL_TGRU (n_classes=%d)", n_classes)
        return model
    except Exception as e:
        raise CustomException(e) from e
