"""
Data pipeline: ingest raw images and turn them into tf.data.Datasets ready
for training.
"""

from dataclasses import dataclass
from typing import List

import tensorflow as tf

from components.data_ingestion import DataIngestion
from data_processing.feature_engineering import augment, make_dataset
from utils.config import BATCH_SIZE, INPUT_DIR
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class DataBundle:
    train_ds: tf.data.Dataset
    val_ds:   tf.data.Dataset
    test_ds:  tf.data.Dataset
    class_names: List[str]
    num_classes: int


def run_data_pipeline(
    input_dir: str = INPUT_DIR, batch_size: int = BATCH_SIZE
) -> DataBundle:
    """End-to-end data prep: returns datasets + class metadata."""
    try:
        ingestion = DataIngestion(input_dir=input_dir)
        data = ingestion.run()

        train_ds = make_dataset(
            data.X_train, data.y_train, augment_fn=augment, batch_size=batch_size
        )
        val_ds = make_dataset(
            data.X_val, data.y_val, augment_fn=None, batch_size=batch_size
        )
        test_ds = make_dataset(
            data.X_test, data.y_test, augment_fn=None, batch_size=batch_size
        )

        logger.info("Data pipeline ready (%d classes)", len(data.class_names))
        return DataBundle(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            class_names=data.class_names,
            num_classes=len(data.class_names),
        )
    except Exception as e:
        raise CustomException(e) from e
