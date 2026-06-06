"""
DataIngestion: discover classes, load every image into memory as numpy arrays,
then split into train / val / test.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_processing.load_data import discover_classes
from data_processing.preprocess import preprocess_image
from utils.config import (
    INPUT_DIR,
    RANDOM_STATE,
    TARGET_SIZE,
    TEST_SIZE,
    VAL_TEST_SPLIT,
)
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class IngestedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val:   np.ndarray
    y_val:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    class_names: List[str]
    class_to_idx: Dict[str, int]


class DataIngestion:
    """End-to-end: discover -> preprocess -> stack -> split."""

    def __init__(self, input_dir: str = INPUT_DIR, target_size: int = TARGET_SIZE):
        self.input_dir = input_dir
        self.target_size = target_size

    def _load_all(
        self, valid_classes: Dict[str, List[str]], class_to_idx: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_all: List[np.ndarray] = []
        y_all: List[int] = []
        failed = 0
        for class_name, paths in tqdm(valid_classes.items(), desc="Loading classes"):
            label = class_to_idx[class_name]
            for path in paths:
                try:
                    X_all.append(preprocess_image(path, target_size=self.target_size))
                    y_all.append(label)
                except Exception:
                    failed += 1
        logger.info("Loaded %d images (failed: %d)", len(X_all), failed)
        return (
            np.array(X_all, dtype=np.float32),
            np.array(y_all, dtype=np.int32),
        )

    def run(self) -> IngestedData:
        try:
            valid_classes = discover_classes(self.input_dir)
            class_names   = sorted(valid_classes.keys())
            class_to_idx  = {name: i for i, name in enumerate(class_names)}

            X_all, y_all = self._load_all(valid_classes, class_to_idx)
            logger.info("Dataset shape: %s, labels: %s", X_all.shape, y_all.shape)

            # 80 / 10 / 10 stratified split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_all, y_all,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_all,
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=VAL_TEST_SPLIT,
                random_state=RANDOM_STATE,
                stratify=y_temp,
            )
            logger.info(
                "Splits — Train: %d, Val: %d, Test: %d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0],
            )

            return IngestedData(
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                X_test=X_test,   y_test=y_test,
                class_names=class_names,
                class_to_idx=class_to_idx,
            )
        except Exception as e:
            raise CustomException(e) from e
