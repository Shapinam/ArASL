"""
Load .npy keypoint sequences for given (split, signers) combinations.
Each .npy is shape (SEQ_LEN, FEAT_DIM) — sequences with any other shape are skipped.
"""

import glob
from typing import List, Tuple

import numpy as np

from utils.config import CLASS_TO_LABEL, EXPECTED_SHAPE, NPY_DIR
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def load_split(
    split: str,
    signers: List[str],
    npy_dir: str = NPY_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk `npy_dir/<split>/<class>/<signer>/*.npy` and stack everything that has
    the expected shape into (N, SEQ_LEN, FEAT_DIM) and labels (N,).
    """
    try:
        X, y = [], []
        bad = 0
        for cls_folder, label in CLASS_TO_LABEL.items():
            for signer in signers:
                pattern = f"{npy_dir}/{split}/{cls_folder}/{signer}/*.npy"
                for f in glob.glob(pattern):
                    seq = np.load(f).astype(np.float32)
                    if seq.shape == EXPECTED_SHAPE:
                        X.append(seq)
                        y.append(label)
                    else:
                        bad += 1
                        logger.warning("Bad shape %s in %s", seq.shape, f)
        logger.info(
            "Loaded split=%s signers=%s -> N=%d (bad=%d)",
            split, signers, len(X), bad,
        )
        return np.array(X), np.array(y)
    except Exception as e:
        raise CustomException(e) from e
