"""
Three ingestion components:

- ArchiveExtractor : 7z-extract chosen classes from per-signer/per-split archives
- VideoProcessor   : turn every .mp4 into a (30, 159) keypoint sequence on disk
- ExperimentSplitter: produce X_train/val/test arrays for each leave-one-signer-out
                     experiment and save them to disk
"""

import gc
import glob
import os
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

from data_processing.feature_engineering import augment_video
from data_processing.load_data import load_split
from data_processing.preprocess import (
    KeypointExtractor,
    normalize_frame,
    read_video,
    standardize_frames,
)
from utils.config import (
    BASE_DIR,
    CLASSES_INT,
    KEEP_FIRST,
    KEEP_SECOND,
    N_AUG,
    NPY_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    SEQ_LEN,
    SIGNER_EXPERIMENTS,
    SIGNERS,
    SPLITS,
    VAL_SPLIT,
)
from utils.exception_handling import CustomException
from utils.helper_functions import ensure_dir, run_7z
from utils.logging_setup import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class ArchiveExtractor:
    """Extract a fixed subset of classes from the KArSL 7z archives."""

    def __init__(self, base_dir: str = BASE_DIR):
        self.base_dir = base_dir

    def _extract_classes(self, archive_path: str, class_list: List[str],
                         out_dir: str, archive_prefix: str) -> None:
        for cls in class_list:
            logger.info("Extracting class %s from %s", cls, os.path.basename(archive_path))
            run_7z(archive_path, f"{archive_prefix}{cls}\\*", out_dir)

    def run(self) -> None:
        try:
            for signer in SIGNERS:
                for split in SPLITS:
                    archive_dir = f"{self.base_dir}/{signer}/{split}"
                    out_dir     = f"{archive_dir}/extracted"

                    # First archive — name varies between 0001-0070.7z and 0001-0071.7z
                    arch1 = f"{archive_dir}/0001-0070.7z"
                    if not os.path.exists(arch1):
                        arch1 = f"{archive_dir}/0001-0071.7z"
                    arch2 = f"{archive_dir}/0071-0170.7z"

                    if os.path.exists(arch1):
                        prefix1 = os.path.basename(arch1).replace(".7z", "") + "\\"
                        self._extract_classes(arch1, KEEP_FIRST, out_dir, prefix1)
                    else:
                        logger.warning("[%s/%s] first archive missing", signer, split)

                    if os.path.exists(arch2):
                        self._extract_classes(arch2, KEEP_SECOND, out_dir, "0071-0170\\")
                    else:
                        logger.warning("[%s/%s] second archive missing", signer, split)

                logger.info("Signer %s done", signer)
            logger.info("All signers extracted")
        except Exception as e:
            raise CustomException(e) from e


# ─────────────────────────────────────────────────────────────────────────────
class VideoProcessor:
    """
    Convert every .mp4 in `base_dir/<signer>/<split>/extracted/<cls>/*.mp4` to a
    normalized keypoint sequence saved as .npy under `npy_dir`.

    For training videos we save the original PLUS N_AUG augmented copies; test
    videos are saved without augmentation.
    """

    def __init__(self, base_dir: str = BASE_DIR, npy_dir: str = NPY_DIR):
        self.base_dir = base_dir
        self.npy_dir  = npy_dir
        self.kp = KeypointExtractor()

    def _process_video(self, vid_path: str, out_dir: str, augment: bool) -> int:
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        frames = read_video(vid_path)
        versions = [frames]
        if augment:
            versions += [augment_video(frames, seed=i) for i in range(N_AUG)]

        saved = 0
        for v_idx, version in enumerate(versions):
            out_path = os.path.join(out_dir, f"{vid_name}_aug{v_idx:02d}.npy")
            if os.path.exists(out_path):
                continue
            std = standardize_frames(version, SEQ_LEN)
            kps = np.stack([self.kp.extract(f) for f in std])
            norm = np.stack([normalize_frame(f) for f in kps])
            np.save(out_path, norm)
            saved += 1
        return saved

    def run(self) -> None:
        try:
            total_saved, total_skipped = 0, 0
            for signer in SIGNERS:
                for cls in CLASSES_INT:
                    cls_str = f"{cls:04d}"
                    for split in SPLITS:
                        vid_dir = f"{self.base_dir}/{signer}/{split}/extracted/{cls_str}/"
                        out_dir = f"{self.npy_dir}/{split}/{cls_str}/{signer}/"
                        ensure_dir(out_dir)

                        videos = glob.glob(vid_dir + "*.mp4")
                        if not videos:
                            total_skipped += 1
                            continue
                        for vid_path in videos:
                            total_saved += self._process_video(
                                vid_path, out_dir, augment=(split == "train")
                            )
                    logger.info("Signer %s | Class %s done", signer, cls_str)
            logger.info("VideoProcessor done. saved=%d skipped=%d",
                        total_saved, total_skipped)
        except Exception as e:
            raise CustomException(e) from e


# ─────────────────────────────────────────────────────────────────────────────
class ExperimentSplitter:
    """Build the three leave-one-signer-out splits and save them to disk."""

    def __init__(self, npy_dir: str = NPY_DIR, save_dir: str = PROCESSED_DIR):
        self.npy_dir = npy_dir
        self.save_dir = save_dir

    def run(self) -> None:
        try:
            for exp in SIGNER_EXPERIMENTS:
                logger.info("Processing %s", exp["name"])
                X_train, y_train = load_split("train", exp["train"], self.npy_dir)
                X_test,  y_test  = load_split("test",  [exp["test"]], self.npy_dir)

                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train,
                    test_size=VAL_SPLIT,
                    random_state=RANDOM_STATE,
                    stratify=y_train,
                )
                logger.info(
                    "  Train: %s | Val: %s | Test: %s",
                    X_tr.shape, X_val.shape, X_test.shape,
                )

                exp_folder = os.path.join(self.save_dir, exp["name"])
                ensure_dir(exp_folder)
                np.save(os.path.join(exp_folder, "X_train.npy"), X_tr)
                np.save(os.path.join(exp_folder, "y_train.npy"), y_tr)
                np.save(os.path.join(exp_folder, "X_val.npy"),   X_val)
                np.save(os.path.join(exp_folder, "y_val.npy"),   y_val)
                np.save(os.path.join(exp_folder, "X_test.npy"),  X_test)
                np.save(os.path.join(exp_folder, "y_test.npy"),  y_test)

                del X_train, y_train, X_test, y_test, X_tr, X_val, y_tr, y_val
                gc.collect()

            logger.info("All experiments saved to %s", self.save_dir)
        except Exception as e:
            raise CustomException(e) from e
