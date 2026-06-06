"""
features/augmentation.py
========================
Runs the full video → .npy preprocessing pipeline across all signers and classes,
saving augmented sequences to NPY_DIR.
"""

import glob
import os

import numpy as np

from config.config import BASE_DIR, CLASS_IDS, NPY_DIR, SIGNERS
from features.keypoints import load_mediapipe_models, process_video


def run_augmentation_pipeline(
    base_dir: str = BASE_DIR,
    npy_dir:  str = NPY_DIR,
    signers:  list = SIGNERS,
    class_ids: list = CLASS_IDS,
    model_dir: str = ".",
) -> None:
    """
    For every (signer, class, split) combination:
      - Read raw .mp4 files from the extracted folder.
      - Run standardise → [augment] → extract_keypoints → normalise.
      - Save each version as a .npy file under npy_dir.

    Skips files that already exist, so it is safe to resume after interruption.

    Args:
        base_dir:   Root of KArSL-502 (contains signer sub-dirs).
        npy_dir:    Destination root for .npy files.
        signers:    List of signer IDs to process.
        class_ids:  List of integer class IDs to process.
        model_dir:  Directory where MediaPipe task files are stored / downloaded.
    """
    models = load_mediapipe_models(model_dir)

    total_saved   = 0
    total_skipped = 0

    for signer in signers:
        for cls in class_ids:
            cls_str = f"{cls:04d}"

            for split in ["train", "test"]:
                vid_dir = f"{base_dir}/{signer}/{split}/extracted/{cls_str}/"
                out_dir = f"{npy_dir}/{split}/{cls_str}/{signer}/"
                os.makedirs(out_dir, exist_ok=True)

                videos = glob.glob(vid_dir + "*.mp4")
                if not videos:
                    print(f"  WARNING: no videos at {vid_dir}")
                    total_skipped += 1
                    continue

                for vid_path in videos:
                    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
                    versions = process_video(vid_path, models, split)

                    for tag, norm_seq in versions:
                        out_path = f"{out_dir}/{vid_name}_{tag}.npy"
                        if os.path.exists(out_path):
                            continue
                        np.save(out_path, norm_seq)
                        total_saved += 1

            print(f"✓ Signer {signer} | Class {cls_str} done")

    print(f"\n🎉 Done! Saved: {total_saved} | Skipped (no videos): {total_skipped}")
