"""
data/loader.py
==============
Loads preprocessed .npy keypoint sequences, builds the class-label map,
splits into train/val/test per cross-signer experiment, and saves results.
"""

import gc
import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split

from config.config import (
    CLASS_IDS,
    NPY_DIR,
    RANDOM_SEED,
    SAVE_DIR,
    SIGNER_EXPERIMENTS,
    VAL_SPLIT,
)

# ── Class → integer label map ─────────────────────────────────────────────────
CLASS_TO_LABEL: dict[str, int] = {f"{cls:04d}": idx for idx, cls in enumerate(CLASS_IDS)}
LABEL_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_LABEL.items()}

print(f"Total classes: {len(CLASS_TO_LABEL)}")   # 59


def load_split(split: str, signers: list, npy_dir: str = NPY_DIR) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all .npy sequence files for a given split and list of signers.

    Args:
        split:    "train" or "test".
        signers:  List of signer IDs (e.g. ["01", "02"]).
        npy_dir:  Root directory of pre-processed .npy files.

    Returns:
        X: float32 array of shape (N, SEQ_LEN, FEAT_DIM)
        y: int array of shape (N,)
    """
    X, y = [], []
    for cls_folder, label in CLASS_TO_LABEL.items():
        for signer in signers:
            pattern = f"{npy_dir}/{split}/{cls_folder}/{signer}/*.npy"
            for f in glob.glob(pattern):
                seq = np.load(f).astype(np.float32)
                if seq.shape == (30, 159):
                    X.append(seq)
                    y.append(label)
                else:
                    print(f"  Bad shape {seq.shape}: {f}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def prepare_experiment(exp: dict, save_dir: str = SAVE_DIR) -> str:
    """
    Load, split, and save data for a single cross-signer experiment.

    Args:
        exp:      Experiment dict with keys "train", "test", "name".
        save_dir: Where to save the resulting .npy arrays.

    Returns:
        Path to the experiment folder.
    """
    print(f"\n--- Processing {exp['name']} ---")

    X_train, y_train = load_split("train", exp["train"])
    X_test,  y_test  = load_split("test",  [exp["test"]])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_train,
    )

    print(f"  Train:      {X_tr.shape}  labels: {y_tr.shape}")
    print(f"  Validation: {X_val.shape} labels: {y_val.shape}")
    print(f"  Test:       {X_test.shape} labels: {y_test.shape}")

    exp_folder = os.path.join(save_dir, exp["name"])
    os.makedirs(exp_folder, exist_ok=True)

    np.save(os.path.join(exp_folder, "X_train.npy"), X_tr)
    np.save(os.path.join(exp_folder, "y_train.npy"), y_tr)
    np.save(os.path.join(exp_folder, "X_val.npy"),   X_val)
    np.save(os.path.join(exp_folder, "y_val.npy"),   y_val)
    np.save(os.path.join(exp_folder, "X_test.npy"),  X_test)
    np.save(os.path.join(exp_folder, "y_test.npy"),  y_test)

    print(f"  Saved to {exp_folder}")

    # Free RAM before next experiment
    del X_train, y_train, X_test, y_test, X_tr, X_val, y_tr, y_val
    gc.collect()

    return exp_folder


def prepare_all_experiments(
    experiments: list = SIGNER_EXPERIMENTS,
    save_dir: str = SAVE_DIR,
) -> None:
    """Run prepare_experiment for every entry in the experiments list."""
    for exp in experiments:
        prepare_experiment(exp, save_dir)
    print("\n✅ All experiments processed and saved.")


def load_experiment(exp_name: str, save_dir: str = SAVE_DIR) -> dict:
    """
    Load a previously saved experiment from disk.

    Returns:
        Dict with keys X_train, y_train, X_val, y_val, X_test, y_test.
    """
    folder = os.path.join(save_dir, exp_name)
    return {
        "X_train": np.load(f"{folder}/X_train.npy"),
        "y_train": np.load(f"{folder}/y_train.npy"),
        "X_val":   np.load(f"{folder}/X_val.npy"),
        "y_val":   np.load(f"{folder}/y_val.npy"),
        "X_test":  np.load(f"{folder}/X_test.npy"),
        "y_test":  np.load(f"{folder}/y_test.npy"),
    }
