"""
Centralized configuration for the Arabic Sign Language (ArSL) CNN project.
All hyperparameters, paths, and constants are kept here so other modules
stay free of magic numbers.
"""

import os


# ── Data paths ────────────────────────────────────────────────────────────────
INPUT_DIR  = os.environ.get("ARSL_INPUT_DIR",  "/kaggle/input")
OUTPUT_DIR = os.environ.get("ARSL_OUTPUT_DIR", "/kaggle/working/arasl_processed")
ARTIFACTS_DIR = os.environ.get("ARSL_ARTIFACTS_DIR", "artifacts")

MODEL_PATH       = os.path.join(ARTIFACTS_DIR, "arsl_cnn_model.keras")
CLASS_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "class_names.json")


# ── Image preprocessing ───────────────────────────────────────────────────────
TARGET_SIZE = 64           # all images resized to TARGET_SIZE x TARGET_SIZE
MIN_IMAGES_PER_CLASS = 10  # ignore folders with fewer than this many images
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


# ── Train / val / test split ─────────────────────────────────────────────────
TEST_SIZE       = 0.20   # 20 % goes to (val + test) combined
VAL_TEST_SPLIT  = 0.50   # of that, 50 % val and 50 % test
RANDOM_STATE    = 42


# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 1e-3

EARLY_STOPPING_PATIENCE = 5
LR_REDUCE_PATIENCE      = 3
LR_REDUCE_FACTOR        = 0.5
LR_MIN                  = 1e-6


# ── Augmentation ──────────────────────────────────────────────────────────────
ROTATION_FACTOR     = 0.05
ZOOM_FACTOR         = 0.10
TRANSLATION_FACTOR  = 0.10
BRIGHTNESS_DELTA    = 0.2
CONTRAST_LOWER      = 0.8
CONTRAST_UPPER      = 1.2
