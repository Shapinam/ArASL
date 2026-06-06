"""
config/config.py
================
Central configuration for the KArSL Arabic Sign Language Recognition pipeline.
Edit this file to change paths, hyperparameters, or experiment settings.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = "/content/drive/MyDrive/arabic_dataset/KArSL-502"
NPY_DIR   = "/content/drive/MyDrive/arabic_dataset/NPY"
SAVE_DIR  = "/content/drive/MyDrive/arabic_dataset/Processed_Experiments"
MODEL_DIR = "/content/drive/MyDrive/arabic_dataset"

# ── Dataset ───────────────────────────────────────────────────────────────────
SIGNERS = ["01", "02", "03"]

# Class IDs to include (maps to 59 classes total)
CLASS_IDS = list(range(1, 11)) + list(range(32, 81))

# Archive extraction ranges
KEEP_FIRST  = list(range(1, 11)) + list(range(32, 71))   # → 0001-0070.7z
KEEP_SECOND = list(range(71, 81))                         # → 0071-0170.7z

# ── Preprocessing ─────────────────────────────────────────────────────────────
SEQ_LEN = 30          # frames per video (after standardisation)
N_AUG   = 10          # augmentation copies per training video

# MediaPipe face landmark indices to keep
FACE_INDICES = [33, 263, 1, 61, 291, 199, 94, 0, 17, 57, 287]

# Feature vector length: 21 (RH) + 21 (LH) + 11 (face) = 53 keypoints × 3 = 159
FEAT_DIM = 159

# ── Model ─────────────────────────────────────────────────────────────────────
N_CLASSES  = 59
NUM_HEADS  = 4
KEY_DIM    = 32
FF_DIM     = 128
DROPOUT    = 0.25
GRU_DROPOUT = 0.5

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS      = 50
BATCH_SIZE  = 16
LR          = 1e-3
LR_MIN      = 1e-6
LR_FACTOR   = 0.5
LR_PATIENCE = 5
ES_PATIENCE = 10
VAL_SPLIT   = 0.1
RANDOM_SEED = 42

# ── Cross-signer experiments ──────────────────────────────────────────────────
SIGNER_EXPERIMENTS = [
    {"train": ["01", "02"], "test": "03", "name": "Exp1_Test_03"},
    {"train": ["01", "03"], "test": "02", "name": "Exp2_Test_02"},
    {"train": ["02", "03"], "test": "01", "name": "Exp3_Test_01"},
]

# ── MediaPipe model URLs ───────────────────────────────────────────────────────
MEDIAPIPE_MODELS = {
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
}
