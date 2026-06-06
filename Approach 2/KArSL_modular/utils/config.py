"""
Centralized configuration for the KArSL (Arabic Sign Language) video project.
"""

import os


# ── Paths ────────────────────────────────────────────────────────────────────
DRIVE_BASE     = os.environ.get("KARSL_DRIVE_BASE", "/content/drive/MyDrive/arabic_dataset")
BASE_DIR       = os.path.join(DRIVE_BASE, "KArSL-502")
NPY_DIR        = os.path.join(DRIVE_BASE, "NPY")
PROCESSED_DIR  = os.path.join(DRIVE_BASE, "Processed_Experiments")
ARTIFACTS_DIR  = os.environ.get("KARSL_ARTIFACTS_DIR", "artifacts")
MODEL_DIR      = os.path.join(DRIVE_BASE, "models")     # checkpoints during training

# Google Drive folder with the raw archives (used for download step)
GDRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1R8ZI6a1xgRNqUThDFA-X4RT7FJh9WeRr"
)

# MediaPipe task files (downloaded on first run)
MEDIAPIPE_DIR = os.environ.get("KARSL_MP_DIR", "mediapipe_models")
POSE_TASK = os.path.join(MEDIAPIPE_DIR, "pose_landmarker.task")
HAND_TASK = os.path.join(MEDIAPIPE_DIR, "hand_landmarker.task")
FACE_TASK = os.path.join(MEDIAPIPE_DIR, "face_landmarker.task")

POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


# ── Signers and classes ──────────────────────────────────────────────────────
SIGNERS = ["01", "02", "03"]
SPLITS  = ["train", "test"]

# 59 classes kept: 1-10, 32-70, 71-80
KEEP_FIRST  = [f"{i:04d}" for i in list(range(1, 11)) + list(range(32, 71))]
KEEP_SECOND = [f"{i:04d}" for i in range(71, 81)]
CLASSES_INT = list(range(1, 11)) + list(range(32, 71)) + list(range(71, 81))
CLASSES     = [f"{c:04d}" for c in CLASSES_INT]
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASSES)}
NUM_CLASSES    = len(CLASSES)   # 59


# ── Signer-independent experiments (leave-one-signer-out) ────────────────────
SIGNER_EXPERIMENTS = [
    {"train": ["01", "02"], "test": "03", "name": "Exp1_Test_03"},
    {"train": ["01", "03"], "test": "02", "name": "Exp2_Test_02"},
    {"train": ["02", "03"], "test": "01", "name": "Exp3_Test_01"},
]


# ── Sequence / feature shape ─────────────────────────────────────────────────
SEQ_LEN  = 30                              # frames per video
FEAT_DIM = 159                             # 53 keypoints * 3 - normalization layout
EXPECTED_SHAPE = (SEQ_LEN, FEAT_DIM)

# Face landmark indices used (from MediaPipe FaceMesh)
FACE_INDICES = [33, 263, 1, 61, 291, 199, 94, 0, 17, 57, 287]


# ── Augmentation ─────────────────────────────────────────────────────────────
N_AUG               = 10        # number of augmented copies per training video
AUG_ROTATION_RANGE  = (-5, 5)   # degrees
AUG_SCALE_RANGE     = (0.8, 1.0)


# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS         = 50
BATCH_SIZE     = 16
LEARNING_RATE  = 1e-3
VAL_SPLIT      = 0.1
RANDOM_STATE   = 42

EARLY_STOPPING_PATIENCE = 10
LR_REDUCE_PATIENCE      = 5
LR_REDUCE_FACTOR        = 0.5
LR_MIN                  = 1e-6


# ── Model hyperparameters ────────────────────────────────────────────────────
TRANSFORMER_BLOCKS    = 4
TRANSFORMER_HEADS     = 4
TRANSFORMER_KEY_DIM   = 32
TRANSFORMER_FF_DIM    = 128
TRANSFORMER_DROPOUT   = 0.25
GRU_DROPOUT           = 0.5
CLASSIFIER_DROPOUT    = 0.5
