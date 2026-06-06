"""
features/keypoints.py
=====================
MediaPipe-based keypoint extraction from video frames.

Pipeline per video:
  read_video → standardize_frames → [augment_video] → extract_keypoints → normalize_frame → save .npy
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.config import FACE_INDICES, MEDIAPIPE_MODELS, N_AUG, SEQ_LEN


# ── Model initialisation ──────────────────────────────────────────────────────

def _download_model(filename: str, url: str) -> None:
    """Download a MediaPipe task file if it doesn't exist locally."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)


def load_mediapipe_models(model_dir: str = ".") -> dict:
    """
    Download (if needed) and initialise all three MediaPipe landmarkers.

    Returns:
        Dict with keys "hand", "face", "pose" → landmarker objects.
    """
    for filename, url in MEDIAPIPE_MODELS.items():
        path = os.path.join(model_dir, filename)
        _download_model(path, url)

    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=os.path.join(model_dir, "hand_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=os.path.join(model_dir, "face_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=os.path.join(model_dir, "pose_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
        )
    )
    print("MediaPipe models loaded ✓")
    return {"hand": hand_landmarker, "face": face_landmarker, "pose": pose_landmarker}


# ── Video utilities ───────────────────────────────────────────────────────────

def read_video(path: str) -> list:
    """Read all frames from a video file. Returns list of BGR numpy arrays."""
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def standardize_frames(frames: list, seq_len: int = SEQ_LEN) -> list:
    """
    Uniformly sample or pad a frame list to exactly seq_len frames.

    Args:
        frames:  Raw list of BGR frames.
        seq_len: Target sequence length.

    Returns:
        List of exactly seq_len frames.
    """
    n = len(frames)
    if n >= seq_len:
        idxs = np.linspace(0, n - 1, seq_len, dtype=int)
        return [frames[i] for i in idxs]
    return frames + [frames[-1]] * (seq_len - n)


def augment_video(frames: list, seed: int = None) -> list:
    """
    Apply a random rotation + scale augmentation to all frames.

    Args:
        frames: List of BGR frames.
        seed:   RNG seed for reproducibility.

    Returns:
        Augmented list of BGR frames.
    """
    rng   = np.random.default_rng(seed)
    angle = rng.uniform(-5, 5)
    scale = rng.uniform(0.8, 1.0)
    h, w  = frames[0].shape[:2]
    M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    return [cv2.warpAffine(f, M, (w, h)) for f in frames]


# ── Keypoint extraction ───────────────────────────────────────────────────────

def extract_keypoints(frame: np.ndarray, models: dict, face_indices: list = FACE_INDICES) -> np.ndarray:
    """
    Extract hand and face keypoints from a single BGR frame.

    Args:
        frame:        BGR image array.
        models:       Dict returned by load_mediapipe_models().
        face_indices: Indices of face landmarks to keep.

    Returns:
        Array of shape (53, 3) — 21 RH + 21 LH + 11 face keypoints, each (x, y, z).
    """
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = models["hand"].detect(mp_img)
    face_result = models["face"].detect(mp_img)

    # Hands
    rh = np.zeros((21, 3))
    lh = np.zeros((21, 3))
    if hand_result.hand_landmarks:
        for hand_lm, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handedness[0].category_name   # "Left" or "Right"
            arr   = np.array([[p.x, p.y, p.z] for p in hand_lm])
            if label == "Right":
                rh = arr
            else:
                lh = arr

    # Face (selected landmarks)
    if face_result.face_landmarks:
        fl   = face_result.face_landmarks[0]
        face = np.array([[fl[i].x, fl[i].y, fl[i].z] for i in face_indices])
    else:
        face = np.zeros((len(face_indices), 3))

    return np.concatenate([rh, lh, face], axis=0)   # (53, 3)


def normalize_frame(kps_53x3: np.ndarray) -> np.ndarray:
    """
    Normalise a single frame's keypoints relative to their own geometry.

    Hands: centred on wrist (index 0), scaled by bounding-box diagonal.
    Face:  centred between eye corners, scaled by inter-eye distance.

    Args:
        kps_53x3: Array of shape (53, 3).

    Returns:
        Flat array of shape (159,).
    """
    def norm_hand(h: np.ndarray) -> np.ndarray:
        h     = h - h[0:1]
        scale = np.linalg.norm(h.max(0) - h.min(0)) + 1e-8
        return h / scale

    def norm_face(f: np.ndarray) -> np.ndarray:
        center = (f[0] + f[1]) / 2
        dist   = np.linalg.norm(f[0] - f[1]) + 1e-8
        return (f - center) / dist

    rh   = norm_hand(kps_53x3[:21])
    lh   = norm_hand(kps_53x3[21:42])
    face = norm_face(kps_53x3[42:53])

    return np.concatenate([rh.flatten(), lh.flatten(), face.flatten()])  # (159,)


# ── Full preprocessing pipeline ───────────────────────────────────────────────

def process_video(vid_path: str, models: dict, split: str, n_aug: int = N_AUG, seq_len: int = SEQ_LEN) -> list:
    """
    Preprocess one video into one or more normalised sequences.

    Args:
        vid_path: Path to .mp4 file.
        models:   MediaPipe landmarker dict.
        split:    "train" (with augmentation) or "test" (no augmentation).
        n_aug:    Number of augmented copies to create for training videos.
        seq_len:  Target sequence length.

    Returns:
        List of (tag, norm_seq) tuples where tag is e.g. "aug00", "aug01", …
        and norm_seq has shape (seq_len, 159).
    """
    frames    = read_video(vid_path)
    versions  = [frames]
    if split == "train":
        for i in range(n_aug):
            versions.append(augment_video(frames, seed=i))

    results = []
    for v_idx, version in enumerate(versions):
        std  = standardize_frames(version, seq_len)
        kps  = [extract_keypoints(f, models) for f in std]
        seq  = np.stack(kps)                              # (seq_len, 53, 3)
        norm = np.stack([normalize_frame(f) for f in seq])  # (seq_len, 159)
        results.append((f"aug{v_idx:02d}", norm))

    return results
