"""
Per-video preprocessing: read frames, standardize length, extract MediaPipe
keypoints (hands + face), and normalize.

A `KeypointExtractor` class lazy-initializes the MediaPipe Tasks landmarkers
so importing this module is cheap and so the heavy models live in a single
place.
"""

from typing import List

import cv2
import numpy as np

from utils.config import (
    FACE_INDICES,
    FACE_TASK,
    FACE_URL,
    HAND_TASK,
    HAND_URL,
    POSE_TASK,
    POSE_URL,
    SEQ_LEN,
)
from utils.exception_handling import CustomException
from utils.helper_functions import download
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def read_video(path: str) -> List[np.ndarray]:
    """Read every frame of a video into a list of BGR ndarrays."""
    try:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    except Exception as e:
        raise CustomException(e) from e


def standardize_frames(frames: List[np.ndarray], seq_len: int = SEQ_LEN) -> List[np.ndarray]:
    """Subsample uniformly to `seq_len`, or pad with the last frame if shorter."""
    n = len(frames)
    if n >= seq_len:
        idxs = np.linspace(0, n - 1, seq_len, dtype=int)
        return [frames[i] for i in idxs]
    return frames + [frames[-1]] * (seq_len - n)


class KeypointExtractor:
    """
    Lazily loads MediaPipe Hand + Face landmarkers and extracts a 53x3 keypoint
    array per frame: 21 right-hand + 21 left-hand + 11 face landmarks.
    """

    def __init__(self):
        self._hand = None
        self._face = None

    def _ensure_loaded(self):
        if self._hand is not None and self._face is not None:
            return
        try:
            # local imports so module import doesn't require mediapipe
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            download(POSE_URL, POSE_TASK)
            download(HAND_URL, HAND_TASK)
            download(FACE_URL, FACE_TASK)

            self._mp = mp
            self._hand = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=HAND_TASK),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=2,
                )
            )
            self._face = mp_vision.FaceLandmarker.create_from_options(
                mp_vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=FACE_TASK),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_faces=1,
                )
            )
            logger.info("MediaPipe landmarkers loaded")
        except Exception as e:
            raise CustomException(e) from e

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Return (53, 3) array of [right-hand | left-hand | face]."""
        try:
            self._ensure_loaded()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

            hand_result = self._hand.detect(mp_img)
            face_result = self._face.detect(mp_img)

            rh = np.zeros((21, 3))
            lh = np.zeros((21, 3))
            if hand_result.hand_landmarks:
                for hand_lm, handedness in zip(
                    hand_result.hand_landmarks, hand_result.handedness
                ):
                    label = handedness[0].category_name  # "Left" or "Right"
                    arr = np.array([[p.x, p.y, p.z] for p in hand_lm])
                    if label == "Right":
                        rh = arr
                    else:
                        lh = arr

            if face_result.face_landmarks:
                fl = face_result.face_landmarks[0]
                face = np.array([[fl[i].x, fl[i].y, fl[i].z] for i in FACE_INDICES])
            else:
                face = np.zeros((len(FACE_INDICES), 3))

            return np.concatenate([rh, lh, face], axis=0)  # (53, 3)
        except Exception as e:
            raise CustomException(e) from e


def normalize_frame(kps_53x3: np.ndarray) -> np.ndarray:
    """
    Normalize a (53, 3) keypoint array:
      - each hand is translated to its wrist and scaled by its bounding box
      - face is centered between landmarks 0/1 and scaled by their distance
    Returns a flat vector of length 53 * 3 = 159.
    """
    def norm_hand(h):
        h = h - h[0:1]
        scale = np.linalg.norm(h.max(0) - h.min(0)) + 1e-8
        return h / scale

    def norm_face(f):
        center = (f[0] + f[1]) / 2
        dist = np.linalg.norm(f[0] - f[1]) + 1e-8
        return (f - center) / dist

    rh   = norm_hand(kps_53x3[:21])
    lh   = norm_hand(kps_53x3[21:42])
    face = norm_face(kps_53x3[42:53])
    return np.concatenate([rh.flatten(), lh.flatten(), face.flatten()])
