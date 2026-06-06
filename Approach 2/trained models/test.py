import cv2
import numpy as np
import collections
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import os
from collections import Counter

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "best_arsl_tgru_exp2.h5"
SEQ_LEN     = 30
CONF_THRESH = 0.45   # slightly lower threshold
VOTE_WINDOW = 7      # more votes = more stable

CLASS_NAMES = (
    [str(i) for i in range(10)] +
    [f"letter_{i}" for i in range(32, 71)] +
    [f"word_{i}"   for i in range(71, 81)]
)

# ── Positional embedding ─────────────────────────────────────────────────────
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, seq_len, dim, **kwargs):
        super().__init__(**kwargs)
        self.embed   = keras.layers.Dense(dim)
        self.seq_len = seq_len
        self.dim     = dim
        pe  = np.zeros((seq_len, dim))
        pos = np.arange(seq_len)[:, np.newaxis]
        div = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.pe = tf.cast(pe, dtype=tf.float32)

    def call(self, x):
        return self.embed(x) + self.pe

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "dim": self.dim})
        return config

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={"PositionalEmbedding": PositionalEmbedding}
)
print("Model loaded")

# ── MediaPipe ────────────────────────────────────────────────────────────────
mp_holistic       = mp.solutions.holistic
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("MediaPipe loaded")

# ── Keypoint extraction ──────────────────────────────────────────────────────
def extract_keypoints(results):
    def lm_to_arr(lm, n):
        if lm:
            return np.array([[p.x, p.y, p.z] for p in lm.landmark[:n]])
        return np.zeros((n, 3))

    rh = lm_to_arr(results.right_hand_landmarks, 21)
    lh = lm_to_arr(results.left_hand_landmarks,  21)

    face_indices = [33, 263, 1, 61, 291, 199, 94, 0, 17, 57, 287]
    if results.face_landmarks:
        fl   = results.face_landmarks.landmark
        face = np.array([[fl[i].x, fl[i].y, fl[i].z] for i in face_indices])
    else:
        face = np.zeros((11, 3))

    return np.concatenate([rh, lh, face], axis=0)

# ── Normalization ────────────────────────────────────────────────────────────
def normalize_frame(kps):
    def norm_hand(h):
        h     = h - h[0:1]
        scale = np.linalg.norm(h.max(0) - h.min(0)) + 1e-8
        return h / scale
    def norm_face(f):
        center = (f[0] + f[1]) / 2
        dist   = np.linalg.norm(f[0] - f[1]) + 1e-8
        return (f - center) / dist
    rh   = norm_hand(kps[:21])
    lh   = norm_hand(kps[21:42])
    face = norm_face(kps[42:53])
    return np.concatenate([rh.flatten(), lh.flatten(), face.flatten()])

# ── Buffers ──────────────────────────────────────────────────────────────────
# deque with maxlen = rolling window — old frames auto-drop as new ones arrive
frame_buffer = collections.deque(maxlen=SEQ_LEN)
vote_buffer  = collections.deque(maxlen=VOTE_WINDOW)

current_prediction = ""
current_confidence = 0.0
frame_count        = 0   # predict every N frames to avoid lag

# ── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nWebcam started — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    # MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True

    # Draw landmarks
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())

    # Add frame to rolling buffer every frame
    kps  = extract_keypoints(results)
    norm = normalize_frame(kps)
    frame_buffer.append(norm)   # auto-drops oldest frame when full

    # Predict every 5 frames (not every frame — reduces lag, keeps it live)
    if len(frame_buffer) == SEQ_LEN and frame_count % 5 == 0:
        seq   = np.array(frame_buffer)[np.newaxis].astype(np.float32)
        probs = model.predict(seq, verbose=0)[0]
        pred  = int(probs.argmax())
        conf  = float(probs[pred])

        vote_buffer.append(pred)  # always vote, filter by conf at display

        if len(vote_buffer) == VOTE_WINDOW:
            majority, count = Counter(vote_buffer).most_common(1)[0]
            if count >= 4 and conf >= CONF_THRESH:
                current_prediction = CLASS_NAMES[majority]
                current_confidence = conf

    # ── UI ───────────────────────────────────────────────────────────────────
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 95), (20, 20, 20), -1)

    # Detection status — use text instead of symbols for Windows
    rh_ok = results.right_hand_landmarks is not None
    lh_ok = results.left_hand_landmarks  is not None
    fc_ok = results.face_landmarks       is not None

    status_color = (0, 220, 0) if (rh_ok or lh_ok) else (0, 0, 220)
    cv2.putText(frame,
                f"R.Hand: {'YES' if rh_ok else 'NO '}   "
                f"L.Hand: {'YES' if lh_ok else 'NO '}   "
                f"Face: {'YES' if fc_ok else 'NO'}",
                (20, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, status_color, 1)

    # Prediction
    if current_prediction:
        cv2.putText(frame,
                    f"Sign: {current_prediction}",
                    (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 120), 2)
        cv2.putText(frame,
                    f"Confidence: {current_confidence:.0%}",
                    (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 1)
    else:
        cv2.putText(frame,
                    "Waiting...",
                    (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (80, 80, 80), 2)

    # Buffer bar
    fill = int((len(frame_buffer) / SEQ_LEN) * 200)
    cv2.rectangle(frame, (w-220, 10), (w-20, 30), (50, 50, 50), -1)
    cv2.rectangle(frame, (w-220, 10), (w-220+fill, 30),
                  (0, 200, 255) if len(frame_buffer) < SEQ_LEN else (0, 255, 100), -1)
    cv2.putText(frame, f"Buffer {len(frame_buffer)}/{SEQ_LEN}",
                (w-220, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 1)

    cv2.putText(frame, "Q = quit",
                (w-90, h-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (80, 80, 80), 1)

    cv2.imshow("ArSL-TGRU Live Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()
print("Demo closed.")