"""
Minimal Flask inference server for the trained KArSL TGRU model.

Run:
    python app.py
POST a video:
    curl -X POST -F "video=@sample.mp4" -F "model_path=artifacts/best.keras" \
         http://localhost:5000/predict
"""

import os
import tempfile

from flask import Flask, jsonify, request

from model.predict import ArSLVideoPredictor
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)

app = Flask(__name__)
_PREDICTOR_CACHE = {}


def _get_predictor(model_path: str) -> ArSLVideoPredictor:
    if model_path not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[model_path] = ArSLVideoPredictor(model_path)
    return _PREDICTOR_CACHE[model_path]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    model_path = request.form.get("model_path") or request.args.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return jsonify({"error": "Provide a valid `model_path`"}), 400

    file = request.files["video"]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            file.save(tmp.name)
            class_name, confidence = _get_predictor(model_path).predict_video(tmp.name)
        os.unlink(tmp.name)
        return jsonify({"class": class_name, "confidence": confidence})
    except CustomException as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
