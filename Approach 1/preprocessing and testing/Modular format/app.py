"""
Minimal Flask inference server for the trained ArSL CNN.

Run:
    python app.py
Then POST an image to /predict:
    curl -X POST -F "image=@sample.jpg" http://localhost:5000/predict
"""

import os
import tempfile

from flask import Flask, jsonify, request

from model.predict import ArSLPredictor
from utils.exception_handling import CustomException
from utils.logging_setup import get_logger

logger = get_logger(__name__)

app = Flask(__name__)
predictor = ArSLPredictor()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "num_classes": len(predictor.class_names)})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            file.save(tmp.name)
            class_name, confidence = predictor.predict_image(tmp.name)
        os.unlink(tmp.name)
        return jsonify({"class": class_name, "confidence": confidence})
    except CustomException as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
