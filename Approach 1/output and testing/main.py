import modal
import base64
import io
import json
import numpy as np
from PIL import Image

app = modal.App("arsl-translator")

image = modal.Image.debian_slim().pip_install(
    "tensorflow-cpu",
    "numpy",
    "pillow",
    "fastapi",
    "python-multipart"
)

# Upload model and class names to Modal's cloud storage
model_volume = modal.Volume.from_name("arsl-model-files", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/model": model_volume},
    cpu=2.0
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import tensorflow as tf

    web_app = FastAPI()

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load model from volume
    model = tf.keras.models.load_model("/model/arsl_cnn_model.keras", compile=False)

    with open("/model/class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)

    arabic_map = {
        "ain":   "ع", "al":    "ال", "aleff": "ا",
        "bb":    "ب", "dal":   "د", "dha":   "ظ",
        "dhad":  "ض", "fa":    "ف", "gaaf":  "ق",
        "ghain": "غ", "ha":    "ح", "haa":   "هـ",
        "jeem":  "ج", "kaaf":  "ك", "khaa":  "خ",
        "la":    "لا","laam":  "ل", "meem":  "م",
        "nun":   "ن", "ra":    "ر", "saad":  "ص",
        "seen":  "س", "sheen": "ش", "ta":    "ت",
        "taa":   "ط", "thaa":  "ث", "thal":  "ذ",
        "toot":  "ة", "waw":   "و", "ya":    "ى",
        "yaa":   "ي", "zay":   "ز"
    }

    TARGET_SIZE = 64

    def preprocess(pil_image: Image.Image) -> np.ndarray:
        img = pil_image.convert('L')
        img_array = np.clip(np.array(img, dtype=np.float32), 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))
        w, h = img.size
        max_dim = max(w, h)
        square = Image.new('L', (max_dim, max_dim), color=0)
        square.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
        square = square.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        arr = np.array(square, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=0)
        return arr

    class ImageData(BaseModel):
        image_base64: str

    @web_app.post("/predict")
    def predict(data: ImageData):
        raw = data.image_base64.split(",")[1] if "," in data.image_base64 else data.image_base64
        pil_image = Image.open(io.BytesIO(base64.b64decode(raw)))
        arr = preprocess(pil_image)
        predictions = model.predict(arr, verbose=0)
        idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][idx]) * 100
        return {
            "letter":     arabic_map.get(class_names[idx], class_names[idx]),
            "label":      class_names[idx],
            "confidence": round(confidence, 2)
        }

    return web_app