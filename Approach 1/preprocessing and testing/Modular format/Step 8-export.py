# export.py
# Save the trained model and class names for deployment

import json
import tensorflow as tf

from config import MODEL_SAVE_PATH, CLASS_NAMES_SAVE_PATH


def export_model(model, class_names,
                 model_path=MODEL_SAVE_PATH,
                 class_names_path=CLASS_NAMES_SAVE_PATH):
    """Save Keras model and class name list to disk."""
    model.save(model_path)
    with open(class_names_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print(f"Saved model:       {model_path}")
    print(f"Saved class names: {class_names_path}")
    print(f"Class order:       {class_names}")


if __name__ == '__main__':
    import pickle

    with open('training_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)

    model = tf.keras.models.load_model('arsl_cnn_model.keras', compile=False)
    export_model(model, artifacts['class_names'])
