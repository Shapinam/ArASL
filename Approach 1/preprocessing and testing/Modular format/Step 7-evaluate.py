# evaluate.py
# Evaluate model on test set: accuracy, classification report, confusion matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import BATCH_SIZE
from dataset import make_dataset


def evaluate_model(model, test_ds):
    """Print test loss and accuracy."""
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")
    return test_loss, test_acc


def get_predictions(model, test_ds):
    """Return (y_true, y_pred) numpy arrays from the test dataset."""
    y_true_list, y_pred_list = [], []
    for X_batch, y_batch in test_ds:
        preds = model.predict(X_batch, verbose=0)
        y_pred_list.extend(np.argmax(preds, axis=1))
        y_true_list.extend(y_batch.numpy())
    return np.array(y_true_list), np.array(y_pred_list)


def print_classification_report(y_true, y_pred, class_names):
    print(f"\nSanity check — unique true labels: {len(np.unique(y_true))}")
    print(f"Sanity check — unique pred labels: {len(np.unique(y_pred))}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import pickle

    # Load saved artifacts from train.py
    with open('training_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)

    class_names = artifacts['class_names']
    X_test      = artifacts['X_test']
    y_test      = artifacts['y_test']

    model   = tf.keras.models.load_model('arsl_cnn_model.keras', compile=False)
    test_ds = make_dataset(X_test, y_test, augment_fn=None, batch_size=BATCH_SIZE)

    evaluate_model(model, test_ds)
    y_true, y_pred = get_predictions(model, test_ds)
    print_classification_report(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names)
