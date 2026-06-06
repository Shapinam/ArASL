# train.py
# Compile, train, and plot training history

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    LEARNING_RATE, EPOCHS,
    LR_REDUCE_FACTOR, LR_REDUCE_PATIENCE, MIN_LR,
    EARLY_STOP_PATIENCE, TARGET_SIZE
)
from dataset import load_dataset, split_dataset, build_pipelines
from model import build_arsl_cnn


def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        ),
    ]


def compile_model(model, learning_rate=LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'],     label='Train accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['loss'],     label='Train loss')
    ax2.plot(history.history['val_loss'], label='Val loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def train(model, train_ds, val_ds, epochs=EPOCHS):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks()
    )
    return history


if __name__ == '__main__':
    # Load data
    X, y, class_names, class_to_idx = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    train_ds, val_ds, test_ds = build_pipelines(X_train, X_val, X_test, y_train, y_val, y_test)

    # Build and train
    model = build_arsl_cnn(
        input_shape=(TARGET_SIZE, TARGET_SIZE, 1),
        num_classes=len(class_names)
    )
    model = compile_model(model)
    history = train(model, train_ds, val_ds)
    plot_history(history)

    # Save for use by evaluate.py and export.py
    import pickle
    with open('training_artifacts.pkl', 'wb') as f:
        pickle.dump({
            'class_names': class_names,
            'X_test': X_test,
            'y_test': y_test,
        }, f)

    model.save('arsl_cnn_model.keras')
    print("Model saved to arsl_cnn_model.keras")
