# dataset.py
# Load all images into memory, split into train/val/test, build tf.data pipelines

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import INPUT_DIR, TARGET_SIZE, BATCH_SIZE, MIN_IMAGES_PER_CLASS, RANDOM_SEED
from preprocessing import preprocess_image


def collect_valid_classes(input_dir=INPUT_DIR, min_images=MIN_IMAGES_PER_CLASS):
    """Return dict of {class_name: [list of image paths]}."""
    valid_classes = {}
    for dirname, dirnames, filenames in os.walk(input_dir):
        if len(dirnames) == 0:
            images = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > min_images:
                class_name = os.path.basename(dirname)
                valid_classes[class_name] = [os.path.join(dirname, f) for f in images]
    return valid_classes


def load_dataset(input_dir=INPUT_DIR, target_size=TARGET_SIZE):
    """
    Load and preprocess all images into memory.
    Returns: X (float32 array), y (int32 array), class_names (list), class_to_idx (dict)
    """
    valid_classes = collect_valid_classes(input_dir)
    class_names   = sorted(valid_classes.keys())
    class_to_idx  = {name: i for i, name in enumerate(class_names)}
    num_classes   = len(class_names)

    print(f"Classes found: {num_classes}")
    print(f"Class mapping: {class_to_idx}")

    X_all, y_all = [], []
    failed = 0

    for class_name, paths in tqdm(valid_classes.items(), desc="Loading classes"):
        label = class_to_idx[class_name]
        for path in paths:
            try:
                arr = preprocess_image(path, target_size=target_size)
                X_all.append(arr)
                y_all.append(label)
            except Exception:
                failed += 1

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)

    print(f"\nDataset shape: {X_all.shape}")
    print(f"Labels shape:  {y_all.shape}")
    print(f"Failed loads:  {failed}")

    return X_all, y_all, class_names, class_to_idx


def split_dataset(X, y, val_size=0.10, test_size=0.10, random_seed=RANDOM_SEED):
    """
    Split into train / val / test sets (stratified).
    Default: 80% train, 10% val, 10% test.
    """
    total_held = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=total_held, random_state=random_seed, stratify=y
    )
    relative_test = test_size / total_held
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=random_seed, stratify=y_temp
    )

    print(f"Train: {X_train.shape[0]} images")
    print(f"Val:   {X_val.shape[0]} images")
    print(f"Test:  {X_test.shape[0]} images")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Augmentation layers (defined once at module level) ──
_aug_rotation    = tf.keras.layers.RandomRotation(0.05, seed=RANDOM_SEED)
_aug_zoom        = tf.keras.layers.RandomZoom(0.10, seed=RANDOM_SEED)
_aug_translation = tf.keras.layers.RandomTranslation(0.10, 0.10, seed=RANDOM_SEED)


def augment(image, label):
    image = tf.image.random_flip_left_right(image, seed=RANDOM_SEED)
    image = tf.image.random_brightness(image, max_delta=0.2, seed=RANDOM_SEED)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2, seed=RANDOM_SEED)
    image = _aug_rotation(image, training=True)
    image = _aug_zoom(image, training=True)
    image = _aug_translation(image, training=True)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def make_dataset(X, y, augment_fn=None, batch_size=BATCH_SIZE):
    """Build a tf.data.Dataset from numpy arrays."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_pipelines(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=BATCH_SIZE):
    """Return (train_ds, val_ds, test_ds) ready for model.fit()."""
    train_ds = make_dataset(X_train, y_train, augment_fn=augment, batch_size=batch_size)
    val_ds   = make_dataset(X_val,   y_val,   augment_fn=None,    batch_size=batch_size)
    test_ds  = make_dataset(X_test,  y_test,  augment_fn=None,    batch_size=batch_size)
    print("Datasets ready.")
    return train_ds, val_ds, test_ds
