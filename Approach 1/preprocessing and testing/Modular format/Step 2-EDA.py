# data_exploration.py
# Explore the dataset: class counts, image modes/sizes, sample visualizations

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from config import INPUT_DIR, MIN_IMAGES_PER_CLASS


def collect_class_info(input_dir=INPUT_DIR, min_images=MIN_IMAGES_PER_CLASS):
    """Walk input directory and collect per-class statistics."""
    class_info = []

    for dirname, dirnames, filenames in os.walk(input_dir):
        if len(dirnames) == 0:
            images = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > min_images:
                class_name = os.path.basename(dirname)
                sample_path = os.path.join(dirname, images[0])
                img = Image.open(sample_path)
                class_info.append({
                    'class': class_name,
                    'count': len(images),
                    'mode':  img.mode,
                    'size':  img.size,
                })

    df = pd.DataFrame(class_info).sort_values('class').reset_index(drop=True)
    return df


def print_summary(df):
    """Print dataset summary statistics."""
    print(f"Total classes: {len(df)}")
    print(f"Total images:  {df['count'].sum()}")
    print(f"\nUnique image modes: {df['mode'].unique()}")
    print(f"Unique image sizes: {df['size'].unique()}")
    print(f"\n{df.to_string()}")


def plot_class_distribution(df):
    """Bar chart of images per class."""
    plt.figure(figsize=(16, 5))
    sns.barplot(data=df, x='class', y='count', hue='class', palette='crest', legend=False)
    plt.title('Images per Arabic Sign Class')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_sample_images(df, input_dir=INPUT_DIR, n=8):
    """Display n random sample images from the dataset."""
    sample_classes = random.sample(list(df['class']), min(n, len(df)))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    for i, cls in enumerate(sample_classes):
        for dirname, dirnames, filenames in os.walk(input_dir):
            if os.path.basename(dirname) == cls and len(dirnames) == 0:
                img_path = os.path.join(dirname, random.choice(filenames))
                img = Image.open(img_path)
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f'{cls}\n{img.mode} {img.size}')
                axes[i].axis('off')
                break

    plt.suptitle('Sample Images from Dataset', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = collect_class_info()
    print_summary(df)
    plot_class_distribution(df)
    plot_sample_images(df)
