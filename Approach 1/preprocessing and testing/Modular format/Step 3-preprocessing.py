# preprocessing.py
# Single-image preprocessing pipeline: grayscale → validate → pad → resize → normalize

import numpy as np
from PIL import Image

from config import TARGET_SIZE


def preprocess_image(image_path, target_size=TARGET_SIZE):
    """
    Robust preprocessing for grayscale sign language images.

    Steps:
    1. Open image and force to true grayscale (handles RGB, P, RGBA, L)
    2. Validate pixel range and clip
    3. Letterbox pad to square (preserves aspect ratio, no squashing)
    4. Resize to target_size x target_size
    5. Return as numpy array with shape (target_size, target_size, 1), float32 in [0, 1]
    """
    img = Image.open(image_path)

    # Step 1: Force to grayscale
    img = img.convert('L')

    # Step 2: Clip any out-of-range pixel values
    img_array = np.array(img, dtype=np.float32)
    img_array = np.clip(img_array, 0, 255)
    img = Image.fromarray(img_array.astype(np.uint8))

    # Step 3: Letterbox pad to square (black background)
    w, h = img.size
    max_dim = max(w, h)
    square_img = Image.new('L', (max_dim, max_dim), color=0)
    square_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))

    # Step 4: Resize to target
    square_img = square_img.resize((target_size, target_size), Image.LANCZOS)

    # Step 5: Normalize and add channel dimension
    final_array = np.array(square_img, dtype=np.float32) / 255.0
    final_array = np.expand_dims(final_array, axis=-1)

    return final_array


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from config import INPUT_DIR

    # Find one test image
    test_path = None
    for dirname, dirnames, filenames in os.walk(INPUT_DIR):
        if len(dirnames) == 0:
            images = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_path = os.path.join(dirname, images[0])
                break

    result = preprocess_image(test_path)
    print(f"Output shape: {result.shape}")       # (64, 64, 1)
    print(f"Min value:    {result.min():.4f}")   # 0.0
    print(f"Max value:    {result.max():.4f}")   # ~1.0
    print(f"Dtype:        {result.dtype}")       # float32

    plt.figure(figsize=(4, 4))
    plt.imshow(result[:, :, 0], cmap='gray')
    plt.title('Preprocessed image (64x64, normalized)')
    plt.axis('off')
    plt.show()
