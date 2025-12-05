# io_utils.py (or first cell)

import os
from glob import glob

import numpy as np
from PIL import Image

def load_dawn_images(
    root_dir,
    target_size=(128, 256),   # (H, W)
    normalize=True,
    max_per_category=None,
):
    """
    Load and preprocess DAWN images from a directory.

    Assumes structure:
        root_dir/
            fog/
                *.png / *.jpg
            rain/
                ...
            snow/
                ...
            clear/
                ...

    Returns
    data_dict : dict
        {weather_label: [X1, X2, ...]}, where each Xi is a 2D np.ndarray (H x W) float32.
    """
    data_dict = {}
    # Obtain weather condition directories and load images
    for weather_dir in sorted(os.listdir(root_dir)):
        full_dir = os.path.join(root_dir, weather_dir)
        if not os.path.isdir(full_dir):
            continue

        images = []
        paths = sorted(
            glob(os.path.join(full_dir, "*.png"))
            + glob(os.path.join(full_dir, "*.jpg"))
            + glob(os.path.join(full_dir, "*.jpeg"))
        )

        # Limit number of images per category if specified
        if max_per_category is not None:
            paths = paths[:max_per_category]
        # Load and preprocess each image
        for path in paths:
            img = Image.open(path)

            # Convert to grayscale (L mode)
            img = img.convert("L")

            # Resize
            img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

            # To numpy
            arr = np.array(img, dtype=np.float32)

            if normalize:
                arr /= 255.0

            images.append(arr)

        if images:
            data_dict[weather_dir] = images

    return data_dict
