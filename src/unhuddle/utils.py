# src/unhuddle/utils.py

import logging
logger = logging.getLogger(__name__)
import numpy as np
from skimage import io
import warnings
import random


def save_image(path, image, description=""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        io.imsave(path, image)
    logger.info(f"Saved {description} at: {path}")


def generate_pseudocolor_mask(mask):
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    random.shuffle(unique_labels)

    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in unique_labels:
        color = np.random.randint(0, 255, size=3)
        out[mask == label] = color
    return out
