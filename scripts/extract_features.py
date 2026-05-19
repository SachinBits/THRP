"""Feature extraction utilities for classical ML experiments."""
from typing import Tuple, Dict
import numpy as np
import cv2
from skimage.feature import hog


def extract_hog(image: np.ndarray, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9) -> np.ndarray:
    # resize to fixed size so HOG output length is deterministic
    target_size = (256, 256)
    if (image.shape[1], image.shape[0]) != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feat = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, block_norm='L2-Hys')
    return feat


def extract_color_hist(image: np.ndarray, bins: int = 32) -> np.ndarray:
    target_size = (256, 256)
    if (image.shape[1], image.shape[0]) != target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    chans = cv2.split(image)
    hist = []
    for ch in chans:
        h = cv2.calcHist([ch], [0], None, [bins], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist.append(h)
    return np.concatenate(hist)


def extract_combined(image: np.ndarray) -> np.ndarray:
    """Return concatenated HOG + color histogram features."""
    h = extract_hog(image)
    c = extract_color_hist(image)
    return np.concatenate([h, c])
