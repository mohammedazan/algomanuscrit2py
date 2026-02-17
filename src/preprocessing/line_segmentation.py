"""
Line Segmentation Module
========================
Detect and extract individual text lines from a handwritten image.
"""

import cv2
import numpy as np


def segment_lines(image):
    """
    Segment image into individual text lines using horizontal projection.

    Args:
        image: Grayscale or binary image

    Returns:
        list of cropped line images
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarization
    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Horizontal projection
    horizontal_sum = np.sum(binary, axis=1)

    lines = []
    start = None

    for i, value in enumerate(horizontal_sum):
        if value > 0 and start is None:
            start = i
        elif value == 0 and start is not None:
            end = i
            line = image[start:end, :]
            if line.shape[0] > 10:  # avoid noise
                lines.append(line)
            start = None

    # Handle last line
    if start is not None:
        line = image[start:, :]
        if line.shape[0] > 10:
            lines.append(line)

    return lines
