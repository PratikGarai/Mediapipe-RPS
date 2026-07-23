"""Convert image arrays into MediaPipe-compatible image objects."""

import cv2
import mediapipe as mp


def prepare_image(image: cv2.Mat) -> mp.Image:
    """Build a MediaPipe image from an OpenCV matrix.

    Args:
        image: Source OpenCV image in RGB-compatible layout.

    Returns:
        MediaPipe image instance ready for model inference.
    """
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
