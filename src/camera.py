"""Provide camera capture helpers."""

import logging

import cv2


def get_webcam(cam_index: int = 0) -> cv2.VideoCapture:
    """Open and return a webcam capture device.

    Args:
        cam_index: Camera index to open.

    Returns:
        Initialized OpenCV capture object.
    """
    webcam_capture = cv2.VideoCapture(cam_index)

    if not webcam_capture.isOpened():
        logging.error("Could not open webcam.")
        exit()
    else:
        logging.info("Webcam opened successfully.")

    return webcam_capture
