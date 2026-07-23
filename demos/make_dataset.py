"""Provide an interactive workflow for capturing hand dataset samples."""

import logging

import cv2

from constants import FRAME_DELAY_MS, MODEL_PATH
from src.camera import get_webcam
from src.converter import prepare_image
from src.dataset import HandClassificationRawDataset
from src.hand_landmarker import get_landmarker
from src.mediapipe_hand_viz import draw_landmarks_on_image, print_landmarks

dataset = HandClassificationRawDataset("dataset", "hand_dataset01")


def make_dataset_demo() -> None:
    """Run webcam capture loop and handle print/capture/quit commands."""
    webcam_capture = get_webcam()
    landmarker = get_landmarker(model_path=MODEL_PATH)

    while webcam_capture.isOpened():
        success, image = webcam_capture.read()
        if not success:
            logging.error("Could not read frame from webcam.")
            break

        image = cv2.flip(image, 1)
        prepared_image = prepare_image(image)
        hand_landmarker_result = landmarker.detect(prepared_image)
        annotated_image = draw_landmarks_on_image(
            image, hand_landmarker_result)
        cv2.imshow('Hand Landmarker Result', annotated_image)

        key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
        if key == ord('p'):
            logging.info("Command : Print")
            print_landmarks(hand_landmarker_result, False, False, True)
        elif key == ord('c'):
            logging.info("Command : Capture")
            try:
                dataset.add_datapoint(
                    image, hand_landmarker_result)
            except Exception as e:
                logging.error(e)
        elif key == ord('q'):
            logging.info("Command : Quit")
            break

    webcam_capture.release()
    cv2.destroyAllWindows()
