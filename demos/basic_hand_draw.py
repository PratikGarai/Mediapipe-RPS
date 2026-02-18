import logging

import cv2

from constants import FRAME_DELAY_MS, MODEL_PATH
from src.commons.camera import get_webcam
from src.commons.converter import prepare_image
from src.commons.hand_landmarker import get_landmarker
from src.commons.mediapipe_hand_viz import draw_landmarks_on_image


def basic_hand_draw_demo():
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

        if cv2.waitKey(FRAME_DELAY_MS) & 0xFF == ord('q'):
            break

    webcam_capture.release()
    cv2.destroyAllWindows()
