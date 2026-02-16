import logging
import os

import cv2
import mediapipe as mp

from visualizer import draw_landmarks_on_image

logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.path.join(os.getcwd(), "models", "hand_landmarker.task")
FRAME_DELAY_MS = 1

def get_webcam() -> cv2.VideoCapture:
    webcam_capture = cv2.VideoCapture(0)

    if not webcam_capture.isOpened():
        logging.error("Could not open webcam.")
        exit()
    else:
        logging.info("Webcam opened successfully.")

    return webcam_capture


def prepare_image(image: cv2.Mat) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

def get_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
    )

    return HandLandmarker.create_from_options(options)

def main():
    webcam_capture = get_webcam()
    landmarker = get_landmarker()

    while webcam_capture.isOpened():
        success, image = webcam_capture.read()
        if not success:
            logging.error("Could not read frame from webcam.")
            break

        image = cv2.flip(image, 1)
        prepared_image = prepare_image(image)
        hand_landmarker_result = landmarker.detect(prepared_image)
        annotated_image = draw_landmarks_on_image(image, hand_landmarker_result)
        cv2.imshow('Hand Landmarker Result', annotated_image)

        if cv2.waitKey(FRAME_DELAY_MS) & 0xFF == ord('q'):
            break

    webcam_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()