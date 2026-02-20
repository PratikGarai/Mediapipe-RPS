import logging

import cv2

from constants import CLASS_LIST, FRAME_DELAY_MS, MODEL_PATH
from src.commons.converter import prepare_image
from src.commons.mediapipe_hand_viz import (draw_landmarks_on_image,
                                            print_landmarks)
from src.dataset import HandClassificationRawDataset

dataset = HandClassificationRawDataset("dataset", "hand_dataset01")

LABEL_MODES = [
    "Full Label",
    "Label Unlabelled"
]

def print_options(question : str, options : list[str]) -> str :
    assert question, "Question cannot be empty"
    assert options, "Options cannot be empty"
    for option in options:
        assert option, "Option cannot be empty"

    options_str = "\n".join(f"[{i+1}] {option.capitalize()}" for i, option in enumerate(options))

    print(question)
    print(options_str)

def label_dataset_demo():
    print_options("Lable image :", CLASS_LIST)
    keys = dataset.get_keys()
    for key in keys:
        try :
            img = dataset.get_image(key)
            print(img.shape)
        except Exception as e :
            logging.error(f"Error loading image {key} : {e}. Skipping...")
            continue
        cv2.imshow("Image", img)
        choice = cv2.waitKey(1) & 0xFF
        cv2.destroyAllWindows()