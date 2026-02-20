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

def input_with_options(question : str, options : list[str]) -> str :
    assert question, "Question cannot be empty"
    assert options, "Options cannot be empty"
    for option in options:
        assert option, "Option cannot be empty"

    options_str = "\n".join(f"[{i+1}] {option.capitalize()}" for i, option in enumerate(options))

    print(question)
    while True :
        try : 
            print(options_str)
            inp = int(input())
            assert inp > 0 and inp < len(options) + 1, "Invalid options index"
            return options[inp-1]
        except : 
            print(f"Invalid option {inp}\n")

def label_dataset_demo():
    # mode = input_with_options("Label dataset : ", LABEL_MODES)
    # Full Labelling by default
    keys = dataset.get_keys()
    print(keys)