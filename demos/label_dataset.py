"""Provide an interactive workflow for reviewing and labeling dataset images."""

import logging

import cv2

from src.dataset import HandClassificationRawDataset
from src.utils import generate_options_string

dataset = HandClassificationRawDataset("dataset", "hand_dataset01")


def label_dataset_demo() -> None:
    """Iterate dataset images and capture key-based labeling input."""
    class_len = len(dataset.classes)
    questionare = generate_options_string("Label image :", dataset.classes)
    print(questionare)

    keys = dataset.get_keys()
    existing_truth = dataset.get_existing_truth_data()
    new_truth = {}

    cv2.namedWindow("Image")

    for key in keys:
        try:
            img = dataset.get_image(key)
        except Exception as e:
            logging.error(f"Error loading image {key} : {e}. Skipping...")
            continue
        cv2.imshow("Image", img)

        while True:
            if key in existing_truth:
                print(f"Existing mapping for image : ",
                      dataset.classes[existing_truth[key]-1])
            print("Enter new mapping : ", end="")
            choice = cv2.waitKey(0)

            if choice & 0xFF == ord('q'):
                print("Breaking out of loop")
                cv2.destroyAllWindows()
                return

            index = choice - ord("1")
            if 0 <= index < class_len:
                print(f"Selected option : {dataset.classes[index]}")
                new_truth[key] = index + 1
                break
            else:
                print(f"Unrecognized key: {choice}")
    
    cv2.destroyAllWindows()
    dataset.update_truth(new_truth)
