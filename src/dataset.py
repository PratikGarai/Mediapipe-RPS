import os
from uuid import uuid4

import cv2
from numpy import ndarray

from constants import FRAME_DELAY_MS
from src.commons.hand_landmarker import (HandLandmarkerResult,
                                         extract_flattened_coordinates,
                                         extract_hand_image_slice)


class HandClassificationRawDataset:
    def __init__(self, dataset_root_path: str, dataset_name: str):
        assert dataset_root_path, "Dataset needs a root path"
        assert dataset_name, "Dataset needs a name"

        self.dataset_root_path = dataset_root_path
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(
            self.dataset_root_path, self.dataset_name)

        self.images_folder = os.path.join(self.dataset_path, "img_dataset")
        self.coords_file = os.path.join(self.dataset_path, "coords.txt")

        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.images_folder, exist_ok=True)

    def _get_class_selection(self, classes: list[str]) -> str:
        assert classes is not None and len(
            classes) != 0, "Classes list cannot be empty or None"

        options_str = ""
        for i, c in enumerate(classes):
            options_str += f"[{i}] {c}\n"
        print(options_str)
        print("Select data class : ")
        key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
        try:
            key = key - ord('0')
            choice = int(key)
            assert choice > 0 and choice < len(classes+1)
            return classes[choice-1]
        except:
            print("Invalid option selected.")

    def add_datapoint(self, image: ndarray, hand_landmarker_result, classes: list[str] = []):
        assert image is not None, "Image cannot be null"
        assert image is not ndarray, "Image has to be an ndarray"
        assert hand_landmarker_result is not None, "Hand landmarker result cannot be null"
        assert classes is not None and len(
            classes) != 0, "Classes list cannot be empty or None"

        try:
            validated_landmarker_result = HandLandmarkerResult(
                hand_landmarker_result)
        except Exception as e:
            raise e

        handedness = validated_landmarker_result.handedness
        landmarks = validated_landmarker_result.landmarks
        world_landmarks = validated_landmarker_result.world_landmarks

        element_id = str(uuid4())
        print(
            "\n\n============================\nGenerating datapoint for id : ", element_id)

        # Image Extraction
        subimage = extract_hand_image_slice(image, landmarks)
        cv2.imshow("Captured subimage", subimage)

        # Flattened Coordinates Extraction
        coords = extract_flattened_coordinates(world_landmarks)

        # option = self._get_class_selection(classes)
        # print(option)
