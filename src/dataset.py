import os

import cv2
from numpy import ndarray

from uuid import uuid4

from src.commons.hand_landmarker import HandLandmarkerResult, extract_hand_image_slice, extract_flattened_coordinates


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

    def add_datapoint(self, image : ndarray, hand_landmarker_result, classes : list[str] = []):
        assert image is not None, "Image cannot be null"
        assert image is not ndarray, "Image has to be an ndarray"
        assert hand_landmarker_result is not None, "Hand landmarker result cannot be null"
        assert classes is not None and len(classes) != 0, "Classes list cannot be empty or None"

        try:
            validated_landmarker_result = HandLandmarkerResult(
                hand_landmarker_result)
        except Exception as e:
            raise e

        handedness = validated_landmarker_result.handedness
        landmarks = validated_landmarker_result.landmarks
        world_landmarks = validated_landmarker_result.world_landmarks

        # Image Extraction
        subimage = extract_hand_image_slice(image, landmarks)
        cv2.imshow("Captured subimage", subimage)

        # Flattened Coordinates Extraction
        coords = extract_flattened_coordinates(world_landmarks)

        element_id = str(uuid4())
        print("Generating")