"""Handle storage and retrieval for raw hand-classification datapoints."""

import logging
import os
from pathlib import Path
from uuid import uuid4

import cv2
from numpy import ndarray

from constants import CLASS_LIST
from src.hand_landmarker import (HandLandmarkerResult,
                                 extract_flattened_coordinates,
                                 extract_hand_image_slice)


class HandClassificationRawDataset:
    """Store and retrieve raw hand-classification image and coordinate data."""

    def __init__(self, dataset_root_path: str, dataset_name: str):
        """Initialize dataset folders and file paths.

        Args:
            dataset_root_path: Root directory that contains datasets.
            dataset_name: Dataset folder name to create or reuse.
        """
        assert dataset_root_path, "Dataset needs a root path"
        assert dataset_name, "Dataset needs a name"

        self.classes = CLASS_LIST

        self.dataset_root_path = dataset_root_path
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(
            self.dataset_root_path, self.dataset_name)

        self.images_folder = os.path.join(self.dataset_path, "img_dataset")
        self.coords_file = os.path.join(self.dataset_path, "coords.txt")
        self.truth_file = os.path.join(self.dataset_path, "truth.txt")

        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.images_folder, exist_ok=True)
        Path(self.coords_file).touch(exist_ok=True)
        Path(self.truth_file).touch(exist_ok=True)

    def add_datapoint(self, image: ndarray, hand_landmarker_result) -> None:
        """Extract and persist a single image/coordinate datapoint.

        Args:
            image: Captured image frame containing a detected hand.
            hand_landmarker_result: Raw MediaPipe hand detection output.

        Raises:
            Exception: Raised when result parsing fails.
        """
        assert image is not None, "Image cannot be null"
        assert image is not ndarray, "Image has to be an ndarray"
        assert hand_landmarker_result is not None, "Hand landmarker result cannot be null"

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

        # Saving data
        image_path = os.path.join(self.images_folder, f"{element_id}.png")
        cv2.imwrite(image_path, subimage)

        text_to_write = f"{element_id}," + ",".join(map(str, coords)) + "\n"
        with open(self.coords_file, "a") as f:
            f.write(text_to_write)

        print("Saved datapoint")

    def get_keys(self) -> list[str]:
        """Return all datapoint identifiers stored in the coordinates file.

        Returns:
            List of datapoint IDs in file order.
        """
        assert os.path.exists(self.coords_file), "Coords file not found"
        keys = []
        with open(self.coords_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                try:
                    keys.append(str(line.split(",")[0]))
                except:
                    logging.warning(f"Error processing line {i} : {line}")

        return keys

    def get_image(self, element_id: str) -> ndarray:
        """Load an image for a previously stored datapoint ID.

        Args:
            element_id: Datapoint identifier mapped to an image file.

        Returns:
            Loaded image matrix in OpenCV format.
        """
        assert element_id, "Requested element_id is empty"
        image_path = os.path.join(self.images_folder, f"{element_id}.png")
        assert os.path.exists(
            image_path), f"Image path {image_path} does not exist"

        img = cv2.imread(image_path)
        return img

    def get_existing_truth_data(self) -> dict[str, int]:
        """Read the stored truth file into a datapoint-to-class mapping.

        Returns:
            Mapping of datapoint ID to its integer class label.

        Raises:
            Exception: Raised when a truth value cannot be parsed as an integer.
        """
        keys = set(self.get_keys())
        lines = []
        with open(self.truth_file, "r+") as f:
            lines = f.readlines()

        res = {}
        for ln in lines:
            splits = ln.split(",")
            assert len(splits) == 2, f"Line {ln} is of incorrect format"
            key, cat = splits[0], splits[1]

            try:
                cat = int(cat)
            except:
                raise f"Value {cat} cannot be converted to int"

            assert key in keys, f"Key {key} not in the list of keys in the dataset"

            res[key] = cat
        return res

    def update_truth(self, truth: dict[str, int]) -> None:
        """Overwrite the truth file with the provided label mapping.

        Args:
            truth: Mapping of datapoint ID to its integer class label.
        """
        truth_content = ""
        keys = self.get_keys()
        for key in keys:
            assert key in truth, f"Key {key} not present in updated truth"
            truth_content += f"{key},{truth[key]}\n"

        with open(self.truth_file, "w+") as f:
            f.write(truth_content)
