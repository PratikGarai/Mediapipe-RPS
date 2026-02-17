import cv2
import mediapipe as mp
from numpy import ndarray

from constants import FRAME_EXTRACTION_PADDING, MAX_INT


def get_landmarker(model_path: str = None):

    assert model_path is not None, "Model path must be provided."

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )

    return HandLandmarker.create_from_options(options)


class HandLandmarkerResult:
    def __init__(self, hand_landmarker_result):
        self.handedness = None
        self.landmarks = None
        self.world_landmarks = None

        try:
            self.handedness = hand_landmarker_result.handedness
            self.landmarks = hand_landmarker_result.hand_landmarks
            self.world_landmarks = hand_landmarker_result.hand_world_landmarks
        except:
            raise "Error extracting propety from hand_landmarker_result. Ensure elements exist"

        assert self.handedness, "Handednees data cannot be null"
        assert self.landmarks, "Landmarks data cannot be null"
        assert self.world_landmarks, "World landmarks data cannot be null"

        assert len(self.handedness) != 0, "Length of handedness variable is 0"
        assert len(self.landmarks) != 0, "Length of landmarks variable is 0"
        assert len(
            self.world_landmarks) != 0, "Length of world_landmarks variable is 0"

        self.handedness = self.handedness[0]
        self.landmarks = self.landmarks[0]
        self.world_landmarks = self.world_landmarks[0]


def extract_hand_image_slice(image: ndarray, landmarks):
    minx = MAX_INT
    maxx = -MAX_INT
    miny = MAX_INT
    maxy = -MAX_INT

    for landmark in landmarks:
        minx = min(minx, landmark.x)
        maxx = max(maxx, landmark.x)
        miny = min(miny, landmark.y)
        maxy = max(maxy, landmark.y)

    height, width, dimensions = image.shape
    minx = max(int(minx * width) - FRAME_EXTRACTION_PADDING, 0)
    maxx = min(int(maxx * width) + FRAME_EXTRACTION_PADDING, width)
    miny = max(int(miny * height) - FRAME_EXTRACTION_PADDING, 0)
    maxy = min(int(maxy * height) + FRAME_EXTRACTION_PADDING, height)

    subimage = image[miny:maxy, minx:maxx]
    subimage = cv2.resize(subimage, (128, 128))

    return subimage


def extract_flattened_coordinates(landmarks, extract_x: bool = True, extract_y: bool = True, extract_z: bool = True):
    results = []
    for landmark in landmarks:
        results.append(landmark.x)
        results.append(landmark.y)
        results.append(landmark.z)
    return results
