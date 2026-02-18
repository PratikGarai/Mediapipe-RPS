import logging

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result, show_handedness=False):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        if show_handedness:
            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def print_landmarks(hand_landmarker_result, print_handedness: bool = True, print_landmarks: bool = True, print_world_landmarks: bool = True):
    if hand_landmarker_result == None:
        logging.warning("Landmark data none. Returning...")
        return

    handedness = hand_landmarker_result.handedness
    landmarks = hand_landmarker_result.hand_landmarks
    world_landmarks = hand_landmarker_result.hand_world_landmarks

    if print_landmarks or print_landmarks or print_world_landmarks:
        print("\n\n============================")
        print("Hand Landmarks Data")
        if print_handedness:
            print("\nHandedness  ---------------")
            print(handedness)
        if print_landmarks:
            print("\nLandmarks   ---------------")
            print(landmarks)
        if print_world_landmarks:
            print("\nW Landmarks ---------------")
            print(world_landmarks)
