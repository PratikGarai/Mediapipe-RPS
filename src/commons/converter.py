import cv2
import mediapipe as mp


def prepare_image(image: cv2.Mat) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
