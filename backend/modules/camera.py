import cv2
import numpy as np


def decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded frame.")
    return image


def image_size(image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    return width, height
