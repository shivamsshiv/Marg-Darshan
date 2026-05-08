import cv2
import numpy as np


class OCRReader:
    def __init__(self) -> None:
        self.mode = "optional"

    def read(self, image: np.ndarray) -> str:
        try:
            import pytesseract

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(rgb)
            return " ".join(text.split())
        except Exception:
            return ""
