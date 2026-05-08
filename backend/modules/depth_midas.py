from pathlib import Path

import cv2
import numpy as np

from modules.schemas import Detection


class MiDaSDepthEstimator:
    def __init__(self, model_path: Path, enable_heavy_models: bool) -> None:
        self.model_path = model_path
        self.enable_heavy_models = enable_heavy_models
        self.mode = "heuristic"

    def estimate(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for detection in detections:
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                detection.distance_hint = "unknown"
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(width * height, 1)
            lower_position = y2 / max(height, 1)
            contrast = float(np.std(gray[y1:y2, x1:x2])) / 128.0
            near_score = min(1.0, area_ratio * 2.2 + lower_position * 0.35 + contrast * 0.15)
            if near_score > 0.78:
                detection.distance_hint = "very_near"
            elif near_score > 0.55:
                detection.distance_hint = "near"
            elif near_score > 0.32:
                detection.distance_hint = "mid"
            else:
                detection.distance_hint = "far"
        return detections
