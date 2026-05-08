from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np

from modules.camera import image_size
from modules.schemas import Detection


class YOLODetector:
    def __init__(self, model_path: Path, enable_heavy_models: bool) -> None:
        self.model_path = model_path
        self.enable_heavy_models = enable_heavy_models
        self.model = None
        self._track_ids = itertools.count(1)
        self.mode = "demo"
        if enable_heavy_models and model_path.exists():
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(model_path))
                self.mode = "yolo"
            except Exception:
                self.model = None
                self.mode = "demo"

    def detect(self, image: np.ndarray) -> list[Detection]:
        if self.model is not None:
            return self._detect_yolo(image)
        return self._detect_demo(image)

    def _detect_yolo(self, image: np.ndarray) -> list[Detection]:
        results = self.model.track(image, persist=True, verbose=False)
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = result.names
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                track_id = int(box.id[0].item()) if box.id is not None else next(self._track_ids)
                if conf >= 0.35:
                    detections.append(
                        Detection(
                            label=names.get(cls_id, "object"),
                            confidence=conf,
                            bbox=[float(v) for v in xyxy],
                            track_id=track_id,
                        )
                    )
        return detections[:5]

    def _detect_demo(self, image: np.ndarray) -> list[Detection]:
        width, height = image_size(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(gray, 90, 210)
        kernel = np.ones((9, 9), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int, float]] = []
        frame_area = width * height
        min_area = max(4_500, int(frame_area * 0.04))
        max_area = int(frame_area * 0.72)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / max(h, 1)
            touches_too_much_border = x <= 4 and y <= 4 and x + w >= width - 4 and y + h >= height - 4
            if area < min_area or area > max_area or touches_too_much_border:
                continue
            if aspect_ratio < 0.18 or aspect_ratio > 5.5:
                continue
            center_x = x + w / 2
            center_weight = 1 - min(abs(center_x - width / 2) / max(width / 2, 1), 1)
            lower_weight = (y + h) / max(height, 1)
            score = min(area / frame_area, 1.0) * 0.55 + center_weight * 0.25 + lower_weight * 0.2
            candidates.append((x, y, w, h, score))
        candidates = sorted(candidates, key=lambda item: item[4], reverse=True)[:2]
        detections = [
            Detection(
                label="object",
                confidence=round(0.52 + min(score, 0.4), 2),
                bbox=[x, y, x + w, y + h],
                track_id=i + 1,
            )
            for i, (x, y, w, h, score) in enumerate(candidates)
        ]
        return detections
