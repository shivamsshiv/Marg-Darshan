from __future__ import annotations

from dataclasses import dataclass

from modules.schemas import Detection


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


@dataclass
class StableTarget:
    track_id: int
    label: str
    confidence: float
    bbox: list[float]
    hits: int = 1
    missed: int = 0


class TargetStabilizer:
    def __init__(self, max_targets: int = 2, alpha: float = 0.68) -> None:
        self.max_targets = max_targets
        self.alpha = alpha
        self._targets: list[StableTarget] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            for target in self._targets:
                target.missed += 1
            self._targets = [target for target in self._targets if target.missed <= 2 and target.hits >= 2]
            return self._to_detections()

        matched_target_indexes: set[int] = set()
        for detection in detections:
            best_index = -1
            best_iou = 0.0
            for index, target in enumerate(self._targets):
                if index in matched_target_indexes:
                    continue
                score = _iou(detection.bbox, target.bbox)
                if score > best_iou:
                    best_iou = score
                    best_index = index

            if best_index >= 0 and best_iou >= 0.18:
                target = self._targets[best_index]
                target.bbox = [
                    target.bbox[i] * self.alpha + detection.bbox[i] * (1 - self.alpha)
                    for i in range(4)
                ]
                target.confidence = max(detection.confidence, target.confidence * 0.92)
                target.label = detection.label
                target.hits += 1
                target.missed = 0
                matched_target_indexes.add(best_index)
            else:
                self._targets.append(
                    StableTarget(
                        track_id=self._next_id,
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox=detection.bbox,
                    )
                )
                self._next_id += 1

        for index, target in enumerate(self._targets):
            if index not in matched_target_indexes and all(_iou(target.bbox, d.bbox) < 0.18 for d in detections):
                target.missed += 1

        self._targets = sorted(
            [target for target in self._targets if target.missed <= 2],
            key=lambda target: (target.hits, target.confidence),
            reverse=True,
        )[: self.max_targets]
        return self._to_detections()

    def _to_detections(self) -> list[Detection]:
        stable = []
        for target in self._targets:
            if target.hits < 2 and target.missed > 0:
                continue
            stable.append(
                Detection(
                    label=target.label,
                    confidence=round(target.confidence, 2),
                    bbox=[round(value, 2) for value in target.bbox],
                    track_id=target.track_id,
                )
            )
        return stable
