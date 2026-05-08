from modules.schemas import Detection


class DirectionDetector:
    def __init__(self, center_zone_ratio: float = 0.34) -> None:
        self.center_zone_ratio = center_zone_ratio

    def annotate(self, detections: list[Detection], image_width: int) -> list[Detection]:
        center_left = image_width * (0.5 - self.center_zone_ratio / 2)
        center_right = image_width * (0.5 + self.center_zone_ratio / 2)
        for detection in detections:
            x1, _, x2, _ = detection.bbox
            center_x = (x1 + x2) / 2
            if center_x < center_left:
                detection.direction = "left"
            elif center_x > center_right:
                detection.direction = "right"
            else:
                detection.direction = "center"
        return detections
