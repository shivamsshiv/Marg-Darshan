from collections import Counter

from modules.schemas import Detection, Intent


class LogicEngine:
    blocking_labels = {
        "person",
        "chair",
        "car",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
        "object",
        "scene",
    }

    def decide(
        self,
        detections: list[Detection],
        intent: Intent,
        ocr_text: str = "",
    ) -> tuple[str, str, list[Detection]]:
        prioritized = self._prioritize(detections)
        if intent.name == "read_text":
            if ocr_text:
                return "ocr", f"I can read: {ocr_text}", prioritized
            return "ocr", "I could not read clear text from this frame.", prioritized

        if intent.name in {"scene_question", "general_question"}:
            return "question", self._answer_scene_question(prioritized), prioritized

        if intent.name == "navigation_query":
            return "question", self._navigation_guidance(prioritized, direct=True), prioritized

        return "navigation", self._navigation_guidance(prioritized, direct=False), prioritized

    def _prioritize(self, detections: list[Detection]) -> list[Detection]:
        for detection in detections:
            direction_score = {"center": 0.35, "left": 0.18, "right": 0.18, "unknown": 0.05}[detection.direction]
            distance_score = {
                "very_near": 0.45,
                "near": 0.32,
                "mid": 0.16,
                "far": 0.05,
                "unknown": 0.08,
            }[detection.distance_hint]
            label_score = 0.15 if detection.label in self.blocking_labels else 0.05
            detection.priority = round(detection.confidence * 0.25 + direction_score + distance_score + label_score, 3)
        return sorted(detections, key=lambda item: item.priority, reverse=True)

    def _answer_scene_question(self, detections: list[Detection]) -> str:
        if not detections:
            return "I do not see a clear object right now."
        counts = Counter(d.label for d in detections[:5])
        summary = ", ".join(f"{count} {label}" for label, count in counts.items())
        nearest = detections[0]
        return f"I can see {summary}. The most important thing is {nearest.label} in the {nearest.direction}."

    def _navigation_guidance(self, detections: list[Detection], direct: bool) -> str:
        if not detections:
            return "Path looks mostly clear. Move forward slowly."
        main = detections[0]
        if main.distance_hint in {"very_near", "near"} and main.direction == "center":
            side_counts = Counter(d.direction for d in detections[1:] if d.distance_hint in {"very_near", "near", "mid"})
            safer_side = "left" if side_counts["right"] > side_counts["left"] else "right"
            return f"Obstacle ahead. Move slightly {safer_side} and go slowly."
        if main.distance_hint in {"very_near", "near"}:
            return f"Obstacle close on your {main.direction}. Keep slightly away and continue slowly."
        if direct:
            return "No close obstacle in the center. You can move forward carefully."
        return f"Path is mostly clear. I see {main.label} toward the {main.direction}."
