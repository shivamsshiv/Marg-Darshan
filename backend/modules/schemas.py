from typing import Any, Literal

from pydantic import BaseModel, Field


Direction = Literal["left", "center", "right", "unknown"]


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float]
    track_id: int | None = None
    direction: Direction = "unknown"
    distance_hint: Literal["very_near", "near", "mid", "far", "unknown"] = "unknown"
    priority: float = 0.0


class Intent(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)
    raw_text: str = ""


class GuidanceResponse(BaseModel):
    status: Literal["ok", "error"] = "ok"
    mode: Literal["navigation", "question", "ocr", "idle"] = "navigation"
    guidance: str
    intent: Intent
    detections: list[Detection] = []
    ocr_text: str = ""
    spoken: bool = False
    debug: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    models: dict[str, str]
    frame_fps: int
