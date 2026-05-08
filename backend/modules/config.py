from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT_DIR.parent


class Settings(BaseSettings):
    app_name: str = "AI Guidance System"
    frame_fps: int = Field(default=3, ge=1, le=15)
    model_dir: Path = ROOT_DIR / "models"
    yolo_model_path: Path = ROOT_DIR / "models" / "yolo.pt"
    midas_model_path: Path = ROOT_DIR / "models" / "midas.pt"
    bert_model_path: Path = ROOT_DIR / "models" / "bert"
    t5_model_path: Path = ROOT_DIR / "models" / "t5"
    enable_heavy_models: bool = False
    enable_server_tts: bool = False
    guidance_cooldown_seconds: float = 1.2
    near_depth_threshold: float = 0.42
    center_zone_ratio: float = 0.34
    ocr_keywords: tuple[str, ...] = ("sign", "text", "board", "label", "document")

    class Config:
        env_file = ROOT_DIR / ".env"
        env_prefix = "GUIDANCE_"


@lru_cache
def get_settings() -> Settings:
    return Settings()
