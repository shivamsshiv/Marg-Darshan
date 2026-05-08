from pathlib import Path

from modules.schemas import Intent
from modules.voice_input import VoiceInput


class BERTIntentClassifier:
    def __init__(self, model_path: Path, enable_heavy_models: bool) -> None:
        self.model_path = model_path
        self.enable_heavy_models = enable_heavy_models
        self.voice = VoiceInput()
        self.mode = "rules"

    def classify(self, voice_text: str) -> Intent:
        return self.voice.quick_intent(voice_text)
