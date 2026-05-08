from pathlib import Path


class T5SentenceGenerator:
    def __init__(self, model_path: Path, enable_heavy_models: bool) -> None:
        self.model_path = model_path
        self.enable_heavy_models = enable_heavy_models
        self.mode = "template"

    def generate(self, decision_text: str) -> str:
        text = decision_text.strip()
        if not text:
            return "I am ready."
        return text
