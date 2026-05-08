from pathlib import Path


class TTSService:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.mode = "browser"

    def speak(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception:
            return False

    def audio_path_for_last_speech(self) -> Path | None:
        return None
