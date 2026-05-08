from modules.schemas import Intent


class VoiceInput:
    def normalize(self, voice_text: str) -> str:
        return " ".join(voice_text.lower().strip().split())

    def has_question(self, voice_text: str) -> bool:
        text = self.normalize(voice_text)
        return bool(text)

    def quick_intent(self, voice_text: str) -> Intent:
        text = self.normalize(voice_text)
        if not text:
            return Intent(name="none", confidence=1.0, raw_text="")
        if any(word in text for word in ("read", "text", "sign", "board", "label", "kya likha")):
            return Intent(name="read_text", confidence=0.82, raw_text=voice_text)
        if any(word in text for word in ("where", "kidhar", "kaha", "left", "right", "rasta", "jana")):
            return Intent(name="navigation_query", confidence=0.78, raw_text=voice_text)
        if any(word in text for word in ("what", "kya", "samne", "front", "around")):
            return Intent(name="scene_question", confidence=0.76, raw_text=voice_text)
        return Intent(name="general_question", confidence=0.62, raw_text=voice_text)
