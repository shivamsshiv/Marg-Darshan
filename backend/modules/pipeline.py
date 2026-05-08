from modules.bert_intent import BERTIntentClassifier
from modules.camera import decode_image, image_size
from modules.config import get_settings
from modules.depth_midas import MiDaSDepthEstimator
from modules.direction_detection import DirectionDetector
from modules.frame_control import FrameController
from modules.logic_engine import LogicEngine
from modules.ocr import OCRReader
from modules.schemas import GuidanceResponse, HealthResponse, Intent
from modules.target_stabilizer import TargetStabilizer
from modules.t5_generator import T5SentenceGenerator
from modules.tts import TTSService
from modules.yolo_detection import YOLODetector


class GuidancePipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.frame_controller = FrameController(self.settings.frame_fps)
        self.detector = YOLODetector(self.settings.yolo_model_path, self.settings.enable_heavy_models)
        self.stabilizer = TargetStabilizer(max_targets=2)
        self.depth = MiDaSDepthEstimator(self.settings.midas_model_path, self.settings.enable_heavy_models)
        self.direction = DirectionDetector(self.settings.center_zone_ratio)
        self.intent = BERTIntentClassifier(self.settings.bert_model_path, self.settings.enable_heavy_models)
        self.ocr = OCRReader()
        self.logic = LogicEngine()
        self.generator = T5SentenceGenerator(self.settings.t5_model_path, self.settings.enable_heavy_models)
        self.tts = TTSService(self.settings.enable_server_tts)

    def health(self) -> HealthResponse:
        return HealthResponse(
            models={
                "yolo": self.detector.mode,
                "midas": self.depth.mode,
                "bert": self.intent.mode,
                "ocr": self.ocr.mode,
                "t5": self.generator.mode,
                "tts": self.tts.mode,
            },
            frame_fps=self.settings.frame_fps,
        )

    def process_frame(self, image_bytes: bytes, voice_text: str = "", speak: bool = False) -> GuidanceResponse:
        image = decode_image(image_bytes)
        width, _ = image_size(image)
        intent = self.intent.classify(voice_text)
        detections = self.detector.detect(image)
        detections = self.stabilizer.update(detections)
        detections = self.depth.estimate(image, detections)
        detections = self.direction.annotate(detections, width)
        needs_ocr = intent.name == "read_text" or any(word in voice_text.lower() for word in self.settings.ocr_keywords)
        ocr_text = self.ocr.read(image) if needs_ocr else ""
        mode, decision_text, detections = self.logic.decide(detections, intent, ocr_text)
        guidance = self.generator.generate(decision_text)
        spoken = self.tts.speak(guidance) if speak else False
        return GuidanceResponse(
            mode=mode,
            guidance=guidance,
            intent=intent,
            detections=detections,
            ocr_text=ocr_text,
            spoken=spoken,
            debug={
                "models": self.health().models,
                "ocr_requested": needs_ocr,
            },
        )

    def process_voice_only(self, voice_text: str, speak: bool = False) -> GuidanceResponse:
        intent = self.intent.classify(voice_text)
        if intent.name == "none":
            guidance = "Camera guidance is active. Ask a question any time."
            mode = "idle"
        else:
            guidance = "I heard you. Please keep the camera pointed forward so I can answer with the scene."
            mode = "question"
        spoken = self.tts.speak(guidance) if speak else False
        return GuidanceResponse(
            mode=mode,
            guidance=guidance,
            intent=intent,
            spoken=spoken,
        )
