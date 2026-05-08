import time


class FrameController:
    def __init__(self, fps: int = 3) -> None:
        self.fps = fps
        self.min_interval = 1.0 / fps
        self._last_processed = 0.0

    def should_process(self) -> bool:
        now = time.monotonic()
        if now - self._last_processed >= self.min_interval:
            self._last_processed = now
            return True
        return False
