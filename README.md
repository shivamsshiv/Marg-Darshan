# Marg-Darshan

AI Guidance System for visually impaired guidance and navigation.

Web-first assistive guidance prototype with camera, 3 FPS frame control, object detection, tracking-ready output, depth hints, direction detection, optional voice questions, intent understanding, OCR-on-demand, decision logic, sentence generation, and voice output.

The app is designed so it works today in demo mode, then upgrades cleanly when you add YOLO, MiDaS, BERT, and T5 model weights.

## Flow

```text
Browser camera + mic
  -> frame captured every 333 ms
  -> FastAPI backend
  -> YOLO detection / demo detector
  -> MiDaS depth / heuristic depth
  -> direction detection
  -> optional voice intent
  -> OCR only when asked
  -> logic engine
  -> T5 sentence generator / template generator
  -> browser text-to-speech
```

## Project Structure

```text
backend/
  main.py
  modules/
    camera.py
    frame_control.py
    yolo_detection.py
    depth_midas.py
    direction_detection.py
    voice_input.py
    bert_intent.py
    ocr.py
    logic_engine.py
    t5_generator.py
    tts.py
    pipeline.py
  models/
    yolo.pt
    midas.pt
    bert/
    t5/

frontend/
  src/
    main.tsx
    styles.css
```

## Run Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Run Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Camera And Mic

Browsers usually allow camera and microphone only on `localhost` or HTTPS. Start the app, press **Start**, allow camera and microphone permissions, then ask questions such as:

- "What is in front of me?"
- "Where should I go?"
- "Read the sign."
- "Kya samne hai?"

When you do not ask a question, the app keeps giving normal navigation guidance from the camera.

## Adding Real Models

Place model files here:

```text
backend/models/yolo.pt
backend/models/midas.pt
backend/models/bert/
backend/models/t5/
```

Then set:

```text
GUIDANCE_ENABLE_HEAVY_MODELS=true
```

The current YOLO wrapper uses `ultralytics.YOLO(...).track(...)`, so ByteTrack-style tracking is ready through Ultralytics tracking once the package and weights are installed.

Install optional heavy dependencies only when needed:

```powershell
pip install ultralytics torch torchvision transformers easyocr pytesseract pyttsx3
```

For OCR with `pytesseract`, install the Tesseract executable separately and make sure it is available in `PATH`.

## API

Health:

```http
GET /health
```

Frame guidance:

```http
POST /api/guidance/frame
multipart/form-data:
  frame: image/jpeg
  voice_text: optional text
  speak: false
```

Voice-only acknowledgement:

```http
POST /api/guidance/voice
multipart/form-data:
  voice_text: text
  speak: false
```

## Future Mobile App Path

This architecture is already app-friendly:

- Keep FastAPI as the AI backend for heavy models.
- Replace React browser camera with React Native or Flutter camera frames.
- Keep the same `/api/guidance/frame` endpoint.
- Move TTS to native mobile TTS for lower latency.
- Add streaming/WebSocket transport when you need faster continuous guidance.
