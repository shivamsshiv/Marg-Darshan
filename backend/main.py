import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from modules.pipeline import GuidancePipeline
from modules.schemas import GuidanceResponse, HealthResponse


app = FastAPI(
    title="AI Guidance System",
    description="Camera, voice, detection, depth, direction, OCR, intent, logic, and guidance API.",
    version="0.1.0",
)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if "*" not in allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = GuidancePipeline()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return pipeline.health()


@app.post("/api/guidance/frame", response_model=GuidanceResponse)
async def process_frame(
    frame: UploadFile = File(...),
    voice_text: str = Form(default=""),
    speak: bool = Form(default=False),
) -> JSONResponse:
    image_bytes = await frame.read()
    result = pipeline.process_frame(
        image_bytes=image_bytes,
        voice_text=voice_text.strip(),
        speak=speak,
    )
    return JSONResponse(result.model_dump())


@app.post("/api/guidance/voice", response_model=GuidanceResponse)
async def process_voice_only(
    voice_text: str = Form(default=""),
    speak: bool = Form(default=False),
) -> JSONResponse:
    result = pipeline.process_voice_only(voice_text=voice_text.strip(), speak=speak)
    return JSONResponse(result.model_dump())


static_dir = Path(os.getenv("STATIC_DIR", ""))
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
