from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.llm_client import AlyaClient
from src.stt import STTClient
from src.tts import TTSClient

load_dotenv()

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "config" / "images"
_AVATAR_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

app = FastAPI(title="Alya-AI")
alya = AlyaClient()
stt = STTClient()
tts = TTSClient()


class ChatRequest(BaseModel):
    message: str


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def _find_avatar() -> Path | None:
    if not IMAGES_DIR.exists():
        return None
    for p in sorted(IMAGES_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in _AVATAR_EXTS:
            return p
    return None


@app.get("/api/avatar")
def avatar() -> FileResponse:
    p = _find_avatar()
    if not p:
        raise HTTPException(404, "no avatar in config/images/")
    return FileResponse(p)


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    def token_stream():
        for piece in alya.stream_reply(req.message):
            yield piece

    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")


@app.post("/api/reset")
def reset() -> dict[str, str]:
    alya.reset()
    return {"status": "ok"}


@app.post("/api/tts")
def tts_endpoint(req: TTSRequest) -> Response:
    if not tts.enabled:
        raise HTTPException(503, "TTS disabled (TTS_ENABLED=false)")
    spoken = tts.strip_for_tts(req.text)
    if not spoken:
        raise HTTPException(400, "nothing speakable after stripping markers/translations")
    try:
        audio = tts.synthesize(spoken, voice=req.voice, speed=req.speed)
    except Exception as e:
        raise HTTPException(502, f"tts upstream error: {e}") from e
    return Response(content=audio, media_type="audio/wav")


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
) -> dict:
    if not stt.enabled:
        raise HTTPException(503, "STT disabled (STT_ENABLED=false)")
    audio = await file.read()
    if not audio:
        raise HTTPException(400, "empty audio")
    try:
        return stt.transcribe(
            audio,
            filename=file.filename or "audio.webm",
            content_type=file.content_type or "audio/webm",
            language=language,
        )
    except Exception as e:
        raise HTTPException(502, f"stt upstream error: {e}") from e


@app.get("/api/state")
def state() -> dict:
    return {
        "model": alya.model,
        "history_messages": len(alya.history),
        "stt_enabled": stt.enabled,
        "tts_enabled": tts.enabled,
    }


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
