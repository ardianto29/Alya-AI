"""faster-whisper HTTP service for Alya voice input. Loads large-v3 fp16 at startup."""
import ctypes
import io
import logging
import os
import sys
import time
from pathlib import Path

# Preload bundled NVIDIA CUDA libs before importing faster_whisper / ctranslate2.
# CTranslate2 dlopens libcublas/libcudnn via the OS loader and does not bundle
# its own preload step (unlike PyTorch). Without this, transcribe() crashes with
# "Library libcublas.so.12 is not found or cannot be loaded" when the libs only
# live inside the venv's site-packages/nvidia/*/lib/.
_venv_root = Path(sys.prefix)
for _lib_dir in _venv_root.glob("lib/python*/site-packages/nvidia/*/lib"):
    for _so in sorted(_lib_dir.glob("*.so*")):
        try:
            ctypes.CDLL(str(_so), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("alya-stt")

MODEL_SIZE = os.getenv("STT_MODEL", "large-v3")
DEVICE = os.getenv("STT_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("STT_COMPUTE", "float16")
# Batas thread saat DEVICE=cpu — biar inference gak monopoli semua core (Ultra 7
# 270K punya 24 core, default 8 supaya sisa 16 free buat OS + service lain).
# Ignored saat DEVICE=cuda.
CPU_THREADS = int(os.getenv("STT_CPU_THREADS", "8"))
DEFAULT_LANG = os.getenv("STT_DEFAULT_LANG", "ja")
# initial_prompt = bias Whisper supaya kenal nama/istilah uncommon.
# Tanpa ini, "Alya" sering ditranskrip jadi "Aliya"/"Alia"/"Aliyah".
# Override via env STT_INITIAL_PROMPT untuk tambah nama personal lain.
DEFAULT_INITIAL_PROMPT = os.getenv(
    "STT_INITIAL_PROMPT",
    "Halo Alya. Aku ngobrol sama Alya Mikhailovna. "
    "Nama yang benar: Alya (A-L-Y-A), bukan Alia, bukan Aliya, bukan Aliyah. "
    "アリサ・ミハイロヴナ. Obrolan santai bahasa Indonesia campur Jepang.",
)

app = FastAPI(title="Alya STT")
model: WhisperModel | None = None


@app.on_event("startup")
def load_model() -> None:
    global model
    log.info("loading faster-whisper %s on %s (%s)", MODEL_SIZE, DEVICE, COMPUTE_TYPE)
    t0 = time.time()
    kwargs = {"device": DEVICE, "compute_type": COMPUTE_TYPE}
    if DEVICE == "cpu":
        kwargs["cpu_threads"] = CPU_THREADS
        log.info("cpu_threads=%d (cap supaya nggak monopoli semua core)", CPU_THREADS)
    model = WhisperModel(MODEL_SIZE, **kwargs)
    log.info("model loaded in %.1fs", time.time() - t0)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "cpu_threads": CPU_THREADS if DEVICE == "cpu" else None,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    task: str = Form("transcribe"),
) -> dict:
    if model is None:
        raise HTTPException(503, "model not loaded yet")
    if task not in {"transcribe", "translate"}:
        raise HTTPException(400, "task must be 'transcribe' or 'translate'")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio payload")

    lang = language or DEFAULT_LANG
    if lang == "auto":
        lang = None

    t0 = time.time()
    segments, info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=lang,
        task=task,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        initial_prompt=DEFAULT_INITIAL_PROMPT or None,
    )
    text = "".join(seg.text for seg in segments).strip()
    elapsed = time.time() - t0

    log.info(
        "transcribe bytes=%d lang=%s detected=%s dur=%.2fs took=%.2fs text=%r",
        len(audio_bytes), lang, info.language, info.duration, elapsed, text[:80],
    )

    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "elapsed": elapsed,
    }
