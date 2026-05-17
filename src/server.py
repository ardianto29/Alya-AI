from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.llm_client import AlyaClient

load_dotenv()

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Alya-AI")
alya = AlyaClient()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


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


@app.get("/api/state")
def state() -> dict[str, int | str]:
    return {
        "model": alya.model,
        "history_messages": len(alya.history),
    }


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
