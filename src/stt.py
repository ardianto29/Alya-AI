from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger(__name__)


class STTClient:
    def __init__(self) -> None:
        try:
            self.base_url = os.environ["STT_URL"].rstrip("/")
        except KeyError as e:
            raise RuntimeError("STT_URL env var required (set in .env)") from e
        self.default_language = os.environ.get("STT_LANGUAGE", "ja")
        self.enabled = os.environ.get("STT_ENABLED", "true").lower() in {"1", "true", "yes"}
        timeout = float(os.environ.get("STT_TIMEOUT", "60"))
        self.client = httpx.Client(timeout=timeout)

    def transcribe(
        self,
        audio: bytes,
        filename: str = "audio.webm",
        content_type: str = "audio/webm",
        language: str | None = None,
    ) -> dict:
        files = {"file": (filename, audio, content_type)}
        data = {"language": language or self.default_language}
        r = self.client.post(f"{self.base_url}/transcribe", files=files, data=data)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()
