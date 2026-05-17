from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

# Strip parenthetical translations like "日本語。(Terjemahan.)" so TTS speaks only Japanese.
# Handles both () and （） full-width parens.
_PAREN_RE = re.compile(r"\s*[（(][^()（）]*[)）]\s*")
# Strip roleplay action markers like *menatap tajam* so TTS doesn't speak them.
_ACTION_RE = re.compile(r"\s*\*[^*\n]+\*\s*")

log = logging.getLogger(__name__)


class TTSClient:
    def __init__(self) -> None:
        try:
            self.base_url = os.environ["TTS_URL"].rstrip("/")
        except KeyError as e:
            raise RuntimeError("TTS_URL env var required (set in .env)") from e
        self.default_voice = os.environ.get("TTS_VOICE", "calm")
        self.enabled = os.environ.get("TTS_ENABLED", "true").lower() in {"1", "true", "yes"}
        timeout = float(os.environ.get("TTS_TIMEOUT", "60"))
        self.client = httpx.Client(timeout=timeout)
        self._player: list[str] | None = None

    @staticmethod
    def _detect_player() -> list[str]:
        for cmd in ("paplay", "aplay", "ffplay"):
            if shutil.which(cmd):
                if cmd == "ffplay":
                    return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
                return [cmd]
        raise RuntimeError("no audio player found (need paplay, aplay, or ffplay)")

    def synthesize(self, text: str, voice: str | None = None, speed: float | None = None) -> bytes:
        payload: dict = {"text": text, "voice": voice or self.default_voice}
        if speed is not None:
            payload["speed"] = speed
        r = self.client.post(f"{self.base_url}/tts", json=payload)
        r.raise_for_status()
        return r.content

    def play(self, audio: bytes) -> None:
        if self._player is None:
            self._player = self._detect_player()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            path = Path(f.name)
        try:
            subprocess.run([*self._player, str(path)], check=True, capture_output=True)
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def strip_for_tts(text: str) -> str:
        """Remove translations and action markers so TTS speaks only the primary language."""
        text = _ACTION_RE.sub(" ", text)
        text = _PAREN_RE.sub(" ", text)
        return text.strip()

    def say(self, text: str, voice: str | None = None, speed: float | None = None) -> None:
        if not self.enabled:
            return
        spoken = self.strip_for_tts(text)
        if not spoken:
            return
        try:
            audio = self.synthesize(spoken, voice=voice, speed=speed)
            self.play(audio)
        except httpx.HTTPError as e:
            log.warning("tts http error: %s", e)
        except subprocess.CalledProcessError as e:
            log.warning("tts playback error: %s", e)

    def list_voices(self) -> list[str]:
        r = self.client.get(f"{self.base_url}/voices")
        r.raise_for_status()
        return r.json()["voices"]

    def health(self) -> dict:
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()
