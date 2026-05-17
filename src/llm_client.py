from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from openai import OpenAI

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_persona() -> str:
    local = CONFIG_DIR / "persona.local.txt"
    path = local if local.exists() else CONFIG_DIR / "persona.txt"
    return path.read_text(encoding="utf-8").strip()


class AlyaClient:
    def __init__(self) -> None:
        self.model = os.environ["MODEL_NAME"]
        self.temperature = float(os.environ.get("TEMPERATURE", "0.8"))
        self.max_tokens = int(os.environ.get("MAX_TOKENS", "512"))
        self.history_turns = int(os.environ.get("HISTORY_TURNS", "12"))
        self.client = OpenAI(
            base_url=os.environ["LM_STUDIO_URL"],
            api_key=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
        )
        self.system_prompt = load_persona()
        self.history: list[dict[str, str]] = []

    def reset(self) -> None:
        self.history.clear()

    def _trimmed_history(self) -> list[dict[str, str]]:
        max_messages = self.history_turns * 2
        if len(self.history) <= max_messages:
            return self.history
        return self.history[-max_messages:]

    def _build_messages(self, user_input: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            *self._trimmed_history(),
            {"role": "user", "content": user_input},
        ]

    def stream_reply(self, user_input: str) -> Iterator[str]:
        messages = self._build_messages(user_input)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        chunks: list[str] = []
        started = False
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            piece = getattr(delta, "content", None)
            if not piece:
                continue
            if not started:
                piece = piece.lstrip()
                if not piece:
                    continue
                started = True
            chunks.append(piece)
            yield piece
        full_reply = "".join(chunks).strip()
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": full_reply})
