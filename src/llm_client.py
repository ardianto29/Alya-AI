from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

from openai import OpenAI

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

_TZ = ZoneInfo(os.environ.get("ALYA_TIMEZONE", "Asia/Jakarta"))
_DAYS_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_MONTHS_ID = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]


def load_persona() -> str:
    local = CONFIG_DIR / "persona.local.txt"
    path = local if local.exists() else CONFIG_DIR / "persona.txt"
    return path.read_text(encoding="utf-8").strip()


def _time_of_day(hour: int) -> str:
    if 5 <= hour < 11:
        return "pagi"
    if 11 <= hour < 15:
        return "siang"
    if 15 <= hour < 18:
        return "sore"
    if 18 <= hour < 24:
        return "malam"
    return "dini hari"


def _qualifier(hour: int, minute: int) -> str:
    """Kalimat factual tentang posisi waktu — biar LLM tau persis di mana
    posisinya (lewat / menuju), tanpa nge-drive topik tertentu."""
    if hour == 0:
        return f"Baru saja lewat tengah malam ({minute} menit lalu)."
    if 1 <= hour <= 3:
        return f"Sudah {hour} jam {minute} menit lewat tengah malam — dini hari."
    if hour == 4:
        return "Menjelang fajar."
    if 5 <= hour < 7:
        return "Pagi sangat awal."
    if 7 <= hour < 10:
        return "Pagi hari."
    if 10 <= hour < 12:
        return "Menjelang siang."
    if 12 <= hour < 14:
        return "Tengah hari."
    if 14 <= hour < 17:
        return "Sore hari."
    if 17 <= hour < 19:
        return "Menjelang malam."
    if 19 <= hour < 22:
        return "Malam hari."
    if 22 <= hour <= 23:
        mins = (24 * 60) - (hour * 60 + minute)
        return f"Malam larut — {mins} menit lagi menuju tengah malam."
    return ""


def current_time_context() -> str:
    now = datetime.now(_TZ)
    day = _DAYS_ID[now.weekday()]
    month = _MONTHS_ID[now.month]
    return (
        f"[Waktu] {day} {now.day} {month} {now.year}, {now.strftime('%H:%M')} WIB.\n"
        "[Cara ngobrol] Engage SPESIFIK ke statement terakhir user. "
        "Kalau user cerita makanan, bahas makanan-nya (rasanya, di mana). "
        "Kalau sapa kasual, sapa balik. Kalau nanya, jawab. "
        "JANGAN auto-comment soal begadang/tidur/kesehatan kecuali user sendiri yang ngangkat. "
        "Pernah ngomentari topik X di session ini? JANGAN ulang lagi — cari sudut lain."
    )


class AlyaClient:
    def __init__(self) -> None:
        self.model = os.environ["MODEL_NAME"]
        self.temperature = float(os.environ.get("TEMPERATURE", "0.8"))
        self.max_tokens = int(os.environ.get("MAX_TOKENS", "512"))
        self.history_turns = int(os.environ.get("HISTORY_TURNS", "12"))
        self.top_p = float(os.environ.get("TOP_P", "0.92"))
        # frequency_penalty mengurangi pengulangan frasa yang sama persis
        self.frequency_penalty = float(os.environ.get("FREQUENCY_PENALTY", "0.6"))
        # presence_penalty mendorong model nyebut topik/kata baru
        self.presence_penalty = float(os.environ.get("PRESENCE_PENALTY", "0.4"))
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
        # /no_think = directive Qwen3 untuk skip thinking/reasoning mode.
        # WAJIB di LAST turn (user message), bukan di system — di system di-ignore.
        # Tanpa ini, qwen3 buang token ke reasoning_content (gak ke-stream ke client)
        # dan UI stuck di typing dots tanpa output.
        system_content = f"{self.system_prompt}\n\n{current_time_context()}"
        return [
            {"role": "system", "content": system_content},
            *self._trimmed_history(),
            {"role": "user", "content": f"{user_input} /no_think"},
        ]

    def stream_reply(self, user_input: str) -> Iterator[str]:
        messages = self._build_messages(user_input)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
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
