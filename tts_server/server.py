"""F5-TTS HTTP service for Alya voice. Loads model + 18 reference voices at startup."""
import io
import logging
import re
from pathlib import Path

import pykakasi
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from f5_tts.api import F5TTS

_HAS_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
_kks = pykakasi.kakasi()


def to_hiragana(text: str) -> str:
    """Convert mixed Japanese (kanji+kana) to pure hiragana for F5-TTS pronunciation."""
    if not _HAS_CJK.search(text):
        return text
    return "".join(item["hira"] for item in _kks.convert(text))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("alya-tts")

VOICES_DIR = Path(__file__).parent / "voices"
JA_MODEL_DIR = Path.home() / "f5-models" / "JA_21999120"
JA_CKPT = JA_MODEL_DIR / "model_21999120.pt"
JA_VOCAB = JA_MODEL_DIR / "vocab_japanese.txt"
USE_JA_MODEL = JA_CKPT.exists() and JA_VOCAB.exists()

VOICES = {
    "calm":        ("01_calm_baseline.wav",       "こんにちは、私はアリサ・ミハイロヴナ・九条よ。今日はとてもいい天気ね。お散歩日和だわ。"),
    "angry":       ("02_angry_outburst.wav",      "もう！何度言ったら分かるのよ！そんなことばかりしてたら、本当に怒るからね！"),
    "sad":         ("03_sad_quiet.wav",           "...本当はね、ずっと寂しかったの。でも誰にも言えなくて、ずっと我慢してた。"),
    "embarrassed": ("04_embarrassed.wav",         "や、やめてよ！そんなにじっと見ないで！か、顔が熱くなるじゃない、もう！"),
    "emphasis":    ("05_emphasis.wav",            "いい？これは絶対に大事なことだから、ちゃんと聞いてね。"),
    "whispering":  ("06_whispering_secret.wav",   "ねえ、内緒の話があるの。でも誰にも言っちゃダメよ、約束ね。"),
    "soft":        ("07_soft_gentle.wav",         "あの...今日は本当にありがとう。あなたがいてくれて、嬉しかった。"),
    "breathy":     ("08_breathy_intimate.wav",    "はぁ...近すぎるわよ、もう少し離れて。...心臓の音、聞こえちゃうじゃない。"),
    "excited":     ("09_excited_celebrate.wav",   "見て見て！また満点よ！私、本当に天才なのかもしれないわ！"),
    "laughing":    ("10_laughing.wav",            "あはは！何それ、おかしすぎる！もう、お腹痛いってば！"),
    "playful":     ("11_chuckling_playful.wav",   "ふふっ、そんなに慌てて可愛いわね。冗談よ、そんなに怒らないで。"),
    "throat":      ("12_clear_throat.wav",        "えっと...ちょっといいかしら？大事な話があるの、聞いてくれる？"),
    "tired":       ("13_sighing_tired.wav",       "はぁ...あなたって本当にいつもこうよね。もう、しょうがない人ね。"),
    "exhausted":   ("14_panting_exhausted.wav",   "はぁ、はぁ...走りすぎたわ。ちょっと、待って...息が..."),
    "frustrated":  ("15_groaning_frustrated.wav", "うぅ...どうしてこんなことになっちゃったのよ。もう、本当に最悪。"),
    "crying":      ("16_sobbing_soft.wav",        "うっ...違うの、泣いてなんかないんだから。ただ、ちょっと目にゴミが入っただけ。"),
    "dramatic":    ("17_pause_dramatic.wav",      "あのね...言おうか言わないか、ずっと迷ってたの。...やっぱり、なんでもないわ。"),
    "russia":      ("18_russia_signature.wav",    "Ты дурак... ふん、何でもないわ。今のは聞かなかったことにして。"),
}

DEFAULT_VOICE = "calm"

app = FastAPI(title="Alya TTS")
model: F5TTS | None = None


class TTSRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    speed: float = 0.6
    nfe_step: int = 64
    cfg_strength: float = 2.0


@app.on_event("startup")
def load_model() -> None:
    global model
    if USE_JA_MODEL:
        log.info("loading F5-TTS Japanese model: %s", JA_CKPT)
        model = F5TTS(model="F5TTS_Base", ckpt_file=str(JA_CKPT), vocab_file=str(JA_VOCAB))
    else:
        log.info("loading F5-TTS default multilingual model (JA model not found at %s)", JA_CKPT)
        model = F5TTS()
    log.info("model loaded (japanese_model=%s)", USE_JA_MODEL)
    for name, (fname, _) in VOICES.items():
        path = VOICES_DIR / fname
        if not path.exists():
            log.warning("voice %s missing: %s", name, path)
        else:
            log.info("voice %s -> %s", name, fname)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None, "voices": list(VOICES.keys())}


@app.get("/voices")
def list_voices() -> dict:
    return {"voices": list(VOICES.keys()), "default": DEFAULT_VOICE}


@app.post("/tts")
def tts(req: TTSRequest) -> Response:
    if model is None:
        raise HTTPException(503, "model not loaded yet")
    if req.voice not in VOICES:
        raise HTTPException(400, f"unknown voice '{req.voice}'. available: {list(VOICES.keys())}")

    ref_fname, ref_text = VOICES[req.voice]
    ref_path = VOICES_DIR / ref_fname

    # Convert mixed JP (kanji+kana) to pure hiragana so F5-TTS reads with Japanese pronunciation,
    # not Chinese pinyin (model's default bilingual fallback).
    gen_text = to_hiragana(req.text)
    ref_text_hira = to_hiragana(ref_text)

    # Leading pause prevents first-phoneme clipping (known F5-TTS edge artifact).
    if not gen_text.startswith(("、", "。", " ")):
        gen_text = "、" + gen_text

    log.info("tts voice=%s len=%d text=%r", req.voice, len(req.text), req.text[:60])

    wav, sr, _ = model.infer(
        ref_file=str(ref_path),
        ref_text=ref_text_hira,
        gen_text=gen_text,
        speed=req.speed,
        nfe_step=req.nfe_step,
        cfg_strength=req.cfg_strength,
        show_info=lambda *_: None,
        progress=None,
    )

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
