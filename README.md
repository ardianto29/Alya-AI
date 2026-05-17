# Alya-AI

Personal chatbot + voice assistant dengan persona **Alya Mikhailovna**
(karakter Roshidere). Kamu chat lewat CLI atau web, Alya jawab dalam
Bahasa Jepang + translation Indonesia, terus suaranya keluar dari speaker
pakai voice clone yang mirip suara aslinya.

> **Note:** Project ini personal & eksperimental. Voice clone-nya pakai
> reference audio yang harus kamu siapin sendiri (legalitas voice = your
> responsibility). Kode-nya open, suaranya enggak.

---

## Cara kerja singkat

```
┌─────────────────┐        ┌──────────────────────┐
│  Laptop kamu    │        │  Server (PC + GPU)   │
│                 │        │                      │
│  Ketik chat ──► │ HTTP   │  LM Studio :1234     │
│                 │ ──────►│  (LLM Brain)         │
│        ◄────────│        │                      │
│                 │        │                      │
│  Text response  │        │  F5-TTS :8001        │
│         │       │ HTTP   │  (Voice synth)       │
│         ▼       │ ──────►│                      │
│  TTS request    │        │                      │
│         ◄───────│ WAV    │                      │
│  Play speaker   │        └──────────────────────┘
└─────────────────┘
```

- **Laptop**: thin client. Ngirim text ke server, terima text + audio.
- **Server (GPU)**: jalanin LM Studio (LLM) + F5-TTS service. Bisa
  satu mesin sama laptop kalau punya GPU bagus.

---

## Yang dibutuhkan

### Server side (mesin dengan GPU)
- **GPU**: NVIDIA dengan CUDA support. Minimal 8GB VRAM. Recommended 16GB.
  - Untuk RTX 50-series (Blackwell, sm_120): wajib PyTorch CUDA 12.8 atau lebih baru.
- **OS**: Linux (tested di Ubuntu 26.04). Bisa juga WSL2 atau native Windows dengan adjustment.
- **Disk**: ~15GB free (LLM model + TTS model + Python deps).
- **LM Studio** (atau OpenAI-compatible API server lain) — untuk LLM.
- **Python 3.10-3.12** untuk F5-TTS env.
- **ffmpeg** terinstal di sistem (untuk audio processing).

### Client side (laptop)
- **Python 3.10+** dengan pip.
- **Audio player**: `paplay` (PipeWire/PulseAudio), `aplay` (ALSA), atau `ffplay`.
- Network akses ke server LLM + TTS.

---

## Setup

### Langkah 1 — Setup LLM (di server)

1. Install [LM Studio](https://lmstudio.ai/) (atau alternatif: Ollama, vLLM, dll).
2. Download model chat (saran: **Qwen3-14B**, **Qwen2.5-14B**, atau yang sejenis).
3. Load model di LM Studio, enable "Local server" di port 1234, bind `0.0.0.0`.
4. Verify: dari client kamu bisa hit `http://<SERVER_IP>:1234/v1/models`.

### Langkah 2 — Setup TTS (di server)

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup venv Python 3.12
mkdir -p ~/f5-tts-env && cd ~/f5-tts-env
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install PyTorch CUDA — sesuaikan CUDA version
# Untuk RTX 50 series (Blackwell): pakai cu128
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio --upgrade

# Install F5-TTS + dependencies
uv pip install f5-tts pykakasi fastapi 'uvicorn[standard]' soundfile

# Install ffmpeg (sudo apt install ffmpeg)
```

#### Siapin voice reference

F5-TTS perlu sample audio yang akan di-clone (~6-15 detik per file, mono, 24kHz WAV).

Suggestion: generate sample dari [Fish Audio](https://fish.audio/) playground (free tier),
download MP3-nya, convert ke WAV 24kHz mono:

```bash
ffmpeg -i sample.mp3 -ar 24000 -ac 1 sample.wav
```

Taruh sample-mu di `~/f5-tts-env/voices/`. Misal: `01_calm.wav`, `02_angry.wav`, dll.

#### (Optional) Pakai model Japanese-specialized

Default F5-TTS multilingual baca Japanese kurang akurat. Untuk Japanese roleplay
yang natural, download model JA-specialized:

```bash
hf download Jmica/F5TTS --include "JA_21999120/*" --local-dir ~/f5-models/
```

Server akan auto-detect dan load model JA kalau ada di `~/f5-models/JA_21999120/`.

#### Run TTS server

Copy `tts_server/server.py` dari repo ini ke `~/f5-tts-env/server.py`, terus:

```bash
cd ~/f5-tts-env
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8001
```

Verify: `curl http://<SERVER_IP>:8001/health` → mestinya keluar `{"status":"ok",...}`

> **Mapping voice di server.py**: di file ada dict `VOICES` yang mapping nama
> ke file + reference text. Sesuaikan nama file & teks dengan sample-mu sendiri.
> Reference text harus exact transcript dari audio sample.

### Langkah 3 — Setup Client (di laptop)

```bash
git clone https://github.com/ardianto29/Alya-AI.git
cd Alya-AI

python3 -m venv .venv
source .venv/bin/activate                # bash/zsh
# source .venv/bin/activate.fish         # fish shell

pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```bash
LM_STUDIO_URL=http://<SERVER_IP>:1234/v1
MODEL_NAME=<model-id-di-lm-studio>       # e.g. qwen/qwen3-14b

TTS_URL=http://<SERVER_IP>:8001
TTS_VOICE=calm                            # nama voice di server.py VOICES dict
TTS_ENABLED=true
```

---

## Run

### CLI

```bash
source .venv/bin/activate
python -m src.chat
```

Chat normal, audio Alya bakal auto-play setelah tiap response.

Commands:
- `/reset` — clear conversation history
- `/voices` — list voice yang tersedia
- `/voice <name>` — ganti voice (misal `/voice angry`, `/voice soft`)
- `/tts on|off` — nyalain/matiin voice output
- `/help`, `/quit`

### Web UI (text-only, gak ada voice)

```bash
uvicorn src.server:app --host 127.0.0.1 --port 8000
```

Buka browser → http://127.0.0.1:8000/

---

## Customize persona

Default `config/persona.txt` itu **template generic**. Kalau mau personalisasi
(nama panggilan, dynamic spesifik, dll), buat file **`config/persona.local.txt`**
— file ini di-gitignored otomatis.

Client loader auto-prefer `.local.txt` kalau ada, fallback ke `persona.txt`.

Contoh struktur:
```
config/
├── persona.txt         ← template (di-commit)
├── persona.local.txt   ← personalisasi kamu (NOT committed, ignored)
└── voice/              ← reference audio (NOT committed, ignored)
```

---

## Struktur project

```
Alya-AI/
├── .env.example        config template
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   └── persona.txt     system prompt persona Alya
├── src/
│   ├── chat.py         CLI chat loop dengan integrasi TTS
│   ├── llm_client.py   wrapper LM Studio / OpenAI-compatible
│   ├── server.py       FastAPI web server (text UI)
│   └── tts.py          F5-TTS HTTP client + audio playback
├── tts_server/
│   └── server.py       F5-TTS service yang jalan di GPU server
└── web/
    └── index.html      web UI (text-only)
```

---

## Tech stack

| Komponen | Pakai apa | Lokasi |
|---|---|---|
| LLM brain | LM Studio (OpenAI-compatible) | Server |
| Voice synth | F5-TTS (open source) | Server |
| Audio playback | paplay/aplay/ffplay | Client |
| Chat UI | Python + rich (CLI) atau FastAPI + vanilla web | Client |
| Voice samples | Bring your own (Fish Audio free tier optional) | Server |

---

## Privacy & legal notes

- **History percakapan in-memory di client only** — gak di-persist ke disk.
- **Inference server stateless** — terima prompt, kirim response, gak nyimpan.
  Tapi log LM Studio panel bisa visible saat runtime, matikan disk logging kalau
  perlu privacy ketat.
- **Voice reference audio**: pastikan kamu punya right untuk pakai. Voice
  cloning karakter komersial tanpa izin = legal gray area. Project ini cuma
  framework — penggunaan voice-nya tanggung jawab user.
- **F5-TTS license**: lihat upstream repo & model card di HuggingFace. Sebagian
  besar variant non-commercial only.
- **Network traffic plaintext HTTP** — kalau lewat untrusted network,
  pertimbangkan SSH tunnel: `ssh -L 1234:localhost:1234 -L 8001:localhost:8001 <server>`.

---

## Roadmap

- [x] Phase A: Text chatbot (CLI + web)
- [x] Phase D: TTS voice output
- [ ] Phase C: STT voice input (next)
- [ ] Phase E: Wake word detection
- [ ] Phase F: Tool calling (open app, search, dll)

---

## Acknowledgments

- Voice clone engine: [F5-TTS](https://github.com/SWivid/F5-TTS) by SWivid
- Japanese model fork: [Jmica/F5TTS](https://huggingface.co/Jmica/F5TTS)
- LLM runtime: [LM Studio](https://lmstudio.ai/)
- Character inspiration: Alisa Mikhailovna Kujou from *Tokidoki Bosotto Russia-go de Dereru Tonari no Alya-san* (Roshidere)
