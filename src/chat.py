from __future__ import annotations

import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.llm_client import AlyaClient
from src.tts import TTSClient

console = Console()

HELP = """
[bold]Commands:[/bold]
  /reset           — clear conversation history
  /voice <name>    — set TTS voice (calm, angry, soft, embarrassed, ...)
  /voices          — list available voices
  /tts on|off      — toggle TTS
  /quit            — exit
  /help            — show this
"""


def banner() -> None:
    console.print(
        Panel.fit(
            "[bold magenta]Alya[/bold magenta] — personal chatbot + voice\n"
            "[dim]LLM: LM Studio (PC), Voice: F5-TTS (PC), History: in-memory (laptop)[/dim]",
            border_style="magenta",
        )
    )


def main() -> int:
    load_dotenv()
    try:
        alya = AlyaClient()
    except KeyError as e:
        console.print(f"[red]Missing env var: {e}. Cek .env file.[/red]")
        return 1

    tts = TTSClient()

    banner()
    console.print(HELP)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            return 0

        if not user_input:
            continue
        if user_input in {"/quit", "/exit"}:
            console.print("[dim]bye.[/dim]")
            return 0
        if user_input == "/reset":
            alya.reset()
            console.print("[yellow]history cleared.[/yellow]")
            continue
        if user_input == "/help":
            console.print(HELP)
            continue
        if user_input == "/voices":
            try:
                voices = tts.list_voices()
                console.print(f"[cyan]voices:[/cyan] {', '.join(voices)}")
            except Exception as e:
                console.print(f"[red]tts error: {e}[/red]")
            continue
        if user_input.startswith("/voice "):
            new_voice = user_input.removeprefix("/voice ").strip()
            tts.default_voice = new_voice
            console.print(f"[cyan]voice set to:[/cyan] {new_voice}")
            continue
        if user_input.startswith("/tts "):
            mode = user_input.removeprefix("/tts ").strip().lower()
            tts.enabled = mode in {"on", "true", "1", "yes"}
            console.print(f"[cyan]tts:[/cyan] {'on' if tts.enabled else 'off'}")
            continue

        console.print("[bold magenta]alya[/bold magenta] ", end="")
        buf: list[str] = []
        try:
            for piece in alya.stream_reply(user_input):
                console.print(piece, end="", soft_wrap=True, highlight=False)
                buf.append(piece)
            console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted.[/yellow]")
            continue
        except Exception as e:
            console.print(f"\n[red]error: {e}[/red]")
            continue

        reply = "".join(buf).strip()
        if reply and tts.enabled:
            tts.say(reply)


if __name__ == "__main__":
    sys.exit(main())
