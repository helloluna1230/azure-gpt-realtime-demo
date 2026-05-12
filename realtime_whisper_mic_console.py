"""
Azure OpenAI Realtime Whisper transcription: microphone to console text.

Whisper transcription is source-language speech to source-language text only.
It does not translate and this script intentionally has no target-language option.

This follows the working transcription session shape captured in useful.md:
connect to a Realtime WebSocket endpoint, then configure the session as
`type: "transcription"` with the model under
`session.audio.input.transcription.model`.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
import urllib.request

import sounddevice as sd
import websockets

# Force UTF-8 on Windows consoles so CJK transcripts don't render as ??.
if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def _emit_text(text: str) -> None:
    """Write text as raw UTF-8 bytes, avoiding console text-codec mid-stream glitches."""
    if not text:
        return
    try:
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()
    except (AttributeError, OSError):
        print(text, end="", flush=True)

ENV_FILE = Path(__file__).resolve().with_name(".env")
DEFAULT_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"
DEFAULT_WS_PATH = "realtime"
DEFAULT_INTENT = "transcription"  # Azure requires intent=transcription on /realtime querystring
TRANSCRIPTION_MODEL_ENV_NAMES = (
    "AZURE_OPENAI_WHISPER_DEPLOYMENT"
)
TRANSCRIPT_DELTA_EVENT_TYPES = {
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.audio_transcription.delta",
}
TRANSCRIPT_COMPLETED_EVENT_TYPES = {
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.audio_transcription.completed",
}
TRANSCRIPT_FAILED_EVENT_TYPES = {
    "conversation.item.input_audio_transcription.failed",
    "conversation.item.audio_transcription.failed",
}


def load_env_file(path: Path, *, override: bool = True) -> None:
    """Load .env file. override=True so shell-inherited stale vars don't mask config."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or (not override and key in os.environ):
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()

        os.environ[key] = value


load_env_file(ENV_FILE)


def env_first_with_source(*names: str, default: str = "") -> tuple[str, str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value, name
    if default:
        return default, "default"
    return "", "missing"


def cli_option_provided(option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])


@dataclass
class Stats:
    input_chunks: int = 0
    events_received: int = 0
    transcript_deltas: int = 0
    completed_segments: int = 0
    failed_segments: int = 0
    dropped_audio_chunks: int = 0


def parse_args() -> argparse.Namespace:
    endpoint_default, endpoint_source = env_first_with_source("AZURE_OPENAI_ENDPOINT")
    api_key_default, api_key_source = env_first_with_source("AZURE_OPENAI_API_KEY")
    transcription_model_default, transcription_model_source = env_first_with_source(
        *TRANSCRIPTION_MODEL_ENV_NAMES,
        default=DEFAULT_TRANSCRIPTION_MODEL,
    )
    language_default, language_source = env_first_with_source(
        "AZURE_OPENAI_TRANSCRIPTION_LANGUAGE",
        "AZURE_OPENAI_WHISPER_LANGUAGE",
    )
    prompt_default, prompt_source = env_first_with_source(
        "AZURE_OPENAI_TRANSCRIPTION_PROMPT",
        "AZURE_OPENAI_WHISPER_PROMPT",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Stream microphone audio to Azure OpenAI Realtime Whisper and print same-language "
            "transcription text. This script does not translate."
        )
    )
    parser.add_argument(
        "--endpoint",
        default=endpoint_default,
        help="Azure OpenAI endpoint or AZURE_OPENAI_ENDPOINT env var",
    )
    parser.add_argument(
        "--api-key",
        default=api_key_default,
        help="Azure OpenAI API key or AZURE_OPENAI_API_KEY env var",
    )
    parser.add_argument(
        "--transcription-model",
        default=transcription_model_default,
        help=(
            "Realtime transcription model placed at session.audio.input.transcription.model. Env vars: "
            + ", ".join(TRANSCRIPTION_MODEL_ENV_NAMES)
        ),
    )
    parser.add_argument(
        "--ws-path",
        default=DEFAULT_WS_PATH,
        help="WebSocket path under /openai/v1, e.g. realtime or realtime/transcriptions.",
    )
    parser.add_argument(
        "--url-model",
        default="",
        help="Optional model= query parameter. Leave empty when --intent is set (Azure expects intent=transcription).",
    )
    parser.add_argument(
        "--intent",
        default=DEFAULT_INTENT,
        help="WebSocket intent query parameter. Azure requires 'transcription' to open a Whisper session on /realtime. Use empty string to omit.",
    )
    parser.add_argument(
        "--auth-mode",
        choices=("api-key", "client-secret"),
        default="api-key",
        help="Authenticate WebSocket with API key header, or first create a realtime client_secret and use Bearer auth.",
    )
    parser.add_argument(
        "--input-language",
        "--language",
        dest="language",
        default=language_default,
        help=(
            "Optional input speech language hint, e.g. en, zh, ja. "
            "This is not a target language and does not translate. Empty means auto-detect."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=prompt_default,
        help="Optional transcription prompt/context for Whisper.",
    )
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument(
        "--audio-event-type",
        choices=("input_audio_buffer.append", "session.input_audio_buffer.append"),
        default="input_audio_buffer.append",
        help="Audio append event type. Use session.input_audio_buffer.append only for legacy endpoints.",
    )
    parser.add_argument(
        "--turn-detection",
        choices=("server_vad", "semantic_vad", "none"),
        default="server_vad",
        help="Realtime turn detection. server_vad is recommended for live mic transcription.",
    )
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-prefix-padding-ms", type=int, default=300)
    parser.add_argument("--vad-silence-duration-ms", type=int, default=500)
    parser.add_argument(
        "--vad-eagerness",
        choices=("low", "medium", "high", "auto"),
        default="auto",
        help="Only used with semantic_vad.",
    )
    parser.add_argument(
        "--debug-events",
        action="store_true",
        help="Print unique realtime event types as they arrive.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Run N seconds then stop. 0 means until Ctrl+C",
    )

    args = parser.parse_args()
    args.endpoint_source = "--endpoint" if cli_option_provided("--endpoint") else endpoint_source
    args.api_key_source = "--api-key" if cli_option_provided("--api-key") else api_key_source
    args.transcription_model_source = (
        "--transcription-model" if cli_option_provided("--transcription-model") else transcription_model_source
    )
    if cli_option_provided("--input-language"):
        args.language_source = "--input-language"
    else:
        args.language_source = language_source
    args.prompt_source = "--prompt" if cli_option_provided("--prompt") else prompt_source

    if not args.endpoint:
        parser.error("--endpoint required or set AZURE_OPENAI_ENDPOINT")
    if not args.api_key:
        parser.error("--api-key required or set AZURE_OPENAI_API_KEY")
    if args.api_key in {"你的key", "your_key", "YOUR_KEY", "<key>", "<api-key>"}:
        parser.error("API key is still placeholder")
    if "translate" in args.transcription_model.lower():
        parser.error(
            "This Whisper script is transcription-only. Do not use a translate model here; "
            "use realtime_translate_mic_console.py for translation."
        )
    if "translations" in args.ws_path.lower() or "translate" in args.url_model.lower():
        parser.error(
            "This Whisper script is transcription-only. Do not use the realtime translations endpoint "
            "or a translate URL model here."
        )
    return args


def build_ws_url(endpoint: str, ws_path: str, url_model: str = "", intent: str = "") -> str:
    base = endpoint.rstrip("/")
    path = ws_path.strip("/")
    if base.endswith("/openai/v1"):
        raw = f"{base}/{path}"
    elif base.endswith("/openai"):
        raw = f"{base}/v1/{path}"
    else:
        raw = f"{base}/openai/v1/{path}"
    if url_model:
        separator = "&" if "?" in raw else "?"
        raw = f"{raw}{separator}model={quote(url_model)}"
    if intent:
        separator = "&" if "?" in raw else "?"
        raw = f"{raw}{separator}intent={quote(intent)}"
    return raw.replace("https://", "wss://").replace("http://", "ws://")


def build_auth_headers(endpoint: str, api_key_or_token: str, auth_mode: str) -> dict[str, str]:
    if auth_mode == "client-secret" or "api.openai.com" in endpoint.lower():
        return {
            "Authorization": f"Bearer {api_key_or_token}",
            "OpenAI-Beta": "realtime=v1",
        }
    return {"api-key": api_key_or_token}


def build_transcription_session(args: argparse.Namespace) -> dict[str, object]:
    transcription: dict[str, str] = {"model": args.transcription_model}
    if args.language:
        transcription["language"] = args.language
    if args.prompt:
        transcription["prompt"] = args.prompt

    input_audio: dict[str, object] = {
        "format": {
            "type": "audio/pcm",
            "rate": args.sample_rate,
        },
        "transcription": transcription,
    }

    if args.turn_detection == "none":
        input_audio["turn_detection"] = None
    else:
        turn_detection: dict[str, object] = {"type": args.turn_detection}
        if args.turn_detection == "server_vad":
            turn_detection.update(
                {
                    "threshold": args.vad_threshold,
                    "prefix_padding_ms": args.vad_prefix_padding_ms,
                    "silence_duration_ms": args.vad_silence_duration_ms,
                }
            )
        elif args.turn_detection == "semantic_vad":
            turn_detection["eagerness"] = args.vad_eagerness
        input_audio["turn_detection"] = turn_detection

    return {
        "type": "transcription",
        "audio": {
            "input": input_audio,
        },
    }


def create_client_secret(endpoint: str, api_key: str, session: dict[str, object]) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/openai/v1"):
        url = f"{base}/realtime/client_secrets"
    elif base.endswith("/openai"):
        url = f"{base}/v1/realtime/client_secrets"
    else:
        url = f"{base}/openai/v1/realtime/client_secrets"

    request = urllib.request.Request(
        url,
        data=json.dumps({"session": session}).encode("utf-8"),
        headers={"api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    value = payload.get("value") or payload.get("client_secret", {}).get("value")
    if not value:
        raise RuntimeError("client_secret response did not include a token value")
    return value


def print_connection_hint(exc: Exception, args: argparse.Namespace) -> None:
    message = str(exc)
    if "HTTP 400" not in message and "HTTP 404" not in message:
        return

    print(
        "[HINT] Azure Whisper WebSocket entry requires querystring intent=transcription. "
        "Use --intent transcription (default) and leave --url-model empty."
    )
    print(
        "[HINT] If you see 'invalid subscription key or wrong API endpoint', a stale "
        "AZURE_OPENAI_ENDPOINT in your shell is overriding the .env. This script forces "
        "override now, but verify the printed endpoint matches your .env."
    )


async def send_session_update(ws: websockets.WebSocketClientProtocol, args: argparse.Namespace) -> None:
    session = build_transcription_session(args)
    await ws.send(json.dumps({"type": "session.update", "session": session}))


async def mic_sender(
    ws: websockets.WebSocketClientProtocol,
    audio_queue: asyncio.Queue[bytes],
    stop_event: asyncio.Event,
    stats: Stats,
    audio_event_type: str,
) -> None:
    while not stop_event.is_set():
        try:
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue

        await ws.send(
            json.dumps(
                {
                    "type": audio_event_type,
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )
        stats.input_chunks += 1


async def event_receiver(
    ws: websockets.WebSocketClientProtocol,
    stop_event: asyncio.Event,
    stats: Stats,
    debug_events: bool,
) -> None:
    seen_event_types: set[str] = set()
    partial_by_item: dict[str, str] = {}

    while not stop_event.is_set():
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except websockets.ConnectionClosed as exc:
            print(f"\n[CONNECTION CLOSED] {exc}")
            stop_event.set()
            break

        event = json.loads(raw)
        event_type = event.get("type", "")
        stats.events_received += 1

        if debug_events and event_type not in seen_event_types:
            seen_event_types.add(event_type)
            print(f"\n[EVENT] {event_type}")

        if event_type in TRANSCRIPT_DELTA_EVENT_TYPES:
            delta = event.get("delta", "")
            if not delta:
                continue
            item_id = event.get("item_id") or event.get("itemId") or "current"
            partial_by_item[item_id] = partial_by_item.get(item_id, "") + delta
            stats.transcript_deltas += 1
            _emit_text(delta)

        elif event_type in TRANSCRIPT_COMPLETED_EVENT_TYPES:
            transcript = event.get("transcript", "")
            item_id = event.get("item_id") or event.get("itemId") or "current"
            partial = partial_by_item.pop(item_id, "")
            stats.completed_segments += 1

            if partial:
                _emit_text("\n")
                if transcript and transcript.strip() != partial.strip():
                    _emit_text(f"[FINAL] {transcript}\n")
            elif transcript:
                _emit_text(f"[FINAL] {transcript}\n")

        elif event_type in TRANSCRIPT_FAILED_EVENT_TYPES:
            stats.failed_segments += 1
            error = event.get("error", {})
            message = error.get("message") or event.get("message") or "Unknown transcription error"
            print(f"\n[TRANSCRIPTION ERROR] {message}")

        elif event_type == "error":
            error = event.get("error", {})
            message = error.get("message") or "Unknown realtime error"
            print(f"\n[ERROR] {message}")
            stop_event.set()


async def main() -> int:
    args = parse_args()
    session = build_transcription_session(args)
    ws_url = build_ws_url(args.endpoint, args.ws_path, args.url_model, args.intent)

    print("=== Azure OpenAI Realtime Whisper Transcription (WebSocket) ===")
    print(f".env: {ENV_FILE} ({'loaded' if ENV_FILE.exists() else 'not found'})")
    print(f"endpoint: {args.endpoint} (from {args.endpoint_source})")
    print(f"api_key: {'set' if args.api_key else 'missing'} (from {args.api_key_source})")
    print(f"auth_mode: {args.auth_mode}")
    print(f"ws_path: {args.ws_path}")
    print(f"url_model: {args.url_model or '(none; model is in session audio input transcription)'}")
    print(f"intent: {args.intent or '(none)'}")
    print(f"input transcription model: {args.transcription_model} (from {args.transcription_model_source})")
    print(f"input_language_hint: {args.language or 'auto'} (from {args.language_source}; not a target language)")
    print(f"prompt: {'set' if args.prompt else 'empty'} (from {args.prompt_source})")
    print(f"ws_url: {ws_url}")
    print(f"audio: pcm16/{args.sample_rate}Hz, chunk={args.chunk_ms}ms")
    print(f"audio_event_type: {args.audio_event_type}")
    print(f"turn_detection: {args.turn_detection}")
    if args.turn_detection == "none":
        print(
            "[WARN] --turn-detection none requires manual input_audio_buffer.commit; "
            "this script is optimized for server_vad."
        )
    if args.duration > 0:
        print(f"duration: {args.duration}s")
    else:
        print("duration: until Ctrl+C")
    print("\nSpeak into your microphone. Same-language transcription will appear below:\n")

    stop_event = asyncio.Event()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    stats = Stats()
    loop = asyncio.get_running_loop()

    def enqueue_audio(chunk: bytes) -> None:
        if audio_queue.full():
            stats.dropped_audio_chunks += 1
            return
        audio_queue.put_nowait(chunk)

    def audio_callback(indata, frames: int, time_info, status) -> None:
        if status:
            if args.debug_events:
                print(f"\n[AUDIO STATUS] {status}")
            return
        loop.call_soon_threadsafe(enqueue_audio, bytes(indata))

    try:
        auth_value = args.api_key
        if args.auth_mode == "client-secret":
            print("creating client_secret: yes")
            auth_value = create_client_secret(args.endpoint, args.api_key, session)

        async with websockets.connect(
            ws_url,
            additional_headers=build_auth_headers(args.endpoint, auth_value, args.auth_mode),
            max_size=None,
        ) as ws:
            await ws.send(json.dumps({"type": "session.update", "session": session}))

            chunk_frames = int(args.sample_rate * args.chunk_ms / 1000)
            with sd.RawInputStream(
                samplerate=args.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=chunk_frames,
                callback=audio_callback,
            ):
                sender_task = asyncio.create_task(
                    mic_sender(ws, audio_queue, stop_event, stats, args.audio_event_type)
                )
                receiver_task = asyncio.create_task(
                    event_receiver(ws, stop_event, stats, args.debug_events)
                )

                try:
                    if args.duration > 0:
                        await asyncio.wait_for(stop_event.wait(), timeout=args.duration)
                    else:
                        await stop_event.wait()
                except asyncio.TimeoutError:
                    pass
                except KeyboardInterrupt:
                    pass
                finally:
                    stop_event.set()

                sender_task.cancel()
                receiver_task.cancel()
                await asyncio.gather(sender_task, receiver_task, return_exceptions=True)

    except Exception as exc:
        print(f"[CONNECTION ERROR] {exc}")
        print_connection_hint(exc, args)
        return 1

    print("\n=== Summary ===")
    print(f"Audio chunks sent: {stats.input_chunks}")
    print(f"Audio chunks dropped: {stats.dropped_audio_chunks}")
    print(f"Events received: {stats.events_received}")
    print(f"Transcript deltas: {stats.transcript_deltas}")
    print(f"Completed segments: {stats.completed_segments}")
    print(f"Failed segments: {stats.failed_segments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
