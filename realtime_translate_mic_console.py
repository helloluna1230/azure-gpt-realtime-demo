"""
Azure realtime translation: microphone to translated audio stream.
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

import numpy as np
import sounddevice as sd
import websockets

ENV_FILE = Path(__file__).resolve().with_name(".env")
DEFAULT_MODEL = "gpt-realtime-translate"
MODEL_ENV_NAMES = (
    "AZURE_OPENAI_TRANSLATE_DEPLOYMENT",
    "AZURE_OPENAI_REALTIME_TRANSLATE_DEPLOYMENT",
    "AZURE_OPENAI_REALTIME_TRANSLATION_DEPLOYMENT",
    "AZURE_OPENAI_REALTIME_DEPLOYMENT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_MODEL",
)
AUDIO_DELTA_EVENT_TYPES = {"response.audio.delta", "session.output_audio.delta"}
OUTPUT_TEXT_DELTA_EVENT_TYPES = {
    "session.output_transcript.delta",
    "response.audio_transcript.delta",
    "response.output_text.delta",
    "response.text.delta",
}
OUTPUT_TEXT_DONE_EVENT_TYPES = {
    "session.output_transcript.done",
    "response.audio_transcript.done",
    "response.output_text.done",
    "response.text.done",
}


def load_env_file(path: Path, *, override: bool = True) -> None:
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


def env_first(*names: str, default: str = "") -> str:
    value, _source = env_first_with_source(*names, default=default)
    return value


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
    output_audio_deltas: int = 0
    events_received: int = 0
    first_output_audio_ms: float | None = None
    output_text_deltas: int = 0
    output_text_started: bool = False


def parse_args() -> argparse.Namespace:
    endpoint_default, endpoint_source = env_first_with_source("AZURE_OPENAI_ENDPOINT")
    api_key_default, api_key_source = env_first_with_source("AZURE_OPENAI_API_KEY")
    model_default, model_source = env_first_with_source(*MODEL_ENV_NAMES, default=DEFAULT_MODEL)

    parser = argparse.ArgumentParser(
        description="Stream microphone audio to Azure realtime translation endpoint"
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
        "--model",
        default=model_default,
        help=(
            "Model/deployment name or one of these env vars: "
            + ", ".join(MODEL_ENV_NAMES)
        ),
    )
    parser.add_argument(
        "--target-language",
        default="zh",
        help="Target language code, e.g. zh, es, fr, ja",
    )
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument(
        "--turn-detection",
        choices=("semantic_vad", "server_vad", "none"),
        default="none",
        help="Optional Realtime turn detection mode. The Azure translations endpoint may reject this.",
    )
    parser.add_argument(
        "--vad-eagerness",
        choices=("low", "medium", "high", "auto"),
        default="high",
        help="Only used with --turn-detection semantic_vad. high starts responses sooner.",
    )
    parser.add_argument(
        "--debug-events",
        action="store_true",
        help="Print realtime event types for latency/debugging.",
    )
    parser.add_argument(
        "--text-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print translated text/transcript events when the service sends them.",
    )
    parser.add_argument("--play-audio", action="store_true", help="Play translated audio")
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Run N seconds then stop. 0 means until Ctrl+C",
    )

    args = parser.parse_args()
    args.endpoint_source = "--endpoint" if cli_option_provided("--endpoint") else endpoint_source
    args.api_key_source = "--api-key" if cli_option_provided("--api-key") else api_key_source
    args.model_source = "--model" if cli_option_provided("--model") else model_source

    if not args.endpoint:
        parser.error("--endpoint required or set AZURE_OPENAI_ENDPOINT")
    if not args.api_key:
        parser.error("--api-key required or set AZURE_OPENAI_API_KEY")
    if args.api_key in {"你的key", "your_key", "YOUR_KEY", "<key>", "<api-key>"}:
        parser.error("API key is still placeholder")
    return args


def build_ws_url(endpoint: str, model: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/openai/v1"):
        raw = f"{base}/realtime/translations?model={quote(model)}"
    elif base.endswith("/openai"):
        raw = f"{base}/v1/realtime/translations?model={quote(model)}"
    else:
        raw = f"{base}/openai/v1/realtime/translations?model={quote(model)}"
    return raw.replace("https://", "wss://").replace("http://", "ws://")


def text_from_event(event: dict[str, object]) -> str:
    for key in ("delta", "transcript", "text"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return ""


async def send_session_update(
    ws: websockets.WebSocketClientProtocol,
    target_language: str,
    turn_detection: str,
    vad_eagerness: str,
) -> None:
    session: dict[str, object] = {
        "audio": {
            "output": {
                "language": target_language,
            }
        }
    }

    if turn_detection != "none":
        detection: dict[str, object] = {
            "type": turn_detection,
            "create_response": True,
            "interrupt_response": True,
        }
        if turn_detection == "semantic_vad":
            detection["eagerness"] = vad_eagerness
        elif turn_detection == "server_vad":
            detection["silence_duration_ms"] = 300
        session["turn_detection"] = detection

    event = {
        "type": "session.update",
        "session": session,
    }
    await ws.send(json.dumps(event))


async def mic_sender(
    ws: websockets.WebSocketClientProtocol,
    audio_queue: asyncio.Queue[bytes],
    stop_event: asyncio.Event,
    stats: Stats,
) -> None:
    while not stop_event.is_set():
        try:
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue

        await ws.send(
            json.dumps(
                {
                    "type": "session.input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )
        stats.input_chunks += 1


async def event_receiver(
    ws: websockets.WebSocketClientProtocol,
    stop_event: asyncio.Event,
    stats: Stats,
    output_stream: sd.RawOutputStream | None,
    start_time: float,
    debug_events: bool,
    text_output: bool,
) -> None:
    seen_event_types: set[str] = set()
    while not stop_event.is_set():
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        event = json.loads(raw)
        event_type = event.get("type", "")
        stats.events_received += 1
        if debug_events and event_type not in seen_event_types:
            seen_event_types.add(event_type)
            print(f"[EVENT] {event_type}")

        if event_type in AUDIO_DELTA_EVENT_TYPES:
            delta = event.get("delta", "")
            if not delta:
                continue
            stats.output_audio_deltas += 1
            if stats.first_output_audio_ms is None:
                stats.first_output_audio_ms = (asyncio.get_running_loop().time() - start_time) * 1000
                print(f"\n[FIRST AUDIO] {stats.first_output_audio_ms:.0f} ms")
            if output_stream is not None:
                try:
                    output_stream.write(base64.b64decode(delta))
                except Exception:
                    pass

        elif event_type in OUTPUT_TEXT_DELTA_EVENT_TYPES:
            delta = text_from_event(event)
            if not delta:
                continue
            stats.output_text_deltas += 1
            if text_output:
                if not stats.output_text_started:
                    stats.output_text_started = True
                    print("\n[TARGET TEXT] ", end="", flush=True)
                print(delta, end="", flush=True)

        elif event_type in OUTPUT_TEXT_DONE_EVENT_TYPES:
            text = text_from_event(event)
            if text_output:
                if text and not stats.output_text_started:
                    print(f"\n[TARGET TEXT] {text}", end="", flush=True)
                if stats.output_text_started or text:
                    print(flush=True)
            stats.output_text_started = False

        elif event_type == "error":
            print(f"\n[ERROR] {event.get('error', {}).get('message', 'Unknown error')}")


async def main() -> int:
    args = parse_args()
    ws_url = build_ws_url(args.endpoint, args.model)

    print("=== Azure Realtime Translation ===")
    print(f".env: {ENV_FILE} ({'loaded' if ENV_FILE.exists() else 'not found'})")
    print(f"endpoint: {args.endpoint} (from {args.endpoint_source})")
    print(f"api_key: {'set' if args.api_key else 'missing'} (from {args.api_key_source})")
    print(f"model: {args.model} (from {args.model_source})")
    if args.model == DEFAULT_MODEL:
        print(
            "[WARN] model/deployment is gpt-realtime-translate. "
            "If your Azure deployment name is different, set "
            "AZURE_OPENAI_TRANSLATE_DEPLOYMENT in .env or pass --model."
        )
    print(f"target_language: {args.target_language}")
    print(f"ws_url: {ws_url}")
    print(f"audio: pcm16/{args.sample_rate}Hz, chunk={args.chunk_ms}ms")
    print(f"turn_detection: {args.turn_detection}, eagerness={args.vad_eagerness}")
    print(f"text_output: {'on' if args.text_output else 'off'}")
    if args.duration > 0:
        print(f"duration: {args.duration}s")
    else:
        print("duration: until Ctrl+C")

    stop_event = asyncio.Event()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    stats = Stats()

    output_stream: sd.RawOutputStream | None = None
    if args.play_audio:
        output_stream = sd.RawOutputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=0,
        )
        output_stream.start()

    loop = asyncio.get_running_loop()

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            return
        pcm16 = np.clip(indata[:, 0], -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
        try:
            loop.call_soon_threadsafe(audio_queue.put_nowait, pcm16)
        except asyncio.QueueFull:
            pass

    try:
        async with websockets.connect(
            ws_url,
            additional_headers={"api-key": args.api_key},
            max_size=None,
        ) as ws:
            await send_session_update(
                ws,
                args.target_language,
                args.turn_detection,
                args.vad_eagerness,
            )

            chunk_frames = int(args.sample_rate * args.chunk_ms / 1000)
            with sd.InputStream(
                samplerate=args.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=chunk_frames,
                callback=audio_callback,
            ):
                start_time = loop.time()
                sender_task = asyncio.create_task(mic_sender(ws, audio_queue, stop_event, stats))
                receiver_task = asyncio.create_task(
                    event_receiver(
                        ws,
                        stop_event,
                        stats,
                        output_stream,
                        start_time,
                        args.debug_events,
                        args.text_output,
                    )
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
        return 1
    finally:
        if output_stream is not None:
            output_stream.stop()
            output_stream.close()

    print("\n=== Summary ===")
    print(f"Audio chunks sent: {stats.input_chunks}")
    print(f"Events received: {stats.events_received}")
    print(f"Audio deltas received: {stats.output_audio_deltas}")
    print(f"Output text deltas received: {stats.output_text_deltas}")
    if stats.first_output_audio_ms is not None:
        print(f"First audio latency: {stats.first_output_audio_ms:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
