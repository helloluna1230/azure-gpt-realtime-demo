"""
Azure OpenAI Realtime Whisper transcription over WebRTC: microphone to console text.

Whisper is source-language speech to source-language text only; no translation.

Flow (per useful.md):
  1) POST {endpoint}/openai/v1/realtime/client_secrets  (Header: api-key)
       body: { "session": { "type": "transcription",
                            "audio": { "input": { "transcription": { "model": "gpt-realtime-whisper" } } } } }
     -> ephemeral token "ek_..."
  2) Create RTCPeerConnection with mic audio track + "oai-events" data channel
  3) POST SDP offer to {endpoint}/openai/v1/realtime/calls
       Header: Authorization: Bearer {ephemeral}, Content-Type: application/sdp
     -> SDP answer
  4) setRemoteDescription, then send session.update over the data channel
     with optional language hint, prompt, and turn detection tweaks.
"""
from __future__ import annotations

import argparse
import asyncio
import fractions
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame

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
    if not text:
        return
    try:
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()
    except (AttributeError, OSError):
        print(text, end="", flush=True)


ENV_FILE = Path(__file__).resolve().with_name(".env")
DEFAULT_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"
TRANSCRIPTION_MODEL_ENV_NAMES = ("AZURE_OPENAI_WHISPER_DEPLOYMENT",)

# WebRTC Opus standard rate.
WEBRTC_SAMPLE_RATE = 48000
WEBRTC_FRAME_MS = 20
WEBRTC_FRAME_SAMPLES = WEBRTC_SAMPLE_RATE * WEBRTC_FRAME_MS // 1000  # 960

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
    events_received: int = 0
    transcript_deltas: int = 0
    completed_segments: int = 0
    failed_segments: int = 0


def parse_args() -> argparse.Namespace:
    endpoint_default, endpoint_source = env_first_with_source("AZURE_OPENAI_ENDPOINT")
    api_key_default, api_key_source = env_first_with_source("AZURE_OPENAI_API_KEY")
    model_default, model_source = env_first_with_source(
        *TRANSCRIPTION_MODEL_ENV_NAMES, default=DEFAULT_TRANSCRIPTION_MODEL
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
            "Stream microphone audio to Azure OpenAI Realtime Whisper over WebRTC "
            "and print same-language transcription. This script does not translate."
        )
    )
    parser.add_argument("--endpoint", default=endpoint_default)
    parser.add_argument("--api-key", default=api_key_default)
    parser.add_argument(
        "--transcription-model",
        default=model_default,
        help="Model placed at session.audio.input.transcription.model",
    )
    parser.add_argument(
        "--input-language",
        "--language",
        dest="language",
        default=language_default,
        help=(
            "Optional source-language hint for Whisper, e.g. en, zh, ja. "
            "Empty (default) means auto-detect. Setting the wrong language causes "
            "conversation.item.input_audio_transcription.failed."
        ),
    )
    parser.add_argument("--prompt", default=prompt_default)
    parser.add_argument(
        "--turn-detection",
        choices=("server_vad", "semantic_vad", "none"),
        default="server_vad",
    )
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-prefix-padding-ms", type=int, default=300)
    parser.add_argument("--vad-silence-duration-ms", type=int, default=500)
    parser.add_argument(
        "--vad-eagerness",
        choices=("low", "medium", "high", "auto"),
        default="auto",
    )
    parser.add_argument("--debug-events", action="store_true")
    parser.add_argument("--text-output", action=argparse.BooleanOptionalAction, default=True,
                        help="Print transcript deltas to stdout. Use --no-text-output to silence.")
    parser.add_argument("--duration", type=int, default=0,
                        help="Run N seconds then stop. 0 means until Ctrl+C")

    args = parser.parse_args()
    args.endpoint_source = "--endpoint" if cli_option_provided("--endpoint") else endpoint_source
    args.api_key_source = "--api-key" if cli_option_provided("--api-key") else api_key_source
    args.transcription_model_source = (
        "--transcription-model"
        if cli_option_provided("--transcription-model")
        else model_source
    )
    args.language_source = (
        "--input-language" if cli_option_provided("--input-language") else language_source
    )
    args.prompt_source = "--prompt" if cli_option_provided("--prompt") else prompt_source

    if not args.endpoint:
        parser.error("--endpoint required or set AZURE_OPENAI_ENDPOINT")
    if not args.api_key:
        parser.error("--api-key required or set AZURE_OPENAI_API_KEY")
    if args.api_key in {"你的key", "your_key", "YOUR_KEY", "<key>", "<api-key>"}:
        parser.error("API key is still placeholder")
    if "translate" in args.transcription_model.lower():
        parser.error(
            "This Whisper script is transcription-only. Use a -whisper model, not -translate."
        )
    return args


def normalize_base(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/openai/v1"):
        return base
    if base.endswith("/openai"):
        return base + "/v1"
    return base + "/openai/v1"


def build_initial_session(model: str) -> dict:
    """Minimal body for /realtime/client_secrets (extra fields can return 500)."""
    return {
        "type": "transcription",
        "audio": {"input": {"transcription": {"model": model}}},
    }


def build_session_update(args: argparse.Namespace) -> dict:
    transcription: dict = {"model": args.transcription_model}
    if args.language:
        transcription["language"] = args.language
    if args.prompt:
        transcription["prompt"] = args.prompt

    audio_input: dict = {"transcription": transcription}

    if args.turn_detection == "none":
        audio_input["turn_detection"] = None
    else:
        detection: dict = {"type": args.turn_detection}
        if args.turn_detection == "server_vad":
            detection.update(
                {
                    "threshold": args.vad_threshold,
                    "prefix_padding_ms": args.vad_prefix_padding_ms,
                    "silence_duration_ms": args.vad_silence_duration_ms,
                }
            )
        elif args.turn_detection == "semantic_vad":
            detection["eagerness"] = args.vad_eagerness
        audio_input["turn_detection"] = detection

    return {
        "type": "transcription",
        "audio": {"input": audio_input},
    }


async def get_ephemeral_token(base: str, api_key: str, session: dict) -> tuple[str, dict]:
    url = f"{base}/realtime/client_secrets"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            headers={"api-key": api_key, "Content-Type": "application/json"},
            json={"session": session},
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"client_secrets failed: HTTP {resp.status_code} {resp.text}")
    data = resp.json()
    token = data.get("value") or data.get("client_secret", {}).get("value")
    if not token:
        raise RuntimeError(f"client_secrets missing token: {data}")
    return token, data


async def exchange_sdp(base: str, ephemeral_token: str, offer_sdp: str) -> str:
    # Azure infers the model from the ephemeral token; do not send ?model=.
    url = f"{base}/realtime/calls"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {ephemeral_token}",
                "Content-Type": "application/sdp",
            },
            content=offer_sdp,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"calls SDP exchange failed: HTTP {resp.status_code} {resp.text}")
    return resp.text


class MicrophoneTrack(MediaStreamTrack):
    """Reads PCM16 mono mic at WEBRTC_SAMPLE_RATE and yields 20ms AudioFrames."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=50)
        self._loop = asyncio.get_running_loop()
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
        self._stream = sd.InputStream(
            samplerate=WEBRTC_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=WEBRTC_FRAME_SAMPLES,
            callback=self._on_audio,
        )
        self._stream.start()
        # Pacing baseline is set on the first recv() so we don't burst-drain the
        # queue if WebRTC starts pulling frames seconds after the mic opens.
        self._start_perf: float | None = None
        self._frames_emitted = 0

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            pass
        chunk = np.ascontiguousarray(indata[:, 0], dtype=np.int16)
        self._loop.call_soon_threadsafe(self._enqueue, chunk)

    def _enqueue(self, chunk: np.ndarray) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    async def recv(self) -> AudioFrame:
        if self._start_perf is None:
            self._start_perf = time.perf_counter()
        target = self._start_perf + (self._frames_emitted + 1) * (WEBRTC_FRAME_MS / 1000)
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        try:
            chunk = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            chunk = np.zeros(WEBRTC_FRAME_SAMPLES, dtype=np.int16)

        if chunk.shape[0] != WEBRTC_FRAME_SAMPLES:
            if chunk.shape[0] < WEBRTC_FRAME_SAMPLES:
                chunk = np.pad(chunk, (0, WEBRTC_FRAME_SAMPLES - chunk.shape[0]))
            else:
                chunk = chunk[:WEBRTC_FRAME_SAMPLES]

        frame = AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = self._time_base
        self._timestamp += WEBRTC_FRAME_SAMPLES
        self._frames_emitted += 1
        return frame

    def stop(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        super().stop()


async def main() -> int:
    args = parse_args()
    base = normalize_base(args.endpoint)
    initial_session = build_initial_session(args.transcription_model)
    update_session = build_session_update(args)

    print("=== Azure OpenAI Realtime Whisper Transcription (WebRTC) ===")
    print(f".env: {ENV_FILE} ({'loaded' if ENV_FILE.exists() else 'not found'})")
    print(f"endpoint: {args.endpoint} (from {args.endpoint_source})")
    print(f"api_key: {'set' if args.api_key else 'missing'} (from {args.api_key_source})")
    print(f"transcription_model: {args.transcription_model} (from {args.transcription_model_source})")
    print(f"input_language_hint: {args.language or 'auto'} (from {args.language_source}; not a target language)")
    print(f"prompt: {'set' if args.prompt else 'empty'} (from {args.prompt_source})")
    print(f"turn_detection: {args.turn_detection}")
    if args.duration > 0:
        print(f"duration: {args.duration}s")
    else:
        print("duration: until Ctrl+C")

    print("[1/3] requesting ephemeral token ...")
    try:
        token, secret_resp = await get_ephemeral_token(base, args.api_key, initial_session)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"  token={token[:12]}... expires_at={secret_resp.get('expires_at')}")

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    stats = Stats()
    stop_event = asyncio.Event()
    seen_event_types: set[str] = set()
    partial_by_item: dict[str, str] = {}

    data_channel = pc.createDataChannel("oai-events")

    @data_channel.on("open")
    def _on_dc_open() -> None:
        print("[data-channel] open, sending session.update")
        msg = {"type": "session.update", "session": update_session}
        try:
            data_channel.send(json.dumps(msg))
        except Exception as exc:
            print(f"[data-channel] send error: {exc}")

    @data_channel.on("message")
    def _on_dc_message(message: str | bytes) -> None:
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8")
            except Exception:
                return
        try:
            event = json.loads(message)
        except Exception:
            return
        stats.events_received += 1
        ev_type = event.get("type", "")
        if args.debug_events and ev_type not in seen_event_types:
            seen_event_types.add(ev_type)
            print(f"\n[EVENT] {ev_type}")

        if ev_type in TRANSCRIPT_DELTA_EVENT_TYPES:
            delta = event.get("delta", "")
            if not delta:
                return
            item_id = event.get("item_id") or event.get("itemId") or "current"
            partial_by_item[item_id] = partial_by_item.get(item_id, "") + delta
            stats.transcript_deltas += 1
            if args.text_output:
                _emit_text(delta)

        elif ev_type in TRANSCRIPT_COMPLETED_EVENT_TYPES:
            transcript = event.get("transcript", "")
            item_id = event.get("item_id") or event.get("itemId") or "current"
            partial = partial_by_item.pop(item_id, "")
            stats.completed_segments += 1
            if args.text_output:
                if partial:
                    _emit_text("\n")
                    if transcript and transcript.strip() != partial.strip():
                        _emit_text(f"[FINAL] {transcript}\n")
                elif transcript:
                    _emit_text(f"[FINAL] {transcript}\n")

        elif ev_type in TRANSCRIPT_FAILED_EVENT_TYPES:
            stats.failed_segments += 1
            err = event.get("error", {})
            print(f"\n[TRANSCRIPTION ERROR] {err.get('message', err)}")

        elif ev_type == "error":
            err = event.get("error", {})
            print(f"\n[ERROR] {err.get('message', err)}")

        elif ev_type.endswith(".failed"):
            print(f"\n[{ev_type}] {json.dumps(event, ensure_ascii=False)}")

    @pc.on("track")
    def _on_track(track: MediaStreamTrack) -> None:
        # Whisper doesn't return audio, but drain just in case.
        async def _drain() -> None:
            try:
                while True:
                    await track.recv()
            except MediaStreamError:
                return
        asyncio.create_task(_drain())

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        print(f"[pc] connectionState={pc.connectionState}")
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            stop_event.set()

    mic_track = MicrophoneTrack()
    pc.addTransceiver(mic_track, direction="sendrecv")

    print("[2/3] creating SDP offer ...")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    if pc.iceGatheringState != "complete":
        gather_done = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def _on_gather() -> None:
            if pc.iceGatheringState == "complete":
                gather_done.set()

        try:
            await asyncio.wait_for(gather_done.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            print(f"[warn] ICE gathering still {pc.iceGatheringState}, sending offer anyway")

    print("[3/3] exchanging SDP with /realtime/calls ...")
    if args.debug_events:
        print("---- SDP OFFER ----")
        print(pc.localDescription.sdp)
        print("---- END SDP ----")
    try:
        answer_sdp = await exchange_sdp(base, token, pc.localDescription.sdp)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        await pc.close()
        mic_track.stop()
        return 1

    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
    print("[ok] WebRTC connected. Speak into the mic. Ctrl+C to stop.\n")

    try:
        if args.duration > 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=args.duration)
            except asyncio.TimeoutError:
                pass
        else:
            await stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        mic_track.stop()
        try:
            await pc.close()
        except Exception:
            pass

    print("\n=== Summary ===")
    print(f"Events received: {stats.events_received}")
    print(f"Transcript deltas: {stats.transcript_deltas}")
    print(f"Completed segments: {stats.completed_segments}")
    print(f"Failed segments: {stats.failed_segments}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(0)
