"""
Azure OpenAI GPT Realtime 2 voice assistant demo: microphone to console/audio.

This is the standard voice-agent Realtime session, not the dedicated Whisper
transcription endpoint and not the realtime translation endpoint. It connects to
the GA WebSocket route:

	/openai/v1/realtime?model=<your gpt-realtime-2 deployment>

Then it streams PCM16 microphone audio with input_audio_buffer.append and listens
for assistant audio/text transcript deltas.
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

import sounddevice as sd
import websockets

# Force UTF-8 on Windows consoles so CJK text does not render as ?? mid-stream.
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
	"""Write text as raw UTF-8 bytes, avoiding Windows console codec glitches."""
	if not text:
		return
	try:
		sys.stdout.buffer.write(text.encode("utf-8"))
		sys.stdout.buffer.flush()
	except (AttributeError, OSError):
		print(text, end="", flush=True)


ENV_FILE = Path(__file__).resolve().with_name(".env")
DEFAULT_MODEL = "gpt-realtime-2"
DEFAULT_INSTRUCTIONS = (
	"你是一个简洁、友好的实时语音助手。默认用中文回答；如果用户用英文提问，"
	"可以用英文回答。回答尽量短，适合直接朗读。"
)
MODEL_ENV_NAMES = (
	"AZURE_OPENAI_REALTIME_2_DEPLOYMENT",
	"AZURE_OPENAI_GPT_REALTIME_2_DEPLOYMENT",
	"AZURE_OPENAI_REALTIME_DEPLOYMENT",
)
INPUT_TRANSCRIPTION_MODEL_ENV_NAMES = (
	"AZURE_OPENAI_INPUT_TRANSCRIPTION_DEPLOYMENT",
	"AZURE_OPENAI_REALTIME_INPUT_TRANSCRIPTION_DEPLOYMENT",
)
INSTRUCTIONS_ENV_NAMES = (
	"AZURE_OPENAI_REALTIME_2_INSTRUCTIONS",
	"AZURE_OPENAI_REALTIME_INSTRUCTIONS",
)

VOICE_CHOICES = (
	"alloy",
	"ash",
	"ballad",
	"coral",
	"echo",
	"sage",
	"shimmer",
	"verse",
	"marin",
	"cedar",
)

OUTPUT_AUDIO_DELTA_EVENT_TYPES = {
	"response.output_audio.delta",
	# Older/beta event name. Kept so the demo is tolerant when testing against
	# different realtime-compatible servers.
	"response.audio.delta",
}
OUTPUT_AUDIO_DONE_EVENT_TYPES = {
	"response.output_audio.done",
	"response.audio.done",
}
OUTPUT_AUDIO_TRANSCRIPT_DELTA_EVENT_TYPES = {
	"response.output_audio_transcript.delta",
	"response.audio_transcript.delta",
}
OUTPUT_AUDIO_TRANSCRIPT_DONE_EVENT_TYPES = {
	"response.output_audio_transcript.done",
	"response.audio_transcript.done",
}
OUTPUT_TEXT_DELTA_EVENT_TYPES = {
	"response.output_text.delta",
	"response.text.delta",
}
OUTPUT_TEXT_DONE_EVENT_TYPES = {
	"response.output_text.done",
	"response.text.done",
}
INPUT_TRANSCRIPT_DELTA_EVENT_TYPES = {
	"conversation.item.input_audio_transcription.delta",
	"conversation.item.audio_transcription.delta",
}
INPUT_TRANSCRIPT_COMPLETED_EVENT_TYPES = {
	"conversation.item.input_audio_transcription.completed",
	"conversation.item.audio_transcription.completed",
}
INPUT_TRANSCRIPT_FAILED_EVENT_TYPES = {
	"conversation.item.input_audio_transcription.failed",
	"conversation.item.audio_transcription.failed",
}


def load_env_file(path: Path, *, override: bool = True) -> None:
	"""Load a simple .env file. override=True keeps repo-local config authoritative."""
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


def text_from_event(event: dict[str, object]) -> str:
	for key in ("delta", "transcript", "text"):
		value = event.get(key)
		if isinstance(value, str):
			return value
	return ""


@dataclass
class Stats:
	input_chunks: int = 0
	dropped_audio_chunks: int = 0
	events_received: int = 0
	output_audio_deltas: int = 0
	assistant_text_deltas: int = 0
	user_transcript_deltas: int = 0
	completed_responses: int = 0
	errors: int = 0
	first_output_audio_ms: float | None = None
	assistant_text_started: bool = False
	user_text_started: bool = False


def _parse_output_modalities(parser: argparse.ArgumentParser, raw: str) -> list[str]:
	modalities = [part.strip().lower() for part in raw.split(",") if part.strip()]
	if not modalities:
		parser.error("--output-modalities must include audio and/or text")
	invalid = sorted(set(modalities) - {"audio", "text"})
	if invalid:
		parser.error(f"Unsupported --output-modalities value(s): {', '.join(invalid)}")
	# Preserve user order while removing duplicates.
	deduped: list[str] = []
	for modality in modalities:
		if modality not in deduped:
			deduped.append(modality)
	return deduped


def parse_args() -> argparse.Namespace:
	endpoint_default, endpoint_source = env_first_with_source("AZURE_OPENAI_ENDPOINT")
	api_key_default, api_key_source = env_first_with_source("AZURE_OPENAI_API_KEY")
	model_default, model_source = env_first_with_source(*MODEL_ENV_NAMES, default=DEFAULT_MODEL)
	instructions_default, instructions_source = env_first_with_source(
		*INSTRUCTIONS_ENV_NAMES,
		default=DEFAULT_INSTRUCTIONS,
	)
	voice_default, voice_source = env_first_with_source(
		"AZURE_OPENAI_REALTIME_VOICE",
		default="marin",
	)
	output_modalities_default, output_modalities_source = env_first_with_source(
		"AZURE_OPENAI_REALTIME_OUTPUT_MODALITIES",
		default="audio",
	)
	input_transcription_model_default, input_transcription_model_source = env_first_with_source(
		*INPUT_TRANSCRIPTION_MODEL_ENV_NAMES,
	)
	input_language_default, input_language_source = env_first_with_source(
		"AZURE_OPENAI_REALTIME_INPUT_LANGUAGE",
		"AZURE_OPENAI_TRANSCRIPTION_LANGUAGE",
		"AZURE_OPENAI_WHISPER_LANGUAGE",
	)
	input_transcription_prompt_default, input_transcription_prompt_source = env_first_with_source(
		"AZURE_OPENAI_REALTIME_INPUT_TRANSCRIPTION_PROMPT",
		"AZURE_OPENAI_TRANSCRIPTION_PROMPT",
		"AZURE_OPENAI_WHISPER_PROMPT",
	)
	reasoning_effort_default, reasoning_effort_source = env_first_with_source(
		"AZURE_OPENAI_REALTIME_REASONING_EFFORT",
		default="low",
	)

	parser = argparse.ArgumentParser(
		description=(
			"Stream microphone audio to Azure OpenAI GPT Realtime 2 and print/play "
			"assistant responses."
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
		"--model",
		default=model_default,
		help=(
			"GPT Realtime deployment name used in /realtime?model=. Env vars: "
			+ ", ".join(MODEL_ENV_NAMES)
		),
	)
	parser.add_argument(
		"--instructions",
		default=instructions_default,
		help="System/developer-style instructions for the voice assistant.",
	)
	parser.add_argument(
		"--voice",
		choices=VOICE_CHOICES,
		default=voice_default if voice_default in VOICE_CHOICES else "marin",
		help="Assistant output voice. marin/cedar are good defaults for realtime voice.",
	)
	parser.add_argument(
		"--output-modalities",
		default=output_modalities_default,
		help="Comma-separated modalities for assistant output: audio, text, or audio,text.",
	)
	parser.add_argument(
		"--reasoning-effort",
		choices=("low", "medium", "high", "none"),
		default=(
			reasoning_effort_default
			if reasoning_effort_default in {"low", "medium", "high", "none"}
			else "low"
		),
		help="Realtime 2 reasoning effort. Use none to omit the reasoning field.",
	)
	parser.add_argument(
		"--input-transcription-model",
		default=input_transcription_model_default,
		help=(
			"Optional deployment name for user speech transcript events. Leave empty to avoid "
			"requiring a separate transcription deployment."
		),
	)
	parser.add_argument(
		"--input-language",
		"--language",
		dest="input_language",
		default=input_language_default,
		help="Optional input language hint for user speech transcription, e.g. zh or en.",
	)
	parser.add_argument(
		"--input-transcription-prompt",
		default=input_transcription_prompt_default,
		help="Optional prompt/context for input transcription if enabled.",
	)
	parser.add_argument("--sample-rate", type=int, default=24000)
	parser.add_argument("--chunk-ms", type=int, default=40)
	parser.add_argument(
		"--turn-detection",
		choices=("semantic_vad", "server_vad"),
		default="semantic_vad",
		help="Server-side VAD mode. semantic_vad works well for natural conversation.",
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
		"--text-output",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Print assistant transcript/text deltas when the service sends them.",
	)
	parser.add_argument(
		"--play-audio",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Play assistant PCM audio deltas as they arrive.",
	)
	parser.add_argument(
		"--auth-mode",
		choices=("auto", "api-key", "bearer"),
		default="auto",
		help="Azure uses api-key. OpenAI public endpoint uses bearer. auto picks from endpoint.",
	)
	parser.add_argument(
		"--safety-identifier",
		default="",
		help="Optional privacy-preserving end-user identifier header.",
	)
	parser.add_argument(
		"--debug-events",
		action="store_true",
		help="Print each unique realtime event type as it arrives.",
	)
	parser.add_argument(
		"--duration",
		type=int,
		default=0,
		help="Run N seconds then stop. 0 means until Ctrl+C",
	)

	args = parser.parse_args()
	args.output_modalities = _parse_output_modalities(parser, args.output_modalities)
	args.endpoint_source = "--endpoint" if cli_option_provided("--endpoint") else endpoint_source
	args.api_key_source = "--api-key" if cli_option_provided("--api-key") else api_key_source
	args.model_source = "--model" if cli_option_provided("--model") else model_source
	args.instructions_source = (
		"--instructions" if cli_option_provided("--instructions") else instructions_source
	)
	args.voice_source = "--voice" if cli_option_provided("--voice") else voice_source
	args.output_modalities_source = (
		"--output-modalities"
		if cli_option_provided("--output-modalities")
		else output_modalities_source
	)
	args.reasoning_effort_source = (
		"--reasoning-effort"
		if cli_option_provided("--reasoning-effort")
		else reasoning_effort_source
	)
	args.input_transcription_model_source = (
		"--input-transcription-model"
		if cli_option_provided("--input-transcription-model")
		else input_transcription_model_source
	)
	args.input_language_source = (
		"--input-language" if cli_option_provided("--input-language") else input_language_source
	)
	args.input_transcription_prompt_source = (
		"--input-transcription-prompt"
		if cli_option_provided("--input-transcription-prompt")
		else input_transcription_prompt_source
	)

	if not args.endpoint:
		parser.error("--endpoint required or set AZURE_OPENAI_ENDPOINT")
	if not args.api_key:
		parser.error("--api-key required or set AZURE_OPENAI_API_KEY")
	if args.api_key in {"你的key", "your_key", "YOUR_KEY", "<key>", "<api-key>"}:
		parser.error("API key is still placeholder")
	if not args.model:
		parser.error("--model required or set AZURE_OPENAI_REALTIME_2_DEPLOYMENT")
	lower_model = args.model.lower()
	if "whisper" in lower_model or "translate" in lower_model:
		parser.error(
			"This script is for GPT Realtime voice-agent deployments. Use a gpt-realtime-2 "
			"deployment, not a whisper/translate deployment."
		)
	if args.sample_rate <= 0:
		parser.error("--sample-rate must be positive")
	if args.chunk_ms <= 0:
		parser.error("--chunk-ms must be positive")
	return args


def build_ws_url(endpoint: str, model: str) -> str:
	base = endpoint.rstrip("/")
	if base.endswith("/openai/v1"):
		raw = f"{base}/realtime?model={quote(model)}"
	elif base.endswith("/openai"):
		raw = f"{base}/v1/realtime?model={quote(model)}"
	else:
		raw = f"{base}/openai/v1/realtime?model={quote(model)}"
	return raw.replace("https://", "wss://").replace("http://", "ws://")


def build_auth_headers(
	endpoint: str,
	api_key: str,
	auth_mode: str,
	safety_identifier: str = "",
) -> dict[str, str]:
	use_bearer = auth_mode == "bearer" or (
		auth_mode == "auto" and "api.openai.com" in endpoint.lower()
	)
	headers = {"Authorization": f"Bearer {api_key}"} if use_bearer else {"api-key": api_key}
	if safety_identifier:
		headers["OpenAI-Safety-Identifier"] = safety_identifier
	return headers


def build_turn_detection(args: argparse.Namespace) -> dict[str, object]:
	detection: dict[str, object] = {
		"type": args.turn_detection,
		"create_response": True,
		"interrupt_response": True,
	}
	if args.turn_detection == "semantic_vad":
		detection["eagerness"] = args.vad_eagerness
	elif args.turn_detection == "server_vad":
		detection.update(
			{
				"threshold": args.vad_threshold,
				"prefix_padding_ms": args.vad_prefix_padding_ms,
				"silence_duration_ms": args.vad_silence_duration_ms,
			}
		)
	return detection


def build_session(args: argparse.Namespace) -> dict[str, object]:
	audio_input: dict[str, object] = {
		"format": {
			"type": "audio/pcm",
			"rate": args.sample_rate,
		},
		"turn_detection": build_turn_detection(args),
	}

	input_transcription_model = args.input_transcription_model.strip()
	if input_transcription_model and input_transcription_model.lower() not in {"none", "off", "false"}:
		transcription: dict[str, str] = {"model": input_transcription_model}
		if args.input_language:
			transcription["language"] = args.input_language
		if args.input_transcription_prompt:
			transcription["prompt"] = args.input_transcription_prompt
		audio_input["transcription"] = transcription

	session: dict[str, object] = {
		"type": "realtime",
		"model": args.model,
		"instructions": args.instructions,
		"output_modalities": args.output_modalities,
		"audio": {
			"input": audio_input,
			"output": {
				"format": {
					"type": "audio/pcm",
					"rate": args.sample_rate,
				},
				"voice": args.voice,
			},
		},
	}

	if args.reasoning_effort != "none":
		session["reasoning"] = {"effort": args.reasoning_effort}

	return session


def websocket_connect(ws_url: str, headers: dict[str, str]):
	"""Return a websockets connection object across websockets 12-15 naming."""
	try:
		return websockets.connect(ws_url, additional_headers=headers, max_size=None)
	except TypeError:
		return websockets.connect(ws_url, extra_headers=headers, max_size=None)


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

		try:
			await ws.send(
				json.dumps(
					{
						"type": "input_audio_buffer.append",
						"audio": base64.b64encode(chunk).decode("ascii"),
					}
				)
			)
			stats.input_chunks += 1
		except websockets.ConnectionClosed:
			stop_event.set()
			break


def _start_prefixed_line(stats: Stats, field_name: str, prefix: str) -> None:
	if not getattr(stats, field_name):
		_emit_text(prefix)
		setattr(stats, field_name, True)


def _finish_prefixed_line(stats: Stats, field_name: str) -> None:
	if getattr(stats, field_name):
		_emit_text("\n")
		setattr(stats, field_name, False)


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
	partial_user_by_item: dict[str, str] = {}

	while not stop_event.is_set():
		try:
			raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
		except asyncio.TimeoutError:
			continue
		except websockets.ConnectionClosed as exc:
			print(f"\n[CONNECTION CLOSED] {exc}")
			stop_event.set()
			break

		try:
			event = json.loads(raw)
		except json.JSONDecodeError:
			if debug_events:
				print(f"\n[NON-JSON EVENT] {raw!r}")
			continue

		event_type = event.get("type", "")
		stats.events_received += 1

		if debug_events and event_type not in seen_event_types:
			seen_event_types.add(event_type)
			print(f"\n[EVENT] {event_type}")

		if event_type in OUTPUT_AUDIO_DELTA_EVENT_TYPES:
			delta = event.get("delta", "")
			if not isinstance(delta, str) or not delta:
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

		elif event_type in OUTPUT_AUDIO_TRANSCRIPT_DELTA_EVENT_TYPES or event_type in OUTPUT_TEXT_DELTA_EVENT_TYPES:
			delta = text_from_event(event)
			if not delta:
				continue
			stats.assistant_text_deltas += 1
			if text_output:
				_start_prefixed_line(stats, "assistant_text_started", "\n[ASSISTANT] ")
				_emit_text(delta)

		elif event_type in OUTPUT_AUDIO_TRANSCRIPT_DONE_EVENT_TYPES or event_type in OUTPUT_TEXT_DONE_EVENT_TYPES:
			if text_output:
				text = text_from_event(event)
				if text and not stats.assistant_text_started:
					_emit_text(f"\n[ASSISTANT] {text}")
					stats.assistant_text_started = True
				_finish_prefixed_line(stats, "assistant_text_started")

		elif event_type in OUTPUT_AUDIO_DONE_EVENT_TYPES:
			# The audio bytes arrive in delta events. The done event is useful
			# only as a lifecycle marker here.
			pass

		elif event_type in INPUT_TRANSCRIPT_DELTA_EVENT_TYPES:
			delta = text_from_event(event)
			if not delta:
				continue
			stats.user_transcript_deltas += 1
			item_id = str(event.get("item_id") or event.get("itemId") or "current")
			partial_user_by_item[item_id] = partial_user_by_item.get(item_id, "") + delta
			if text_output:
				_start_prefixed_line(stats, "user_text_started", "\n[YOU] ")
				_emit_text(delta)

		elif event_type in INPUT_TRANSCRIPT_COMPLETED_EVENT_TYPES:
			item_id = str(event.get("item_id") or event.get("itemId") or "current")
			partial = partial_user_by_item.pop(item_id, "")
			transcript = text_from_event(event)
			if text_output:
				if not partial and transcript:
					_emit_text(f"\n[YOU] {transcript}")
					stats.user_text_started = True
				_finish_prefixed_line(stats, "user_text_started")

		elif event_type in INPUT_TRANSCRIPT_FAILED_EVENT_TYPES:
			stats.errors += 1
			error = event.get("error", {})
			if isinstance(error, dict):
				message = error.get("message") or error
			else:
				message = error
			print(f"\n[INPUT TRANSCRIPTION ERROR] {message}")

		elif event_type == "input_audio_buffer.speech_started":
			if debug_events:
				print("\n[VAD] speech started")

		elif event_type == "input_audio_buffer.speech_stopped":
			if debug_events:
				print("\n[VAD] speech stopped")

		elif event_type == "response.done":
			stats.completed_responses += 1
			response = event.get("response", {})
			if isinstance(response, dict):
				status = response.get("status")
				if status and status not in {"completed", "incomplete"}:
					print(f"\n[RESPONSE] status={status} details={response.get('status_details')}")

		elif event_type == "error":
			stats.errors += 1
			error = event.get("error", {})
			if isinstance(error, dict):
				message = error.get("message") or error
			else:
				message = error
			print(f"\n[ERROR] {message}")

		elif isinstance(event_type, str) and event_type.endswith(".failed"):
			stats.errors += 1
			print(f"\n[{event_type}] {json.dumps(event, ensure_ascii=False)}")


def print_connection_hint(exc: Exception) -> None:
	message = str(exc)
	if "HTTP 400" not in message and "HTTP 401" not in message and "HTTP 404" not in message:
		return
	print(
		"[HINT] GPT Realtime 2 GA WebSocket should look like "
		"wss://<resource>.openai.azure.com/openai/v1/realtime?model=<deployment>."
	)
	print(
		"[HINT] For Azure, use the deployment name in --model / "
		"AZURE_OPENAI_REALTIME_2_DEPLOYMENT and authenticate with the api-key header."
	)


async def main() -> int:
	args = parse_args()
	session = build_session(args)
	ws_url = build_ws_url(args.endpoint, args.model)
	headers = build_auth_headers(args.endpoint, args.api_key, args.auth_mode, args.safety_identifier)

	print("=== Azure OpenAI GPT Realtime 2 Voice Demo (WebSocket) ===")
	print(f".env: {ENV_FILE} ({'loaded' if ENV_FILE.exists() else 'not found'})")
	print(f"endpoint: {args.endpoint} (from {args.endpoint_source})")
	print(f"api_key: {'set' if args.api_key else 'missing'} (from {args.api_key_source})")
	print(f"auth_mode: {args.auth_mode}")
	print(f"model/deployment: {args.model} (from {args.model_source})")
	print(f"instructions: {'set' if args.instructions else 'empty'} (from {args.instructions_source})")
	print(f"voice: {args.voice} (from {args.voice_source})")
	print(
		"output_modalities: "
		f"{','.join(args.output_modalities)} (from {args.output_modalities_source})"
	)
	print(f"reasoning_effort: {args.reasoning_effort} (from {args.reasoning_effort_source})")
	print(
		"input_transcription_model: "
		f"{args.input_transcription_model or '(disabled)'} "
		f"(from {args.input_transcription_model_source})"
	)
	print(f"input_language_hint: {args.input_language or 'auto'} (from {args.input_language_source})")
	print(f"input_transcription_prompt: {'set' if args.input_transcription_prompt else 'empty'}")
	print(f"ws_url: {ws_url}")
	print(f"audio input: pcm16/{args.sample_rate}Hz, chunk={args.chunk_ms}ms")
	print(f"turn_detection: {args.turn_detection}")
	print(f"text_output: {'on' if args.text_output else 'off'}")
	print(f"play_audio: {'on' if args.play_audio else 'off'}")
	if args.play_audio and "audio" not in args.output_modalities:
		print("[WARN] --play-audio is on but output_modalities does not include audio.")
	if args.duration > 0:
		print(f"duration: {args.duration}s")
	else:
		print("duration: until Ctrl+C")
	print("\nSpeak into your microphone. The assistant response will appear below.\n")

	stop_event = asyncio.Event()
	audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
	stats = Stats()
	loop = asyncio.get_running_loop()

	output_stream: sd.RawOutputStream | None = None
	if args.play_audio and "audio" in args.output_modalities:
		output_stream = sd.RawOutputStream(
			samplerate=args.sample_rate,
			channels=1,
			dtype="int16",
			blocksize=0,
		)
		output_stream.start()

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
		async with websocket_connect(ws_url, headers) as ws:
			await ws.send(json.dumps({"type": "session.update", "session": session}))

			chunk_frames = int(args.sample_rate * args.chunk_ms / 1000)
			with sd.RawInputStream(
				samplerate=args.sample_rate,
				channels=1,
				dtype="int16",
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
		print_connection_hint(exc)
		return 1
	finally:
		if output_stream is not None:
			output_stream.stop()
			output_stream.close()

	_finish_prefixed_line(stats, "assistant_text_started")
	_finish_prefixed_line(stats, "user_text_started")
	print("\n=== Summary ===")
	print(f"Audio chunks sent: {stats.input_chunks}")
	print(f"Audio chunks dropped: {stats.dropped_audio_chunks}")
	print(f"Events received: {stats.events_received}")
	print(f"Output audio deltas: {stats.output_audio_deltas}")
	print(f"Assistant text deltas: {stats.assistant_text_deltas}")
	print(f"User transcript deltas: {stats.user_transcript_deltas}")
	print(f"Completed responses: {stats.completed_responses}")
	print(f"Errors: {stats.errors}")
	if stats.first_output_audio_ms is not None:
		print(f"First audio latency: {stats.first_output_audio_ms:.0f} ms")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(asyncio.run(main()))
	except KeyboardInterrupt:
		raise SystemExit(0)
