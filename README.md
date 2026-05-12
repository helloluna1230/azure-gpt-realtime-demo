# gpt-realtime

Azure OpenAI Realtime API 示例：将麦克风音频实时送往 Whisper 转写与翻译模型，分别基于 WebSocket 和 WebRTC 两种通道。

## 脚本一览

| 脚本 | 通道 | 模型 | 功能 |
|---|---|---|---|
| `realtime_whisper_mic_console.py` | WebSocket `/realtime` | `gpt-realtime-whisper` | 同语言转写（不翻译） |
| `realtime_whisper_mic_webrtc.py` | WebRTC `/realtime/calls` | `gpt-realtime-whisper` | 通过 WebRTC 做同语言转写 |
| `realtime_translate_mic_console.py` | WebSocket `/realtime/translations` | `gpt-realtime-translate` | 将麦克风语音翻译为目标语言（文本 + 音频） |

> WebRTC 翻译版本 (`realtime_translate_mic_webrtc.py`) 仍在调试中，本文档暂不涉及。

## 环境准备

需要 Python 3.11+（开发环境为 3.14）。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

在脚本同目录下创建 `.env`：

```dotenv
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/openai/v1
AZURE_OPENAI_API_KEY=<your-key>

AZURE_OPENAI_TRANSLATE_DEPLOYMENT=gpt-realtime-translate
AZURE_OPENAI_WHISPER_DEPLOYMENT=gpt-realtime-whisper
```

`AZURE_OPENAI_ENDPOINT` 可以是 `https://<resource>.openai.azure.com`、`.../openai` 或 `.../openai/v1`，脚本会自动补全。

## Whisper 转写 — WebSocket

```powershell
python .\realtime_whisper_mic_console.py
```

对着麦克风说话，转写文本会随讲随出。按 Ctrl+C 停止。

常用参数：

- `--input-language zh`：源语言提示。留空则自动识别。**填错语言会触发服务端报错**（`conversation.item.input_audio_transcription.failed`）。
- `--prompt "技术名词: Kubernetes, Azure"`：给 Whisper 一些上下文提示，提高专业词汇命中率。
- `--turn-detection server_vad`（默认） | `semantic_vad` | `none`。
- `--debug-events`：每种事件类型首次出现时打印一行，便于排查。
- `--duration 30`：运行 N 秒后自动停止。

## Whisper 转写 — WebRTC

```powershell
python .\realtime_whisper_mic_webrtc.py
```

同样的模型，改走 WebRTC（`/realtime/calls`）。麦克风音频以 48 kHz Opus 推送，`oai-events` 数据通道承载 `session.update` 与转写事件。

参数与 WebSocket 版一致：`--input-language`、`--prompt`、`--turn-detection`、`--debug-events`、`--duration`，另外多了 `--text-output / --no-text-output` 控制是否打印文本。

## 翻译 — WebSocket

```powershell
python .\realtime_translate_mic_console.py --target-language zh --play-audio
```

把翻译后的文本流式输出到控制台；加 `--play-audio` 还会播放翻译后的语音。

常用参数：

- `--target-language zh|en|es|fr|ja|...`：目标语言代码。
- `--play-audio`：播放翻译音频。
- `--chunk-ms 40`：更小的分片可降低首音延迟。
- `--turn-detection none`（该端点默认） | `server_vad` | `semantic_vad`。
- `--no-text-output`：不打印翻译文本。
- `--debug-events`：打印事件类型。

## 常见问题

- **`API key is still placeholder`**：`.env` 里 key 还是占位符，没替换。
- **`client_secrets failed: HTTP 500`**：`client_secrets` 请求体要保持最小（仅 `type: transcription` + `transcription.model`），多余字段会报 500。
- **`calls SDP exchange failed: HTTP 400`**：通常是 ICE 候选还没收集完，或 URL 上多带了 `?model=`。脚本会等待最多 5 秒收集 ICE，并且不带 `?model=`。
- **`conversation.item.input_audio_transcription.failed`**：源语言提示填错，去掉 `--input-language` 让服务自动识别。
- **找不到音频设备**：用 `python -c "import sounddevice as sd; print(sd.query_devices())"` 看一下设备列表，再通过 `sounddevice.default.device` 指定。
