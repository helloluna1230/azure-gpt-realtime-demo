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

## `--turn-detection` 怎么选

服务端用什么策略判定「这一句说完了」从而触发转写/翻译。

| 选项 | 行为 | 相关参数 | 适合 |
|---|---|---|---|
| `server_vad`(默认) | 声学 VAD,静音够久就切句 | `--vad-threshold`、`--vad-prefix-padding-ms`、`--vad-silence-duration-ms` | 安静环境、命令式短句 |
| `semantic_vad` | 语义模型判断整句是否完整 | `--vad-eagerness low/medium/high/auto` | 自然口语、长句、有停顿 |
| `none` | 关闭服务端 VAD,由客户端手动 commit | — | Push-to-Talk(本仓库脚本暂未实现快捷键) |

调参经验:

- 噪音多 → `server_vad` + `--vad-threshold 0.6~0.7`
- 句子被切碎 → `--vad-silence-duration-ms 700~900`,或换 `semantic_vad --vad-eagerness low`
- 首字被吞 → `--vad-prefix-padding-ms 400~600`
- 演讲/长句 → `semantic_vad --vad-eagerness low`
- 短问答 → `semantic_vad --vad-eagerness high`

`semantic_vad` 强烈建议同时指定 `--input-language`,否则容易判错语种把句子提前切断。

## `--prompt` 写得好的几个原则

`--prompt` 会作为前置上下文影响 Whisper 的词表偏置和输出风格,主要用来**降低专有名词/术语误识**。它**不会**翻译、不会强制语种(语种用 `--input-language`)、也不是「指令」。

写法建议:

1. **用同语种的自然句子**承载术语,比逗号裸列表更有效。
2. **保持正确的大小写和官方写法**:`Kubernetes`、`Azure OpenAI`、`Cosmos DB`,模型会照抄。
3. **长度控制在几十~一两百字**,塞太长反而稀释偏置。
4. **音频是中文就用中文 prompt,英文就用英文 prompt**。中英混说时把英文术语嵌进中文句子。
5. **易混词成对列出**消歧:`请区分:Azure(云平台)与 ASR(语音识别)`。
6. **长 prompt 写进 `.env`**:`AZURE_OPENAI_TRANSCRIPTION_PROMPT=...`,命令行就不用再传。

## CLI 使用示例

### Whisper 转写(WebSocket)

```powershell
# 中文,自动识别语言
python .\realtime_whisper_mic_console.py --input-language zh --debug-events

# 英文 + 热词
python .\realtime_whisper_mic_console.py --input-language en --prompt "Azure, Kubernetes, Foundry, Realtime API"
```

### Whisper 转写(WebRTC)

```powershell
# 安静环境,声学 VAD
python .\realtime_whisper_mic_webrtc.py `
  --input-language zh `
  --turn-detection server_vad `
  --vad-threshold 0.5 --vad-silence-duration-ms 600 `
  --debug-events

# 中文技术分享,语义 VAD + 长 prompt
python .\realtime_whisper_mic_webrtc.py `
  --input-language zh `
  --turn-detection semantic_vad --vad-eagerness low `
  --prompt "本次分享主题包括 Azure、Microsoft Foundry、Azure OpenAI、Realtime API、Whisper、Kubernetes、AKS、Cosmos DB、Service Bus、Event Hubs、RAG、向量检索、Embedding、推理与训练。" `
  --debug-events

# 英文短问答,语义 VAD 低延迟
python .\realtime_whisper_mic_webrtc.py `
  --input-language en `
  --turn-detection semantic_vad --vad-eagerness high `
  --prompt "This demo covers Azure, Microsoft Foundry, Azure OpenAI, Realtime API, Whisper, Kubernetes, AKS, Cosmos DB, RAG, embeddings, and inference."
```

### 翻译(WebSocket)

```powershell
# 中→英,带音频
python .\realtime_translate_mic_console.py --target-language en --play-audio --debug-events

# 英→中,语义 VAD + 源语热词
python .\realtime_translate_mic_console.py `
  --target-language zh `
  --turn-detection semantic_vad --vad-eagerness low `
  --prompt "This talk covers Azure, Microsoft Foundry, Kubernetes, AKS, Cosmos DB, Realtime API." `
  --play-audio
```

### 用 `--debug-events` 看 VAD 行为

跑的时候关注这些事件类型:

- `input_audio_buffer.speech_started` / `speech_stopped`:VAD 切句节点
- `conversation.item.input_audio_transcription.delta` / `.completed`:转写陆续到达 / 该句最终文本
- `conversation.item.input_audio_transcription.failed`:通常是语种判错或音质太差,补一个明确的 `--input-language`

## 常见问题

- **`API key is still placeholder`**：`.env` 里 key 还是占位符，没替换。
- **`client_secrets failed: HTTP 500`**：`client_secrets` 请求体要保持最小（仅 `type: transcription` + `transcription.model`），多余字段会报 500。
- **`calls SDP exchange failed: HTTP 400`**：通常是 ICE 候选还没收集完，或 URL 上多带了 `?model=`。脚本会等待最多 5 秒收集 ICE，并且不带 `?model=`。
- **`conversation.item.input_audio_transcription.failed`**：源语言提示填错，去掉 `--input-language` 让服务自动识别。
- **找不到音频设备**：用 `python -c "import sounddevice as sd; print(sd.query_devices())"` 看一下设备列表，再通过 `sounddevice.default.device` 指定。
