# 语音播放与屏幕展示不一致 — 排查步骤

## 1. 看浏览器控制台（前端）

1. 打开页面，按 **F12**（或右键 → 检查）→ 切到 **Console**。
2. 发一条消息，等角色回复。
3. 在控制台里找以 `[TTS 排查]` 开头的几行：

   - **后端返回**：`replyLen` / `tts_textLen` 是否相同？若 `是否相同=false`，说明后端返回的 `reply` 和 `tts_text` 就不一致（理论上不应出现）。
   - **前端统一使用**：`replyRawLen` 和前面 80 字，这是**真正用来展示和播放**的同一段文本。
   - **即将播放**：TTS 请求里发送的文本长度和前 80 字，应和「前端统一使用」完全一致。

**判断**：

- 若「后端返回」里 `reply` ≠ `tts_text` → 问题在后端或 Grok 返回结构，看下一步服务端日志。
- 若「后端返回」一致，但「即将播放」和「前端统一使用」不一致 → 说明中间有别的逻辑改动了文本（目前设计上应不会）。
- 若三者都一致，但听到的和看到的不一样 → 可能是 **TTS 接口（Grok Voice）** 读错了内容，或网络/缓存导致请求串了，可对比服务端 `error_messages.txt` 里 `/chat` 与 `/tts` 的预览。

---

## 2. 看服务端日志（后端）

1. 在项目目录下打开 **`error_messages.txt`**（与 `chat_web_cm.py` 同目录）。
2. 在文件里搜索 **`TTS_DEBUG`**：

   - **`reply_len=... preview=...`**：这是 **`/chat`** 返回给前端的「角色回复」长度和内容预览（前 120 字）。
   - **`tts_request len=... preview=...`**：这是 **`/tts`** 收到的「要读的文本」长度和内容预览。

**判断**：

- 若某次对话的 `reply_len` / `preview` 与紧接着的 `tts_request len` / `preview` **不一致** → 前端发给 TTS 的文本和当时 `/chat` 返回的回复不一致（例如请求顺序错乱、用了旧回复）。
- 若两者**一致**，但听到的内容和预览不同 → 可能是 Grok Voice API 把同一段文字读错/改写了，或音频被缓存/串台。

---

## 3. 可能原因与对应处理

| 现象 | 可能原因 | 建议 |
|------|----------|------|
| 控制台里 `reply` ≠ `tts_text` | 后端某处把两者设成不同来源 | 检查 `chat_web_cm.py` 里 `/chat` 的 `return jsonify(...)`，确认 `reply` 和 `tts_text` 都来自同一个 `character_reply`。 |
| 控制台一致，但 `error_messages.txt` 里 chat 与 tts 的 preview 不一致 | 前端用「上一次」的回复去调 TTS，或请求顺序错乱 | 确认发消息后只触发一次「展示 + 播放」，且 `playTts` 用的是本次 `replyRaw`，而不是缓存的旧文本。 |
| 后端 chat 与 tts 的 preview 一致，但听到的和看到的不一样 | Grok Voice 复述时改写了内容，或音频顺序错乱 | 检查 Voice API 的 instructions 是否要求「一字不差复述」；或暂时关掉语音，确认展示是否稳定一致。 |
| `response.content` 不是纯字符串 | Grok SDK 返回了 content blocks 列表 | 已在后端做兼容：若 `response.content` 是 list，会拼接所有 `type: "text"` 的 `text` 再给展示和 TTS。 |

---

## 4. 可选：带调试信息的请求

在「发消息」的请求 URL 上加上 **`?tts_debug=1`**（需在代码里给 `fetch('/chat', ...)` 的 URL 加上该参数），则本次 `/chat` 的 JSON 里会多一个 `_tts_debug` 字段，包含 `reply_len`、`tts_len` 和 `preview`，便于在 Network 里直接对比。

---

排查完或确认问题后，若不再需要这些日志，可：

- 删除或注释掉前端 `console.log('[TTS 排查] ...')`。
- 删除或注释掉后端 `append_error_log("TTS_DEBUG", ...)`。
