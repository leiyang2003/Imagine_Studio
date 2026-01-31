import os
import json
import time
import asyncio
import base64
import struct

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, Response, send_file
from xai_sdk import Client
from xai_sdk.chat import system, user, image as chat_image

# 加载 .env 文件（默认找当前工作目录下的 .env）
load_dotenv()  # 或 load_dotenv(".env.local") 指定其他名字


def read_system_prompt(file_path: str) -> str:
    """Read system prompt text from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise RuntimeError("System prompt file not found, please check the path.")
    except Exception as e:
        raise RuntimeError(f"Error reading system prompt file: {e}")


# 三阶段 System Prompt 分隔符（角色文件内：阶段1 / 阶段2 / 阶段3）
STAGE_DELIMITER_2 = "--- 阶段2 ---"
STAGE_DELIMITER_3 = "--- 阶段3 ---"


def read_system_prompt_staged(file_path: str) -> tuple:
    """
    读取包含三阶段人设的角色文件。用 --- 阶段2 --- 和 --- 阶段3 --- 分隔。
    返回 (stage1, stage2, stage3)；若无分隔符则整篇为阶段1，(stage2, stage3) 为空字符串。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError("System prompt file not found, please check the path.")
    except Exception as e:
        raise RuntimeError(f"Error reading system prompt file: {e}")
    if STAGE_DELIMITER_2 not in content:
        return (content, "", "")
    parts = content.split(STAGE_DELIMITER_2, 1)
    stage1 = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    if STAGE_DELIMITER_3 not in rest:
        return (stage1, rest, "")
    parts3 = rest.split(STAGE_DELIMITER_3, 1)
    stage2 = parts3[0].strip()
    stage3 = parts3[1].strip() if len(parts3) > 1 else ""
    return (stage1, stage2, stage3)


def append_jsonl(path: str, record: dict) -> None:
    """Append one JSON record to a JSONL file (utf-8)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_error_log(context: str, message: str) -> None:
    """Append an error line to error_messages.txt (utf-8) for user review."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(app_dir, "error_messages.txt")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{context}] {message}\n")
    except Exception:
        pass


def read_json(path: str):
    """Read a JSON file; return None if missing/invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_sessions(log_dir: str, prompt_file: str = None):
    """List saved sessions; if prompt_file is set, include sessions with that prompt_file or with no prompt_file (legacy). Sorted by updated_at desc."""
    sessions = []
    if not os.path.isdir(log_dir):
        return sessions
    for name in os.listdir(log_dir):
        if not name.endswith(".state.json"):
            continue
        session_id = name[: -len(".state.json")]
        state_path = os.path.join(log_dir, name)
        state = read_json(state_path) or {}
        if not isinstance(state, dict):
            continue
        # When filtering by person: include sessions with that prompt_file OR with no prompt_file (legacy 历史记录)
        if prompt_file is not None:
            s_pf = state.get("prompt_file")
            if s_pf is not None and s_pf != prompt_file:
                continue
        sessions.append({
            "session_id": session_id,
            "name": state.get("name") or None,
            "updated_at": state.get("updated_at") or 0,
            "prompt_file": state.get("prompt_file"),
        })
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


def list_prompt_files(prompt_dir: str):
    """List .txt files in prompt_dir; return [{ id, name }] where name is filename without .txt."""
    files = []
    if not os.path.isdir(prompt_dir):
        return files
    for name in sorted(os.listdir(prompt_dir)):
        if not name.endswith(".txt"):
            continue
        files.append({"id": name, "name": name[:-4]})
    return files


def read_history_jsonl(log_path: str):
    """Read message entries from a session's JSONL; return [{ role, content }]."""
    history = []
    if not os.path.isfile(log_path):
        return history
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") == "message" and "role" in rec and "content" in rec:
                        history.append({
                            "role": "user" if rec["role"] == "user" else "bot",
                            "content": rec["content"],
                        })
                except Exception:
                    continue
    except Exception:
        pass
    return history


def count_assistant_messages(log_path: str) -> int:
    """Count messages with role assistant/bot in the session log (for trigger every 3 rounds)."""
    history = read_history_jsonl(log_path)
    return sum(1 for m in history if m.get("role") == "bot")


def chat_log_to_text(log_path: str, last_n_rounds: int = None) -> str:
    """
    Build a plain-text transcript from JSONL for the evaluation agent.
    If last_n_rounds is set (e.g. 5), use only the last N rounds (1 round = 1 user + 1 assistant);
    if the log has fewer than N rounds, use the entire log.
    """
    history = read_history_jsonl(log_path)
    if last_n_rounds is not None and last_n_rounds > 0:
        need = last_n_rounds * 2  # 5 rounds = 10 messages
        if len(history) > need:
            history = history[-need:]
    lines = []
    for m in history:
        who = "用户" if m["role"] == "user" else "角色"
        lines.append(f"{who}: {m['content']}")
    return "\n\n".join(lines) if lines else "（暂无对话）"


EVALUATOR_SYSTEM_PROMPT = """你是一个基于聊天记录的分析助手。你的任务是根据一段「用户」与「角色」（由 system prompt 定义的虚拟人物）的对话记录，从角色的视角评估角色对用户的感觉。

请从以下 4 个维度打分，每个维度 0–10 分（0 最低，10 最高）。若对话刚起步、关系尚中性，各维度可从 3–5 分起评，不必一律打 0–2。

1. 情感亲密度（emotional_intimacy）：角色从疏离到亲密的程度，包括分享脆弱、情感表达的深度。
2. 占有欲与嫉妒（possessiveness_jealousy）：角色是否表现出独占欲，反映对用户的重视和不安全感。
3. 试探与信任构建（testing_trust_building）：本维度打「角色对用户的信任程度」。0 = 角色仍在大量试探、推拉、不信任、缺乏安全感；10 = 角色已较信任用户、安全感高、较少试探、愿意放下防备。
4. 性吸引与身体亲密（sexual_attraction_physical_intimacy）：对话涉及身体/亲密时的热情度，体现角色的欲求和舒适度。

你必须只输出一个合法的 JSON 对象，不要输出任何其他文字、解释或 markdown 标记。JSON 格式如下：
{
  "emotional_intimacy": <0-10 的数字>,
  "possessiveness_jealousy": <0-10 的数字>,
  "testing_trust_building": <0-10 的数字>,
  "sexual_attraction_physical_intimacy": <0-10 的数字>
}
"""


def run_evaluation(client: Client, log_path: str, character_name: str) -> dict | None:
    """
    Run the evaluation agent in a new session (no continuation). Uses the last 5 rounds
    of dialogue (or the full log if fewer than 5 rounds). Returns a dict with
    keys emotional_intimacy, possessiveness_jealousy, testing_trust_building,
    sexual_attraction_physical_intimacy (0-10 each). If any dimension is missing or
    invalid, returns None so the caller can retry.
    """
    transcript = chat_log_to_text(log_path, last_n_rounds=5)
    user_message = f"当前对话的角色名：{character_name or '未命名角色'}\n\n对话记录：\n\n{transcript}"
    chat = client.chat.create(
        model="grok-4-1-fast-reasoning",
        store_messages=False,
        tools=[],
    )
    chat.append(system(EVALUATOR_SYSTEM_PROMPT))
    chat.append(user(user_message))
    response = chat.sample()
    raw = (response.content or "").strip()
    # Strip markdown code block if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    try:
        data = json.loads(raw)
        out = {}
        keys_required = (
            "emotional_intimacy",
            "possessiveness_jealousy",
            "testing_trust_building",
            "sexual_attraction_physical_intimacy",
        )
        for key in keys_required:
            v = data.get(key)
            if v is not None and isinstance(v, (int, float)):
                out[key] = max(0, min(10, int(round(float(v)))))
            else:
                return None  # 任一维度缺失或无效则返回 None，由调用方再问一次
        return out
    except Exception:
        return None


# 评估维度顺序（用于计算总评值 S = 四维平均）
EVAL_DIM_KEYS = (
    "emotional_intimacy",
    "possessiveness_jealousy",
    "testing_trust_building",
    "sexual_attraction_physical_intimacy",
)

# --- TTS via Grok Voice Agent API (female voice Ara) ---
TTS_WS_URL = "wss://api.x.ai/v1/realtime"
TTS_VOICE = "Ara"  # Female, warm and friendly
TTS_SAMPLE_RATE = 24000


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Build a minimal WAV file from raw PCM 16-bit mono little-endian bytes."""
    n_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * n_channels * (bits_per_sample // 8)
    block_align = n_channels * (bits_per_sample // 8)
    data_size = len(pcm_bytes)
    chunk_size = 4 + 8 + 16 + 8 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        n_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_bytes


async def _tts_via_voice_api(text: str, api_key: str) -> bytes:
    """
    Use Grok Voice Agent WebSocket to synthesize speech from text.
    Sends text as user message with instruction to repeat exactly; collects output audio deltas and returns WAV bytes.
    Requires: pip install websockets
    """
    import websockets

    pcm_chunks: list[bytes] = []
    done = False
    session_updated = False
    last_error: str | None = None

    async def on_message(ws, message):
        nonlocal done, session_updated, last_error
        try:
            data = json.loads(message)
        except Exception:
            return
        msg_type = data.get("type") or ""
        if msg_type == "session.updated":
            session_updated = True
        elif msg_type == "response.output_audio.delta":
            delta_b64 = data.get("delta")
            if delta_b64:
                try:
                    pcm_chunks.append(base64.b64decode(delta_b64))
                except Exception:
                    pass
        elif msg_type == "response.output_audio.done":
            done = True
        elif msg_type == "error" or "error" in msg_type:
            last_error = data.get("message") or data.get("error", {}).get("message") or json.dumps(data)[:200]

    async with websockets.connect(
        TTS_WS_URL,
        ssl=True,
        additional_headers={"Authorization": f"Bearer {api_key}"},
        close_timeout=10,
        open_timeout=20,
    ) as ws:
        # Session: female voice Ara, instruction to repeat exactly, no VAD (we send text only)
        session_config = {
            "type": "session.update",
            "session": {
                "voice": TTS_VOICE,
                "instructions": "You are a text-to-speech agent. The user will send you text. Reply with exactly that text and nothing else—no additions, no commentary.",
                "turn_detection": None,
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": TTS_SAMPLE_RATE}},
                    "output": {"format": {"type": "audio/pcm", "rate": TTS_SAMPLE_RATE}},
                },
            },
        }
        await ws.send(json.dumps(session_config))
        # Wait for session.updated (server may send conversation.created first)
        wait_deadline = asyncio.get_running_loop().time() + 10.0
        while not session_updated:
            try:
                remaining = max(0.5, wait_deadline - asyncio.get_running_loop().time())
                msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            await on_message(ws, msg)
            if last_error:
                raise RuntimeError(last_error)

        if last_error:
            raise RuntimeError(last_error)

        # One user message: the text to speak
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "response.create", "response": {"modalities": ["text", "audio"]}}))
        # Collect audio deltas until output_audio.done (with timeout)
        timeout_sec = 120
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec
        while not done:
            try:
                remaining = max(0.1, deadline - loop.time())
                msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            await on_message(ws, msg)
            if last_error:
                raise RuntimeError(last_error)

    if last_error:
        raise RuntimeError(last_error)
    pcm_bytes = b"".join(pcm_chunks)
    if not pcm_bytes:
        raise RuntimeError("Voice API 未返回音频，请确认账号已开通 Realtime/Voice 权限与 us-east-1 区域")
    return _pcm_to_wav(pcm_bytes)


# API key：环境变量可选。不设置时由访客在页面填写自己的 Key，部署者不分享自己的 Key
_env_api_key = os.getenv("XAI_API_KEY")
default_client = Client(api_key=_env_api_key, timeout=3600) if _env_api_key else None
client = default_client  # 兼容后续可能仍引用 client 的模块级代码；请求内应使用 get_client_for_request()

# System prompt: default file in current dir; person-specific prompts in systemprompt/ directory
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH = os.path.join(_APP_DIR, "SystemPrompt.txt")
PROMPT_DIR = os.path.join(_APP_DIR, "systemprompt")
system_prompt = read_system_prompt(SYSTEM_PROMPT_PATH)

# Persist chat to local files (single-user).
# - chat_logs/<session_id>.jsonl: append-only transcript
# - chat_logs/<session_id>.state.json: stores previous_response_id so next run can continue
# Vercel serverless 下项目目录只读，日志写到 /tmp
LOG_DIR = os.getenv("CHAT_LOG_DIR", "/tmp/chat_logs" if os.getenv("VERCEL") else "chat_logs")
LAST_SESSION_PATH = os.path.join(LOG_DIR, "last_session.txt")

# If CHAT_SESSION_ID is not provided, reuse the last session id to continue chatting next time.
SESSION_ID = os.getenv("CHAT_SESSION_ID")
if not SESSION_ID:
    try:
        with open(LAST_SESSION_PATH, "r", encoding="utf-8") as f:
            SESSION_ID = (f.read() or "").strip() or None
    except FileNotFoundError:
        SESSION_ID = None
if not SESSION_ID:
    SESSION_ID = f"{int(time.time())}"
os.makedirs(LOG_DIR, exist_ok=True)
with open(LAST_SESSION_PATH, "w", encoding="utf-8") as f:
    f.write(SESSION_ID)

CHAT_LOG_PATH = os.path.join(LOG_DIR, f"{SESSION_ID}.jsonl")
STATE_PATH = os.path.join(LOG_DIR, f"{SESSION_ID}.state.json")

# Load previous_response_id from disk (so the model can continue the same conversation).
previous_response_id = None
state = read_json(STATE_PATH) or {}
if isinstance(state, dict):
    previous_response_id = state.get("previous_response_id") or None

# 仅当部署者配置了环境变量 Key 时，预创建全局 chat 并写一条 system 日志
chat = None
if default_client:
    chat = default_client.chat.create(
        model="grok-4-1-fast-reasoning",
        store_messages=True,
        tools=[],
    )
    chat.append(system(system_prompt))
    append_jsonl(
        CHAT_LOG_PATH,
        {
            "type": "system",
            "timestamp": time.time(),
            "content": system_prompt,
            "model": "grok-4-1-fast-reasoning",
            "session_id": SESSION_ID,
            "resumed": bool(previous_response_id),
        },
    )

app = Flask(__name__)


def get_api_key_from_request():
    """从请求头 X-Api-Key 或 JSON body 的 api_key、或环境变量取 Key。未设置返回 None。"""
    key = request.headers.get("X-Api-Key") or (request.headers.get("Authorization") or "").replace("Bearer ", "").strip()
    if not key and request.is_json:
        data = request.get_json(silent=True) or {}
        key = (data.get("api_key") or "").strip()
    return key or os.getenv("XAI_API_KEY") or None


def get_client_for_request():
    """用于本次请求的 xAI Client：优先请求里的 Key，否则用环境变量。无 Key 返回 None。"""
    key = get_api_key_from_request()
    if not key:
        return None
    return Client(api_key=key, timeout=3600)

# --- 每三轮根据对话生成 16:9 风格图，替换 character-visual 展示 ---
IMAGE_GEN_MODEL = "grok-imagine-image"
DISPLAY_IMAGE_FILENAME = "{}_display.png"

# 生成图片时的风格依据（可根据对话微调场景/情绪，但整体保持此风格）
DISPLAY_IMAGE_STYLE_PROMPT = (
    "with heavy eyeliner, full lips, erotic expression, lying on luxurious deep maroon silk bedsheets, "
    "wearing sheer red lace lingerie set with garter belts large breasts with visible nipples through sheer fabric, "
    "seductive bedroom setting at night, dark romantic atmosphere, "
    "burning rose vines and thorny stems forming a fiery crown of flames above her head, intense orange fire glow illuminating her body, "
    "scattered red roses around, cinematic lighting, rim light, volumetric god rays, highly detailed textures, "
    "ultra-realistic rendering, masterpiece, 8k, --ar 16:9 --stylize 250 --v 6 --q 2"
)


def _get_character_image_data_url(state_path: str) -> str | None:
    """从当前会话 state 的 prompt_file 找到 systemprompt 下的角色图，返回 data URL；无图返回 None。"""
    state = read_json(state_path) or {}
    if not isinstance(state, dict):
        return None
    pf = state.get("prompt_file")
    if not pf or not isinstance(pf, str) or not pf.endswith(".txt"):
        return None
    basename = pf[:-4].strip()
    if not basename or ".." in basename or "/" in basename:
        return None
    for ext in (".png", ".jpg", ".jpeg", ".gif"):
        path = os.path.join(PROMPT_DIR, basename + ext)
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                b64 = base64.b64encode(raw).decode("ascii")
                mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in (".jpg", ".jpeg") else "image/gif")
                return f"data:{mime};base64,{b64}"
            except Exception:
                return None
    return None


def _describe_person_from_image(data_url: str, client: "Client") -> str | None:
    """用视觉模型描述图中人物外貌，供后续生图保持同一人。失败返回 None。"""
    try:
        chat_desc = client.chat.create(
            model="grok-4-1-fast-reasoning",
            store_messages=False,
            tools=[],
        )
        chat_desc.append(
            system(
                "You are an expert at describing a person's appearance so another AI can draw the same person. "
                "Output only a concise description in English: face shape, eyes, hair color and style, skin tone, "
                "body type, distinctive features. No preamble, no scene description."
            )
        )
        chat_desc.append(
            user(
                "Describe this person's appearance in detail so that another AI can draw the exact same woman in a different scene.",
                chat_image(data_url, detail="high"),
            )
        )
        response = chat_desc.sample()
        text = (response.content or "").strip()
        return text if len(text) > 20 else None
    except Exception as e:
        append_error_log("describe_person_from_image", str(e))
        return None


def _generate_display_image_from_chat(
    log_path: str, state_path: str, session_id: str, log_dir: str, *, client: "Client | None" = None
) -> bool:
    """
    根据最近三轮对话 + 风格依据生成 16:9 图，保存为 log_dir/<session_id>_display.png，
    女生外貌依据原图（systemprompt 角色图）保持同一人。client 为 None 时使用 default_client。
    成功返回 True，失败返回 False。
    """
    c = client or default_client
    if not c:
        return False
    history = read_history_jsonl(log_path)
    last_n = 6
    if len(history) < last_n:
        return False
    recent = history[-last_n:]
    parts = []
    for m in recent:
        who = "用户" if m.get("role") == "user" else "角色"
        content = (m.get("content") or "").strip()
        if content:
            parts.append(f"{who}: {content[:200]}")
    if not parts:
        return False
    conversation_text = "\n".join(parts)

    # 原图女生外貌描述：用 systemprompt 角色图做视觉描述，后续生图保持同一人
    person_desc = None
    data_url = _get_character_image_data_url(state_path)
    if data_url:
        person_desc = _describe_person_from_image(data_url, c)
    sfw_prefix = "SFW only. Safe for work. No nudity, no explicit sexual content. Keep the image tasteful and suitable for all audiences. "
    if person_desc:
        prompt = (
            sfw_prefix
            + "The woman in the image MUST look exactly like this person (same face, hair, body): "
            + person_desc
            + ".\n\n"
            + DISPLAY_IMAGE_STYLE_PROMPT
            + "\n\nAdapt the mood or small details (expression, pose nuance) based on this conversation: "
            + conversation_text
        )
    else:
        prompt = (
            sfw_prefix
            + DISPLAY_IMAGE_STYLE_PROMPT
            + "\n\nAdapt the mood or small details (expression, pose nuance) based on this conversation, keep the style above: "
            + conversation_text
        )
    try:
        # xAI image generation: SDK 仅支持 prompt, model, image_format（无 aspect_ratio），16:9 写进 prompt
        response = c.image.sample(
            prompt=prompt,
            model=IMAGE_GEN_MODEL,
            image_format="base64",
        )
        # 取图数据：response.image 为 bytes（SDK 解码 base64）；失败则用 response.base64 自解码
        image_data = None
        try:
            image_data = response.image
        except Exception:
            try:
                b64 = getattr(response, "base64", None) or (
                    getattr(response, "_image", None) and getattr(response._image, "base64", None)
                )
                if b64 and isinstance(b64, str):
                    if "base64," in b64:
                        b64 = b64.split("base64,", 1)[-1]
                    image_data = base64.b64decode(b64)
            except Exception:
                pass
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        if not image_data or len(image_data) < 100:
            _set_display_image_to_prompt(state_path)
            return False
        out_path = os.path.join(log_dir, DISPLAY_IMAGE_FILENAME.format(session_id))
        os.makedirs(log_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(image_data)
        state = read_json(state_path) or {}
        if not isinstance(state, dict):
            state = {}
        state["display_image"] = "generated"
        state["updated_at"] = time.time()
        write_json(state_path, state)
        return True
    except Exception as e:
        append_error_log("display_image_gen", str(e))
        _set_display_image_to_prompt(state_path)
        return False


def _set_display_image_to_prompt(state_path: str) -> None:
    """生成失败时回退为使用原图：将 state.display_image 设为 prompt。"""
    try:
        state = read_json(state_path) or {}
        if not isinstance(state, dict):
            return
        state["display_image"] = "prompt"
        state["updated_at"] = time.time()
        write_json(state_path, state)
    except Exception:
        pass


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <title>Imagine Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
      :root {
        /* 极简色彩：大量留白 + 中性灰/白/深蓝，仅关键操作与 AI 状态用亮色 */
        --bg: #f8fafc;
        --surface: #ffffff;
        --surface-2: #f1f5f9;
        --border: #e2e8f0;
        --border-subtle: #f1f5f9;
        --text: #0f172a;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --accent: #22c55e;
        --accent-dim: #16a34a;
        --shadow: 0 1px 2px rgba(15,23,42,.04);
        --shadow-md: 0 4px 12px rgba(15,23,42,.06);
        --radius: 10px;
        --radius-sm: 6px;
        /* 细宋体：Noto Serif SC Light(300) + 系统宋体回退 */
        --font-song: "Noto Serif SC", "Songti SC", "STSong", "SimSun", "NSimSun", "Serif";
        --font-display: var(--font-song);
        --font-body: var(--font-song);
        --icon-size: 18px;
        --icon-color: var(--text-secondary);
      }
      * { box-sizing: border-box; }
      html, body {
        height: 100%;
        margin: 0;
      }
      body {
        font-family: var(--font-body);
        font-weight: 300;
        width: 100%;
        padding: 24px 32px;
        background: var(--bg);
        color: var(--text);
        font-size: 15px;
        line-height: 1.55;
      }
      h1 {
        font-family: var(--font-display);
        font-weight: 500;
        font-size: 18px;
        margin: 0 0 8px;
        letter-spacing: 0.02em;
        color: var(--text);
      }
      p.subtitle {
        margin: 0 0 16px;
        color: var(--text-secondary);
        font-size: 13px;
      }
      .icon { width: var(--icon-size); height: var(--icon-size); flex-shrink: 0; color: var(--icon-color); }
      #chat-screen.active {
        display: flex;
        flex-direction: column;
        min-height: calc(100vh - 48px);
        width: 100%;
      }
      #chat-screen.active #chat-layout {
        flex: 1;
        min-height: 0;
        align-items: stretch;
      }
      #chat-screen.active #chat-left {
        display: flex;
        flex-direction: column;
        min-height: 0;
      }
      #character-visual {
        position: relative;
        width: 100%;
        border-radius: var(--radius);
        overflow: hidden;
        background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
        border: 1px solid var(--border);
      }
      #character-visual img, #character-visual video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center top;
      }
      #character-visual .no-image {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 160px;
        color: var(--text-muted);
        font-family: var(--font-display);
        font-size: 15px;
      }
      #character-name-bar {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 12px 16px;
        background: linear-gradient(transparent, rgba(15,23,42,.6));
        color: #fff;
        font-family: var(--font-display);
        font-weight: 600;
        font-size: 16px;
      }
      #character-status {
        font-family: var(--font-body);
        font-size: 12px;
        font-weight: 400;
        color: rgba(255,255,255,.85);
        margin-top: 2px;
      }
      #character-status .dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent);
        margin-right: 6px;
        vertical-align: middle;
      }
      #chat-dialog-wrap #chat-container {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
        border-radius: var(--radius);
        background: var(--surface);
        padding: 20px;
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
      }
      #chat-dialog-wrap #chat-container #chat {
        flex: 1;
        min-height: 80px;
        border-radius: var(--radius-sm);
        background: var(--surface-2);
        padding: 16px;
        overflow-y: auto;
        scroll-behavior: smooth;
        border: 1px solid var(--border-subtle);
      }
      #chat-dialog-wrap #chat-container form,
      #chat-dialog-wrap #chat-container #status {
        flex-shrink: 0;
      }
      .msg {
        margin-bottom: 14px;
        display: flex;
      }
      .msg-inner {
        max-width: 82%;
        padding: 10px 14px;
        border-radius: var(--radius);
        font-size: 14px;
        line-height: 1.5;
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      .user {
        justify-content: flex-end;
      }
      .user .msg-inner {
        background: var(--primary);
        color: #fff;
        border-bottom-right-radius: 4px;
      }
      .bot {
        justify-content: flex-start;
      }
      .bot .msg-inner {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text);
        border-bottom-left-radius: 4px;
      }
      .label {
        font-size: 11px;
        color: var(--text-secondary);
        margin-bottom: 4px;
      }
      .user .label { text-align: right; }
      .bot .label { text-align: left; }
      form {
        margin-top: 14px;
        display: flex;
        gap: 8px;
        align-items: center;
      }
      input[type=text] {
        font-family: var(--font-body);
        flex: 1;
        padding: 12px 16px;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        font-size: 14px;
        font-weight: 300;
        outline: none;
        transition: border-color .15s, box-shadow .15s;
      }
      input[type=text]:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(37, 99, 235, .12);
      }
      button {
        font-family: var(--font-body);
        padding: 8px 14px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text-secondary);
        cursor: pointer;
        font-weight: 400;
        font-size: 13px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: background .15s, border-color .15s, color .15s;
      }
      button .icon { color: inherit; }
      button:hover {
        background: var(--surface-2);
        color: var(--text);
        border-color: var(--text-muted);
      }
      button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      /* 仅关键操作使用亮色：发送、新对话（欢迎页）、确定 */
      button[type="submit"],
      #btn-new-chat-welcome,
      #name-modal-ok {
        background: var(--primary);
        border-color: var(--primary);
        color: #fff;
      }
      button[type="submit"]:hover:not(:disabled),
      #btn-new-chat-welcome:hover,
      #name-modal-ok:hover {
        background: var(--primary-hover);
        border-color: var(--primary-hover);
        color: #fff;
      }
      #new-chat, #back-home {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text-secondary);
      }
      #new-chat:hover, #back-home:hover {
        background: var(--surface-2);
        border-color: var(--border);
        color: var(--text);
      }
      #status {
        margin-top: 8px;
        font-size: 12px;
        color: var(--text-muted);
      }
      .nsfw-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 10px;
        font-size: 12px;
        color: var(--text-muted);
      }
      .nsfw-row span.nsfw-label {
        margin-right: 4px;
      }
      .nsfw-toggle {
        display: inline-flex;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        overflow: hidden;
        background: var(--surface-2);
      }
      .nsfw-toggle button {
        padding: 4px 10px;
        border: none;
        border-radius: 0;
        font-size: 11px;
        cursor: pointer;
        background: transparent;
        color: var(--text-muted);
        transition: background .15s, color .15s;
      }
      .nsfw-toggle button:first-child {
        border-right: 1px solid var(--border);
      }
      .nsfw-toggle button.active {
        background: var(--surface);
        color: var(--text);
        font-weight: 500;
      }
      .nsfw-toggle button:not(.active):hover {
        background: var(--surface);
        color: var(--text-secondary);
      }
      #name-modal-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(15,23,42,.25);
        z-index: 100;
        align-items: center;
        justify-content: center;
      }
      #name-modal-overlay.show { display: flex; }
      #name-modal {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
        min-width: 340px;
        box-shadow: var(--shadow-md);
      }
      #name-modal h3 {
        margin: 0 0 6px;
        font-size: 18px;
        font-weight: 600;
        color: var(--text);
      }
      #name-modal p {
        margin: 0 0 16px;
        font-size: 14px;
        color: var(--text-secondary);
      }
      #name-modal input {
        width: 100%;
        padding: 12px 14px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        font-size: 14px;
        margin-bottom: 20px;
      }
      #name-modal input:focus {
        border-color: var(--primary);
        outline: none;
        box-shadow: 0 0 0 2px rgba(37,99,235,.2);
      }
      #name-modal-actions {
        display: flex;
        gap: 10px;
        justify-content: flex-end;
      }
      #name-modal-actions button {
        padding: 10px 18px;
        border-radius: var(--radius-sm);
        border: none;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
      }
      #name-modal-skip {
        background: var(--surface);
        color: var(--text-secondary);
        border: 1px solid var(--border);
      }
      #name-modal-skip:hover { background: var(--bg); }
      #name-modal-ok {
        background: var(--primary);
        color: #fff;
      }
      #name-modal-ok:hover { background: var(--primary-hover); }
      #chat-screen {
        display: none;
      }
      #chat-layout {
        display: flex;
        flex-direction: row;
        gap: 16px;
        align-items: stretch;
        width: 100%;
        max-width: 100%;
        min-height: 0;
        flex: 1;
      }
      #chat-left {
        flex: 0 0 80%;
        width: 80%;
        min-width: 0;
        display: flex;
        flex-direction: column;
        min-height: 0;
      }
      #character-visual {
        flex: 7 1 0;
        min-height: 120px;
      }
      #chat-dialog-wrap {
        flex: 3 1 0;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        min-height: 0;
      }
      #evaluation-panel {
        flex: 0 0 20%;
        width: 20%;
        min-width: 0;
        align-self: stretch;
        border-radius: var(--radius);
        background: var(--surface);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        padding: 18px;
        overflow-y: auto;
      }
      #evaluation-panel h3 {
        margin: 0 0 14px;
        font-size: 13px;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: -0.01em;
      }
      .eval-stage-ps {
        margin-bottom: 12px;
        padding: 10px 12px;
        background: var(--surface-2);
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-subtle);
        font-size: 12px;
        color: var(--text-secondary);
      }
      .eval-ps-line {
        margin-bottom: 4px;
      }
      .eval-ps-line:last-of-type { margin-bottom: 0; }
      .eval-downgrade {
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid var(--border);
        font-weight: 600;
        color: #dc2626;
      }
      .eval-system-prompt-stage {
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid var(--border);
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
      }
      .eval-dim {
        margin-bottom: 14px;
      }
      .eval-dim:last-child {
        margin-bottom: 0;
      }
      .eval-dim-name {
        font-size: 12px;
        color: var(--text-secondary);
        margin-bottom: 4px;
        line-height: 1.35;
      }
      .eval-dim-bar-wrap {
        height: 8px;
        background: var(--bg);
        border-radius: 4px;
        overflow: hidden;
      }
      .eval-dim-bar {
        height: 100%;
        background: var(--primary);
        border-radius: 4px;
        width: 0%;
        transition: width .3s ease;
      }
      .eval-dim-score {
        font-size: 11px;
        color: var(--text-muted);
        margin-top: 2px;
      }
      #evaluation-loading, #evaluation-empty {
        font-size: 13px;
        color: var(--text-secondary);
        padding: 12px 0;
      }
      #evaluation-error {
        font-size: 12px;
        color: #dc2626;
        padding: 8px 0;
      }
      #welcome-screen {
        display: none;
      }
      #welcome-screen.active {
        display: block;
      }
      #person-picker-screen {
        max-width: 520px;
        margin: 0 auto;
        padding: 40px 36px;
        border-radius: var(--radius);
        background: var(--surface);
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
      }
      #person-picker-screen h2 {
        margin: 0 0 8px;
        font-size: 18px;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.02em;
      }
      #person-picker-screen .picker-sub {
        margin: 0 0 28px;
        font-size: 14px;
        color: var(--text-secondary);
      }
      #person-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .person-item {
        display: flex;
        align-items: center;
        gap: 10px;
        width: 100%;
        padding: 14px 18px;
        text-align: left;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        background: var(--surface);
        color: var(--text);
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: background .15s, border-color .15s;
      }
      .person-item:hover {
        background: var(--surface-2);
        border-color: var(--border);
      }
      .person-item:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(37,99,235,.12);
      }
      .person-item .icon { flex-shrink: 0; }
      .picker-empty {
        padding: 32px;
        text-align: center;
        color: var(--text-muted);
        font-size: 14px;
      }
      #welcome-screen {
        max-width: 520px;
        margin: 0 auto;
        padding: 40px 36px;
        border-radius: var(--radius);
        background: var(--surface);
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
      }
      #welcome-screen h2 {
        margin: 0 0 8px;
        font-size: 18px;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.02em;
      }
      #welcome-screen .welcome-sub {
        margin: 0 0 12px;
        font-size: 14px;
        color: var(--text-secondary);
      }
      .back-to-picker-btn {
        margin-bottom: 24px;
        padding: 8px 12px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text-secondary);
        font-size: 13px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: background .15s, color .15s;
      }
      .back-to-picker-btn:hover {
        background: var(--surface-2);
        color: var(--text);
      }
      .back-to-picker-btn .icon { color: inherit; }
      .welcome-actions {
        display: flex;
        gap: 12px;
        margin-bottom: 28px;
      }
      .welcome-actions button {
        flex: 1;
        padding: 14px 20px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        transition: background .15s, border-color .15s;
      }
      #btn-new-chat-welcome {
        background: var(--primary);
        border-color: var(--primary);
        color: #fff;
      }
      #btn-new-chat-welcome:hover { background: var(--primary-hover); border-color: var(--primary-hover); }
      #btn-history-welcome {
        background: var(--surface);
        color: var(--text-secondary);
      }
      #btn-history-welcome:hover {
        background: var(--surface-2);
        color: var(--text);
        border-color: var(--border);
      }
      .welcome-actions .icon { color: inherit; }
      #history-list-wrap {
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid var(--border);
      }
      #history-list-wrap h3 {
        margin: 0 0 12px;
        font-size: 13px;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: .04em;
      }
      #history-list {
        max-height: 280px;
        overflow-y: auto;
        border-radius: var(--radius-sm);
        background: var(--bg);
        border: 1px solid var(--border);
      }
      .history-item {
        display: block;
        width: 100%;
        padding: 12px 14px;
        text-align: left;
        border: none;
        border-bottom: 1px solid var(--border);
        background: transparent;
        color: var(--text);
        font-size: 14px;
        cursor: pointer;
        transition: background .15s;
      }
      .history-item:last-child { border-bottom: none; }
      .history-item:hover { background: var(--surface); }
      .history-item .name {
        font-weight: 500;
        color: var(--text);
      }
      .history-item .meta {
        font-size: 12px;
        color: var(--text-secondary);
        margin-top: 2px;
      }
      .history-item-row {
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid var(--border);
      }
      .history-item-row:last-child { border-bottom: none; }
      .history-item-row .history-item {
        flex: 1;
        border: none;
        border-radius: 0;
        margin: 0;
      }
      .history-item-delete {
        flex-shrink: 0;
        width: 32px;
        height: 32px;
        padding: 0;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        background: var(--surface);
        color: var(--text-secondary);
        font-size: 18px;
        line-height: 1;
        cursor: pointer;
        transition: background .15s, color .15s, border-color .15s;
      }
      .history-item-delete:hover {
        background: var(--surface-2);
        color: #dc2626;
        border-color: var(--border);
      }
      #delete-modal-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,.4);
        z-index: 100;
        align-items: center;
        justify-content: center;
      }
      #delete-modal-overlay.show { display: flex; }
      #delete-modal {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
        min-width: 340px;
        box-shadow: var(--shadow-md);
      }
      #delete-modal h3 {
        margin: 0 0 8px;
        font-size: 18px;
        font-weight: 600;
        color: var(--text);
      }
      #delete-modal p {
        margin: 0 0 20px;
        font-size: 14px;
        color: var(--text-secondary);
      }
      #delete-modal-actions {
        display: flex;
        gap: 10px;
        justify-content: flex-end;
      }
      #delete-modal-actions button {
        padding: 10px 18px;
        border-radius: var(--radius-sm);
        border: none;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
      }
      #delete-modal-cancel {
        background: var(--surface);
        color: var(--text-secondary);
        border: 1px solid var(--border);
      }
      #delete-modal-cancel:hover { background: var(--bg); }
      #delete-modal-confirm {
        background: #dc2626;
        color: #fff;
      }
      #delete-modal-confirm:hover { background: #b91c1c; }
      .history-empty {
        padding: 24px;
        text-align: center;
        color: #64748b;
        font-size: 14px;
      }
      .history-item-delete .icon { color: var(--text-muted); }
      .history-item-delete:hover .icon { color: #dc2626; }
      .history-empty {
        padding: 24px;
        text-align: center;
        color: var(--text-muted);
        font-size: 14px;
      }
      #api-key-modal-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,.4);
        z-index: 101;
        align-items: center;
        justify-content: center;
      }
      #api-key-modal-overlay.show { display: flex; }
      #api-key-modal {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
        min-width: 360px;
        box-shadow: var(--shadow-md);
      }
      #api-key-modal h3 { margin: 0 0 8px; font-size: 18px; font-weight: 600; color: var(--text); }
      #api-key-modal p { margin: 0 0 16px; font-size: 14px; color: var(--text-secondary); }
      #api-key-modal input {
        width: 100%;
        padding: 10px 12px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        font-size: 14px;
        margin-bottom: 16px;
      }
      #api-key-modal-actions { display: flex; gap: 10px; justify-content: flex-end; }
      #api-key-modal-actions button { padding: 8px 16px; border-radius: var(--radius-sm); font-size: 14px; cursor: pointer; }
      #api-key-save { background: var(--primary); color: #fff; border: none; }
      #api-key-save:hover { background: var(--primary-hover); }
      .api-key-link { font-size: 12px; color: var(--text-muted); cursor: pointer; margin-left: 12px; }
      .api-key-link:hover { color: var(--primary); }
    </style>
  </head>
  <body>
    <svg xmlns="http://www.w3.org/2000/svg" style="position:absolute;width:0;height:0;">
      <symbol id="icon-home" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></symbol>
      <symbol id="icon-plus" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></symbol>
      <symbol id="icon-send" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></symbol>
      <symbol id="icon-users" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></symbol>
      <symbol id="icon-history" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></symbol>
      <symbol id="icon-chevron-left" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></symbol>
      <symbol id="icon-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></symbol>
      <symbol id="icon-trash" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></symbol>
      <symbol id="icon-user" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></symbol>
    </svg>
    <div id="name-modal-overlay">
      <div id="name-modal">
        <h3>给这段对话起个名字</h3>
        <p>方便以后找回并继续聊～</p>
        <input id="session-name-input" type="text" placeholder="例如：工作周报、学英语、点子记录..." maxlength="64" autocomplete="off" />
        <div id="name-modal-actions">
          <button id="name-modal-skip" type="button">跳过</button>
          <button id="name-modal-ok" type="button"><svg class="icon" viewBox="0 0 24 24"><use href="#icon-check"/></svg>确定</button>
        </div>
      </div>
    </div>
    <div id="delete-modal-overlay">
      <div id="delete-modal">
        <h3>确认删除</h3>
        <p>是否确认要删除该条历史记录？删除后无法恢复。</p>
        <div id="delete-modal-actions">
          <button id="delete-modal-cancel" type="button">取消</button>
          <button id="delete-modal-confirm" type="button"><svg class="icon" viewBox="0 0 24 24"><use href="#icon-trash"/></svg>确认删除</button>
        </div>
      </div>
    </div>
    <div id="api-key-modal-overlay">
      <div id="api-key-modal">
        <h3>xAI API Key</h3>
        <p>未配置环境变量时，请填写你自己的 xAI API Key，仅保存在本机，不会上传。</p>
        <input id="api-key-input" type="password" placeholder="sk-..." autocomplete="off" />
        <div id="api-key-modal-actions">
          <button id="api-key-cancel" type="button">取消</button>
          <button id="api-key-save" type="button">保存</button>
        </div>
      </div>
    </div>
    <div id="person-picker-screen">
      <h2>选择与谁聊天</h2>
      <p class="picker-sub">从 systemprompt 目录中选择一个角色，开始或继续对话</p>
      <div id="person-list">
        <div class="picker-empty">加载中…</div>
      </div>
    </div>
    <div id="welcome-screen">
      <h2 id="welcome-title" style="display:flex;align-items:center;">和谁聊天<span class="api-key-link" id="btn-set-api-key-welcome" title="设置 xAI API Key">设置 API Key</span></h2>
      <p class="welcome-sub">从历史对话继续，或开始新对话</p>
      <button id="btn-back-to-picker" type="button" class="back-to-picker-btn"><svg class="icon" viewBox="0 0 24 24"><use href="#icon-chevron-left"/></svg>换一个人</button>
      <div class="welcome-actions">
        <button id="btn-history-welcome" type="button"><svg class="icon" viewBox="0 0 24 24"><use href="#icon-history"/></svg>从历史对话继续</button>
        <button id="btn-new-chat-welcome" type="button"><svg class="icon" viewBox="0 0 24 24"><use href="#icon-plus"/></svg>新对话</button>
      </div>
      <div id="history-list-wrap">
        <h3>历史对话</h3>
        <div id="history-list">
          <div class="history-empty">加载中…</div>
        </div>
      </div>
    </div>
    <div id="chat-screen">
      <h1 style="display:flex;align-items:center;">Imagine Studio<span class="api-key-link" id="btn-set-api-key" title="设置 xAI API Key">设置 API Key</span></h1>
      <div id="chat-layout">
        <div id="chat-left">
          <div id="character-visual">
            <div class="no-image" id="character-no-image">选择角色后将显示立绘或动图</div>
            <img id="character-img" alt="" style="display:none;" />
            <video id="character-video" loop muted playsinline style="display:none;"></video>
            <div id="character-name-bar" style="display:none;">
              <span id="character-display-name"></span>
              <div id="character-status"><span class="dot"></span>准备就绪</div>
            </div>
          </div>
          <div id="chat-dialog-wrap">
            <div id="chat-container">
              <div id="chat"></div>
              <form id="chat-form">
                <input id="msg" type="text" placeholder="说点什么..." autocomplete="off" />
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:8px;">
                  <button id="back-home" type="button" title="回到首页，选择聊天对象">
                    <svg class="icon" viewBox="0 0 24 24"><use href="#icon-home"/></svg>
                    <span>回到首页</span>
                  </button>
                  <button id="new-chat" type="button" title="开始一个全新的对话（不再续上之前上下文）">
                    <svg class="icon" viewBox="0 0 24 24"><use href="#icon-plus"/></svg>
                    <span>新对话</span>
                  </button>
                  <button type="submit">
                    <svg class="icon" viewBox="0 0 24 24"><use href="#icon-send"/></svg>
                    <span>发送</span>
                  </button>
                  <div class="nsfw-row" style="margin-top:0;margin-left:auto;">
                    <span class="nsfw-label">NSFW</span>
                    <div class="nsfw-toggle">
                      <button id="nsfw-on" type="button" title="允许成人内容">ON</button>
                      <button id="nsfw-off" type="button" title="仅限安全内容">OFF</button>
                    </div>
                  </div>
                </div>
              </form>
              <div id="status">准备就绪。</div>
            </div>
          </div>
        </div>
        <div id="evaluation-panel">
          <h3>角色对用户的观感</h3>
          <div id="evaluation-content">
            <div id="evaluation-loading">评估中…</div>
          </div>
        </div>
      </div>
    </div>
    <script>
      function apiHeaders() {
        var h = { 'Content-Type': 'application/json' };
        var k = localStorage.getItem('xai_api_key');
        if (k) h['X-Api-Key'] = k;
        return h;
      }
      function showApiKeyModal() {
        var el = document.getElementById('api-key-input');
        if (el) el.value = localStorage.getItem('xai_api_key') || '';
        document.getElementById('api-key-modal-overlay').classList.add('show');
        if (el) el.focus();
      }
      function hideApiKeyModal() { document.getElementById('api-key-modal-overlay').classList.remove('show'); }
      document.getElementById('btn-set-api-key').addEventListener('click', showApiKeyModal);
      var btnSetApiKeyWelcome = document.getElementById('btn-set-api-key-welcome');
      if (btnSetApiKeyWelcome) btnSetApiKeyWelcome.addEventListener('click', showApiKeyModal);
      document.getElementById('api-key-save').addEventListener('click', function() {
        var v = (document.getElementById('api-key-input').value || '').trim();
        if (v) localStorage.setItem('xai_api_key', v);
        hideApiKeyModal();
      });
      document.getElementById('api-key-cancel').addEventListener('click', hideApiKeyModal);
      document.getElementById('api-key-modal-overlay').addEventListener('click', function(e) {
        if (e.target.id === 'api-key-modal-overlay') hideApiKeyModal();
      });

      const form = document.getElementById('chat-form');
      const input = document.getElementById('msg');
      const chatDiv = document.getElementById('chat');
      const newChatBtn = document.getElementById('new-chat');
      const button = form.querySelector('button[type="submit"]');
      const status = document.getElementById('status');
      const welcomeScreen = document.getElementById('welcome-screen');
      const chatScreen = document.getElementById('chat-screen');
      const historyListEl = document.getElementById('history-list');
      const personPickerScreen = document.getElementById('person-picker-screen');
      const personListEl = document.getElementById('person-list');
      const welcomeTitleEl = document.getElementById('welcome-title');
      const evaluationContentEl = document.getElementById('evaluation-content');
      let selectedPromptFile = null;
      let selectedPromptName = '';
      let currentBotName = '小丙';
      let nsfwAllowed = true;

      function setNsfwUI(on) {
        nsfwAllowed = on;
        document.getElementById('nsfw-on').classList.toggle('active', on);
        document.getElementById('nsfw-off').classList.toggle('active', !on);
      }
      document.getElementById('nsfw-on').addEventListener('click', function() { setNsfwUI(true); });
      document.getElementById('nsfw-off').addEventListener('click', function() { setNsfwUI(false); });
      setNsfwUI(true);

      var EVAL_DIM_LABELS = {
        emotional_intimacy: '情感亲密度',
        possessiveness_jealousy: '占有欲与嫉妒',
        testing_trust_building: '试探与信任构建',
        sexual_attraction_physical_intimacy: '性吸引与身体亲密'
      };
      var EVAL_DIM_ORDER = [
        'emotional_intimacy',
        'possessiveness_jealousy',
        'testing_trust_building',
        'sexual_attraction_physical_intimacy'
      ];

      function renderEvaluation(scores, stageData) {
        if (!scores || typeof scores !== 'object') {
          evaluationContentEl.innerHTML = '<div id="evaluation-empty">暂无评估</div>';
          return;
        }
        var html = '';
        if (stageData) {
          var prevPs = stageData.previous_stage_ps;
          var newPs = stageData.new_stage_ps;
          var downgraded = stageData.stage_downgraded;
          var effStage = stageData.effective_stage;
          if (prevPs != null || newPs != null) {
            html += '<div class="eval-stage-ps">';
            if (prevPs != null) html += '<div class="eval-ps-line">上个阶段 PS: ' + prevPs + '</div>';
            if (newPs != null) html += '<div class="eval-ps-line">本阶段 PS: ' + newPs + '</div>';
            if (downgraded && effStage != null) html += '<div class="eval-downgrade">已降级至阶段' + effStage + '</div>';
            html += '</div>';
          }
        }
        EVAL_DIM_ORDER.forEach(function(key) {
          var val = scores[key];
          if (val == null) val = 0;
          var label = EVAL_DIM_LABELS[key] || key;
          html += '<div class="eval-dim">';
          html += '<div class="eval-dim-name">' + label + '</div>';
          html += '<div class="eval-dim-bar-wrap"><div class="eval-dim-bar" style="width:' + (val * 10) + '%"></div></div>';
          html += '<div class="eval-dim-score">' + val + ' / 10</div>';
          html += '</div>';
        });
        if (stageData && (stageData.effective_stage != null && stageData.effective_stage !== undefined)) {
          html += '<div class="eval-system-prompt-stage">System Prompt 阶段: 阶段' + stageData.effective_stage + '</div>';
        }
        evaluationContentEl.innerHTML = html;
      }

      function setEvaluationLoading() {
        evaluationContentEl.innerHTML = '<div id="evaluation-loading">评估中…</div>';
      }

      function setEvaluationError(msg) {
        evaluationContentEl.innerHTML = '<div id="evaluation-error">' + (msg || '评估失败') + '</div>';
      }

      function fetchEvaluation() {
        setEvaluationLoading();
        fetch('/evaluate', {
          method: 'POST',
          headers: apiHeaders(),
          body: JSON.stringify({})
        })
          .then(function(res) {
            if (res.status === 401) {
              res.json().then(function(d) { showApiKeyModal(); status.textContent = d.message || '请设置 xAI API Key'; });
              return null;
            }
            return res.json();
          })
          .then(function(data) {
            if (!data) return;
            if (data.scores) {
              var stageData = {
                previous_stage_ps: data.previous_stage_ps,
                new_stage_ps: data.new_stage_ps,
                stage_downgraded: data.stage_downgraded,
                effective_stage: data.effective_stage
              };
              renderEvaluation(data.scores, stageData);
            } else setEvaluationEmpty();
          })
          .catch(function(err) {
            console.error(err);
            setEvaluationError('评估失败，请稍后再试');
          });
      }

      function setEvaluationEmpty() {
        evaluationContentEl.innerHTML = '<div id="evaluation-empty">暂无评估</div>';
      }

      function restoreEvaluationState() {
        fetch('/evaluation-state').then(function(res) { return res.json(); }).then(function(data) {
          if (data.scores) {
            var stageData = {
              previous_stage_ps: data.previous_stage_ps,
              new_stage_ps: data.new_stage_ps,
              stage_downgraded: data.stage_downgraded,
              effective_stage: data.effective_stage
            };
            renderEvaluation(data.scores, stageData);
          } else setEvaluationEmpty();
        }).catch(function() { setEvaluationEmpty(); });
      }

      function setCurrentBotName(name) {
        currentBotName = name || '小丙';
        var nameEl = document.getElementById('character-display-name');
        if (nameEl) nameEl.textContent = currentBotName;
      }

      function updateCharacterVisual() {
        var noImg = document.getElementById('character-no-image');
        var img = document.getElementById('character-img');
        var bar = document.getElementById('character-name-bar');
        if (!noImg || !img || !bar) return;
        var pf = selectedPromptFile;
        if (!pf || typeof pf !== 'string') {
          noImg.style.display = 'block';
          noImg.textContent = '选择角色后将显示立绘或动图';
          img.style.display = 'none';
          bar.style.display = 'none';
          return;
        }
        var basename = pf.replace(/\.txt$/i, '');
        bar.style.display = 'block';
        var nameEl = document.getElementById('character-display-name');
        if (nameEl) nameEl.textContent = currentBotName || basename;
        img.style.display = 'none';
        img.onload = function() {
          noImg.style.display = 'none';
          img.style.display = 'block';
        };
        var characterImageUrl = '/character-image/' + encodeURIComponent(basename) + '?t=' + Date.now();
        img.onerror = function() {
          if (img.src && img.src.indexOf('session-display-image') !== -1) {
            img.src = characterImageUrl;
          } else {
            noImg.style.display = 'block';
            noImg.textContent = '暂无立绘，可在 systemprompt 目录添加同名 .png/.jpg/.gif';
            img.style.display = 'none';
          }
        };
        fetch('/current-session').then(function(r) { return r.json(); }).then(function(data) {
          if (data.display_image === 'generated') {
            img.src = '/session-display-image?t=' + Date.now();
          } else {
            img.src = characterImageUrl;
          }
        }).catch(function() {
          img.src = characterImageUrl;
        });
      }

      function showPersonPicker() {
        personPickerScreen.style.display = 'block';
        welcomeScreen.classList.remove('active');
        welcomeScreen.style.display = 'none';
        chatScreen.classList.remove('active');
      }

      function showWelcomeScreen() {
        personPickerScreen.style.display = 'none';
        welcomeScreen.style.display = 'block';
        welcomeScreen.classList.add('active');
        chatScreen.classList.remove('active');
      }

      function fetchSystemPrompt() {
        var el = document.getElementById('system-prompt-content');
        if (!el) return;
        el.textContent = '加载中…';
        fetch('/current-session-prompt').then(function(res) { return res.json(); }).then(function(data) {
          el.textContent = (data.system_prompt || '').trim() || '（无内容）';
        }).catch(function() {
          el.textContent = '加载失败';
        });
      }

      function showChatScreen() {
        personPickerScreen.style.display = 'none';
        welcomeScreen.style.display = 'none';
        welcomeScreen.classList.remove('active');
        chatScreen.classList.add('active');
        updateCharacterVisual();
        scrollChatToBottom();
      }

      function formatHistoryDate(ts) {
        if (!ts) return '';
        const d = new Date(ts * 1000);
        const now = new Date();
        const sameDay = d.getDate() === now.getDate() && d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
        if (sameDay) return '今天 ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
        return d.toLocaleDateString('zh-CN') + ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
      }

      let pendingDeleteSessionId = null;

      function renderHistoryList(sessions) {
        historyListEl.innerHTML = '';
        if (!sessions || sessions.length === 0) {
          const empty = document.createElement('div');
          empty.className = 'history-empty';
          empty.textContent = '暂无历史对话';
          historyListEl.appendChild(empty);
          return;
        }
        sessions.forEach(function(s) {
          const row = document.createElement('div');
          row.className = 'history-item-row';
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'history-item';
          btn.dataset.sessionId = s.session_id;
          const nameSpan = document.createElement('span');
          nameSpan.className = 'name';
          nameSpan.textContent = s.name || ('对话 ' + s.session_id.slice(-8));
          const metaSpan = document.createElement('span');
          metaSpan.className = 'meta';
          metaSpan.textContent = formatHistoryDate(s.updated_at) + ' · ' + s.session_id.slice(0, 8) + '…';
          btn.appendChild(nameSpan);
          btn.appendChild(document.createElement('br'));
          btn.appendChild(metaSpan);
          btn.addEventListener('click', function() {
            const id = btn.dataset.sessionId;
            fetch('/switch-session', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ session_id: id })
            }).then(function(res) {
              if (!res.ok) throw new Error('HTTP ' + res.status);
              return fetch('/history?session_id=' + encodeURIComponent(id));
            }).then(function(res) {
              if (!res.ok) throw new Error('HTTP ' + res.status);
              return res.json();
            }).then(function(data) {
              return fetch('/current-session').then(function(r) { return r.json(); }).then(function(cur) {
                if (cur.prompt_file) selectedPromptFile = cur.prompt_file;
                setCurrentBotName(cur.prompt_name);
                chatDiv.innerHTML = '';
                (data.history || []).forEach(function(msg) {
                  addMessage(msg.role, msg.content);
                });
                showChatScreen();
                restoreEvaluationState();
                input.focus();
              });
            }).catch(function(err) {
              console.error(err);
              alert('切换对话失败，请重试。');
            });
          });
          const delBtn = document.createElement('button');
          delBtn.type = 'button';
          delBtn.className = 'history-item-delete';
          delBtn.title = '删除';
          delBtn.innerHTML = '<svg class="icon" viewBox="0 0 24 24"><use href="#icon-trash"/></svg>';
          delBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            pendingDeleteSessionId = s.session_id;
            document.getElementById('delete-modal-overlay').classList.add('show');
          });
          row.appendChild(btn);
          row.appendChild(delBtn);
          historyListEl.appendChild(row);
        });
      }

      document.getElementById('delete-modal-cancel').addEventListener('click', function() {
        pendingDeleteSessionId = null;
        document.getElementById('delete-modal-overlay').classList.remove('show');
      });
      document.getElementById('delete-modal-confirm').addEventListener('click', function() {
        if (!pendingDeleteSessionId) {
          document.getElementById('delete-modal-overlay').classList.remove('show');
          return;
        }
        const id = pendingDeleteSessionId;
        pendingDeleteSessionId = null;
        document.getElementById('delete-modal-overlay').classList.remove('show');
        fetch('/session/' + encodeURIComponent(id), { method: 'DELETE' })
          .then(function(res) {
            if (!res.ok) throw new Error('HTTP ' + res.status);
            loadSessionsForPerson();
          })
          .catch(function(err) {
            console.error(err);
            alert('删除失败，请重试。');
          });
      });
      document.getElementById('delete-modal-overlay').addEventListener('click', function(e) {
        if (e.target.id === 'delete-modal-overlay') {
          pendingDeleteSessionId = null;
          document.getElementById('delete-modal-overlay').classList.remove('show');
        }
      });

      function loadSessionsForPerson() {
        var url = '/sessions';
        if (selectedPromptFile) url += '?prompt_file=' + encodeURIComponent(selectedPromptFile);
        fetch(url).then(function(res) { return res.json(); }).then(function(data) {
          renderHistoryList(data.sessions || []);
        }).catch(function() {
          historyListEl.innerHTML = '<div class="history-empty">加载失败</div>';
        });
      }

      function renderPersonList(files) {
        personListEl.innerHTML = '';
        if (!files || files.length === 0) {
          personListEl.innerHTML = '<div class="picker-empty">暂无可选角色，请在 systemprompt 目录下添加 .txt 文件</div>';
          return;
        }
        files.forEach(function(f) {
          var btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'person-item';
          btn.innerHTML = '<svg class="icon" viewBox="0 0 24 24"><use href="#icon-user"/></svg><span>' + (f.name || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</span>';
          btn.dataset.id = f.id;
          btn.dataset.name = f.name;
          btn.addEventListener('click', function() {
            selectedPromptFile = btn.dataset.id;
            selectedPromptName = btn.dataset.name || selectedPromptFile.replace(/\.txt$/, '');
            welcomeTitleEl.textContent = '和 ' + selectedPromptName + ' 聊天';
            showWelcomeScreen();
            loadSessionsForPerson();
          });
          personListEl.appendChild(btn);
        });
      }

      fetch('/prompt-files').then(function(res) { return res.json(); }).then(function(data) {
        renderPersonList(data.prompt_files || []);
      }).catch(function() {
        personListEl.innerHTML = '<div class="picker-empty">加载失败</div>';
      });

      function tryShowCurrentSessionChat() {
        fetch('/current-session').then(function(res) { return res.json(); }).then(function(cur) {
          if (!cur || !cur.session_id || !cur.prompt_file) return;
          selectedPromptFile = cur.prompt_file;
          setCurrentBotName(cur.prompt_name);
          return fetch('/history');
        }).then(function(res) {
          if (!res || !res.ok) return null;
          return res.json();
        }).then(function(data) {
          if (!data || !data.history || !Array.isArray(data.history)) return;
          chatDiv.innerHTML = '';
          data.history.forEach(function(msg) {
            addMessage(msg.role, msg.content);
          });
          showChatScreen();
          restoreEvaluationState();
          requestAnimationFrame(function() { scrollChatToBottom(); });
        }).catch(function() {});
      }
      tryShowCurrentSessionChat();

      document.getElementById('btn-back-to-picker').addEventListener('click', function() {
        showPersonPicker();
      });

      document.getElementById('btn-new-chat-welcome').addEventListener('click', function() {
        var body = {};
        if (selectedPromptFile) body.prompt_file = selectedPromptFile;
        fetch('/new', {
          method: 'POST',
          headers: apiHeaders(),
          body: JSON.stringify(body)
        })
          .then(function(res) {
            if (!res.ok) {
              if (res.status === 401) {
                res.json().then(function(d) { showApiKeyModal(); status.textContent = d.message || '请设置 xAI API Key'; });
                return;
              }
              throw new Error('HTTP ' + res.status);
            }
            return res.json();
          })
          .then(function(data) {
            if (!data) return;
            setCurrentBotName(data.prompt_name);
            chatDiv.innerHTML = '';
            showChatScreen();
            fetchEvaluation();
            input.value = '';
            input.focus();
          })
          .catch(function(err) {
            console.error(err);
            alert('开启新对话失败，请重试。');
          });
      });

      document.getElementById('btn-history-welcome').addEventListener('click', function() {
        historyListEl.scrollIntoView({ behavior: 'smooth' });
      });

      document.getElementById('back-home').addEventListener('click', function() {
        showPersonPicker();
      });

      function addMessage(role, text) {
        const wrapper = document.createElement('div');
        wrapper.className = 'msg ' + role;

        const inner = document.createElement('div');
        inner.className = 'msg-inner';

        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = role === 'user' ? '你' : currentBotName;

        const content = document.createElement('div');
        content.textContent = text;

        inner.appendChild(label);
        inner.appendChild(content);
        wrapper.appendChild(inner);
        chatDiv.appendChild(wrapper);
        scrollChatToBottom();
      }

      function scrollChatToBottom() {
        var el = document.getElementById('chat');
        if (!el) return;
        function doScroll() {
          var lastMsg = el.lastElementChild;
          if (lastMsg) {
            lastMsg.scrollIntoView({ block: 'end', behavior: 'auto' });
          } else {
            el.scrollTop = el.scrollHeight;
          }
        }
        doScroll();
        requestAnimationFrame(doScroll);
      }

      var ttsGeneration = 0;
      var ttsAbortController = null;
      var ttsCurrentAudio = null;

      function playTts(text) {
        ttsGeneration += 1;
        var myGen = ttsGeneration;
        if (ttsAbortController) ttsAbortController.abort();
        ttsAbortController = new AbortController();
        if (ttsCurrentAudio) {
          ttsCurrentAudio.pause();
          ttsCurrentAudio.currentTime = 0;
          ttsCurrentAudio = null;
        }
        status.textContent = '正在生成语音…';
        return fetch('/tts', {
          method: 'POST',
          headers: apiHeaders(),
          body: JSON.stringify({ text: text }),
          signal: ttsAbortController.signal
        }).then(function (res) {
          if (myGen !== ttsGeneration) return null;
          if (!res.ok) {
            if (res.status === 401) {
              res.json().then(function(d) { showApiKeyModal(); status.textContent = d.message || '请设置 xAI API Key'; });
              return Promise.reject(new Error('missing_api_key'));
            }
            return res.text().then(function (t) {
              var err = 'TTS ' + res.status;
              try { var d = JSON.parse(t); if (d.error) err = d.error; } catch (e) {}
              throw new Error(err);
            });
          }
          return res.blob();
        }).then(function (blob) {
          if (myGen !== ttsGeneration) return;
          if (!blob || (blob.size !== undefined && blob.size < 100)) {
            if (blob && blob.size < 100) throw new Error('语音数据为空');
            return;
          }
          var url = URL.createObjectURL(blob);
          var audio = new Audio(url);
          ttsCurrentAudio = audio;
          audio.onended = function () {
            URL.revokeObjectURL(url);
            if (ttsCurrentAudio === audio) ttsCurrentAudio = null;
            status.textContent = '准备就绪。';
          };
          return audio.play().then(function () {
            if (myGen !== ttsGeneration) { audio.pause(); return; }
            status.textContent = '语音播放中…';
          }).catch(function (err) {
            URL.revokeObjectURL(url);
            if (err.name === 'NotAllowedError') {
              addTtsPlayButton(url, blob);
              status.textContent = '准备就绪。（点击「播放语音」可听回复）';
            } else {
              status.textContent = '准备就绪。';
              throw err;
            }
          });
        }).catch(function (err) {
          if (err.name === 'AbortError' || myGen !== ttsGeneration) return;
          if (err.message === 'missing_api_key') return;
          console.warn('TTS:', err);
          var msg = err.message || String(err);
          status.textContent = '语音生成失败: ' + msg;
          setTimeout(function () { status.textContent = '准备就绪。'; }, 3000);
          fetch('/log-error', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, context: 'TTS' })
          }).catch(function () {});
        });
      }

      function addTtsPlayButton(objectUrl, blob) {
        var lastBot = document.querySelector('#chat .msg.bot:last-child .msg-inner');
        if (!lastBot) return;
        var wrap = document.createElement('div');
        wrap.style.marginTop = '8px';
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = '🔊 点击播放语音';
        btn.style.cssText = 'font-size:12px;padding:4px 8px;border-radius:6px;border:1px solid var(--border);background:var(--bg);cursor:pointer;';
        btn.onclick = function () {
          var a = new Audio(objectUrl);
          a.onended = function () { URL.revokeObjectURL(objectUrl); };
          a.play().catch(function () {});
          wrap.remove();
        };
        wrap.appendChild(btn);
        lastBot.appendChild(wrap);
      }

      function startNewChat() {
        newChatBtn.disabled = true;
        button.disabled = true;
        status.textContent = '正在开始新对话…';
        var body = {};
        fetch('/current-session').then(function(res) { return res.json(); }).then(function(data) {
          if (data.prompt_file) body.prompt_file = data.prompt_file;
          return fetch('/new', {
            method: 'POST',
            headers: apiHeaders(),
            body: JSON.stringify(body)
          });
        }).catch(function() {
          if (selectedPromptFile) body.prompt_file = selectedPromptFile;
          return fetch('/new', {
            method: 'POST',
            headers: apiHeaders(),
            body: JSON.stringify(body)
          });
        }).then(function(res) {
          if (!res.ok) {
            if (res.status === 401) {
              res.json().then(function(d) { showApiKeyModal(); status.textContent = d.message || '请设置 xAI API Key'; });
              return null;
            }
            throw new Error('HTTP ' + res.status);
          }
          return res.json();
        }).then(function(data) {
          if (!data) return;
          setCurrentBotName(data.prompt_name);
          chatDiv.innerHTML = '';
          status.textContent = '新对话已就绪。';
          restoreEvaluationState();
          input.value = '';
          input.focus();
        }).catch(function(err) {
          console.error(err);
          status.textContent = '开启新对话失败，请刷新页面重试。';
        }).finally(function() {
          newChatBtn.disabled = false;
          button.disabled = false;
        });
      }

      newChatBtn.addEventListener('click', () => {
        const overlay = document.getElementById('name-modal-overlay');
        const nameInput = document.getElementById('session-name-input');
        nameInput.value = '';
        overlay.classList.add('show');
        nameInput.focus();
      });

      document.getElementById('name-modal-skip').addEventListener('click', () => {
        document.getElementById('name-modal-overlay').classList.remove('show');
        startNewChat();
      });

      document.getElementById('name-modal-ok').addEventListener('click', async () => {
        const nameInput = document.getElementById('session-name-input');
        const name = (nameInput.value || '').trim();
        document.getElementById('name-modal-overlay').classList.remove('show');
        if (name) {
          newChatBtn.disabled = true;
          button.disabled = true;
          status.textContent = '正在保存对话名称…';
          try {
            const res = await fetch('/name-session', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ name: name })
            });
            if (!res.ok) throw new Error('HTTP ' + res.status);
          } catch (err) {
            console.error(err);
          }
        }
        startNewChat();
      });

      document.getElementById('session-name-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          document.getElementById('name-modal-ok').click();
        }
        if (e.key === 'Escape') {
          document.getElementById('name-modal-overlay').classList.remove('show');
        }
      });

      document.getElementById('name-modal-overlay').addEventListener('click', (e) => {
        if (e.target.id === 'name-modal-overlay') {
          document.getElementById('name-modal-overlay').classList.remove('show');
        }
      });

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = input.value.trim();
        if (!text) return;

        addMessage('user', text);
        input.value = '';
        button.disabled = true;
        newChatBtn.disabled = true;
        status.textContent = currentBotName + '思考中…';

        try {
          const res = await fetch('/chat', {
            method: 'POST',
            headers: apiHeaders(),
            body: JSON.stringify({ message: text, nsfw_allowed: nsfwAllowed })
          });
          if (!res.ok) {
            if (res.status === 401) {
              var d = await res.json();
              showApiKeyModal();
              status.textContent = d.message || '请设置 xAI API Key';
              return;
            }
            throw new Error('HTTP ' + res.status);
          }
          const data = await res.json();
          const replyText = data.reply || '(空回复)';
          addMessage('bot', replyText);
          if (data.display_image_updated) updateCharacterVisual();
          status.textContent = '准备就绪。';
          fetchEvaluation();
          // 同时请求 TTS 并播放语音（女声 Ara）
          if (replyText && replyText !== '(空回复)') {
            playTts(replyText).catch(function (err) { console.warn('TTS:', err); });
          }
        } catch (err) {
          console.error(err);
          addMessage('bot', '和服务器对话时出错了，请稍后再试。');
          status.textContent = '发生错误，请刷新页面重试。';
        } finally {
          button.disabled = false;
          newChatBtn.disabled = false;
          input.focus();
        }
      });
    </script>
  </body>
</html>
"""


@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template_string(INDEX_HTML)


@app.route("/prompt-files", methods=["GET"])
def get_prompt_files():
    """List system prompt files in systemprompt/ (for 'choose who to chat with')."""
    files = list_prompt_files(PROMPT_DIR)
    return jsonify({"prompt_files": files})


@app.route("/session-display-image")
def session_display_image():
    """
    当前会话的展示图：若 state.display_image 为 generated 且已存在生成图文件，返回该图；
    否则 404，前端回退到 character-image。
    """
    state = read_json(STATE_PATH) or {}
    if not isinstance(state, dict) or state.get("display_image") != "generated":
        return jsonify({"error": "no generated image"}), 404
    path = os.path.join(LOG_DIR, DISPLAY_IMAGE_FILENAME.format(SESSION_ID))
    if not os.path.isfile(path):
        return jsonify({"error": "file not found"}), 404
    return send_file(path, mimetype="image/png", last_modified=os.path.getmtime(path))


@app.route("/character-image/<path:basename>")
def character_image(basename):
    """
    按角色名返回 systemprompt 目录下同名的图/动图（.png / .jpg / .gif），便于按 system prompt 配置立绘。
    basename 为去掉 .txt 后的文件名，如 晴雪。
    """
    basename = (basename or "").strip().rstrip("/")
    if not basename or ".." in basename or "/" in basename or basename.startswith("."):
        return jsonify({"error": "invalid basename"}), 400
    for ext in (".png", ".jpg", ".jpeg", ".gif"):
        path = os.path.join(PROMPT_DIR, basename + ext)
        if os.path.isfile(path):
            mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in (".jpg", ".jpeg") else "image/gif")
            return send_file(path, mimetype=mime)
    return jsonify({"error": "not found"}), 404


@app.route("/current-session", methods=["GET"])
def current_session():
    """Return current session_id, prompt_file, and prompt_name (for back-home and chat UI label)."""
    state = read_json(STATE_PATH) or {}
    if not isinstance(state, dict):
        state = {}
    pf = state.get("prompt_file")
    prompt_name = (pf[:-4] if pf and isinstance(pf, str) and pf.endswith(".txt") else (pf or "")) or None
    display_image = state.get("display_image") or "prompt"
    return jsonify({
        "session_id": SESSION_ID,
        "prompt_file": pf,
        "prompt_name": prompt_name,
        "display_image": display_image,
    })


@app.route("/current-session-prompt", methods=["GET"])
def current_session_prompt():
    """Return the effective system prompt text for the current session (for display in UI)."""
    state = read_json(STATE_PATH) or {}
    if not isinstance(state, dict):
        state = {}
    s_val = float(state.get("evaluation_S", 0))
    eff = state.get("effective_stage")
    if eff is None:
        eff = 1 if s_val < 3 else (2 if s_val < 6 else 3)
    pf = state.get("prompt_file")
    content = ""
    if pf:
        prompt_path = os.path.join(PROMPT_DIR, pf)
        if os.path.isfile(prompt_path):
            try:
                stage1, stage2, stage3 = read_system_prompt_staged(prompt_path)
                if eff == 1:
                    content = stage1
                elif eff == 2:
                    content = stage2 if stage2 else stage1
                else:
                    content = stage3 if stage3 else stage1
            except Exception:
                content = read_system_prompt(prompt_path)
        else:
            content = "（未找到文件）"
    else:
        content = read_system_prompt(SYSTEM_PROMPT_PATH)
    return jsonify({"system_prompt": content or "（无内容）"})


@app.route("/sessions", methods=["GET"])
def get_sessions():
    """List saved chat sessions; optional ?prompt_file=xxx filters by that person."""
    prompt_file = request.args.get("prompt_file") or None
    sessions = list_sessions(LOG_DIR, prompt_file=prompt_file)
    return jsonify({"sessions": sessions})


@app.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a chat session (remove its .state.json and .jsonl files)."""
    if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id:
        return jsonify({"error": "invalid session_id"}), 400
    state_path = os.path.join(LOG_DIR, f"{session_id}.state.json")
    log_path = os.path.join(LOG_DIR, f"{session_id}.jsonl")
    display_path = os.path.join(LOG_DIR, DISPLAY_IMAGE_FILENAME.format(session_id))
    deleted = False
    try:
        if os.path.isfile(state_path):
            os.remove(state_path)
            deleted = True
        if os.path.isfile(log_path):
            os.remove(log_path)
            deleted = True
        if os.path.isfile(display_path):
            os.remove(display_path)
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    if not deleted:
        return jsonify({"error": "session not found"}), 404
    return jsonify({"ok": True})


@app.route("/switch-session", methods=["POST"])
def switch_session():
    """Switch current session to the given one; only update session state，下次 /chat 会用请求的 client 续接。"""
    global SESSION_ID, CHAT_LOG_PATH, STATE_PATH, previous_response_id
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    state_path = os.path.join(LOG_DIR, f"{session_id}.state.json")
    if not os.path.isfile(state_path):
        return jsonify({"error": "session not found"}), 404
    state = read_json(state_path) or {}
    if not isinstance(state, dict):
        state = {}
    prev_id = state.get("previous_response_id") or None
    SESSION_ID = session_id
    CHAT_LOG_PATH = os.path.join(LOG_DIR, f"{session_id}.jsonl")
    STATE_PATH = state_path
    previous_response_id = prev_id
    with open(LAST_SESSION_PATH, "w", encoding="utf-8") as f:
        f.write(SESSION_ID)
    return jsonify({"ok": True, "session_id": SESSION_ID})


@app.route("/history")
def get_history():
    """Get message history for a session (for displaying in UI). session_id defaults to current."""
    session_id = request.args.get("session_id") or SESSION_ID
    log_path = os.path.join(LOG_DIR, f"{session_id}.jsonl")
    history = read_history_jsonl(log_path)
    return jsonify({"session_id": session_id, "history": history})


@app.route("/evaluation-state", methods=["GET"])
def get_evaluation_state():
    """Return the current session's last saved evaluation (four dimensions, S, PS) from state for restore."""
    state = read_json(STATE_PATH) or {}
    if not isinstance(state, dict):
        state = {}
    stored = state.get("evaluation_dimensions")
    if not stored or len(stored) != len(EVAL_DIM_KEYS):
        return jsonify({"session_id": SESSION_ID, "scores": None})
    scores = {k: round(float(stored[i]), 1) for i, k in enumerate(EVAL_DIM_KEYS)}
    return jsonify({
        "session_id": SESSION_ID,
        "scores": scores,
        "evaluation_S": state.get("evaluation_S"),
        "previous_stage_ps": state.get("previous_stage_ps"),
        "new_stage_ps": state.get("last_stage_ps"),
        "stage_downgraded": state.get("stage_downgraded", False),
        "effective_stage": state.get("effective_stage"),
    })


@app.route("/evaluate", methods=["POST"])
def evaluate_session():
    """
    Run the evaluation agent in a new session (separate from chat). Analyzes the given
    session's chat log and returns 5 dimension scores (0-10) for how the character views the user.
    """
    req_client = get_client_for_request()
    if not req_client:
        return jsonify({"error": "missing_api_key", "message": "请设置 xAI API Key"}), 401
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip() or SESSION_ID
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        return jsonify({"error": "invalid session_id"}), 400
    log_path = os.path.join(LOG_DIR, f"{session_id}.jsonl")
    state_path = os.path.join(LOG_DIR, f"{session_id}.state.json")
    if not os.path.isfile(log_path):
        return jsonify({"error": "session not found", "scores": None}), 404
    state = read_json(state_path) or {}
    if not isinstance(state, dict):
        state = {}
    pf = state.get("prompt_file")
    character_name = (pf[:-4] if pf and isinstance(pf, str) and pf.endswith(".txt") else (pf or "")) or "角色"
    try:
        history = read_history_jsonl(log_path)
        round_count = len(history) // 2
        # PS 只在每 5 轮（5、10、15…）计算一次：非 5 的倍数或不足 5 轮时不调 LLM，直接返回当前 state
        if round_count < 5 or round_count % 5 != 0:
            state["updated_at"] = time.time()
            write_json(state_path, state)
            stored = state.get("evaluation_dimensions")
            if stored and len(stored) == len(EVAL_DIM_KEYS):
                scores_to_return = {k: round(float(stored[i]), 1) for i, k in enumerate(EVAL_DIM_KEYS)}
            else:
                scores_to_return = None
            payload = {
                "ok": True,
                "session_id": session_id,
                "scores": scores_to_return,
                "previous_stage_ps": state.get("previous_stage_ps"),
                "new_stage_ps": state.get("last_stage_ps"),
                "stage_downgraded": state.get("stage_downgraded", False),
                "effective_stage": state.get("effective_stage"),
            }
            return jsonify(payload)
        max_retries = 3
        scores = None
        for _ in range(max_retries):
            scores = run_evaluation(req_client, log_path, character_name)
            if scores:
                break
        if scores:
            s_val = sum(float(scores.get(k, 5.0)) for k in EVAL_DIM_KEYS) / len(EVAL_DIM_KEYS)
            state["evaluation_S"] = round(s_val, 2)
            state["evaluation_dimensions"] = [float(scores.get(k, 5.0)) for k in EVAL_DIM_KEYS]
            stage_from_s = 1 if s_val < 3 else (2 if s_val < 6 else 3)
            previous_stage_ps = state.get("last_stage_ps")
            ps_new = round(s_val, 2)
            new_stage_ps = ps_new
            stage_downgraded = False
            effective_stage = stage_from_s
            if previous_stage_ps is not None and ps_new < previous_stage_ps:
                effective_stage = max(1, effective_stage - 1)
                stage_downgraded = True
            state["previous_stage_ps"] = previous_stage_ps
            state["last_stage_ps"] = ps_new
            state["effective_stage"] = effective_stage
            state["stage_downgraded"] = stage_downgraded
            state["updated_at"] = time.time()
            write_json(state_path, state)
            scores_to_return = {k: round(float(state["evaluation_dimensions"][i]), 1) for i, k in enumerate(EVAL_DIM_KEYS)}
            payload = {
                "ok": True,
                "session_id": session_id,
                "scores": scores_to_return,
                "previous_stage_ps": state.get("previous_stage_ps"),
                "new_stage_ps": state.get("last_stage_ps"),
                "stage_downgraded": state.get("stage_downgraded", False),
                "effective_stage": state.get("effective_stage"),
            }
            return jsonify(payload)
        # 重试后仍无完整数据：若有上一次各维度数值，则按上一次该维度的数值返回
        stored = state.get("evaluation_dimensions")
        if stored and len(stored) == len(EVAL_DIM_KEYS):
            scores = {k: round(float(stored[i]), 1) for i, k in enumerate(EVAL_DIM_KEYS)}
            payload = {
                "ok": True,
                "session_id": session_id,
                "scores": scores,
                "previous_stage_ps": state.get("previous_stage_ps"),
                "new_stage_ps": state.get("last_stage_ps"),
                "stage_downgraded": state.get("stage_downgraded", False),
                "effective_stage": state.get("effective_stage"),
            }
            return jsonify(payload)
        return jsonify({"ok": False, "error": "评估返回不完整（缺少维度），已重试 3 次，请稍后再试", "scores": None}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "scores": None}), 500


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Handle chat messages from the browser."""
    global previous_response_id, chat

    req_client = get_client_for_request()
    if not req_client:
        return jsonify({"error": "missing_api_key", "message": "请设置 xAI API Key（在页面设置中填写，或由部署者配置环境变量 XAI_API_KEY）"}), 401

    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()
    nsfw_allowed = data.get("nsfw_allowed", True)
    if nsfw_allowed is None:
        nsfw_allowed = True

    if not user_input:
        return jsonify({"reply": "你什么都没说呀，我听不到。"}), 400

    # 按 effective_stage 选择阶段人设（可因 PS 降级）；无则按 S 推导
    message_to_model = user_input
    state_for_chat = read_json(STATE_PATH) or {}
    pf = state_for_chat.get("prompt_file") if isinstance(state_for_chat, dict) else None
    if pf:
        prompt_path = os.path.join(PROMPT_DIR, pf)
        if os.path.isfile(prompt_path):
            try:
                stage1, stage2, stage3 = read_system_prompt_staged(prompt_path)
                s_val = float(state_for_chat.get("evaluation_S", 0))
                eff = state_for_chat.get("effective_stage")
                if eff is None:
                    eff = 1 if s_val < 3 else (2 if s_val < 6 else 3)
                if eff == 1:
                    stage_prefix = f"[当前总评值 S={s_val:.1f}，阶段1。]\n\n"
                elif eff == 2:
                    stage_prefix = f"[当前总评值 S={s_val:.1f}，阶段2。请按以下人设回复：\n\n{stage2}\n]\n\n" if stage2 else f"[当前总评值 S={s_val:.1f}。]\n\n"
                else:
                    stage_prefix = f"[当前总评值 S={s_val:.1f}，阶段3。请按以下人设回复：\n\n{stage3}\n]\n\n" if stage3 else f"[当前总评值 S={s_val:.1f}。]\n\n"
                message_to_model = stage_prefix + message_to_model
                dims = state_for_chat.get("evaluation_dimensions")
                if dims and len(dims) == len(EVAL_DIM_KEYS) and float(dims[0]) < 7:
                    nsfw_allowed = False
            except Exception:
                pass
    if not nsfw_allowed:
        message_to_model = "[Keep your reply SFW; avoid explicit sexual or adult content.] " + message_to_model

    append_jsonl(
        CHAT_LOG_PATH,
        {
            "type": "message",
            "role": "user",
            "timestamp": time.time(),
            "content": user_input,
        },
    )

    # 使用本次请求的 client 创建/续接对话
    if previous_response_id is None:
        prompt_content = system_prompt
        if pf:
            prompt_path = os.path.join(PROMPT_DIR, pf)
            if os.path.isfile(prompt_path):
                try:
                    stage1, stage2, stage3 = read_system_prompt_staged(prompt_path)
                    prompt_content = stage1
                except Exception:
                    pass
        chat_req = req_client.chat.create(model="grok-4-1-fast-reasoning", store_messages=True, tools=[])
        chat_req.append(system(prompt_content))
        chat_req.append(user(message_to_model))
        response = chat_req.sample()
    else:
        chat_continue = req_client.chat.create(
            model="grok-4-1-fast-reasoning",
            previous_response_id=previous_response_id,
            store_messages=True,
            tools=[],
        )
        chat_continue.append(user(message_to_model))
        response = chat_continue.sample()

    previous_response_id = response.id
    state = read_json(STATE_PATH) or {}
    if not isinstance(state, dict):
        state = {}
    state.update({
        "session_id": SESSION_ID,
        "previous_response_id": previous_response_id,
        "updated_at": time.time(),
        "model": "grok-4-1-fast-reasoning",
    })
    write_json(STATE_PATH, state)
    append_jsonl(
        CHAT_LOG_PATH,
        {
            "type": "message",
            "role": "assistant",
            "timestamp": time.time(),
            "content": response.content,
            "response_id": getattr(response, "id", None),
        },
    )
    # 每三轮根据对话生成 16:9 漫画风格图并替换展示
    display_image_updated = False
    bot_count = count_assistant_messages(CHAT_LOG_PATH)
    if bot_count >= 3 and bot_count % 3 == 0:
        if _generate_display_image_from_chat(CHAT_LOG_PATH, STATE_PATH, SESSION_ID, LOG_DIR, client=req_client):
            display_image_updated = True
    return jsonify({"reply": response.content, "display_image_updated": display_image_updated})


@app.route("/tts", methods=["POST"])
def tts_endpoint():
    """
    Synthesize speech from text using Grok Voice Agent API (female voice Ara).
    Body: { "text": "..." }. Returns WAV audio for browser playback.
    Requires: pip install websockets
    """
    tts_api_key = get_api_key_from_request()
    if not tts_api_key:
        return jsonify({"error": "missing_api_key", "message": "请设置 xAI API Key"}), 401
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "missing or empty text"}), 400
    try:
        wav_bytes = asyncio.run(_tts_via_voice_api(text, tts_api_key))
    except Exception as e:
        append_error_log("TTS", str(e))
        return jsonify({"error": str(e)}), 500
    if not wav_bytes:
        append_error_log("TTS", "no audio generated")
        return jsonify({"error": "no audio generated"}), 502
    return Response(wav_bytes, mimetype="audio/wav", headers={"Content-Disposition": "inline; filename=tts.wav"})


@app.route("/log-error", methods=["POST"])
def log_error():
    """Append client-reported error to error_messages.txt. Body: { \"message\": \"...\", \"context\": \"...\" }."""
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip() or "(empty)"
    context = (data.get("context") or "").strip() or "Frontend"
    append_error_log(context, message)
    return jsonify({"ok": True})


@app.route("/name-session", methods=["POST"])
def name_session():
    """Save a display name for the current chat session (called before starting a new one)."""
    global STATE_PATH
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": True})
    state = read_json(STATE_PATH) or {}
    if not isinstance(state, dict):
        state = {}
    state["name"] = name
    state["updated_at"] = time.time()
    write_json(STATE_PATH, state)
    return jsonify({"ok": True, "name": name})


@app.route("/new", methods=["POST"])
def new_chat():
    """Start a brand new chat session. Optional body: { "prompt_file": "xxx.txt" } to use that person's system prompt."""
    global SESSION_ID, CHAT_LOG_PATH, STATE_PATH, previous_response_id, chat

    req_client = get_client_for_request()
    c = req_client or default_client
    if not c:
        return jsonify({"error": "missing_api_key", "message": "请设置 xAI API Key"}), 401

    data = request.get_json(silent=True) or {}
    prompt_file = (data.get("prompt_file") or "").strip() or None
    prompt_content = system_prompt
    if prompt_file:
        prompt_path = os.path.join(PROMPT_DIR, prompt_file)
        if os.path.isfile(prompt_path):
            stage1, stage2, stage3 = read_system_prompt_staged(prompt_path)
            prompt_content = stage1
        else:
            prompt_file = None

    SESSION_ID = f"{int(time.time())}"
    CHAT_LOG_PATH = os.path.join(LOG_DIR, f"{SESSION_ID}.jsonl")
    STATE_PATH = os.path.join(LOG_DIR, f"{SESSION_ID}.state.json")
    previous_response_id = None

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LAST_SESSION_PATH, "w", encoding="utf-8") as f:
        f.write(SESSION_ID)

    chat = c.chat.create(
        model="grok-4-1-fast-reasoning",
        store_messages=True,
        tools=[],
    )
    chat.append(system(prompt_content))

    state = {
        "session_id": SESSION_ID,
        "previous_response_id": None,
        "updated_at": time.time(),
        "model": "grok-4-1-fast-reasoning",
        "evaluation_S": 0,
        "evaluation_dimensions": [0.0, 0.0, 0.0, 0.0],
        "effective_stage": 1,
        "last_stage_ps": None,
        "previous_stage_ps": None,
        "stage_downgraded": False,
        "display_image": "prompt",
    }
    if prompt_file:
        state["prompt_file"] = prompt_file
    write_json(STATE_PATH, state)
    append_jsonl(
        CHAT_LOG_PATH,
        {
            "type": "system",
            "timestamp": time.time(),
            "content": prompt_content,
            "model": "grok-4-1-fast-reasoning",
            "session_id": SESSION_ID,
            "resumed": False,
        },
    )

    prompt_name = (prompt_file[:-4] if prompt_file and prompt_file.endswith(".txt") else (prompt_file or "")) or None
    return jsonify({"ok": True, "session_id": SESSION_ID, "prompt_file": prompt_file, "prompt_name": prompt_name})


if __name__ == "__main__":
    # 访问 http://127.0.0.1:5000 即可在浏览器中聊天
    app.run(host="127.0.0.1", port=5001, debug=True)

