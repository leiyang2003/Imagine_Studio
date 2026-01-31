import os
import json
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from xai_sdk import Client
from xai_sdk.chat import system, user

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


def append_jsonl(path: str, record: dict) -> None:
    """Append one JSON record to a JSONL file (utf-8)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


# Get API key from environment variable (recommended for security)
api_key = os.getenv("XAI_API_KEY")
if not api_key:
    raise RuntimeError("XAI_API_KEY not set in environment or .env file.")

# Initialize the client
client = Client(api_key=api_key, timeout=3600)  # Longer timeout for potential agentic responses

# Load system prompt from file in current directory
SYSTEM_PROMPT_PATH = "SystemPrompt.txt"
system_prompt = read_system_prompt(SYSTEM_PROMPT_PATH)

# Persist chat to local files (single-user).
# - chat_logs/<session_id>.jsonl: append-only transcript
# - chat_logs/<session_id>.state.json: stores previous_response_id so next run can continue
LOG_DIR = os.getenv("CHAT_LOG_DIR", "chat_logs")
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

# Start a new stateful chat object for first turn / fallback.
chat = client.chat.create(
    model="grok-4-1-fast-reasoning",  # Use a reasoning model for agentic potential
    store_messages=True,
    tools=[],  # Add agent tools here if desired
)
chat.append(system(system_prompt))

# Log the system prompt (duplicates are harmless).
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

INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <title>Grok Web Chat</title>
    <style>
      :root {
        color-scheme: light dark;
      }
      body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     "Segoe UI", sans-serif;
        max-width: 800px;
        margin: 40px auto;
        padding: 0 16px;
        background: #0b1020;
        color: #f5f5f5;
      }
      h1 {
        font-weight: 600;
        margin-bottom: 8px;
      }
      p.subtitle {
        margin-top: 0;
        color: #a6accd;
        font-size: 14px;
      }
      #chat-container {
        border-radius: 16px;
        background: radial-gradient(circle at top left, #1f2937, #020617);
        padding: 16px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.8);
      }
      #chat {
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.9);
        padding: 12px 12px 4px;
        height: 420px;
        overflow-y: auto;
        scroll-behavior: smooth;
      }
      .msg {
        margin-bottom: 10px;
        display: flex;
      }
      .msg-inner {
        max-width: 80%;
        padding: 8px 11px;
        border-radius: 10px;
        font-size: 14px;
        line-height: 1.5;
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      .user {
        justify-content: flex-end;
      }
      .user .msg-inner {
        background: linear-gradient(135deg, #22d3ee, #4f46e5);
        color: #0b1120;
        border-bottom-right-radius: 2px;
      }
      .bot {
        justify-content: flex-start;
      }
      .bot .msg-inner {
        background: rgba(30, 64, 175, 0.4);
        border: 1px solid rgba(129, 140, 248, 0.5);
        border-bottom-left-radius: 2px;
      }
      .label {
        font-size: 11px;
        opacity: 0.8;
        margin-bottom: 2px;
      }
      .user .label {
        text-align: right;
      }
      .bot .label {
        text-align: left;
      }
      form {
        margin-top: 8px;
        display: flex;
        gap: 8px;
      }
      input[type=text] {
        flex: 1;
        padding: 10px 12px;
        border-radius: 999px;
        border: 1px solid #1f2937;
        background: rgba(15, 23, 42, 0.95);
        color: #f9fafb;
        font-size: 14px;
        outline: none;
      }
      input[type=text]:focus {
        border-color: #22d3ee;
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.6);
      }
      button {
        padding: 0 18px;
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #4f46e5, #22d3ee);
        color: white;
        cursor: pointer;
        font-weight: 500;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      button:disabled {
        opacity: 0.6;
        cursor: default;
      }
      #new-chat {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.35);
        color: #e2e8f0;
      }
      #new-chat:hover {
        background: rgba(148, 163, 184, 0.18);
      }
      #status {
        margin-top: 6px;
        font-size: 12px;
        color: #a6accd;
      }
    </style>
  </head>
  <body>
    <h1>Grok Web Chat</h1>
    <p class="subtitle">在浏览器里和你的「小丙」聊天，不用再开命令行啦～</p>
    <div id="chat-container">
      <div id="chat"></div>
      <form id="chat-form">
        <input id="msg" type="text" placeholder="说点什么..." autocomplete="off" />
        <button id="new-chat" type="button" title="开始一个全新的对话（不再续上之前上下文）">
          <span>新对话</span>
        </button>
        <button type="submit">
          <span>发送</span>
        </button>
      </form>
      <div id="status">准备就绪。</div>
    </div>
    <script>
      const form = document.getElementById('chat-form');
      const input = document.getElementById('msg');
      const chatDiv = document.getElementById('chat');
      const newChatBtn = document.getElementById('new-chat');
      const button = form.querySelector('button[type="submit"]');
      const status = document.getElementById('status');

      function addMessage(role, text) {
        const wrapper = document.createElement('div');
        wrapper.className = 'msg ' + role;

        const inner = document.createElement('div');
        inner.className = 'msg-inner';

        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = role === 'user' ? '你' : '小丙';

        const content = document.createElement('div');
        content.textContent = text;

        inner.appendChild(label);
        inner.appendChild(content);
        wrapper.appendChild(inner);
        chatDiv.appendChild(wrapper);
        chatDiv.scrollTop = chatDiv.scrollHeight;
      }

      newChatBtn.addEventListener('click', async () => {
        newChatBtn.disabled = true;
        button.disabled = true;
        status.textContent = '正在开始新对话…';
        try {
          const res = await fetch('/new', { method: 'POST' });
          if (!res.ok) throw new Error('HTTP ' + res.status);
          chatDiv.innerHTML = '';
          status.textContent = '新对话已就绪。';
          input.value = '';
          input.focus();
        } catch (err) {
          console.error(err);
          status.textContent = '开启新对话失败，请刷新页面重试。';
        } finally {
          newChatBtn.disabled = false;
          button.disabled = false;
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
        status.textContent = '小丙思考中…';

        try {
          const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
          });
          if (!res.ok) {
            throw new Error('HTTP ' + res.status);
          }
          const data = await res.json();
          addMessage('bot', data.reply || '(空回复)');
          status.textContent = '准备就绪。';
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


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Handle chat messages from the browser."""
    global previous_response_id, chat

    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()

    if not user_input:
        return jsonify({"reply": "你什么都没说呀，我听不到。"}), 400

    # Append user message
    chat.append(user(user_input))
    append_jsonl(
        CHAT_LOG_PATH,
        {
            "type": "message",
            "role": "user",
            "timestamp": time.time(),
            "content": user_input,
        },
    )

    # For the first response
    if previous_response_id is None:
        response = chat.sample()
    else:
        # Continue the stateful conversation
        chat_continue = client.chat.create(
            model="grok-4-1-fast-reasoning",
            previous_response_id=previous_response_id,
            store_messages=True,
            tools=[],  # Same tools as above
        )
        chat_continue.append(user(user_input))
        response = chat_continue.sample()

    previous_response_id = response.id
    write_json(
        STATE_PATH,
        {
            "session_id": SESSION_ID,
            "previous_response_id": previous_response_id,
            "updated_at": time.time(),
            "model": "grok-4-1-fast-reasoning",
        },
    )
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
    return jsonify({"reply": response.content})


@app.route("/new", methods=["POST"])
def new_chat():
    """Start a brand new chat session (clears continuation state)."""
    global SESSION_ID, CHAT_LOG_PATH, STATE_PATH, previous_response_id, chat

    SESSION_ID = f"{int(time.time())}"
    CHAT_LOG_PATH = os.path.join(LOG_DIR, f"{SESSION_ID}.jsonl")
    STATE_PATH = os.path.join(LOG_DIR, f"{SESSION_ID}.state.json")
    previous_response_id = None

    # Mark this as the latest session so next run continues from here by default.
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LAST_SESSION_PATH, "w", encoding="utf-8") as f:
        f.write(SESSION_ID)

    # Fresh chat object (first message will not reference old context)
    chat = client.chat.create(
        model="grok-4-1-fast-reasoning",
        store_messages=True,
        tools=[],
    )
    chat.append(system(system_prompt))

    # Write initial state + log
    write_json(
        STATE_PATH,
        {
            "session_id": SESSION_ID,
            "previous_response_id": None,
            "updated_at": time.time(),
            "model": "grok-4-1-fast-reasoning",
        },
    )
    append_jsonl(
        CHAT_LOG_PATH,
        {
            "type": "system",
            "timestamp": time.time(),
            "content": system_prompt,
            "model": "grok-4-1-fast-reasoning",
            "session_id": SESSION_ID,
            "resumed": False,
        },
    )

    return jsonify({"ok": True, "session_id": SESSION_ID})


if __name__ == "__main__":
    # 访问 http://127.0.0.1:5000 即可在浏览器中聊天
    app.run(host="127.0.0.1", port=5001, debug=True)

