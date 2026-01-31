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


def list_sessions(log_dir: str):
    """List all saved sessions: scan for *.state.json, return [{ session_id, name, updated_at }] sorted by updated_at desc."""
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
        sessions.append({
            "session_id": session_id,
            "name": state.get("name") or None,
            "updated_at": state.get("updated_at") or 0,
        })
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


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
      #new-chat, #back-home {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.35);
        color: #e2e8f0;
      }
      #new-chat:hover, #back-home:hover {
        background: rgba(148, 163, 184, 0.18);
      }
      #status {
        margin-top: 6px;
        font-size: 12px;
        color: #a6accd;
      }
      #name-modal-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.6);
        z-index: 100;
        align-items: center;
        justify-content: center;
      }
      #name-modal-overlay.show {
        display: flex;
      }
      #name-modal {
        background: linear-gradient(180deg, #1e293b, #0f172a);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 16px;
        padding: 24px;
        min-width: 320px;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
      }
      #name-modal h3 {
        margin: 0 0 8px;
        font-size: 16px;
        color: #e2e8f0;
      }
      #name-modal p {
        margin: 0 0 16px;
        font-size: 13px;
        color: #94a3b8;
      }
      #name-modal input {
        width: 100%;
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid #334155;
        background: rgba(15, 23, 42, 0.95);
        color: #f9fafb;
        font-size: 14px;
        box-sizing: border-box;
        margin-bottom: 16px;
      }
      #name-modal input:focus {
        border-color: #22d3ee;
        outline: none;
      }
      #name-modal-actions {
        display: flex;
        gap: 8px;
        justify-content: flex-end;
      }
      #name-modal-actions button {
        padding: 8px 16px;
        border-radius: 8px;
        border: none;
        font-size: 13px;
        cursor: pointer;
      }
      #name-modal-skip {
        background: transparent;
        color: #94a3b8;
        border: 1px solid #475569;
      }
      #name-modal-skip:hover {
        background: rgba(71, 85, 105, 0.3);
      }
      #name-modal-ok {
        background: linear-gradient(135deg, #4f46e5, #22d3ee);
        color: white;
      }
      #name-modal-ok:hover {
        opacity: 0.9;
      }
      #chat-screen {
        display: none;
      }
      #chat-screen.active {
        display: block;
      }
      #welcome-screen {
        max-width: 560px;
        margin: 40px auto;
        padding: 24px;
        border-radius: 16px;
        background: radial-gradient(circle at top left, #1f2937, #020617);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
      }
      #welcome-screen h2 {
        margin: 0 0 8px;
        font-size: 20px;
        color: #e2e8f0;
      }
      #welcome-screen .welcome-sub {
        margin: 0 0 24px;
        font-size: 14px;
        color: #94a3b8;
      }
      .welcome-actions {
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
      }
      .welcome-actions button {
        flex: 1;
        padding: 14px 20px;
        border-radius: 12px;
        border: none;
        font-size: 15px;
        font-weight: 500;
        cursor: pointer;
        transition: opacity 0.2s;
      }
      .welcome-actions button:hover {
        opacity: 0.9;
      }
      #btn-new-chat-welcome {
        background: linear-gradient(135deg, #4f46e5, #22d3ee);
        color: white;
      }
      #btn-history-welcome {
        background: rgba(51, 65, 85, 0.8);
        color: #e2e8f0;
        border: 1px solid #475569;
      }
      #history-list-wrap {
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid rgba(71, 85, 105, 0.5);
      }
      #history-list-wrap h3 {
        margin: 0 0 12px;
        font-size: 14px;
        color: #94a3b8;
      }
      #history-list {
        max-height: 280px;
        overflow-y: auto;
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.6);
      }
      .history-item {
        display: block;
        width: 100%;
        padding: 12px 14px;
        text-align: left;
        border: none;
        border-bottom: 1px solid rgba(51, 65, 85, 0.6);
        background: transparent;
        color: #e2e8f0;
        font-size: 14px;
        cursor: pointer;
        transition: background 0.15s;
        box-sizing: border-box;
      }
      .history-item:last-child {
        border-bottom: none;
      }
      .history-item:hover {
        background: rgba(51, 65, 85, 0.4);
      }
      .history-item .name {
        font-weight: 500;
        color: #f1f5f9;
      }
      .history-item .meta {
        font-size: 12px;
        color: #64748b;
        margin-top: 2px;
      }
      .history-empty {
        padding: 24px;
        text-align: center;
        color: #64748b;
        font-size: 14px;
      }
    </style>
  </head>
  <body>
    <div id="name-modal-overlay">
      <div id="name-modal">
        <h3>给这段对话起个名字</h3>
        <p>方便以后找回并继续聊～</p>
        <input id="session-name-input" type="text" placeholder="例如：工作周报、学英语、点子记录..." maxlength="64" autocomplete="off" />
        <div id="name-modal-actions">
          <button id="name-modal-skip" type="button">跳过</button>
          <button id="name-modal-ok" type="button">确定</button>
        </div>
      </div>
    </div>
    <div id="welcome-screen">
      <h2>和小丙聊天</h2>
      <p class="welcome-sub">从历史对话继续，或开始新对话</p>
      <div class="welcome-actions">
        <button id="btn-history-welcome" type="button">从历史对话继续</button>
        <button id="btn-new-chat-welcome" type="button">新对话</button>
      </div>
      <div id="history-list-wrap">
        <h3>历史对话</h3>
        <div id="history-list">
          <div class="history-empty">加载中…</div>
        </div>
      </div>
    </div>
    <div id="chat-screen">
      <h1>Grok Web Chat</h1>
      <p class="subtitle">在浏览器里和你的「小丙」聊天，不用再开命令行啦～</p>
      <div id="chat-container">
      <div id="chat"></div>
      <form id="chat-form">
        <input id="msg" type="text" placeholder="说点什么..." autocomplete="off" />
        <button id="back-home" type="button" title="回到首页，选择历史对话或新对话">
          <span>回到首页</span>
        </button>
        <button id="new-chat" type="button" title="开始一个全新的对话（不再续上之前上下文）">
          <span>新对话</span>
        </button>
        <button type="submit">
          <span>发送</span>
        </button>
      </form>
      <div id="status">准备就绪。</div>
      </div>
    </div>
    <script>
      const form = document.getElementById('chat-form');
      const input = document.getElementById('msg');
      const chatDiv = document.getElementById('chat');
      const newChatBtn = document.getElementById('new-chat');
      const button = form.querySelector('button[type="submit"]');
      const status = document.getElementById('status');
      const welcomeScreen = document.getElementById('welcome-screen');
      const chatScreen = document.getElementById('chat-screen');
      const historyListEl = document.getElementById('history-list');

      function showChatScreen() {
        welcomeScreen.style.display = 'none';
        chatScreen.classList.add('active');
      }

      function showWelcomeScreen() {
        welcomeScreen.style.display = 'block';
        chatScreen.classList.remove('active');
      }

      function formatHistoryDate(ts) {
        if (!ts) return '';
        const d = new Date(ts * 1000);
        const now = new Date();
        const sameDay = d.getDate() === now.getDate() && d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
        if (sameDay) return '今天 ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
        return d.toLocaleDateString('zh-CN') + ' ' + d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
      }

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
              chatDiv.innerHTML = '';
              (data.history || []).forEach(function(msg) {
                addMessage(msg.role, msg.content);
              });
              showChatScreen();
              input.focus();
            }).catch(function(err) {
              console.error(err);
              alert('切换对话失败，请重试。');
            });
          });
          historyListEl.appendChild(btn);
        });
      }

      fetch('/sessions').then(function(res) { return res.json(); }).then(function(data) {
        renderHistoryList(data.sessions || []);
      }).catch(function() {
        historyListEl.innerHTML = '<div class="history-empty">加载失败</div>';
      });

      document.getElementById('btn-new-chat-welcome').addEventListener('click', function() {
        fetch('/new', { method: 'POST' })
          .then(function(res) {
            if (!res.ok) throw new Error('HTTP ' + res.status);
            chatDiv.innerHTML = '';
            showChatScreen();
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
        showWelcomeScreen();
        fetch('/sessions').then(function(res) { return res.json(); }).then(function(data) {
          renderHistoryList(data.sessions || []);
        }).catch(function() {
          renderHistoryList([]);
        });
      });

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

      function startNewChat() {
        newChatBtn.disabled = true;
        button.disabled = true;
        status.textContent = '正在开始新对话…';
        fetch('/new', { method: 'POST' })
          .then(res => {
            if (!res.ok) throw new Error('HTTP ' + res.status);
            chatDiv.innerHTML = '';
            status.textContent = '新对话已就绪。';
            input.value = '';
            input.focus();
          })
          .catch(err => {
            console.error(err);
            status.textContent = '开启新对话失败，请刷新页面重试。';
          })
          .finally(() => {
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


@app.route("/sessions", methods=["GET"])
def get_sessions():
    """List all saved chat sessions (for history picker)."""
    sessions = list_sessions(LOG_DIR)
    return jsonify({"sessions": sessions})


@app.route("/switch-session", methods=["POST"])
def switch_session():
    """Switch current session to the given one; load state and continuation chat."""
    global SESSION_ID, CHAT_LOG_PATH, STATE_PATH, previous_response_id, chat
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
    if prev_id:
        chat = client.chat.create(
            model="grok-4-1-fast-reasoning",
            previous_response_id=prev_id,
            store_messages=True,
            tools=[],
        )
    else:
        chat = client.chat.create(
            model="grok-4-1-fast-reasoning",
            store_messages=True,
            tools=[],
        )
        chat.append(system(system_prompt))
    return jsonify({"ok": True, "session_id": SESSION_ID})


@app.route("/history")
def get_history():
    """Get message history for a session (for displaying in UI). session_id defaults to current."""
    session_id = request.args.get("session_id") or SESSION_ID
    log_path = os.path.join(LOG_DIR, f"{session_id}.jsonl")
    history = read_history_jsonl(log_path)
    return jsonify({"session_id": session_id, "history": history})


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
    return jsonify({"reply": response.content})


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

