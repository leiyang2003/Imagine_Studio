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


# Get API key from environment variable (recommended for security)
api_key = os.getenv("XAI_API_KEY")
if not api_key:
    raise RuntimeError("XAI_API_KEY not set in environment or .env file.")

# Initialize the client
client = Client(api_key=api_key, timeout=3600)  # Longer timeout for potential agentic responses

# System prompt: default file in current dir; person-specific prompts in systemprompt/ directory
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH = os.path.join(_APP_DIR, "SystemPrompt.txt")
PROMPT_DIR = os.path.join(_APP_DIR, "systemprompt")
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
        --bg: #f8fafc;
        --surface: #ffffff;
        --border: #e2e8f0;
        --text: #1e293b;
        --text-secondary: #64748b;
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --shadow: 0 1px 3px rgba(0,0,0,.06);
        --shadow-md: 0 4px 12px rgba(0,0,0,.08);
        --radius: 12px;
        --radius-sm: 8px;
      }
      * { box-sizing: border-box; }
      body {
        font-family: "Google Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        max-width: 720px;
        margin: 0 auto;
        padding: 32px 24px;
        background: var(--bg);
        color: var(--text);
        font-size: 15px;
        line-height: 1.55;
      }
      h1 {
        font-weight: 600;
        font-size: 22px;
        margin: 0 0 4px;
        letter-spacing: -0.02em;
      }
      p.subtitle {
        margin: 0 0 20px;
        color: var(--text-secondary);
        font-size: 14px;
      }
      #chat-container {
        border-radius: var(--radius);
        background: var(--surface);
        padding: 20px;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border);
      }
      #chat {
        border-radius: var(--radius-sm);
        background: var(--bg);
        padding: 16px;
        height: 420px;
        overflow-y: auto;
        scroll-behavior: smooth;
        border: 1px solid var(--border);
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
        margin-top: 12px;
        display: flex;
        gap: 10px;
        align-items: center;
      }
      input[type=text] {
        flex: 1;
        padding: 12px 16px;
        border-radius: 24px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        font-size: 14px;
        outline: none;
        transition: border-color .15s, box-shadow .15s;
      }
      input[type=text]:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(37, 99, 235, .2);
      }
      button {
        padding: 10px 18px;
        border-radius: 24px;
        border: none;
        background: var(--primary);
        color: #fff;
        cursor: pointer;
        font-weight: 500;
        font-size: 14px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: background .15s;
      }
      button:hover { background: var(--primary-hover); }
      button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      #new-chat, #back-home {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text);
      }
      #new-chat:hover, #back-home:hover {
        background: var(--bg);
        border-color: var(--text-secondary);
      }
      #status {
        margin-top: 8px;
        font-size: 12px;
        color: var(--text-secondary);
      }
      #name-modal-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,.4);
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
      #chat-screen.active {
        display: block;
      }
      #welcome-screen {
        display: none;
      }
      #welcome-screen.active {
        display: block;
      }
      #person-picker-screen {
        max-width: 480px;
        margin: 0 auto;
        padding: 28px;
        border-radius: var(--radius);
        background: var(--surface);
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border);
      }
      #person-picker-screen h2 {
        margin: 0 0 6px;
        font-size: 20px;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.02em;
      }
      #person-picker-screen .picker-sub {
        margin: 0 0 24px;
        font-size: 14px;
        color: var(--text-secondary);
      }
      #person-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .person-item {
        display: block;
        width: 100%;
        padding: 14px 18px;
        text-align: left;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        background: var(--surface);
        color: var(--text);
        font-size: 15px;
        font-weight: 500;
        cursor: pointer;
        transition: background .15s, border-color .15s;
      }
      .person-item:hover {
        background: var(--bg);
        border-color: var(--text-secondary);
      }
      .person-item:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(37,99,235,.2);
      }
      .picker-empty {
        padding: 28px;
        text-align: center;
        color: var(--text-secondary);
        font-size: 14px;
      }
      #welcome-screen {
        max-width: 480px;
        margin: 0 auto;
        padding: 28px;
        border-radius: var(--radius);
        background: var(--surface);
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border);
      }
      #welcome-screen h2 {
        margin: 0 0 6px;
        font-size: 20px;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.02em;
      }
      #welcome-screen .welcome-sub {
        margin: 0 0 8px;
        font-size: 14px;
        color: var(--text-secondary);
      }
      .back-to-picker-btn {
        margin-bottom: 20px;
        padding: 8px 14px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text-secondary);
        font-size: 13px;
        cursor: pointer;
        transition: background .15s, color .15s;
      }
      .back-to-picker-btn:hover {
        background: var(--bg);
        color: var(--text);
      }
      .welcome-actions {
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
      }
      .welcome-actions button {
        flex: 1;
        padding: 14px 20px;
        border-radius: var(--radius-sm);
        border: none;
        font-size: 15px;
        font-weight: 500;
        cursor: pointer;
        transition: background .15s;
      }
      #btn-new-chat-welcome {
        background: var(--primary);
        color: #fff;
      }
      #btn-new-chat-welcome:hover { background: var(--primary-hover); }
      #btn-history-welcome {
        background: var(--surface);
        color: var(--text);
        border: 1px solid var(--border);
      }
      #btn-history-welcome:hover {
        background: var(--bg);
        border-color: var(--text-secondary);
      }
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
        background: #fef2f2;
        color: #dc2626;
        border-color: #fecaca;
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
    <div id="delete-modal-overlay">
      <div id="delete-modal">
        <h3>确认删除</h3>
        <p>是否确认要删除该条历史记录？删除后无法恢复。</p>
        <div id="delete-modal-actions">
          <button id="delete-modal-cancel" type="button">取消</button>
          <button id="delete-modal-confirm" type="button">确认删除</button>
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
      <h2 id="welcome-title">和谁聊天</h2>
      <p class="welcome-sub">从历史对话继续，或开始新对话</p>
      <button id="btn-back-to-picker" type="button" class="back-to-picker-btn">换一个人</button>
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
      <p class="subtitle" id="chat-subtitle">在浏览器里和你的「小丙」聊天，不用再开命令行啦～</p>
      <div id="chat-container">
      <div id="chat"></div>
      <form id="chat-form">
        <input id="msg" type="text" placeholder="说点什么..." autocomplete="off" />
        <button id="back-home" type="button" title="回到首页，选择聊天对象">
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
      const personPickerScreen = document.getElementById('person-picker-screen');
      const personListEl = document.getElementById('person-list');
      const welcomeTitleEl = document.getElementById('welcome-title');
      const chatSubtitleEl = document.getElementById('chat-subtitle');
      let selectedPromptFile = null;
      let selectedPromptName = '';
      let currentBotName = '小丙';

      function setCurrentBotName(name) {
        currentBotName = name || '小丙';
        if (chatSubtitleEl) chatSubtitleEl.textContent = '在浏览器里和你的「' + currentBotName + '」聊天，不用再开命令行啦～';
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

      function showChatScreen() {
        personPickerScreen.style.display = 'none';
        welcomeScreen.style.display = 'none';
        welcomeScreen.classList.remove('active');
        chatScreen.classList.add('active');
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
                setCurrentBotName(cur.prompt_name);
                chatDiv.innerHTML = '';
                (data.history || []).forEach(function(msg) {
                  addMessage(msg.role, msg.content);
                });
                showChatScreen();
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
          delBtn.textContent = '×';
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
          btn.textContent = f.name;
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

      document.getElementById('btn-back-to-picker').addEventListener('click', function() {
        showPersonPicker();
      });

      document.getElementById('btn-new-chat-welcome').addEventListener('click', function() {
        var body = {};
        if (selectedPromptFile) body.prompt_file = selectedPromptFile;
        fetch('/new', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        })
          .then(function(res) {
            if (!res.ok) throw new Error('HTTP ' + res.status);
            return res.json();
          })
          .then(function(data) {
            setCurrentBotName(data.prompt_name);
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
        chatDiv.scrollTop = chatDiv.scrollHeight;
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
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          });
        }).catch(function() {
          if (selectedPromptFile) body.prompt_file = selectedPromptFile;
          return fetch('/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          });
        }).then(function(res) {
          if (!res.ok) throw new Error('HTTP ' + res.status);
          return res.json();
        }).then(function(data) {
          setCurrentBotName(data.prompt_name);
          chatDiv.innerHTML = '';
          status.textContent = '新对话已就绪。';
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


@app.route("/prompt-files", methods=["GET"])
def get_prompt_files():
    """List system prompt files in systemprompt/ (for 'choose who to chat with')."""
    files = list_prompt_files(PROMPT_DIR)
    return jsonify({"prompt_files": files})


@app.route("/current-session", methods=["GET"])
def current_session():
    """Return current session_id, prompt_file, and prompt_name (for back-home and chat UI label)."""
    state = read_json(STATE_PATH) or {}
    if not isinstance(state, dict):
        state = {}
    pf = state.get("prompt_file")
    prompt_name = (pf[:-4] if pf and isinstance(pf, str) and pf.endswith(".txt") else (pf or "")) or None
    return jsonify({
        "session_id": SESSION_ID,
        "prompt_file": pf,
        "prompt_name": prompt_name,
    })


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
    deleted = False
    try:
        if os.path.isfile(state_path):
            os.remove(state_path)
            deleted = True
        if os.path.isfile(log_path):
            os.remove(log_path)
            deleted = True
    except OSError as e:
        return jsonify({"error": str(e)}), 500
    if not deleted:
        return jsonify({"error": "session not found"}), 404
    return jsonify({"ok": True})


@app.route("/switch-session", methods=["POST"])
def switch_session():
    """Switch current session to the given one; load state and continuation chat."""
    global SESSION_ID, CHAT_LOG_PATH, STATE_PATH, previous_response_id, chat, system_prompt
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
        prompt_content = system_prompt
        pf = state.get("prompt_file")
        if pf:
            prompt_path = os.path.join(PROMPT_DIR, pf)
            if os.path.isfile(prompt_path):
                prompt_content = read_system_prompt(prompt_path)
                system_prompt = prompt_content
        chat = client.chat.create(
            model="grok-4-1-fast-reasoning",
            store_messages=True,
            tools=[],
        )
        chat.append(system(prompt_content))
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
    """Start a brand new chat session. Optional body: { "prompt_file": "xxx.txt" } to use that person's system prompt."""
    global SESSION_ID, CHAT_LOG_PATH, STATE_PATH, previous_response_id, chat, system_prompt

    data = request.get_json(silent=True) or {}
    prompt_file = (data.get("prompt_file") or "").strip() or None
    prompt_content = system_prompt
    if prompt_file:
        prompt_path = os.path.join(PROMPT_DIR, prompt_file)
        if os.path.isfile(prompt_path):
            prompt_content = read_system_prompt(prompt_path)
        else:
            prompt_file = None

    SESSION_ID = f"{int(time.time())}"
    CHAT_LOG_PATH = os.path.join(LOG_DIR, f"{SESSION_ID}.jsonl")
    STATE_PATH = os.path.join(LOG_DIR, f"{SESSION_ID}.state.json")
    previous_response_id = None

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LAST_SESSION_PATH, "w", encoding="utf-8") as f:
        f.write(SESSION_ID)

    chat = client.chat.create(
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

