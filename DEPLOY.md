# Imagine Studio 部署指南（最简单方式）

用 **Render**、**Railway** 或 **Vercel** 可以零服务器、连 GitHub 就上线，别人通过链接即可访问。

---

## API Key 怎么处理（必读）

应用使用 **xAI API**。支持两种方式，**二选一**：

### 方式 A：不分享自己的 Key（推荐给公开部署）

- **部署时不要设置** `XAI_API_KEY` 环境变量（Vercel / Render / Railway 里都不填）。
- 访客打开你的链接后，在页面点击「**设置 API Key**」，填写自己的 xAI API Key 并保存。
- Key 只存在访客本机（localStorage），不会上传到你的服务器，你也不会看到。
- 这样你不需要分享自己的 Grok API，每个人用自己的 Key。

### 方式 B：用你自己的 Key 给所有人用

- 在平台 **Environment Variables** 里设置：`XAI_API_KEY` = 你的 xAI API Key。
- 所有访客共用你的 Key，无需在页面填写；适合仅自己或小范围使用。

### 本地开发

1. 在项目根目录新建 `.env` 文件（若还没有）：
   ```bash
   echo "XAI_API_KEY=你的xAI密钥" > .env
   ```
2. **不要**把 `.env` 提交到 Git（`.gitignore` 已忽略 `.env`）。

### 安全小结

| 场景           | 做法 |
|----------------|------|
| 公开部署、不想分享 Key | 不设置 `XAI_API_KEY`，让访客在页面「设置 API Key」填自己的 |
| 仅自己/熟人用   | 在平台环境变量里设置 `XAI_API_KEY` |
| 代码仓库       | 永远不要提交 `.env` 或任何包含密钥的文件 |

---

## 方式一：Vercel（推荐，让别人用链接访问）

### 1. 代码先推到 GitHub

若还没推过，在项目目录执行：

```bash
cd /Users/leiyang/Desktop/Coding/TEST
git init
git add .
git commit -m "Imagine Studio: chat + display image gen"
```

在 [github.com/new](https://github.com/new) 新建一个仓库（不要勾选 README），然后：

```bash
git remote add origin https://github.com/你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

### 2. 在 Vercel 部署

1. 打开 **[vercel.com](https://vercel.com)**，用 GitHub 登录。
2. 点击 **Add New** → **Project**。
3. **Import** 你刚推送的 GitHub 仓库，选好分支（如 `main`）。
4. **Environment Variables** 里添加（必填）：
   - **Name**: `XAI_API_KEY`
   - **Value**: 你的 xAI API Key  
   保存。
5. 点击 **Deploy**，等构建完成。

### 3. 访问与分享

- 部署成功后会有 **Production URL**，例如：`https://xxx.vercel.app`。
- 把这个链接发给别人，对方打开即可使用 App。
- 之后每次往该仓库 `git push`，Vercel 会自动重新部署。

**说明**：  
- 项目里已有 `index.py`（Flask 入口）和 `vercel.json`（Flask 框架），Vercel 会自动识别。  
- 在 Vercel 上会话和生成图会写到 `/tmp/chat_logs`，实例回收后可能清空，属正常。  
- 本地开发不受影响（非 Vercel 环境仍用 `chat_logs/`）。

---

## 方式二：Render（有免费额度）

1. **代码推送到 GitHub**  
   确保项目在 GitHub 上，且包含：
   - `chat_web_cm.py`
   - `requirements.txt`
   - `Procfile`
   - `systemprompt/` 目录（角色 .txt 和立绘 .png/.jpg/.gif）

2. **注册并创建 Web Service**  
   - 打开 [render.com](https://render.com) 注册/登录  
   - Dashboard → **New** → **Web Service**  
   - 连接你的 GitHub 仓库，选好分支

3. **配置**  
   - **Build Command**: `pip install -r requirements.txt`（或留空，Render 会自动识别）  
   - **Start Command**: 留空即可（会按 Procfile 的 `web` 启动）  
   - **Environment** 里添加变量：  
     - `XAI_API_KEY` = 你的 xAI API Key（必填）

4. **部署**  
   点击 **Create Web Service**，等构建完成会得到一个 `https://xxx.onrender.com` 的链接，别人用这个链接即可访问。

> 免费实例一段时间无访问会休眠，首次打开可能稍慢；如需常驻可升级付费。

---

## 方式三：Railway

1. 打开 [railway.app](https://railway.app)，用 GitHub 登录  
2. **New Project** → **Deploy from GitHub repo**，选中本仓库  
3. 在项目里点 **Variables**，添加 `XAI_API_KEY`  
4. Railway 会自动检测 Procfile 并部署，部署完成后在 **Settings** 里生成一个公网域名

---

## 注意事项

- **不要**把 `.env` 或 API Key 提交到 Git；只在平台的环境变量里配置 `XAI_API_KEY`。  
- `chat_logs/` 会在运行时自动创建；在 Render/Railway 上免费实例重启后日志可能清空；在 Vercel 上使用 `/tmp/chat_logs`，实例回收后也会清空，属正常。  
- 若需要持久化对话记录，可后续把 `CHAT_LOG_DIR` 指向云存储或数据库。
