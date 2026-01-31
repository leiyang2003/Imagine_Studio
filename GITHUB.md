# 把项目推到 GitHub

按下面步骤做即可。**推送前请确认 `.env` 已在 `.gitignore` 里（项目已配置），这样 API Key 不会被推上去。**

---

## 1. 在 GitHub 上新建仓库

1. 打开 [github.com](https://github.com) 登录，右上角 **+** → **New repository**
2. **Repository name** 填一个名字，例如：`imagine-studio` 或 `chat-memory`
3. 选 **Public**
4. **不要**勾选 "Add a README file"（本地已有文件，直接推即可）
5. 点 **Create repository**

创建后会看到一个空仓库页面，记下仓库地址，例如：  
`https://github.com/你的用户名/imagine-studio.git`

---

## 2. 在本地初始化 Git 并推送

在终端里进入项目目录，依次执行（把 `你的用户名/imagine-studio` 换成你的仓库地址）：

```bash
cd "/Users/leiyang/Desktop/Coding/Chat Memory"

# 初始化仓库
git init

# 添加所有文件（.gitignore 会排除 .env、chat_logs、venv 等）
git add .

# 第一次提交
git commit -m "Initial commit: Imagine Studio"

# 主分支命名为 main（可选，GitHub 默认用 main）
git branch -M main

# 添加远程仓库（替换成你自己的仓库地址）
git remote add origin https://github.com/你的用户名/仓库名.git

# 推送到 GitHub
git push -u origin main
```

推送时可能会要求输入 GitHub 用户名和密码；现在 GitHub 要求用 **Personal Access Token** 代替密码，在 [GitHub → Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens) 生成一个，用 token 当密码即可。

---

## 3. 之后每次改完代码再推送

```bash
git add .
git commit -m "简短说明你改了什么"
git push
```

---

## 提醒

- 推送前可执行 `git status` 看一下，**不要出现 `.env`**；若出现说明没被忽略，不要 `git add .env`。
- 若仓库地址用 SSH（如 `git@github.com:用户名/仓库名.git`），把上面 `git remote add origin` 里的 URL 换成你的 SSH 地址即可。
