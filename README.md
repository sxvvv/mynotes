# HPC/Triton 技术笔记

一个简约科技风格的个人技术笔记网站，用于分享高性能计算（HPC）、Triton 编程和 GPU 优化的学习笔记。

## 特性

- **简约设计**：干净、现代的界面风格
- **深色模式**：支持明暗主题切换
- **代码高亮**：支持 Python、C++、CUDA、Bash 等语言
- **全文搜索**：快速搜索文章和标签
- **标签分类**：按标签和分类浏览文章
- **纯静态**：无需后端，可部署到任何静态托管服务
- **响应式**：完美支持桌面端和移动端

## 技术栈

- **HTML5/CSS3/JavaScript (ES6+)**
- **marked.js** - Markdown 解析
- **Prism.js** - 代码语法高亮
- **GitHub Pages** - 免费托管

## 快速开始

### 本地预览

使用 Python：

```bash
python -m http.server 8000
```

或使用 Node.js：

```bash
npx live-server
```

然后访问 http://localhost:8000

### 添加新文章

1. 在 `posts/` 目录下创建新的 `.md` 文件
2. 使用 Front Matter 添加元数据：

```markdown
---
title: 文章标题
date: 2024-01-15
tags: [tag1, tag2, tag3]
category: 分类名称
description: 文章简介（用于搜索和预览）
---

# 文章内容

正文...
```

3. 更新 `data/posts.json`，添加文章元数据：

```json
{
  "posts": [
    {
      "id": "article-id",
      "title": "文章标题",
      "date": "2024-01-15",
      "tags": ["tag1", "tag2", "tag3"],
      "category": "分类名称",
      "description": "文章简介",
      "file": "posts/your-article.md",
      "readTime": 5
    }
  ],
  ...
}
```

## 项目结构

```
triton/
├── index.html              # 主页面（SPA）
├── css/
│   ├── main.css           # 主样式
│   ├── theme.css          # 主题变量
│   └── prism.css          # 代码高亮样式
├── js/
│   ├── app.js             # 主应用逻辑
│   ├── router.js          # Hash 路由
│   ├── search.js          # 搜索功能
│   ├── theme.js           # 主题切换
│   └── lib/               # 第三方库
│       ├── marked.min.js
│       └── prism.min.js
├── posts/                  # Markdown 文章
├── data/
│   └── posts.json         # 文章索引
├── assets/
│   └── images/            # 图片资源
├── .gitignore
├── .nojekyll              # GitHub Pages 配置
└── README.md
```

## 上传到 GitHub 并部署（个人笔记网站）

### 第一步：在 GitHub 创建仓库

1. 登录 [GitHub](https://github.com)，点击右上角 **+** → **New repository**
2. 仓库名可填 `triton` 或任意名称（如 `notes`）
3. 选择 **Public**，不勾选 “Add a README”
4. 点击 **Create repository**

### 第二步：在本地推送代码（上传）

在项目目录打开终端，执行：

```bash
# 若尚未初始化 Git
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: 个人笔记网站"

# 添加远程仓库（把 yourusername 和 triton 换成你的用户名和仓库名）
git remote add origin https://github.com/yourusername/triton.git

# 推送到 main 分支（若本地是 master，可先执行：git branch -M main）
git branch -M main
git push -u origin main
```

若推送时要求登录，请使用 **Personal Access Token** 或 **SSH**，不要用账号密码（GitHub 已不支持密码推送）。

### 第三步：开启 GitHub Pages

1. 打开你的仓库页面，点击 **Settings**
2. 左侧找到 **Pages**
3. 在 **Source** 里选择 **Deploy from a branch**
4. **Branch** 选 `main`，文件夹选 **/ (root)**，点 **Save**
5. 等待 1～2 分钟，页面会显示站点地址

### 第四步：访问你的笔记网站

- 若仓库名为 `triton`，地址为：**https://你的用户名.github.io/triton/**
- 若仓库名为 `notes`，地址为：**https://你的用户名.github.io/notes/**

之后每次修改笔记，只需在本地执行：

```bash
git add .
git commit -m "更新笔记"
git push
```

GitHub Pages 会自动重新部署，几分钟后刷新网站即可看到更新。

### 常见问题

- **推送时报错（无法上传）**：GitHub 已不支持账号密码推送。请使用 [Personal Access Token](https://github.com/settings/tokens)（在 GitHub → Settings → Developer settings 中创建），或配置 [SSH 密钥](https://docs.github.com/cn/authentication/connecting-to-github-with-ssh)。
- **打开网站白屏/加载失败**：确认 GitHub Pages 已开启，且访问地址为 `https://你的用户名.github.io/仓库名/`（末尾要有斜杠）。若仍失败，打开浏览器开发者工具（F12）查看 Console 是否有 404 等报错。
- **想用根域名**：若仓库名为 `你的用户名.github.io`，则站点地址为 `https://你的用户名.github.io/`，无需子路径。

## 自定义

### 修改主题颜色

编辑 `css/theme.css` 中的 CSS 变量：

```css
:root {
  --accent-primary: #2563eb;  /* 主题色 */
  --accent-secondary: #06b6d4; /* 次要色 */
}
```

### 添加新标签/分类

在 `data/posts.json` 中更新 `tags` 和 `categories` 数组。

### 修改网站标题

编辑 `index.html` 中的 `<title>` 和相关文本。

## 示例文章

网站已包含以下示例文章：

- **Triton 入门指南** - 介绍 Triton 编程语言的基础
- **CUDA 内存优化实战** - 深入理解 CUDA 内存优化
- **GPU 架构浅析** - 从硬件层面理解 GPU 工作原理

## 性能

- 首页加载：< 500ms
- 资源总大小：< 100KB（gzip 压缩后）
- Lighthouse 性能评分：95+

## License

MIT License - Feel free to use and modify.

## 作者

Your Name - [GitHub](https://github.com/yourusername)
