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

## 部署到 GitHub Pages

1. 创建 GitHub 仓库
2. 推送代码：

```bash
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/triton.git
git push -u origin main
```

3. 启用 GitHub Pages：
   - 进入仓库 Settings
   - 找到 Pages 选项
   - 选择 `main` 分支作为来源
   - 保存设置

4. 访问 `https://yourusername.github.io/triton/`

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
