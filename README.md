# mynotes 个人技术笔记

一个简约科技风格的个人技术笔记网站，用于整理和分享你的技术学习笔记（当前示例内容以 HPC / Triton / GPU 为主，你可以按需替换为自己的主题）。

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
