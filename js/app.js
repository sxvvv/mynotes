/**
 * Main Application
 * Core application logic for the blog
 */

class BlogApp {
  constructor() {
    this.posts = [];
    this.categories = [];
    this.tags = [];
    this.currentPost = null;

    this.init();
  }

  /**
   * Initialize the application
   */
  async init() {
    try {
      // Load posts data
      await this.loadPostsData();

      // Initialize router
      this.initRouter();

      // Initialize search
      this.initSearch();

      // Render navigation
      this.renderNavigation();

      // Add copy button functionality
      this.addCopyButtons();

    } catch (error) {
      console.error('Failed to initialize app:', error);
      this.showError('加载数据失败，请刷新页面重试。');
    }
  }

  /**
   * Load posts metadata from JSON
   */
  async loadPostsData() {
    const response = await fetch('data/posts.json');
    const data = await response.json();

    this.posts = data.posts || [];
    this.categories = data.categories || [];
    this.tags = data.tags || [];
  }

  /**
   * Initialize hash router
   */
  initRouter() {
    const router = new Router({
      '/': () => this.renderHome(),
      '/post/:id': (params) => this.renderPost(params.id),
      '/tag/:tag': (params) => this.renderTagPage(params.tag),
      '/category/:category': (params) => this.renderCategoryPage(params.category)
    });

    router.setNotFoundHandler(() => {
      this.render404();
    });

    this.router = router;
  }

  /**
   * Initialize search engine
   */
  initSearch() {
    this.searchEngine = new SearchEngine(this.posts);
  }

  /**
   * Render navigation (categories and tags)
   */
  renderNavigation() {
    this.renderCategories();
    this.renderTagCloud();
  }

  /**
   * Render category list in sidebar
   */
  renderCategories() {
    const categoryList = document.getElementById('category-list');
    if (!categoryList) return;

    // Count posts per category
    const categoryCount = {};
    this.posts.forEach(post => {
      const category = post.category || '未分类';
      categoryCount[category] = (categoryCount[category] || 0) + 1;
    });

    // Build category list HTML
    const categoryItems = Object.entries(categoryCount).map(([name, count]) => `
      <li class="nav-item">
        <a href="#/category/${encodeURIComponent(name)}" class="nav-link" data-category="${name}">
          <span>${name}</span>
          <span class="nav-count">${count}</span>
        </a>
      </li>
    `).join('');

    categoryList.innerHTML = categoryItems;
  }

  /**
   * Render tag cloud in sidebar
   */
  renderTagCloud() {
    const tagCloud = document.getElementById('tag-cloud');
    if (!tagCloud) return;

    // Count posts per tag
    const tagCount = {};
    this.posts.forEach(post => {
      (post.tags || []).forEach(tag => {
        tagCount[tag] = (tagCount[tag] || 0) + 1;
      });
    });

    // Sort by count and take top tags
    const sortedTags = Object.entries(tagCount)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20);

    // Build tag cloud HTML
    tagCloud.innerHTML = sortedTags.map(([tag, count]) => `
      <span class="tag" data-tag="${tag}">${tag} (${count})</span>
    `).join('');

    // Add click handlers
    tagCloud.querySelectorAll('.tag').forEach(el => {
      el.addEventListener('click', () => {
        const tag = el.dataset.tag;
        window.location.hash = `#/tag/${encodeURIComponent(tag)}`;
      });
    });
  }

  /**
   * Render home page (post list)
   */
  renderHome() {
    const content = document.getElementById('content');
    if (!content) return;

    // Update page title
    document.title = 'HPC/TritON 技术笔记';

    // Clear active states
    this.clearActiveStates();

    // Sort posts by date (newest first)
    const sortedPosts = [...this.posts].sort((a, b) =>
      new Date(b.date) - new Date(a.date)
    );

    // Build post list HTML
    const postCards = sortedPosts.map(post => this.renderPostCard(post)).join('');

    content.innerHTML = `
      <div class="hero">
        <h1 class="hero-title">HPC/Triton 技术笔记</h1>
        <p class="hero-description">
          分享高性能计算、Triton 编程和 GPU 优化的学习笔记与实践经验
        </p>
      </div>

      <div class="section-header">
        <h2 class="section-title">最新文章</h2>
      </div>

      <div class="post-list">
        ${postCards || '<div class="empty-state"><div class="empty-state-icon">📝</div><div class="empty-state-title">暂无文章</div></div>'}
      </div>
    `;

    // Add click handlers to post cards
    content.querySelectorAll('.post-card').forEach(card => {
      card.addEventListener('click', () => {
        const postId = card.dataset.postId;
        window.location.hash = `#/post/${postId}`;
      });
    });
  }

  /**
   * Render a single post card
   */
  renderPostCard(post) {
    const date = new Date(post.date).toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });

    const tagsHtml = (post.tags || []).slice(0, 3).map(tag =>
      `<span class="tag">${tag}</span>`
    ).join('');

    return `
      <article class="post-card" data-post-id="${post.id}">
        <div class="post-card-header">
          <h2 class="post-card-title">${post.title}</h2>
          <div class="post-card-meta">
            <span>📅 ${date}</span>
            <span>⏱️ ${post.readTime || 5} 分钟</span>
          </div>
        </div>
        <p class="post-card-description">${post.description}</p>
        <div class="post-card-tags">${tagsHtml}</div>
      </article>
    `;
  }

  /**
   * Render post detail page
   */
  async renderPost(postId) {
    const content = document.getElementById('content');
    if (!content) return;

    const post = this.posts.find(p => p.id === postId);

    if (!post) {
      this.render404();
      return;
    }

    this.currentPost = post;

    // Update page title
    document.title = `${post.title} - HPC/Triton 技术笔记`;

    // Clear active states
    this.clearActiveStates();

    // Show loading
    content.innerHTML = '<div class="loading"><div class="spinner"></div>加载中...</div>';

    try {
      // Fetch markdown content
      const response = await fetch(post.file);
      const markdown = await response.text();

      // Render markdown to HTML
      const htmlContent = this.renderMarkdown(markdown);

      // Render post detail
      content.innerHTML = this.buildPostDetailHTML(post, htmlContent);

      // Add copy buttons to code blocks
      this.addCopyButtons();

      // Highlight code
      if (window.Prism) {
        Prism.highlightAll();
      }

      // Update active tag states
      (post.tags || []).forEach(tag => {
        const tagEl = document.querySelector(`.tag[data-tag="${tag}"]`);
        if (tagEl) tagEl.classList.add('active');
      });

    } catch (error) {
      console.error('Failed to load post:', error);
      content.innerHTML = '<div class="empty-state"><div class="empty-state-icon">❌</div><div class="empty-state-title">加载失败</div><p>无法加载文章内容，请稍后重试。</p></div>';
    }
  }

  /**
   * Render markdown content
   */
  renderMarkdown(markdown) {
    // Configure marked
    marked.setOptions({
      gfm: true,
      breaks: true,
      highlight: function(code, lang) {
        if (Prism.languages[lang]) {
          return Prism.highlight(code, Prism.languages[lang], lang);
        }
        return code;
      }
    });

    return marked.parse(markdown);
  }

  /**
   * Build post detail HTML
   */
  buildPostDetailHTML(post, content) {
    const date = new Date(post.date).toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });

    const tagsHtml = (post.tags || []).map(tag =>
      `<span class="post-tag" data-tag="${tag}">${tag}</span>`
    ).join('');

    // Find previous and next posts
    const sortedPosts = [...this.posts].sort((a, b) =>
      new Date(b.date) - new Date(a.date)
    );
    const currentIndex = sortedPosts.findIndex(p => p.id === post.id);
    const prevPost = currentIndex < sortedPosts.length - 1 ? sortedPosts[currentIndex + 1] : null;
    const nextPost = currentIndex > 0 ? sortedPosts[currentIndex - 1] : null;

    return `
      <article class="post-detail">
        <header class="post-header">
          <h1 class="post-title">${post.title}</h1>
          <div class="post-meta">
            <span class="post-meta-item">📅 ${date}</span>
            <span class="post-meta-item">⏱️ ${post.readTime || 5} 分钟</span>
            <span class="post-meta-item">📁 ${post.category || '未分类'}</span>
          </div>
        </header>

        <div class="post-content">
          ${content}
        </div>

        <div class="post-tags">
          ${tagsHtml}
        </div>

        <nav class="post-navigation">
          ${nextPost ? `
            <a href="#/post/${nextPost.id}" class="post-nav-link next">
              <span class="post-nav-label">下一篇</span>
              <span class="post-nav-title">${nextPost.title}</span>
            </a>
          ` : '<div></div>'}
          ${prevPost ? `
            <a href="#/post/${prevPost.id}" class="post-nav-link">
              <span class="post-nav-label">上一篇</span>
              <span class="post-nav-title">${prevPost.title}</span>
            </a>
          ` : '<div></div>'}
        </nav>
      </article>
    `;
  }

  /**
   * Render tag filtered page
   */
  renderTagPage(tag) {
    const content = document.getElementById('content');
    if (!content) return;

    // Update page title
    document.title = `标签: ${tag} - HPC/Triton 技术笔记`;

    // Clear active states
    this.clearActiveStates();

    // Set active tag
    const tagEl = document.querySelector(`.tag[data-tag="${tag}"]`);
    if (tagEl) tagEl.classList.add('active');

    // Filter posts by tag
    const filteredPosts = this.posts.filter(post =>
      (post.tags || []).includes(tag)
    ).sort((a, b) => new Date(b.date) - new Date(a.date));

    // Render
    content.innerHTML = `
      <div class="hero">
        <h1 class="hero-title">标签: ${tag}</h1>
        <p class="hero-description">共 ${filteredPosts.length} 篇文章</p>
      </div>

      <div class="post-list">
        ${filteredPosts.map(post => this.renderPostCard(post)).join('') || '<div class="empty-state"><div class="empty-state-icon">📭</div><div class="empty-state-title">暂无文章</div></div>'}
      </div>
    `;

    // Add click handlers
    content.querySelectorAll('.post-card').forEach(card => {
      card.addEventListener('click', () => {
        window.location.hash = `#/post/${card.dataset.postId}`;
      });
    });
  }

  /**
   * Render category filtered page
   */
  renderCategoryPage(category) {
    const content = document.getElementById('content');
    if (!content) return;

    // Update page title
    document.title = `分类: ${category} - HPC/Triton 技术笔记`;

    // Clear active states
    this.clearActiveStates();

    // Set active category
    const categoryEl = document.querySelector(`.nav-link[data-category="${category}"]`);
    if (categoryEl) categoryEl.classList.add('active');

    // Filter posts by category
    const filteredPosts = this.posts.filter(post =>
      (post.category || '未分类') === category
    ).sort((a, b) => new Date(b.date) - new Date(a.date));

    // Render
    content.innerHTML = `
      <div class="hero">
        <h1 class="hero-title">分类: ${category}</h1>
        <p class="hero-description">共 ${filteredPosts.length} 篇文章</p>
      </div>

      <div class="post-list">
        ${filteredPosts.map(post => this.renderPostCard(post)).join('') || '<div class="empty-state"><div class="empty-state-icon">📭</div><div class="empty-state-title">暂无文章</div></div>'}
      </div>
    `;

    // Add click handlers
    content.querySelectorAll('.post-card').forEach(card => {
      card.addEventListener('click', () => {
        window.location.hash = `#/post/${card.dataset.postId}`;
      });
    });
  }

  /**
   * Render 404 page
   */
  render404() {
    const content = document.getElementById('content');
    if (!content) return;

    content.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <div class="empty-state-title">页面未找到</div>
        <p>抱歉，您访问的页面不存在。</p>
        <a href="#/" class="post-tag" style="margin-top: 1rem; display: inline-block;">返回首页</a>
      </div>
    `;
  }

  /**
   * Show error message
   */
  showError(message) {
    const content = document.getElementById('content');
    if (content) {
      content.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">❌</div>
          <div class="empty-state-title">出错了</div>
          <p>${message}</p>
        </div>
      `;
    }
  }

  /**
   * Clear all active states in navigation
   */
  clearActiveStates() {
    document.querySelectorAll('.nav-link.active, .tag.active').forEach(el => {
      el.classList.remove('active');
    });
  }

  /**
   * Add copy buttons to code blocks
   */
  addCopyButtons() {
    document.querySelectorAll('pre').forEach(pre => {
      if (pre.querySelector('.copy-btn')) return;

      const button = document.createElement('button');
      button.className = 'copy-btn';
      button.textContent = '复制';
      button.addEventListener('click', () => this.copyCode(button, pre));

      pre.style.position = 'relative';
      pre.appendChild(button);
    });
  }

  /**
   * Copy code to clipboard
   */
  copyCode(button, pre) {
    const code = pre.querySelector('code');
    if (!code) return;

    const text = code.textContent;

    navigator.clipboard.writeText(text).then(() => {
      button.textContent = '已复制';
      button.classList.add('copied');

      setTimeout(() => {
        button.textContent = '复制';
        button.classList.remove('copied');
      }, 2000);
    }).catch(() => {
      button.textContent = '失败';
    });
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.blogApp = new BlogApp();
});
