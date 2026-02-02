/**
 * Search Engine
 * Lightweight client-side search functionality
 */

class SearchEngine {
  constructor(posts = []) {
    this.posts = posts;
    this.index = this.buildIndex();
    this.init();
  }

  /**
   * Build search index from posts
   */
  buildIndex() {
    return this.posts.map(post => ({
      id: post.id,
      title: post.title,
      description: post.description,
      tags: post.tags || [],
      category: post.category || '',
      content: '', // Lazy loaded on demand
      searchText: this.createSearchText(post)
    }));
  }

  /**
   * Create searchable text from post
   */
  createSearchText(post) {
    const parts = [
      post.title,
      post.description,
      ...(post.tags || []),
      post.category
    ];
    return parts.join(' ').toLowerCase();
  }

  /**
   * Initialize search functionality
   */
  init() {
    this.searchInput = document.getElementById('search-input');
    this.searchResults = document.getElementById('search-results');

    if (!this.searchInput || !this.searchResults) return;

    // Debounced search
    let debounceTimer;
    this.searchInput.addEventListener('input', (e) => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        this.search(e.target.value);
      }, 200);
    });

    // Focus/blur events
    this.searchInput.addEventListener('focus', () => {
      if (this.searchInput.value.trim()) {
        this.showResults();
      }
    });

    document.addEventListener('click', (e) => {
      if (!e.target.closest('.search-container')) {
        this.hideResults();
      }
    });

    // Keyboard navigation
    this.searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.hideResults();
        this.searchInput.blur();
      } else if (e.key === 'Enter') {
        const firstResult = this.searchResults.querySelector('.search-result-item');
        if (firstResult) {
          firstResult.click();
        }
      }
    });
  }

  /**
   * Perform search
   */
  search(query) {
    query = query.trim().toLowerCase();

    if (!query) {
      this.hideResults();
      return;
    }

    const keywords = query.split(/\s+/).filter(Boolean);
    const results = this.index.filter(post => {
      return keywords.every(keyword => post.searchText.includes(keyword));
    });

    this.renderResults(results, query);
  }

  /**
   * Render search results
   */
  renderResults(results, query) {
    if (!this.searchResults) return;

    if (results.length === 0) {
      this.searchResults.innerHTML = `
        <div class="search-no-results">
          没有找到匹配的文章
        </div>
      `;
      this.showResults();
      return;
    }

    this.searchResults.innerHTML = results.slice(0, 10).map(post => `
      <div class="search-result-item" data-post-id="${post.id}">
        <div class="search-result-title">${this.highlightMatch(post.title, query)}</div>
        <div class="search-result-excerpt">${this.highlightMatch(post.description, query)}</div>
      </div>
    `).join('');

    // Add click handlers
    this.searchResults.querySelectorAll('.search-result-item').forEach(item => {
      item.addEventListener('click', () => {
        const postId = item.dataset.postId;
        window.location.hash = `#/post/${postId}`;
        this.hideResults();
        this.searchInput.value = '';
      });
    });

    this.showResults();
  }

  /**
   * Highlight matching text
   */
  highlightMatch(text, query) {
    if (!query) return text;
    const keywords = query.split(/\s+/).filter(Boolean);
    let result = text;
    keywords.forEach(keyword => {
      const regex = new RegExp(`(${this.escapeRegex(keyword)})`, 'gi');
      result = result.replace(regex, '<mark>$1</mark>');
    });
    return result;
  }

  /**
   * Escape regex special characters
   */
  escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  /**
   * Show search results dropdown
   */
  showResults() {
    this.searchResults?.classList.add('active');
  }

  /**
   * Hide search results dropdown
   */
  hideResults() {
    this.searchResults?.classList.remove('active');
  }

  /**
   * Update index with new posts
   */
  updatePosts(posts) {
    this.posts = posts;
    this.index = this.buildIndex();
  }
}
