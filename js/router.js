/**
 * Simple Hash Router
 * SPA routing using window.location.hash
 */

class Router {
  constructor(routes) {
    this.routes = routes;
    this.notFoundHandler = null;

    // Listen for hash changes
    window.addEventListener('hashchange', () => this.route());
    window.addEventListener('load', () => this.route());
  }

  /**
   * Match route pattern against current hash
   * Supports patterns like: /post/:id, /tag/:name
   */
  matchRoute(pattern, hash) {
    const patternParts = pattern.split('/').filter(Boolean);
    const hashParts = hash.split('/').filter(Boolean);

    // Check if the number of parts match
    if (patternParts.length !== hashParts.length) {
      return null;
    }

    const params = {};

    for (let i = 0; i < patternParts.length; i++) {
      const patternPart = patternParts[i];
      const hashPart = hashParts[i];

      // Check for dynamic parameter
      if (patternPart.startsWith(':')) {
        const paramName = patternPart.slice(1);
        params[paramName] = hashPart;
      } else if (patternPart !== hashPart) {
        return null;
      }
    }

    return { params };
  }

  /**
   * Navigate to a specific hash
   */
  navigate(hash) {
    window.location.hash = hash;
  }

  /**
   * Get current route info
   */
  getCurrentRoute() {
    const hash = window.location.hash.slice(1) || '/';
    return hash;
  }

  /**
   * Route to appropriate handler based on current hash
   */
  route() {
    const hash = window.location.hash.slice(1) || '/';

    for (const [pattern, handler] of Object.entries(this.routes)) {
      const match = this.matchRoute(pattern, hash);
      if (match) {
        handler(match.params);
        return;
      }
    }

    // 404 - no matching route
    if (this.notFoundHandler) {
      this.notFoundHandler();
    } else {
      this.render404();
    }
  }

  /**
   * Set custom 404 handler
   */
  setNotFoundHandler(handler) {
    this.notFoundHandler = handler;
  }

  /**
   * Default 404 render
   */
  render404() {
    const content = document.getElementById('content');
    content.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <div class="empty-state-title">页面未找到</div>
        <p>抱歉，您访问的页面不存在。</p>
        <a href="#/" class="post-tag" style="margin-top: 1rem; display: inline-block;">返回首页</a>
      </div>
    `;
  }
}
