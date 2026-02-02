/**
 * Theme Manager
 * Handles light/dark theme switching with localStorage persistence
 */

class ThemeManager {
  constructor() {
    this.theme = 'light';
    this.init();
  }

  init() {
    // Get saved theme or detect system preference
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    this.theme = savedTheme || (prefersDark ? 'dark' : 'light');
    this.applyTheme();

    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      if (!localStorage.getItem('theme')) {
        this.theme = e.matches ? 'dark' : 'light';
        this.applyTheme();
      }
    });

    // Bind toggle button
    this.bindToggle();
  }

  applyTheme() {
    document.documentElement.setAttribute('data-theme', this.theme);
    this.updatePrismTheme();
  }

  updatePrismTheme() {
    const lightTheme = document.getElementById('prism-theme-light');
    const darkTheme = document.getElementById('prism-theme-dark');

    if (this.theme === 'dark') {
      lightTheme?.setAttribute('disabled', 'true');
      darkTheme?.removeAttribute('disabled');
    } else {
      lightTheme?.removeAttribute('disabled');
      darkTheme?.setAttribute('disabled', 'true');
    }
  }

  toggle() {
    this.theme = this.theme === 'light' ? 'dark' : 'light';
    localStorage.setItem('theme', this.theme);
    this.applyTheme();
  }

  bindToggle() {
    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => this.toggle());
    }
  }

  getTheme() {
    return this.theme;
  }
}

// Initialize theme manager
const themeManager = new ThemeManager();
