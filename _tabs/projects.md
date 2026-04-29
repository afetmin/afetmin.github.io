---
title: 项目
icon: fas fa-laptop-code
order: 4
---

<style>
  .projects-intro {
    margin-bottom: 2rem;
    color: var(--bs-secondary-color, #6c757d);
  }

  .projects-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1rem;
  }

  .project-card {
    height: 100%;
    padding: 1.2rem;
    border: 1px solid var(--bs-border-color, rgba(127, 127, 127, 0.18));
    border-radius: 1rem;
    background: rgba(127, 127, 127, 0.04);
    transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
  }

  .project-card:hover {
    transform: translateY(-2px);
    background: rgba(127, 127, 127, 0.07);
    border-color: rgba(127, 127, 127, 0.24);
  }

  .project-card h2 {
    margin-bottom: 0.45rem;
    font-size: 1.2rem;
  }

  .project-meta,
  .project-links {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
  }

  .project-meta {
    margin-bottom: 0.8rem;
    font-size: 0.92rem;
    color: var(--bs-secondary-color, #6c757d);
  }

  .project-card p {
    margin-bottom: 1rem;
  }

  .project-links a {
    display: inline-flex;
    align-items: center;
    padding: 0.34rem 0.72rem;
    border: 1px solid var(--bs-border-color, rgba(127, 127, 127, 0.2));
    border-radius: 999px;
    text-decoration: none;
    color: var(--bs-secondary-color, #6c757d);
    background: transparent;
    transition: border-color 0.2s ease, background 0.2s ease, color 0.2s ease;
  }

  .project-links a:hover {
    text-decoration: none;
    color: var(--link-color, #2f6fed);
    background: rgba(127, 127, 127, 0.06);
  }
</style>

<div class="projects-grid">
  <article class="project-card">
    <h2>vue-music-next</h2>
    <div class="project-meta">
      <span>Vue</span>
    </div>
    <p>一个用 Vue 全家桶实现的音乐播放器项目。</p>
    <div class="project-links">
      <a href="https://github.com/afetmin/vue-music-next" target="_blank" rel="noopener noreferrer">仓库</a>
    </div>
  </article>

  <article class="project-card">
    <h2>FlashSeal</h2>
    <div class="project-meta">
      <span>TypeScript</span>
    </div>
    <p>一个基于 Cloudflare 的加密阅后即焚分享工具。</p>
    <div class="project-links">
      <a href="https://github.com/afetmin/FlashSeal" target="_blank" rel="noopener noreferrer">仓库</a>
      <a href="https://www.flashseal.space" target="_blank" rel="noopener noreferrer">Demo</a>
    </div>
  </article>

  <article class="project-card">
    <h2>ai-job-copilot</h2>
    <div class="project-meta">
      <span>TypeScript / FastAPI</span>
    </div>
    <p>一个用大模型做简历分析与优化的 AI 应用。</p>
    <div class="project-links">
      <a href="https://github.com/afetmin/ai-job-copilot" target="_blank" rel="noopener noreferrer">仓库</a>
      <a href="http://43.135.177.195/" target="_blank" rel="noopener noreferrer">Demo</a>
    </div>
  </article>

  <article class="project-card">
    <h2>AutoSort-Bookmarks</h2>
    <div class="project-meta">
      <span>TypeScript</span>
    </div>
    <p>一个用 AI 自动整理浏览器收藏夹的工具。</p>
    <div class="project-links">
      <a href="https://github.com/afetmin/AutoSort-Bookmarks" target="_blank" rel="noopener noreferrer">仓库</a>
    </div>
  </article>
</div>
