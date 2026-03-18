#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const targetRoot = path.resolve(__dirname, '..');
const workspaceRoot = path.resolve(targetRoot, '..');
const sourceRoot = path.join(workspaceRoot, 'blog', 'source');
const sourcePostsRoot = path.join(sourceRoot, '_posts');
const targetPostsRoot = path.join(targetRoot, '_posts');
const targetImgRoot = path.join(targetRoot, 'assets', 'img', 'posts');
const targetMediaRoot = path.join(targetRoot, 'assets', 'media', 'posts');
const reportPath = path.join(targetRoot, 'tools', 'migration-report.json');
const SLUG_OVERRIDES = new Map([
  ['齐次坐标到底是什么', 'homogeneous-coordinates'],
  ['03 为什么深度学习现在才真正起飞？', 'why-deep-learning-takes-off-now'],
  ['前端性能优化方法', 'frontend-performance-optimization'],
  ['现代浏览器架构', 'modern-browser-architecture'],
]);

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp', '.avif']);
const MEDIA_EXTS = new Set(['.mp4', '.webm', '.ogg', '.mov', '.m4v']);

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walk(fullPath));
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      files.push(fullPath);
    }
  }
  return files.sort();
}

function stripQuotes(value) {
  if (typeof value !== 'string') return value;
  const trimmed = value.trim();
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function parseArrayLiteral(raw) {
  const inner = raw.trim().replace(/^\[/, '').replace(/\]$/, '');
  if (!inner.trim()) return [];
  return inner
    .split(',')
    .map((item) => stripQuotes(item))
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseFrontMatter(content) {
  if (!content.startsWith('---\n')) {
    return { data: {}, body: content };
  }

  const end = content.indexOf('\n---\n', 4);
  if (end === -1) {
    return { data: {}, body: content };
  }

  const raw = content.slice(4, end);
  const body = content.slice(end + 5);
  const data = {};
  let currentKey = null;

  for (const line of raw.split('\n')) {
    if (/^\s*#/.test(line) || !line.trim()) continue;

    const listMatch = line.match(/^\s*-\s+(.*)$/);
    if (listMatch && currentKey) {
      if (!Array.isArray(data[currentKey])) data[currentKey] = [];
      data[currentKey].push(stripQuotes(listMatch[1]));
      continue;
    }

    const keyMatch = line.match(/^([A-Za-z0-9_]+):\s*(.*)$/);
    if (!keyMatch) {
      currentKey = null;
      continue;
    }

    const key = keyMatch[1];
    const rawValue = keyMatch[2];
    currentKey = key;

    if (!rawValue) {
      data[key] = [];
      continue;
    }

    if (rawValue.startsWith('[') && rawValue.endsWith(']')) {
      data[key] = parseArrayLiteral(rawValue);
      continue;
    }

    const normalized = stripQuotes(rawValue);
    if (normalized === 'true') data[key] = true;
    else if (normalized === 'false') data[key] = false;
    else data[key] = normalized;
  }

  return { data, body };
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function hashText(text) {
  return crypto.createHash('sha1').update(text).digest('hex').slice(0, 10);
}

function slugify(input, fallbackSeed) {
  const normalized = (input || '')
    .normalize('NFKC')
    .replace(/[\/\\]/g, '-')
    .replace(/[（）()【】\[\]{}“”"'`]+/g, '')
    .replace(/[：:，,。!！?？;；]+/g, '-')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .replace(/[^\p{Letter}\p{Number}-]+/gu, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .toLowerCase();

  if (normalized) return normalized;
  return `post-${hashText(fallbackSeed)}`;
}

function getDatePrefix(rawDate) {
  const match = String(rawDate || '').match(/\d{4}-\d{2}-\d{2}/);
  if (match) return match[0];
  return '1970-01-01';
}

function toArray(value) {
  if (Array.isArray(value)) return value.filter(Boolean);
  if (typeof value === 'string' && value.trim()) return [value.trim()];
  return [];
}

function cleanupBody(body) {
  return body
    .replace(/\r\n/g, '\n')
    .replace(/[\u200b\u200c\u200d\ufeff]/g, '')
    .replace(/\n{3,}/g, '\n\n');
}

function resolveAssetPath(mdPath, rawRelativePath) {
  const relativePath = rawRelativePath.replace(/^\.\//, '');
  const decodedRelativePath = decodeURIComponent(relativePath);
  const sameDir = path.resolve(path.dirname(mdPath), rawRelativePath);
  if (fs.existsSync(sameDir)) return sameDir;
  const decodedSameDir = path.resolve(path.dirname(mdPath), `./${decodedRelativePath}`);
  if (fs.existsSync(decodedSameDir)) return decodedSameDir;

  const postAssetDir = path.join(path.dirname(mdPath), path.basename(mdPath, '.md'));
  const siblingAsset = path.join(postAssetDir, relativePath);
  if (fs.existsSync(siblingAsset)) return siblingAsset;
  const decodedSiblingAsset = path.join(postAssetDir, decodedRelativePath);
  if (fs.existsSync(decodedSiblingAsset)) return decodedSiblingAsset;

  const sourceScoped = path.join(sourceRoot, relativePath);
  if (fs.existsSync(sourceScoped)) return sourceScoped;
  const decodedSourceScoped = path.join(sourceRoot, decodedRelativePath);
  if (fs.existsSync(decodedSourceScoped)) return decodedSourceScoped;

  return null;
}

function copyAsset(sourceFile, slug) {
  const ext = path.extname(sourceFile).toLowerCase();
  const fileName = path.basename(sourceFile);
  const isMedia = MEDIA_EXTS.has(ext);
  const isImage = IMAGE_EXTS.has(ext) || !isMedia;
  const outputDir = isMedia
    ? path.join(targetMediaRoot, slug)
    : path.join(targetImgRoot, slug);

  ensureDir(outputDir);
  const targetFile = path.join(outputDir, fileName);
  fs.copyFileSync(sourceFile, targetFile);

  return isMedia
    ? `/assets/media/posts/${slug}/${fileName}`
    : `/assets/img/posts/${slug}/${fileName}`;
}

function writePost(filePath, frontMatterLines, body) {
  const content = ['---', ...frontMatterLines, '---', '', body.trim(), ''].join('\n');
  fs.writeFileSync(filePath, content);
}

function buildFrontMatter(meta, body) {
  const lines = [];
  lines.push(`title: ${JSON.stringify(meta.title)}`);
  lines.push(`date: ${meta.date}`);

  if (meta.lastModifiedAt) {
    lines.push(`last_modified_at: ${meta.lastModifiedAt}`);
  }

  if (meta.description) {
    lines.push(`description: ${JSON.stringify(meta.description)}`);
  }

  if (meta.categories.length) {
    lines.push('categories:');
    for (const category of meta.categories) {
      lines.push(`  - ${category}`);
    }
  }

  if (meta.tags.length) {
    lines.push('tags:');
    for (const tag of meta.tags) {
      lines.push(`  - ${tag}`);
    }
  }

  if (meta.math) {
    lines.push('math: true');
  }

  if (meta.mermaid) {
    lines.push('mermaid: true');
  }

  if (body.includes('<iframe')) {
    lines.push('render_with_liquid: false');
  }

  return lines;
}

function main() {
  ensureDir(targetPostsRoot);
  ensureDir(targetImgRoot);
  ensureDir(targetMediaRoot);

  const sourcePosts = walk(sourcePostsRoot);
  const slugSet = new Set();
  const titleMap = new Map();
  const metadata = [];

  for (const sourcePath of sourcePosts) {
    const content = fs.readFileSync(sourcePath, 'utf8');
    const { data, body } = parseFrontMatter(content);
    const title = data.title || path.basename(sourcePath, '.md');
    const date = data.date || '1970-01-01 00:00:00';
    const fallbackSeed = path.relative(sourcePostsRoot, sourcePath);
    let slug =
      SLUG_OVERRIDES.get(title) || SLUG_OVERRIDES.get(path.basename(sourcePath, '.md')) || slugify(path.basename(sourcePath, '.md'), fallbackSeed);
    if (!slug || slugSet.has(slug)) {
      slug = `${slug || 'post'}-${hashText(fallbackSeed)}`;
    }
    slugSet.add(slug);

    const datePrefix = getDatePrefix(date);
    const targetName = `${datePrefix}-${slug}.md`;
    const targetPath = path.join(targetPostsRoot, targetName);
    const categories = toArray(data.categories);
    const tags = toArray(data.tags);
    const item = {
      sourcePath,
      targetPath,
      title,
      slug,
      permalink: `/posts/${slug}/`,
      date,
      lastModifiedAt: data.lastmod || null,
      description: typeof data.description === 'string' ? data.description : null,
      categories,
      tags,
      math: Boolean(data.mathjax) || body.includes('$$'),
      mermaid: body.includes('```mermaid'),
    };
    metadata.push(item);
    titleMap.set(title, item.permalink);
    titleMap.set(path.basename(sourcePath, '.md'), item.permalink);
  }

  const report = {
    migrated: 0,
    unresolvedAssets: [],
    unresolvedPostLinks: [],
    postsWithVideo: [],
    postsWithIframe: [],
  };

  for (const item of metadata) {
    const content = fs.readFileSync(item.sourcePath, 'utf8');
    const { body: rawBody } = parseFrontMatter(content);
    let body = cleanupBody(rawBody);
    const copiedAssets = new Map();

    body = body.replace(/\]\((\.\/[^)\n]+?)(?:\s+"[^"]*")?\)/g, (match, relPath) => {
      const sourceAsset = resolveAssetPath(item.sourcePath, relPath);
      if (!sourceAsset) {
        report.unresolvedAssets.push({ post: item.title, path: relPath });
        return match;
      }

      const webPath = copiedAssets.get(sourceAsset) || copyAsset(sourceAsset, item.slug);
      copiedAssets.set(sourceAsset, webPath);
      return match.replace(relPath, webPath);
    });

    body = body.replace(/(src|data-src)="(\.\/[^"]+)"/g, (match, attr, relPath) => {
      const sourceAsset = resolveAssetPath(item.sourcePath, relPath);
      if (!sourceAsset) {
        report.unresolvedAssets.push({ post: item.title, path: relPath });
        return match;
      }

      const webPath = copiedAssets.get(sourceAsset) || copyAsset(sourceAsset, item.slug);
      copiedAssets.set(sourceAsset, webPath);
      return `${attr}="${webPath}"`;
    });

    body = body.replace(/\{%\s*post_link\s+([^%]+?)\s*%\}/g, (match, rawTarget) => {
      const targetTitle = rawTarget.trim();
      const permalink = titleMap.get(targetTitle);
      if (!permalink) {
        report.unresolvedPostLinks.push({ post: item.title, target: targetTitle });
        return targetTitle;
      }
      return permalink;
    });

    if (body.includes('<video')) {
      report.postsWithVideo.push(item.title);
    }

    if (body.includes('<iframe')) {
      report.postsWithIframe.push(item.title);
    }

    const frontMatterLines = buildFrontMatter(item, body);
    writePost(item.targetPath, frontMatterLines, body);
    report.migrated += 1;
  }

  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');

  console.log(
    JSON.stringify(
      {
        migrated: report.migrated,
        unresolvedAssets: report.unresolvedAssets.length,
        unresolvedPostLinks: report.unresolvedPostLinks.length,
        postsWithVideo: report.postsWithVideo.length,
        postsWithIframe: report.postsWithIframe.length,
        reportPath,
      },
      null,
      2
    )
  );
}

main();
