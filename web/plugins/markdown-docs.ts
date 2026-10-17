/**
 * Build-time Markdown → HTML for the documentation page.
 *
 * Pages live in web/content/docs/*.md with a small front-matter block. They
 * are rendered in Node (no Markdown parser ships to the browser) and exposed
 * as the virtual module `virtual:hedonic-docs`.
 *
 * Extensions:
 *   $…$ and $$…$$          KaTeX math (fails the build on invalid TeX)
 *   [@key]                  citation linking to the reference cards
 *   ```references```        citation cards from src/generated/references.json
 *   repo:path#L1            link to a file in the GitHub repository
 *   story:#anchor           link to a section of the explainer page
 */

import { readFileSync, readdirSync } from 'node:fs';
import { basename, join, relative, resolve, sep } from 'node:path';
import katex from 'katex';
import MarkdownIt from 'markdown-it';
import type { MarkdownIt as Markdown, StateBlock, StateInline, Token } from 'markdown-it';
import type { Plugin } from 'vite';
import type { DocHeading, DocPage, DocsBundle } from './docs-types.ts';

export const DOCS_MODULE_ID = 'virtual:hedonic-docs';
const RESOLVED_ID = `\0${DOCS_MODULE_ID}`;

// ---------------------------------------------------------------------------
// Generated data shapes (subset of what the generators write)
// ---------------------------------------------------------------------------

export interface SiteData {
  repositoryUrl: string;
  sourceRef: string;
}

interface Reference {
  key: string;
  title: string;
  authors: string[];
  year: string | null;
  venue: string | null;
  volume: string | null;
  number: string | null;
  pages: string | null;
  publisher: string | null;
  note: string | null;
  keywords: string[];
  links: { kind: string; label: string; href: string }[];
}

export interface ReferencesData {
  references: Reference[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function slugify(text: string): string {
  return text
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/<[^>]+>/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function renderMath(tex: string, displayMode: boolean): string {
  return katex.renderToString(tex, { displayMode, throwOnError: true, output: 'htmlAndMathml', strict: 'ignore' });
}

interface FrontMatter {
  title?: string;
  summary?: string;
  order?: string;
  group?: string;
  slug?: string;
}

export function splitFrontMatter(source: string): { data: FrontMatter; body: string } {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(source);
  if (!match) return { data: {}, body: source };
  const data: Record<string, string> = {};
  for (const line of match[1].split(/\r?\n/)) {
    const pair = /^([A-Za-z]+):\s*(.*)$/.exec(line.trim());
    if (pair) data[pair[1]] = pair[2].replace(/^["']|["']$/g, '');
  }
  return { data, body: source.slice(match[0].length) };
}

// ---------------------------------------------------------------------------
// markdown-it extensions
// ---------------------------------------------------------------------------

function mathPlugin(md: Markdown): void {
  md.inline.ruler.after('escape', 'math_inline', (state: StateInline, silent: boolean) => {
    const { src, pos } = state;
    if (src.charCodeAt(pos) !== 0x24 || src.charCodeAt(pos + 1) === 0x24) return false;
    const first = src.charAt(pos + 1);
    if (!first || /\s/.test(first)) return false;
    let end = pos + 1;
    for (;;) {
      end = src.indexOf('$', end);
      if (end === -1) return false;
      if (src.charAt(end - 1) === '\\') {
        end += 1;
        continue;
      }
      break;
    }
    if (/\s/.test(src.charAt(end - 1)) || /\d/.test(src.charAt(end + 1))) return false;
    if (!silent) {
      const token = state.push('math_inline', 'math', 0);
      token.content = src.slice(pos + 1, end);
    }
    state.pos = end + 1;
    return true;
  });

  md.block.ruler.after(
    'blockquote',
    'math_block',
    (state: StateBlock, startLine: number, endLine: number, silent: boolean) => {
      const start = state.bMarks[startLine] + state.tShift[startLine];
      const firstLine = state.src.slice(start, state.eMarks[startLine]);
      if (!firstLine.startsWith('$$')) return false;
      if (silent) return true;
      let content = firstLine.slice(2);
      let line = startLine;
      let closed = content.trimEnd().endsWith('$$');
      if (closed) content = content.trimEnd().slice(0, -2);
      while (!closed && ++line < endLine) {
        const text = state.src.slice(state.bMarks[line] + state.tShift[line], state.eMarks[line]);
        if (text.trimEnd().endsWith('$$')) {
          content += `\n${text.trimEnd().slice(0, -2)}`;
          closed = true;
        } else {
          content += `\n${text}`;
        }
      }
      if (!closed) return false;
      state.line = line + 1;
      const token = state.push('math_block', 'math', 0);
      token.block = true;
      token.content = content.trim();
      token.map = [startLine, state.line];
      return true;
    },
    { alt: ['paragraph', 'reference', 'blockquote', 'list'] },
  );

  md.renderer.rules.math_inline = (tokens: Token[], idx: number) => renderMath(tokens[idx].content, false);
  md.renderer.rules.math_block = (tokens: Token[], idx: number) =>
    `<div class="math-display" role="math">${renderMath(tokens[idx].content, true)}</div>\n`;
}

// ---------------------------------------------------------------------------
// Renderers for generated blocks
// ---------------------------------------------------------------------------

interface RenderContext {
  site: SiteData;
  references: ReferencesData;
  base: string;
  md: Markdown;
}

const REFERENCE_GROUPS: [string, string][] = [
  ['project', 'This project'],
  ['cpm', 'Constant Potts Model'],
  ['leiden', 'Louvain and Leiden'],
  ['community-detection', 'Community detection'],
  ['hedonic-games', 'Hedonic and potential games'],
  ['frustration', 'Frustration and the Ising model'],
];

function referenceCard(ref: Reference): string {
  const authors = ref.authors.length > 4 ? `${ref.authors.slice(0, 3).join(', ')}, et al.` : ref.authors.join(', ');
  const details = [
    ref.venue,
    ref.volume ? `vol. ${ref.volume}${ref.number ? ` (${ref.number})` : ''}` : null,
    ref.pages ? `pp. ${ref.pages}` : null,
    ref.publisher,
  ]
    .filter(Boolean)
    .join(' · ');
  const links = ref.links.length
    ? `<p class="ref-links">${ref.links.map((l) => `<a href="${escapeHtml(l.href)}">${escapeHtml(l.label)}</a>`).join(' ')}</p>`
    : '<p class="ref-links ref-links-none">No DOI or URL is recorded in the project bibliography.</p>';
  return `<article class="ref-card" id="ref-${escapeHtml(ref.key)}"><p class="ref-meta">${escapeHtml(authors)}${ref.year ? ` · ${escapeHtml(ref.year)}` : ''}</p><h4 class="ref-title">${escapeHtml(ref.title)}</h4>${details ? `<p class="ref-venue">${escapeHtml(details)}</p>` : ''}${ref.note ? `<p class="ref-note">${escapeHtml(ref.note)}</p>` : ''}${links}<p class="ref-key"><code>${escapeHtml(ref.key)}</code></p></article>`;
}

function renderReferences(ctx: RenderContext): string {
  const seen = new Set<string>();
  let html = '';
  for (const [keyword, title] of REFERENCE_GROUPS) {
    const refs = ctx.references.references.filter((r) => !seen.has(r.key) && r.keywords[0] === keyword);
    if (!refs.length) continue;
    refs.forEach((r) => seen.add(r.key));
    html += `<h3 class="ref-group">${escapeHtml(title)}</h3><div class="ref-grid">${refs.map(referenceCard).join('')}</div>`;
  }
  const rest = ctx.references.references.filter((r) => !seen.has(r.key));
  if (rest.length) html += `<h3 class="ref-group">Other</h3><div class="ref-grid">${rest.map(referenceCard).join('')}</div>`;
  return `${html}\n`;
}

function citationLabel(ref: Reference): string {
  const last = (name: string) => name.split(' ').slice(-1)[0];
  const who = ref.authors.length > 2 ? `${last(ref.authors[0])} et al.` : ref.authors.map(last).join(' & ');
  return `${who} ${ref.year ?? ''}`.trim();
}

// ---------------------------------------------------------------------------
// Page rendering
// ---------------------------------------------------------------------------

function createMarkdown(ctx: Omit<RenderContext, 'md'>, slug: string, headings: DocHeading[]): Markdown {
  const md = new MarkdownIt({ html: true, linkify: false, typographer: true });
  mathPlugin(md);
  const full: RenderContext = { ...ctx, md };

  // Heading anchors, unique within the page, collected for the table of contents.
  // Page titles are <h2> on the docs page, so Markdown "##" renders as <h3>.
  md.core.ruler.push('heading_ids', (state) => {
    const used = new Set<string>();
    state.tokens.forEach((token, index) => {
      if (token.type !== 'heading_open') return;
      const level = Number(token.tag.slice(1));
      const shifted = `h${Math.min(6, level + 1)}`;
      token.tag = shifted;
      const close = state.tokens.slice(index).find((candidate) => candidate.type === 'heading_close');
      if (close) close.tag = shifted;
      const inline = state.tokens[index + 1];
      const text = (inline.children ?? []).map((child) => child.content).join('');
      let id = `${slug}--${slugify(text)}`;
      for (let n = 2; used.has(id); n += 1) id = `${slug}--${slugify(text)}-${n}`;
      used.add(id);
      token.attrSet('id', id);
      if (level === 2 || level === 3) headings.push({ id, text, level });
    });
  });

  const defaultLink = md.renderer.rules.link_open ?? ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const href = String(token.attrGet('href') ?? '');
    if (href.startsWith('repo:')) {
      const [path, anchor] = href.slice(5).split('#');
      token.attrSet('href', `${ctx.site.repositoryUrl}/blob/${ctx.site.sourceRef}/${path}${anchor ? `#${anchor}` : ''}`);
    } else if (href.startsWith('story:')) {
      token.attrSet('href', `${ctx.base}${href.slice(6)}`);
    } else if (href.startsWith('docs:')) {
      token.attrSet('href', href.slice(5));
    }
    return defaultLink(tokens, idx, options, env, self);
  };

  // Wide tables scroll inside their own container instead of widening the page.
  md.renderer.rules.table_open = () => '<div class="table-scroll"><table>\n';
  md.renderer.rules.table_close = () => '</table></div>\n';

  const defaultFence = md.renderer.rules.fence ?? ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const [kind] = token.info.trim().split(/\s+/);
    switch (kind) {
      case 'references':
        return renderReferences(full);
      case 'math':
        return `<div class="math-display" role="math">${renderMath(token.content, true)}</div>\n`;
      default:
        return defaultFence(tokens, idx, options, env, self);
    }
  };
  return md;
}

function resolveCitations(body: string, references: ReferencesData): string {
  return body.replace(/\[@([\w-]+)\]/g, (_match, key: string) => {
    const ref = references.references.find((r) => r.key === key);
    if (!ref) throw new Error(`docs: unknown citation [@${key}]`);
    return `<a class="cite" href="#ref-${key}">${escapeHtml(citationLabel(ref))}</a>`;
  });
}

export interface RenderInput {
  files: { path: string; source: string }[];
  site: SiteData;
  references: ReferencesData;
  base: string;
}

export function renderDocs({ files, site, references, base }: RenderInput): DocsBundle {
  const pages: DocPage[] = files.map(({ path, source }) => {
    const { data, body } = splitFrontMatter(source);
    const slug = data.slug ?? slugify(basename(path, '.md').replace(/^\d+-/, ''));
    const headings: DocHeading[] = [];
    const md = createMarkdown({ site, references, base }, slug, headings);
    const html = md.render(resolveCitations(body, references));
    return {
      slug,
      title: data.title ?? slug,
      summary: data.summary ?? '',
      group: data.group ?? 'Guide',
      order: Number(data.order ?? 999),
      html,
      headings,
      sourcePath: path,
    };
  });
  pages.sort((a, b) => a.order - b.order || a.slug.localeCompare(b.slug));
  const slugs = new Set<string>();
  for (const page of pages) {
    if (slugs.has(page.slug)) throw new Error(`docs: duplicate page slug "${page.slug}"`);
    slugs.add(page.slug);
  }
  return {
    pages,
    repository: { url: site.repositoryUrl, ref: site.sourceRef },
  };
}

export interface MarkdownDocsOptions {
  contentDir: string;
  siteFile: string;
  referencesFile: string;
}

export function loadDocs(options: MarkdownDocsOptions, base: string): DocsBundle {
  const repoRoot = resolve(options.contentDir, '..', '..', '..');
  const files = readdirSync(options.contentDir)
    .filter((name) => name.endsWith('.md'))
    .sort()
    .map((name) => {
      const full = join(options.contentDir, name);
      return { path: relative(repoRoot, full).split(sep).join('/'), source: readFileSync(full, 'utf8') };
    });
  const site = JSON.parse(readFileSync(options.siteFile, 'utf8')) as SiteData;
  const references = JSON.parse(readFileSync(options.referencesFile, 'utf8')) as ReferencesData;
  return renderDocs({ files, site, references, base });
}

export function markdownDocs(options: MarkdownDocsOptions): Plugin {
  let base = '/';
  const watched = (file: string) =>
    (file.startsWith(options.contentDir) && file.endsWith('.md')) || file === options.siteFile || file === options.referencesFile;
  return {
    name: 'hedonic-markdown-docs',
    configResolved(config) {
      base = config.base;
    },
    resolveId(id) {
      return id === DOCS_MODULE_ID ? RESOLVED_ID : undefined;
    },
    load(id) {
      if (id !== RESOLVED_ID) return undefined;
      this.addWatchFile(options.siteFile);
      this.addWatchFile(options.referencesFile);
      const bundle = loadDocs(options, base);
      bundle.pages.forEach((page) => this.addWatchFile(resolve(options.contentDir, basename(page.sourcePath))));
      return `export default ${JSON.stringify(bundle)};`;
    },
    configureServer(server) {
      server.watcher.add([options.contentDir, options.siteFile, options.referencesFile]);
      const refresh = (file: string) => {
        if (!watched(file)) return;
        const mod = server.moduleGraph.getModuleById(RESOLVED_ID);
        if (mod) server.moduleGraph.invalidateModule(mod);
        server.ws.send({ type: 'full-reload' });
      };
      server.watcher.on('add', refresh);
      server.watcher.on('change', refresh);
      server.watcher.on('unlink', refresh);
    },
  };
}
