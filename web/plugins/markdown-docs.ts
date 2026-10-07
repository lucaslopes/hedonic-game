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
 *   ```api Game.member```   generated API block from src/generated/api.json
 *   ```api-members```       table of the public `Game` members
 *   ```api-constants```     public module constants of Game.py
 *   ```package-info```      version, Python and native-dependency pins
 *   ```cli-commands```      experiments CLI registry (optional prefix filter)
 *   ```sweep-methods```     disjoint SBM sweep method table
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

interface ApiParameter {
  name: string;
  kind: string;
  annotation: string | null;
  default: string | null;
  description?: string;
  docType?: string;
}

interface ApiDoc {
  summary: string;
  description: string;
  sections: Record<string, string>;
}

interface ApiMember {
  kind: 'method' | 'property' | 'alias' | 'function';
  name: string;
  qualname: string;
  signature?: string;
  parameters?: ApiParameter[];
  returns?: string | null;
  doc?: ApiDoc | null;
  aliasOf?: string;
  settable?: boolean;
  path: string;
  line: number;
  endLine: number;
  module?: string;
}

export interface ApiData {
  repository: { url: string; ref: string };
  package: { version: string | null; nativeDependency: string | null; requiresPython: string | null };
  exports: string[];
  classes: { name: string; qualname: string; bases: string[]; doc: ApiDoc | null; path: string; line: number; endLine: number; members: ApiMember[] }[];
  functions: ApiMember[];
  constants: { name: string; value: unknown; path: string; line: number }[];
  cli: { path: string; commands: { name: string; module: string | null; summary: string; needsData: string | null }[] };
  experiments: {
    disjoint: {
      sbmSweep: { path: string; methods: Record<string, { method_call_name: string; parameters: Record<string, unknown> }> };
    };
  };
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

/** Convert the reStructuredText idioms used in the Python docstrings to Markdown. */
export function rstToMarkdown(text: string): string {
  return text
    .replace(/:(?:meth|func|class|mod|attr|data):`~?([^`]+)`/g, '`$1`')
    .replace(/``([^`]+)``/g, '`$1`')
    .replace(/::\s*$/gm, ':');
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
  api: ApiData;
  references: ReferencesData;
  base: string;
  md: Markdown;
}

function sourceHref(ctx: RenderContext, path: string, line?: number, endLine?: number): string {
  const anchor = line ? `#L${line}${endLine && endLine !== line ? `-L${endLine}` : ''}` : '';
  return `${ctx.api.repository.url}/blob/${ctx.api.repository.ref}/${path}${anchor}`;
}

function sourceLink(ctx: RenderContext, path: string, line?: number, endLine?: number): string {
  return `<a class="api-source" href="${escapeHtml(sourceHref(ctx, path, line, endLine))}">${escapeHtml(path)}${line ? `:${line}` : ''}</a>`;
}

function findMember(ctx: RenderContext, qualname: string): ApiMember {
  const [owner, name] = qualname.includes('.') ? qualname.split('.') : [null, qualname];
  if (owner) {
    const cls = ctx.api.classes.find((c) => c.name === owner);
    const member = cls?.members.find((m) => m.name === name);
    if (member) return member;
  } else {
    const fn = ctx.api.functions.find((f) => f.name === name);
    if (fn) return { ...fn, kind: 'function' };
  }
  throw new Error(`docs: unknown API member "${qualname}" (run npm run docs:generate?)`);
}

function renderSignature(member: ApiMember, owner: string | null): string {
  const params = member.parameters ?? [];
  const call = owner ? `${owner}.${member.name}` : member.name;
  if (member.kind === 'property') {
    return `${call}: ${member.returns ?? 'object'}`;
  }
  const parts: string[] = [];
  let starEmitted = false;
  for (const p of params) {
    let text = p.name;
    if (p.kind === 'var-positional') {
      text = `*${text}`;
      starEmitted = true;
    } else if (p.kind === 'var-keyword') {
      text = `**${text}`;
    } else if (p.kind === 'keyword-only' && !starEmitted) {
      parts.push('*');
      starEmitted = true;
    }
    if (p.annotation) text += `: ${p.annotation}`;
    if (p.default !== null) text += p.annotation ? ` = ${p.default}` : `=${p.default}`;
    parts.push(text);
  }
  const returns = member.returns ? ` -> ${member.returns}` : '';
  const oneLine = `${call}(${parts.join(', ')})${returns}`;
  if (oneLine.length <= 72) return oneLine;
  return `${call}(\n${parts.map((part) => `    ${part},`).join('\n')}\n)${returns}`;
}

function renderApiBlock(ctx: RenderContext, qualname: string): string {
  const member = findMember(ctx, qualname);
  const owner = qualname.includes('.') ? qualname.split('.')[0] : null;
  const id = `api-${slugify(qualname)}`;
  if (member.kind === 'alias') {
    return `<section class="api-block" id="${id}"><div class="api-head"><code class="api-name">${escapeHtml(qualname)}</code><span class="api-kind">alias</span>${sourceLink(ctx, member.path, member.line)}</div><p>Alias of <a href="#api-${slugify(`${owner}.${member.aliasOf}`)}"><code>${escapeHtml(`${owner}.${member.aliasOf}`)}</code></a>; both names share one implementation.</p></section>\n`;
  }
  const doc = member.doc;
  const kindLabel = member.kind === 'property' ? (member.settable ? 'property · settable' : 'property') : member.kind;
  let html = `<section class="api-block" id="${id}">`;
  html += `<div class="api-head"><code class="api-name">${escapeHtml(qualname)}</code><span class="api-kind">${escapeHtml(kindLabel)}</span>${sourceLink(ctx, member.path, member.line, member.endLine)}</div>`;
  html += `<pre class="api-signature"><code class="language-python">${escapeHtml(renderSignature(member, owner))}</code></pre>`;
  if (doc?.summary) html += ctx.md.render(rstToMarkdown(doc.summary));
  if (doc?.description) html += `<div class="api-description">${ctx.md.render(rstToMarkdown(doc.description))}</div>`;
  const params = (member.parameters ?? []).filter((p) => !['args', 'kwargs'].includes(p.name) || p.description);
  if (params.length && member.kind !== 'property') {
    html += '<div class="table-scroll"><table class="api-params"><thead><tr><th scope="col">Parameter</th><th scope="col">Type</th><th scope="col">Default</th><th scope="col">Description</th></tr></thead><tbody>';
    for (const p of params) {
      const type = p.annotation ?? p.docType ?? '';
      const prefix = p.kind === 'var-positional' ? '*' : p.kind === 'var-keyword' ? '**' : '';
      const defaultCell =
        p.default === null
          ? '<span class="api-required">—</span>'
          : p.default === '_UNSET'
            ? '<span title="Omitted: reuse Game.memberships when valid">omitted</span>'
            : `<code>${escapeHtml(p.default)}</code>`;
      const description = p.description ? ctx.md.render(rstToMarkdown(p.description)) : '';
      html += `<tr><th scope="row"><code>${prefix}${escapeHtml(p.name).replace(/_/g, '_<wbr>')}</code>${p.kind === 'keyword-only' ? ' <span class="api-badge">keyword</span>' : ''}</th><td>${type ? `<code>${escapeHtml(type)}</code>` : ''}</td><td>${defaultCell}</td><td>${description}</td></tr>`;
    }
    html += '</tbody></table></div>';
  }
  const returns = doc?.sections?.Returns;
  if (returns) html += `<div class="api-returns"><h4>Returns</h4>${ctx.md.render(rstToMarkdown(returns))}</div>`;
  const notes = doc?.sections?.Notes;
  if (notes) html += `<div class="api-notes"><h4>Notes</h4>${ctx.md.render(rstToMarkdown(notes))}</div>`;
  html += '</section>\n';
  return html;
}

function renderApiMembers(ctx: RenderContext): string {
  const game = ctx.api.classes.find((c) => c.name === 'Game');
  if (!game) throw new Error('docs: Game class missing from api.json');
  const rows = game.members
    .map((m) => {
      const summary = m.kind === 'alias' ? `Alias of <code>${escapeHtml(m.aliasOf ?? '')}</code>.` : ctx.md.renderInline(rstToMarkdown(m.doc?.summary ?? ''));
      return `<tr><th scope="row"><a href="#api-${slugify(`Game.${m.name}`)}"><code>${escapeHtml(m.name)}</code></a></th><td>${escapeHtml(m.kind)}</td><td>${summary}</td></tr>`;
    })
    .join('');
  return `<div class="table-scroll"><table class="api-members"><thead><tr><th scope="col">Member</th><th scope="col">Kind</th><th scope="col">Summary</th></tr></thead><tbody>${rows}</tbody></table></div>\n`;
}

function renderConstants(ctx: RenderContext): string {
  const rows = ctx.api.constants
    .map(
      (c) =>
        `<tr><th scope="row"><code>${escapeHtml(c.name)}</code></th><td><code>${escapeHtml(JSON.stringify(c.value))}</code></td><td>${sourceLink(ctx, c.path, c.line)}</td></tr>`,
    )
    .join('');
  return `<div class="table-scroll"><table><thead><tr><th scope="col">Constant</th><th scope="col">Value</th><th scope="col">Source</th></tr></thead><tbody>${rows}</tbody></table></div>\n`;
}

function renderPackageInfo(ctx: RenderContext): string {
  const pkg = ctx.api.package;
  const rows: [string, string][] = [
    ['Package version', pkg.version ?? 'unknown'],
    ['Python', pkg.requiresPython ?? 'unknown'],
    ['Native detector', pkg.nativeDependency ?? 'unknown'],
    ['Public exports', ctx.api.exports.join(', ')],
    ['Source reference', ctx.api.repository.ref],
  ];
  return `<div class="table-scroll"><table><tbody>${rows
    .map(([key, value]) => `<tr><th scope="row">${escapeHtml(key)}</th><td><code>${escapeHtml(value)}</code></td></tr>`)
    .join('')}</tbody></table></div><p class="api-caption">Generated from <code>pyproject.toml</code> and <code>src/hedonic/__init__.py</code>.</p>\n`;
}

function renderCliCommands(ctx: RenderContext, filter: string): string {
  const prefixes = filter.split(/\s+/).filter(Boolean);
  const commands = ctx.api.cli.commands.filter((c) => prefixes.length === 0 || prefixes.some((p) => c.name.startsWith(p)));
  const rows = commands
    .map(
      (c) =>
        `<tr><th scope="row"><code>hedonic-exp ${escapeHtml(c.name)}</code></th><td>${escapeHtml(c.summary)}</td><td>${c.needsData ? escapeHtml(c.needsData) : 'none'}</td></tr>`,
    )
    .join('');
  return `<div class="table-scroll"><table class="cli-table"><thead><tr><th scope="col">Command</th><th scope="col">Summary</th><th scope="col">Data</th></tr></thead><tbody>${rows}</tbody></table></div><p class="api-caption">Generated from the <code>COMMANDS</code> registry in ${sourceLink(ctx, ctx.api.cli.path)}.</p>\n`;
}

function renderSweepMethods(ctx: RenderContext): string {
  const sweep = ctx.api.experiments.disjoint.sbmSweep;
  const rows = Object.entries(sweep.methods)
    .map(([name, spec]) => {
      const params = Object.entries(spec.parameters)
        .filter(([, value]) => value !== null)
        .map(([key, value]) => `<code>${escapeHtml(key)}=${escapeHtml(JSON.stringify(value))}</code>`)
        .join(' ');
      return `<tr><th scope="row">${escapeHtml(name)}</th><td><code>${escapeHtml(spec.method_call_name)}</code></td><td>${params || '—'}</td></tr>`;
    })
    .join('');
  return `<div class="table-scroll"><table class="methods-table"><thead><tr><th scope="col">Method</th><th scope="col">Call</th><th scope="col">Fixed parameters</th></tr></thead><tbody>${rows}</tbody></table></div><p class="api-caption">Generated from <code>METHODS</code> in ${sourceLink(ctx, sweep.path)}.</p>\n`;
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
      token.attrSet('href', `${ctx.api.repository.url}/blob/${ctx.api.repository.ref}/${path}${anchor ? `#${anchor}` : ''}`);
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
    const [kind, ...args] = token.info.trim().split(/\s+/);
    switch (kind) {
      case 'api':
        return renderApiBlock(full, args[0] ?? '');
      case 'api-members':
        return renderApiMembers(full);
      case 'cli-commands':
        return renderCliCommands(full, args.join(' '));
      case 'sweep-methods':
        return renderSweepMethods(full);
      case 'api-constants':
        return renderConstants(full);
      case 'package-info':
        return renderPackageInfo(full);
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
  api: ApiData;
  references: ReferencesData;
  base: string;
}

export function renderDocs({ files, api, references, base }: RenderInput): DocsBundle {
  const pages: DocPage[] = files.map(({ path, source }) => {
    const { data, body } = splitFrontMatter(source);
    const slug = data.slug ?? slugify(basename(path, '.md').replace(/^\d+-/, ''));
    const headings: DocHeading[] = [];
    const md = createMarkdown({ api, references, base }, slug, headings);
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
    repository: api.repository,
    packageVersion: api.package.version,
    nativeDependency: api.package.nativeDependency,
  };
}

export interface MarkdownDocsOptions {
  contentDir: string;
  apiFile: string;
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
  const api = JSON.parse(readFileSync(options.apiFile, 'utf8')) as ApiData;
  const references = JSON.parse(readFileSync(options.referencesFile, 'utf8')) as ReferencesData;
  return renderDocs({ files, api, references, base });
}

export function markdownDocs(options: MarkdownDocsOptions): Plugin {
  let base = '/';
  const watched = (file: string) =>
    (file.startsWith(options.contentDir) && file.endsWith('.md')) || file === options.apiFile || file === options.referencesFile;
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
      this.addWatchFile(options.apiFile);
      this.addWatchFile(options.referencesFile);
      const bundle = loadDocs(options, base);
      bundle.pages.forEach((page) => this.addWatchFile(resolve(options.contentDir, basename(page.sourcePath))));
      return `export default ${JSON.stringify(bundle)};`;
    },
    configureServer(server) {
      server.watcher.add([options.contentDir, options.apiFile, options.referencesFile]);
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
