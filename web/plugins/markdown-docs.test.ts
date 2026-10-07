import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadDocs, rstToMarkdown, slugify, splitFrontMatter } from './markdown-docs';

const root = resolve(__dirname, '..');
const bundle = loadDocs(
  {
    contentDir: resolve(root, 'content/docs'),
    apiFile: resolve(root, 'src/generated/api.json'),
    referencesFile: resolve(root, 'src/generated/references.json'),
  },
  '/hedonic-game/',
);
const page = (slug: string) => {
  const found = bundle.pages.find((p) => p.slug === slug);
  if (!found) throw new Error(`missing page ${slug}`);
  return found;
};

describe('documentation bundle', () => {
  it('renders every Markdown page in order with unique slugs', () => {
    const slugs = bundle.pages.map((p) => p.slug);
    expect(slugs).toEqual([
      'getting-started',
      'concepts',
      'game-api',
      'community-hedonic',
      'disjoint-vs-overlapping',
      'resolution',
      'familiarity-index',
      'metagraph',
      'better-response',
      'reproducibility',
      'limitations',
      'glossary',
      'references',
    ]);
    bundle.pages.forEach((p) => {
      expect(p.title).not.toBe(p.slug);
      expect(p.html.length).toBeGreaterThan(200);
    });
  });

  it('generates the API reference from the Python source', () => {
    const html = page('community-hedonic').html;
    expect(html).toContain('id="api-game-community-hedonic"');
    for (const parameter of ['initial_<wbr>membership', 'max_<wbr>memberships', 'resolution', 'allow_<wbr>isolation', 'local_<wbr>move_<wbr>only']) {
      expect(html).toContain(parameter);
    }
    expect(html).toContain('src/hedonic/Game.py');
    expect(page('game-api').html).toContain('id="api-game-to-igraph"');
    expect(page('reproducibility').html).toContain('hedonic-exp reproduce-disjoint');
  });

  it('renders math with KaTeX and resolves citations and links', () => {
    const html = page('familiarity-index').html;
    expect(html).toContain('class="katex"');
    expect(html).not.toMatch(/\$\\Delta/);
    expect(page('concepts').html).toContain('href="#ref-traag2011cpm"');
    expect(page('concepts').html).toContain('href="/hedonic-game/#dilemma"');
    expect(page('game-api').html).toMatch(/href="https:\/\/github\.com\/[^"]+\/blob\/[^"]+\/src\/hedonic\/Game\.py"/);
  });

  it('builds reference cards without inventing links', () => {
    const html = page('references').html;
    expect(html).toContain('id="ref-felipe2025hedonic"');
    expect(html).toContain('https://arxiv.org/abs/2509.03834');
    expect(html).toContain('https://doi.org/10.1186/s40649-018-0059-5');
    const doiLinks = html.match(/https:\/\/doi\.org\//g) ?? [];
    expect(doiLinks).toHaveLength(1);
  });

  it('collects headings for the table of contents', () => {
    const headings = page('better-response').headings;
    expect(headings.map((h) => h.text)).toContain('Sinks of the example');
    headings.forEach((h) => expect(h.id.startsWith('better-response--')).toBe(true));
  });
});

describe('helpers', () => {
  it('parses front matter and slugs', () => {
    expect(splitFrontMatter('---\ntitle: A title\norder: 3\n---\nBody')).toEqual({ data: { title: 'A title', order: '3' }, body: 'Body' });
    expect(slugify('Sinks of the example (γ ≤ 1)')).toBe('sinks-of-the-example-1');
  });

  it('converts docstring reStructuredText to Markdown', () => {
    expect(rstToMarkdown('Use ``max_memberships`` and :meth:`community_hedonic`.')).toBe('Use `max_memberships` and `community_hedonic`.');
  });
});
