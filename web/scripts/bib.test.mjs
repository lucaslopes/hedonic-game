import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { formatAuthor, latexToText, parseBibtex, toReferences } from './bib.mjs';

describe('BibTeX reader', () => {
  it('parses nested braces, quotes and bare numbers', () => {
    const [entry] = parseBibtex('@article{key1,\n  title = {{Nested} title},\n  year = 2020,\n  note = "quoted {value}"\n}');
    expect(entry).toMatchObject({ type: 'article', key: 'key1' });
    expect(entry.fields).toEqual({ title: '{Nested} title', year: '2020', note: 'quoted {value}' });
  });

  it('converts LaTeX accents and dashes', () => {
    expect(latexToText('Reichardt, J{\\"o}rg')).toBe('Reichardt, Jörg');
    expect(latexToText('pages 1--12')).toBe('pages 1–12');
    expect(latexToText('{From Louvain to Leiden}')).toBe('From Louvain to Leiden');
  });

  it('formats author names without inventing initials', () => {
    expect(formatAuthor('Traag, Vincent A')).toBe('Vincent A. Traag');
    expect(formatAuthor('Hwang, D-U')).toBe('D-U Hwang');
    expect(formatAuthor('Lucas Lopes Felipe')).toBe('Lucas Lopes Felipe');
  });

  it('only links to identifiers present in the source', () => {
    const source = readFileSync(resolve(__dirname, '../content/references.bib'), 'utf8');
    const references = toReferences(parseBibtex(source));
    expect(references.length).toBe(14);
    const byKey = Object.fromEntries(references.map((r) => [r.key, r]));
    expect(byKey.traag2011cpm.links).toEqual([]);
    expect(byKey.avrachenkov2017cooperative.links.map((l) => l.href)).toEqual(['https://doi.org/10.1186/s40649-018-0059-5']);
    expect(byKey.felipe2025hedonic.links.map((l) => l.href)).toEqual(['https://arxiv.org/abs/2509.03834']);
    expect(byKey.reichardt2006statistical.authors[0]).toBe('Jörg Reichardt');
  });
});
