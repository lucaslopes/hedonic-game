/**
 * A small, dependency-free BibTeX reader for the site's reference cards.
 *
 * It supports the subset used by web/content/references.bib: @type{key, …}
 * entries with brace- or quote-delimited values, bare numbers, `%` comments,
 * nested braces and common LaTeX accents. Links are only produced from
 * fields that exist in the source (doi, url, arXiv eprint) — never invented.
 */

const ACCENTS = {
  '"': { a: 'ä', e: 'ë', i: 'ï', o: 'ö', u: 'ü', A: 'Ä', E: 'Ë', I: 'Ï', O: 'Ö', U: 'Ü' },
  "'": { a: 'á', e: 'é', i: 'í', o: 'ó', u: 'ú', A: 'Á', E: 'É', I: 'Í', O: 'Ó', U: 'Ú', c: 'ć' },
  '`': { a: 'à', e: 'è', i: 'ì', o: 'ò', u: 'ù', A: 'À', E: 'È' },
  '^': { a: 'â', e: 'ê', i: 'î', o: 'ô', u: 'û' },
  '~': { a: 'ã', o: 'õ', n: 'ñ', A: 'Ã', O: 'Õ', N: 'Ñ' },
  c: { c: 'ç', C: 'Ç' },
};

/** Convert the LaTeX used in bibliographies to plain Unicode text. */
export function latexToText(value) {
  let text = value;
  // \"{o}, {\"o}, \"o, \c{c}, {\c c}
  text = text.replace(/\{?\\([`'^"~])\s*\{?([A-Za-z])\}?\}?/g, (match, accent, letter) => ACCENTS[accent]?.[letter] ?? letter);
  text = text.replace(/\{?\\c\s*\{?([A-Za-z])\}?\}?/g, (match, letter) => ACCENTS.c[letter] ?? letter);
  text = text.replace(/\\(?:url|textit|emph|textbf|mathrm)\{([^{}]*)\}/g, '$1');
  text = text.replace(/\\&/g, '&').replace(/\\%/g, '%').replace(/\\_/g, '_');
  text = text.replace(/---/g, '—').replace(/--/g, '–').replace(/~/g, ' ');
  text = text.replace(/[{}]/g, '');
  return text.replace(/\s+/g, ' ').trim();
}

function readValue(source, start) {
  let i = start;
  while (/\s/.test(source[i])) i += 1;
  const open = source[i];
  if (open === '{') {
    let depth = 0;
    for (let j = i; j < source.length; j += 1) {
      if (source[j] === '{') depth += 1;
      else if (source[j] === '}') {
        depth -= 1;
        if (depth === 0) return { raw: source.slice(i + 1, j), end: j + 1 };
      }
    }
    throw new SyntaxError(`unterminated brace value at ${i}`);
  }
  if (open === '"') {
    let depth = 0;
    for (let j = i + 1; j < source.length; j += 1) {
      if (source[j] === '{') depth += 1;
      else if (source[j] === '}') depth -= 1;
      else if (source[j] === '"' && depth === 0) return { raw: source.slice(i + 1, j), end: j + 1 };
    }
    throw new SyntaxError(`unterminated quoted value at ${i}`);
  }
  const match = /^[\w.-]+/.exec(source.slice(i));
  if (!match) throw new SyntaxError(`unexpected character ${JSON.stringify(open)} at ${i}`);
  return { raw: match[0], end: i + match[0].length };
}

/** Parse BibTeX source into raw entries: { type, key, fields }. */
export function parseBibtex(source) {
  const withoutComments = source
    .split('\n')
    .filter((line) => !line.trimStart().startsWith('%'))
    .join('\n');
  const entries = [];
  const header = /@(\w+)\s*\{\s*([^,\s]+)\s*,/g;
  let match;
  while ((match = header.exec(withoutComments))) {
    const fields = {};
    let i = header.lastIndex;
    for (;;) {
      while (/[\s,]/.test(withoutComments[i])) i += 1;
      if (withoutComments[i] === '}') {
        i += 1;
        break;
      }
      const name = /^([A-Za-z][\w-]*)\s*=/.exec(withoutComments.slice(i));
      if (!name) throw new SyntaxError(`malformed field in entry ${match[2]}`);
      i += name[0].length;
      const { raw, end } = readValue(withoutComments, i);
      fields[name[1].toLowerCase()] = raw;
      i = end;
    }
    header.lastIndex = i;
    entries.push({ type: match[1].toLowerCase(), key: match[2], fields });
  }
  return entries;
}

/** "Last, First" → "First Last"; bare initials get a period. */
export function formatAuthor(name) {
  const clean = latexToText(name);
  const [last, first] = clean.includes(',') ? clean.split(',').map((part) => part.trim()) : [null, null];
  const full = last !== null ? `${first} ${last}`.trim() : clean;
  return full.replace(/(^|\s)([A-Z])(?=\s|$)/g, '$1$2.');
}

/** Reference-card data, sorted by year (newest first) then key. */
export function toReferences(entries) {
  return entries
    .map(({ type, key, fields }) => {
      const text = (name) => (fields[name] === undefined ? undefined : latexToText(fields[name]));
      const links = [];
      if (fields.doi) links.push({ kind: 'doi', label: `doi:${text('doi')}`, href: `https://doi.org/${text('doi')}` });
      if (fields.eprint && /arxiv/i.test(fields.archiveprefix ?? '')) {
        links.push({ kind: 'arxiv', label: `arXiv:${text('eprint')}`, href: `https://arxiv.org/abs/${text('eprint')}` });
      }
      const url = text('url');
      if (url && !links.some((link) => link.href === url)) links.push({ kind: 'url', label: url.replace(/^https?:\/\//, ''), href: url });
      return {
        key,
        type,
        title: text('title') ?? key,
        authors: (fields.author ?? '').split(/\s+and\s+/).filter(Boolean).map(formatAuthor),
        year: text('year') ?? null,
        venue: text('journal') ?? text('booktitle') ?? text('howpublished') ?? null,
        volume: text('volume') ?? null,
        number: text('number') ?? null,
        pages: text('pages') ?? null,
        publisher: text('publisher') ?? null,
        note: text('note') ?? null,
        keywords: (text('keywords') ?? '').split(',').map((k) => k.trim()).filter(Boolean),
        links,
      };
    })
    .sort((a, b) => Number(b.year ?? 0) - Number(a.year ?? 0) || a.key.localeCompare(b.key));
}
