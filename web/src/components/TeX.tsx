import katex from 'katex';
import { useMemo } from 'react';

interface TeXProps {
  readonly math: string;
  readonly display?: boolean;
  readonly className?: string;
}

const cache = new Map<string, string>();

/** Render TeX with KaTeX; the output includes MathML for screen readers. */
export function renderTeX(math: string, display = false): string {
  const key = `${display ? 'D' : 'I'}:${math}`;
  let html = cache.get(key);
  if (html === undefined) {
    html = katex.renderToString(math, { displayMode: display, throwOnError: false, output: 'htmlAndMathml', strict: 'ignore' });
    cache.set(key, html);
  }
  return html;
}

export function TeX({ math, display = false, className }: TeXProps) {
  const html = useMemo(() => renderTeX(math, display), [math, display]);
  const Tag = display ? 'div' : 'span';
  return <Tag className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}
