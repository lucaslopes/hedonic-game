import { Fragment, type ReactNode } from 'react';
import { TeX } from './TeX';

/**
 * Render the tiny markup used by the dictionaries:
 * **bold**, *italic*, `code` and $TeX$. Everything else is plain text, so
 * translated strings never inject HTML.
 */
const TOKEN = /(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;

export function parseRich(text: string): ReactNode[] {
  return text.split(TOKEN).map((part, index) => {
    if (!part) return null;
    if (part.startsWith('$') && part.endsWith('$') && part.length > 2) return <TeX key={index} math={part.slice(1, -1)} />;
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) return <code key={index}>{part.slice(1, -1)}</code>;
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) return <em key={index}>{part.slice(1, -1)}</em>;
    return <Fragment key={index}>{part}</Fragment>;
  });
}

interface RichProps {
  readonly text: string;
  readonly as?: 'p' | 'span' | 'div';
  readonly className?: string;
}

export function Rich({ text, as: Tag = 'span', className }: RichProps) {
  return <Tag className={className}>{parseRich(text)}</Tag>;
}
