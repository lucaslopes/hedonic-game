import { useT } from '../../i18n';
import styles from './Legend.module.css';

type LegendMode = 'nodes' | 'edges' | 'types' | 'oriented' | 'walk';

function Swatch({ kind }: { kind: string }) {
  const arrow = (color: string) => <path d="M22 7 L30 10 L22 13 z" fill={color} />;
  switch (kind) {
    case 'grand':
    case 'singletons':
    case 'intermediate': {
      const tone = kind === 'grand' ? 'friend' : kind === 'singletons' ? 'stranger' : 'mid';
      return <rect x="3" y="3" width="26" height="14" rx="4" fill={`var(--${tone}-soft)`} stroke={`var(--${tone})`} strokeWidth="1.5" />;
    }
    case 'neutral':
      return <line x1="2" y1="10" x2="30" y2="10" stroke="var(--rule-strong)" strokeWidth="2" />;
    case 'clear':
      return (
        <>
          <line x1="2" y1="10" x2="24" y2="10" stroke="var(--clear)" strokeWidth="2.2" />
          {arrow('var(--clear)')}
        </>
      );
    case 'frustrated':
      return (
        <>
          <path d="M10 7 L2 10 L10 13 z" fill="var(--friend)" />
          <line x1="8" y1="10" x2="15" y2="10" stroke="var(--friend)" strokeWidth="2.4" />
          <line x1="15" y1="10" x2="24" y2="10" stroke="var(--stranger)" strokeWidth="2.4" />
          <circle cx="15" cy="10" r="2.4" fill="var(--card)" stroke="var(--ink)" strokeWidth="1.2" />
          {arrow('var(--stranger)')}
        </>
      );
    case 'friends':
      return (
        <>
          <line x1="2" y1="10" x2="24" y2="10" stroke="var(--friend)" strokeWidth="2.4" />
          <circle cx="12" cy="10" r="4" fill="var(--friend)" stroke="var(--card)" strokeWidth="1.6" />
          {arrow('var(--friend)')}
        </>
      );
    case 'strangers':
      return (
        <>
          <line x1="2" y1="10" x2="24" y2="10" stroke="var(--stranger)" strokeWidth="2.4" />
          <rect x="9" y="6.5" width="7" height="7" fill="var(--stranger)" stroke="var(--card)" strokeWidth="1.6" />
          {arrow('var(--stranger)')}
        </>
      );
    case 'tie':
      return <line x1="2" y1="10" x2="30" y2="10" stroke="var(--tie)" strokeWidth="2" strokeDasharray="4 3" />;
    case 'sink':
      return (
        <g transform="translate(16 10)">
          <circle r="7.5" fill="var(--ink)" />
          <path d="M-3.4 -3.2 L0 0.6 L3.4 -3.2 M-3.6 3.2 H3.6" fill="none" stroke="var(--paper)" strokeWidth="1.6" strokeLinecap="round" />
        </g>
      );
    case 'walk':
      return (
        <>
          <line x1="2" y1="10" x2="24" y2="10" stroke="var(--threshold)" strokeWidth="3.4" />
          {arrow('var(--threshold)')}
        </>
      );
    default:
      return null;
  }
}

export function MetagraphLegend({ mode }: { mode: LegendMode }) {
  const t = useT();
  const items: [string, string][] = [
    ['grand', t.metagraph.legend.grand],
    ['intermediate', t.metagraph.legend.intermediate],
    ['singletons', t.metagraph.legend.singletons],
  ];
  if (mode === 'edges') items.push(['neutral', t.metagraph.legend.neutral]);
  if (mode === 'types') {
    items.push(['clear', t.metagraph.legend.clear], ['frustrated', t.metagraph.legend.frustrated], ['tie', t.metagraph.legend.tie]);
  }
  if (mode === 'oriented' || mode === 'walk') {
    items.push(
      ['clear', t.metagraph.legend.clear],
      ['friends', t.metagraph.legend.friends],
      ['strangers', t.metagraph.legend.strangers],
      ['tie', t.metagraph.legend.tie],
      ['sink', t.metagraph.legend.sink],
    );
  }
  if (mode === 'walk') items.push(['walk', t.metagraph.legend.walk]);
  return (
    <ul className={styles.legend} aria-label={t.metagraph.legend.title}>
      {items.map(([kind, label]) => (
        <li key={kind}>
          <svg width="32" height="20" viewBox="0 0 32 20" aria-hidden="true">
            <Swatch kind={kind} />
          </svg>
          {label}
        </li>
      ))}
    </ul>
  );
}
