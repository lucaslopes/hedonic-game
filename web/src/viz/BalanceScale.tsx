import { communityUtility, signOf, type CommunityCounts } from '../model';
import { useSpring } from '../hooks/useSpring';
import styles from './BalanceScale.module.css';

export interface BalanceLabels {
  readonly a: string;
  readonly b: string;
  readonly moreFriends: string;
  readonly fewerStrangers: string;
  readonly sameFriends: string;
  readonly sameStrangers: string;
}

interface BalanceScaleProps {
  readonly a: CommunityCounts;
  readonly b: CommunityCounts;
  readonly gamma: number;
  /**
   * Progressive disclosure: 0 anonymous members, 1 friends and strangers
   * (scale locked), 2 weighted by γ (scale tips), 3 same plus threshold cues.
   */
  readonly level: 0 | 1 | 2 | 3;
  readonly reducedMotion: boolean;
  readonly labels: BalanceLabels;
}

const PIVOT = { x: 320, y: 118 };
const ARM = 184;
const MAX_TILT = 16;

function friendSlots(count: number) {
  return Array.from({ length: count }, (_, i) => {
    const row = Math.floor(i / 4);
    const col = i % 4;
    return { x: -45 + col * 30 + (row % 2) * 15, y: 104 - row * 19 };
  });
}

function balloonSlots(count: number) {
  return Array.from({ length: count }, (_, i) => {
    const spread = Math.min(26, 110 / Math.max(1, count));
    return { x: (i - (count - 1) / 2) * spread, y: 26 + (i % 2) * 20 };
  });
}

function Pan({
  counts,
  gamma,
  level,
  side,
  winner,
}: {
  counts: CommunityCounts;
  gamma: number;
  level: BalanceScaleProps['level'];
  side: 'A' | 'B';
  winner: boolean;
}) {
  const weighted = level >= 2;
  const friendR = weighted ? 6 + 6 * (1 - gamma) : 9;
  const balloonR = weighted ? 6 + 8 * gamma : 9;
  const friends = friendSlots(counts.friends);
  const balloons = balloonSlots(counts.strangers);
  const classified = level >= 1;
  return (
    <g>
      {/* Hanging strings and the bowl. */}
      <path d="M0 0 L-64 116 M0 0 L64 116" stroke="var(--metal)" strokeWidth={1.4} fill="none" />
      <path d="M-80 116 Q0 150 80 116 Z" fill={winner ? 'var(--threshold-soft)' : 'var(--card-2)'} stroke="var(--metal)" strokeWidth={2} />
      <line x1={-82} y1={116} x2={82} y2={116} stroke="var(--metal)" strokeWidth={2.4} strokeLinecap="round" />
      <text x={0} y={138} textAnchor="middle" className={styles.panLetter}>
        {side}
      </text>
      {/* Strangers: balloons tied to the rim (they lift the pan as γ grows). */}
      {balloons.map((slot, i) => (
        <g key={`s-${i}`} className={styles.member} style={{ transform: `translate(${slot.x}px, ${classified ? slot.y : 104 - Math.floor((counts.friends + i) / 4) * 19}px)` }}>
          {classified && <path d={`M0 ${balloonR} Q ${i % 2 ? 6 : -6} ${(116 - slot.y) / 2} ${slot.x > 0 ? -8 : 8} ${116 - slot.y}`} stroke="var(--stranger)" strokeWidth={1} fill="none" opacity={0.7} />}
          <circle
            r={classified ? balloonR : 8}
            fill={classified ? (weighted && gamma === 0 ? 'none' : 'var(--stranger-soft)') : 'var(--tie-soft)'}
            stroke={classified ? 'var(--stranger)' : 'var(--tie)'}
            strokeWidth={2}
            strokeDasharray={classified && weighted && gamma === 0 ? '3 2' : undefined}
          />
          {classified && <path d={`M-3 ${balloonR - 1} L3 ${balloonR - 1} L0 ${balloonR + 4} Z`} fill="var(--stranger)" />}
        </g>
      ))}
      {/* Friends: weights resting in the pan (lighter as γ grows). */}
      {friends.map((slot, i) => (
        <g key={`f-${i}`} className={styles.member} style={{ transform: `translate(${slot.x}px, ${slot.y}px)` }}>
          <circle
            r={classified ? friendR : 8}
            fill={classified ? (weighted && gamma === 1 ? 'none' : 'var(--friend)') : 'var(--tie-soft)'}
            stroke={classified ? 'var(--friend)' : 'var(--tie)'}
            strokeWidth={2}
            strokeDasharray={classified && weighted && gamma === 1 ? '3 2' : undefined}
          />
        </g>
      ))}
    </g>
  );
}

export function BalanceScale({ a, b, gamma, level, reducedMotion, labels }: BalanceScaleProps) {
  const uA = communityUtility(a.friends, a.strangers, gamma);
  const uB = communityUtility(b.friends, b.strangers, gamma);
  const direction = level >= 2 ? signOf(uB - uA) : 0;
  const target = direction === 0 ? 0 : Math.max(-MAX_TILT, Math.min(MAX_TILT, (uB - uA) * 7));
  const angle = useSpring(target, reducedMotion);
  const rad = (angle * Math.PI) / 180;
  const hook = (sign: -1 | 1) => ({ x: PIVOT.x + sign * ARM * Math.cos(rad), y: PIVOT.y + sign * ARM * Math.sin(rad) });
  const left = hook(-1);
  const right = hook(1);

  const friendSide = a.friends === b.friends ? null : a.friends > b.friends ? 'A' : 'B';
  const strangerSide = a.strangers === b.strangers ? null : a.strangers < b.strangers ? 'A' : 'B';

  const badge = (text: string, side: 'A' | 'B' | null, tone: 'friend' | 'stranger', row: number) => {
    const x = side === 'A' ? 136 : side === 'B' ? 504 : 320;
    const y = 18 + row * 30;
    const width = Math.max(96, text.length * 7.4 + 34);
    return (
      <g className={styles.badge} style={{ transform: `translate(${x}px, ${y}px)` }}>
        <rect x={-width / 2} y={-12} width={width} height={24} rx={12} fill={`var(--${tone}-soft)`} stroke={`var(--${tone})`} />
        {tone === 'friend' ? (
          <circle cx={-width / 2 + 14} cy={0} r={5} fill="var(--friend)" />
        ) : (
          <g transform={`translate(${-width / 2 + 14} 0)`}>
            <circle r={5} fill="var(--stranger-soft)" stroke="var(--stranger)" strokeWidth={1.6} />
          </g>
        )}
        <text x={-width / 2 + 26} y={0} dy="0.35em" className={styles.badgeText} fill={`var(--${tone}-ink)`}>
          {text}
        </text>
      </g>
    );
  };

  const winner = direction > 0 ? 'B' : direction < 0 ? 'A' : null;

  return (
    <svg className={styles.scale} viewBox="0 0 640 380" role="presentation" aria-hidden="true" focusable="false">
      {level >= 1 && (
        <g>
          {badge(friendSide ? labels.moreFriends : labels.sameFriends, friendSide, 'friend', 0)}
          {badge(strangerSide ? labels.fewerStrangers : labels.sameStrangers, strangerSide, 'stranger', friendSide === strangerSide ? 1 : 0)}
        </g>
      )}
      {/* Stand */}
      <path d="M250 368 Q320 350 390 368 Z" fill="var(--metal-2)" stroke="var(--metal)" strokeWidth={2} />
      <rect x={314} y={PIVOT.y} width={12} height={246} rx={4} fill="var(--metal-2)" stroke="var(--metal)" strokeWidth={1.5} />
      {/* Beam */}
      <g style={{ transform: `rotate(${angle}deg)`, transformOrigin: `${PIVOT.x}px ${PIVOT.y}px` }}>
        <rect x={PIVOT.x - ARM - 16} y={PIVOT.y - 6} width={(ARM + 16) * 2} height={12} rx={6} fill="var(--metal-2)" stroke="var(--metal)" strokeWidth={1.5} />
        <circle cx={PIVOT.x - ARM} cy={PIVOT.y} r={4} fill="var(--metal)" />
        <circle cx={PIVOT.x + ARM} cy={PIVOT.y} r={4} fill="var(--metal)" />
      </g>
      <g style={{ transform: `translate(${left.x}px, ${left.y}px)` }}>
        <Pan counts={a} gamma={gamma} level={level} side="A" winner={winner === 'A'} />
      </g>
      <g style={{ transform: `translate(${right.x}px, ${right.y}px)` }}>
        <Pan counts={b} gamma={gamma} level={level} side="B" winner={winner === 'B'} />
      </g>
      {/* The agent sits on the pivot and leans toward its choice. */}
      <g style={{ transform: `translate(${PIVOT.x}px, ${PIVOT.y - 34}px)` }}>
        <circle r={20} fill="var(--ink)" stroke="var(--card)" strokeWidth={3} />
        <text y={0} dy="0.36em" textAnchor="middle" className={styles.agent}>
          {level >= 2 ? (winner === 'A' ? '←' : winner === 'B' ? '→' : '=') : level === 1 && friendSide && strangerSide && friendSide !== strangerSide ? '?' : 'i'}
        </text>
      </g>
      <circle cx={PIVOT.x} cy={PIVOT.y} r={6} fill="var(--metal)" />
    </svg>
  );
}
