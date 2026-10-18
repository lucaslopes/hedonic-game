import type { MouseEvent } from 'react';
import { useMediaQuery } from '../app/settings';
import { useT } from '../i18n';
import { formatNumber } from '../model';
import { FRESH, auditVertex, bestResponse, labelName, type Cover } from '../model/overlap';
import { GAME, labelColor } from './overlapData';
import styles from './Overlap.module.css';

const M = { left: 44, right: 34, top: 14, bottom: 30 };
const SAMPLES = 160;

interface UtilityChartProps {
  readonly cover: Cover;
  readonly vertex: number;
  readonly gamma: number;
  readonly cap: number;
  readonly isolation: boolean;
  readonly onGamma: (gamma: number) => void;
}

/**
 * The focus vertex's utility as γ sweeps [0, 1] with the cover frozen:
 * one line per candidate label (its gain a − γb), the current set, the best
 * response, and the exact stable intervals from the audit shaded.
 */
export function UtilityChart({ cover, vertex, gamma, cap, isolation, onGamma }: UtilityChartProps) {
  const t = useT();
  // A narrower drawing on phones keeps the tick labels legible.
  const compact = useMediaQuery('(max-width: 560px)');
  const W = compact ? 380 : 600;
  const H = compact ? 260 : 290;
  const audit = auditVertex(GAME, cover, vertex, { cap, allowIsolation: isolation });
  const grid = new Set<number>(audit.cuts);
  for (let i = 0; i <= SAMPLES; i += 1) grid.add(i / SAMPLES);
  const xs = [...grid].sort((a, b) => a - b);
  const best = xs.map((x) => [x, bestResponse(GAME, cover, vertex, { gamma: x, cap, allowIsolation: isolation }).best.value] as const);
  const current = (x: number) => audit.curA - x * audit.curB;
  const lines = audit.entries.map((entry) => ({ ...entry, at: (x: number) => entry.a - x * entry.b }));

  const values = [current(0), current(1), ...best.map(([, y]) => y), ...lines.flatMap((line) => [line.at(0), line.at(1)])];
  let lo = Math.min(0, ...values);
  let hi = Math.max(0, ...values);
  const pad = (hi - lo) * 0.08 || 0.5;
  lo -= pad;
  hi += pad;
  const sx = (x: number) => M.left + x * (W - M.left - M.right);
  const sy = (y: number) => M.top + ((hi - y) / (hi - lo)) * (H - M.top - M.bottom);
  const path = (points: readonly (readonly [number, number])[]) => points.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(2)},${sy(y).toFixed(2)}`).join(' ');

  const step = hi - lo > 4 ? 1 : hi - lo > 2 ? 0.5 : hi - lo > 0.8 ? 0.25 : 0.1;
  const ticks: number[] = [];
  for (let y = Math.ceil(lo / step) * step; y <= hi; y += step) ticks.push(Number(y.toFixed(6)));

  // End labels of the gain lines, nudged apart where lines end together.
  const labelY = new Map<number, number>();
  let previous = -Infinity;
  for (const line of [...lines].sort((a, b) => sy(a.at(1)) - sy(b.at(1)))) {
    const y = Math.max(sy(line.at(1)), previous + 12);
    labelY.set(line.label, y);
    previous = y;
  }

  const onClick = (event: MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * W;
    const value = (x - M.left) / (W - M.left - M.right);
    onGamma(Math.min(1, Math.max(0, Math.round(value * 100) / 100)));
  };

  const bestAt = bestResponse(GAME, cover, vertex, { gamma, cap, allowIsolation: isolation }).best.value;

  return (
    <figure className={styles.chartFigure}>
      <svg viewBox={`0 0 ${W} ${H}`} className={styles.chart} role="img" aria-label={t.overlap.intervals.chart(vertex)} onClick={onClick}>
        {audit.intervals.map(([a, b]) => (
          <rect
            key={`${a}-${b}`}
            x={sx(a) - (b - a < 1e-9 ? 1.5 : 0)}
            width={Math.max(3, sx(b) - sx(a))}
            y={M.top}
            height={H - M.top - M.bottom}
            className={styles.stableBand}
          />
        ))}
        {ticks.map((y) => (
          <g key={y}>
            <line x1={M.left} x2={W - M.right} y1={sy(y)} y2={sy(y)} className={y === 0 ? styles.axisZero : styles.gridLine} />
            <text x={M.left - 6} y={sy(y)} dy="0.32em" className={styles.tickY}>
              {formatNumber(y, step < 0.25 ? 1 : 2)}
            </text>
          </g>
        ))}
        {(compact ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map((x) => (
          <text key={x} x={sx(x)} y={H - 10} className={styles.tickX}>
            {formatNumber(x, 2)}
          </text>
        ))}
        <text x={W - 4} y={H - 10} className={styles.axisName} textAnchor="end">
          γ
        </text>
        {lines.map((line) => (
          <g key={line.label} style={{ color: line.label === FRESH ? 'var(--ink-3)' : labelColor(line.label) }}>
            <line x1={sx(0)} y1={sy(line.at(0))} x2={sx(1)} y2={sy(line.at(1))} className={styles.gainLine} data-own={line.own} />
            <text x={sx(1) + 4} y={labelY.get(line.label)} dy="0.32em" className={styles.gainName}>
              {line.label === FRESH ? '∗' : labelName(line.label)}
            </text>
          </g>
        ))}
        <path d={path(best)} className={styles.bestLine} />
        <line x1={sx(0)} y1={sy(current(0))} x2={sx(1)} y2={sy(current(1))} className={styles.currentLine} />
        <line x1={sx(gamma)} x2={sx(gamma)} y1={M.top} y2={H - M.bottom} className={styles.marker} />
        <circle cx={sx(gamma)} cy={sy(bestAt)} r={5} className={styles.bestDot} />
        <circle cx={sx(gamma)} cy={sy(current(gamma))} r={4.5} className={styles.currentDot} />
      </svg>
      <figcaption className={styles.chartLegend}>
        <span className={styles.legendItem}>
          <svg width="22" height="10" aria-hidden="true">
            <line x1="1" x2="21" y1="5" y2="5" className={styles.currentLine} />
          </svg>
          {t.overlap.intervals.current}
        </span>
        <span className={styles.legendItem}>
          <svg width="22" height="10" aria-hidden="true">
            <line x1="1" x2="21" y1="5" y2="5" className={styles.bestLine} />
          </svg>
          {t.overlap.intervals.best}
        </span>
        <span className={styles.legendItem}>
          <svg width="22" height="10" aria-hidden="true">
            <line x1="1" x2="21" y1="5" y2="5" className={styles.gainLine} style={{ color: 'var(--ink-3)' }} />
          </svg>
          {t.overlap.intervals.gains}
        </span>
        <span className={styles.legendItem}>
          <span className={styles.stableSwatch} aria-hidden="true" />
          {t.overlap.intervals.stable}
        </span>
        <span className="muted">{t.overlap.intervals.clickHint}</span>
      </figcaption>
    </figure>
  );
}
