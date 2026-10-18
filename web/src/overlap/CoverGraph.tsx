import type { KeyboardEvent } from 'react';
import { useT } from '../i18n';
import { OVERLAP_POSITIONS, labelMembers, labelName, type Cover } from '../model/overlap';
import { GAME, labelColor, rowText } from './overlapData';
import styles from './CoverGraph.module.css';

const SCALE = 100;
const R = 21;
const POINTS = OVERLAP_POSITIONS.map(([x, y]) => [x * SCALE, y * SCALE] as const);

interface CoverGraphProps {
  readonly cover: Cover;
  /** The vertex whose friends and strangers are marked. */
  readonly focus?: number | null;
  /** Vertices to ring (e.g. the vertex just visited). */
  readonly highlight?: number | null;
  /** Vertices waiting in a queue (drawn with a small dot). */
  readonly queued?: readonly number[];
  readonly onVertex?: (v: number) => void;
  readonly className?: string;
}

type Point = readonly [number, number];

/** Convex hull (monotone chain), counter-clockwise, no repeated end point. */
function hull(points: Point[]): Point[] {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (sorted.length <= 2) return sorted;
  const cross = (o: Point, a: Point, b: Point) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Point[] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: Point[] = [];
  for (const p of [...sorted].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

function sector(cx: number, cy: number, r: number, from: number, to: number): string {
  const a = [cx + r * Math.cos(from), cy + r * Math.sin(from)];
  const b = [cx + r * Math.cos(to), cy + r * Math.sin(to)];
  const large = to - from > Math.PI ? 1 : 0;
  return `M${cx},${cy} L${a[0].toFixed(2)},${a[1].toFixed(2)} A${r},${r} 0 ${large} 1 ${b[0].toFixed(2)},${b[1].toFixed(2)} Z`;
}

/**
 * The six-vertex graph with a cover drawn on it: one translucent hull per
 * label with two or more members, and each vertex split into one sector per
 * label it holds. Label letters sit beside every vertex, so colour is never
 * the only cue.
 */
export function CoverGraph({ cover, focus = null, highlight = null, queued = [], onVertex, className }: CoverGraphProps) {
  const t = useT();
  const members = [...labelMembers(cover).entries()].filter(([, list]) => list.length > 1);
  // Larger communities sit behind with a wider margin, so identical bodies stay visible as nested bands.
  members.sort((a, b) => b[1].length - a[1].length || a[0] - b[0]);
  const description = t.overlap.graph.communities(
    [...labelMembers(cover).entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([label, list]) => `${labelName(label)} = {${list.join(', ')}}`)
      .join('; '),
  );
  const onKey = (event: KeyboardEvent, v: number) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onVertex?.(v);
    }
  };

  return (
    <svg viewBox="-180 -150 360 300" className={`${styles.svg} ${className ?? ''}`} role="group" aria-label={t.overlap.graph.label(description)}>
      <g aria-hidden="true">
        {members.map(([label, list], index) => {
          const pad = 30 + 7 * (members.length - 1 - index);
          const points = hull(list.map((v) => POINTS[v]));
          const d = `M${points.map((p) => `${p[0]},${p[1]}`).join(' L')}${points.length > 2 ? ' Z' : ''}`;
          return (
            <g key={label} className={styles.hull} style={{ color: labelColor(label) }}>
              <path d={d} strokeWidth={2 * pad} />
            </g>
          );
        })}
        {GAME.graph.edges.map(([a, b]) => {
          const touches = focus !== null && (a === focus || b === focus);
          return (
            <line
              key={`${a}-${b}`}
              x1={POINTS[a][0]}
              y1={POINTS[a][1]}
              x2={POINTS[b][0]}
              y2={POINTS[b][1]}
              className={touches ? styles.edgeFocus : styles.edge}
            />
          );
        })}
      </g>
      {cover.map((row, v) => {
        const [x, y] = POINTS[v];
        const isFocus = v === focus;
        const friend = focus !== null && !isFocus && GAME.adj[focus][v];
        const stranger = focus !== null && !isFocus && !GAME.adj[focus][v];
        const k = row.length;
        const top = y < 0;
        const ring = isFocus ? styles.ringFocus : friend ? styles.ringFriend : stranger ? styles.ringStranger : styles.ring;
        const interactive = Boolean(onVertex);
        return (
          <g
            key={v}
            className={`${styles.vertex} ${interactive ? styles.interactive : ''}`}
            role={interactive ? 'button' : 'img'}
            tabIndex={interactive ? 0 : undefined}
            aria-label={interactive ? t.overlap.graph.select(v, rowText(row)) : t.overlap.graph.vertex(v, rowText(row))}
            aria-pressed={interactive ? isFocus : undefined}
            onClick={interactive ? () => onVertex?.(v) : undefined}
            onKeyDown={interactive ? (event) => onKey(event, v) : undefined}
          >
            {(highlight === v || isFocus) && <circle cx={x} cy={y} r={R + 9} className={styles.halo} />}
            {k === 1 ? (
              <circle cx={x} cy={y} r={R} fill={labelColor(row[0])} />
            ) : (
              row.map((label, index) => (
                <path
                  key={label}
                  d={sector(x, y, R, -Math.PI / 2 + (2 * Math.PI * index) / k, -Math.PI / 2 + (2 * Math.PI * (index + 1)) / k)}
                  fill={labelColor(label)}
                />
              ))
            )}
            <circle cx={x} cy={y} r={R} className={styles.ring} />
            {ring !== styles.ring && <circle cx={x} cy={y} r={R + 3} className={ring} />}
            <circle cx={x} cy={y} r={R * 0.5} className={styles.core} />
            <text x={x} y={y} className={styles.number} dy="0.35em">
              {v}
            </text>
            <text x={x} y={top ? y - R - 9 : y + R + 9} dy={top ? '0' : '0.7em'} className={styles.row}>
              {row.map(labelName).join('·')}
            </text>
            {queued.includes(v) && <circle cx={x + R * 0.78} cy={y + (top ? R * 0.78 : -R * 0.78)} r={4.5} className={styles.queued} />}
          </g>
        );
      })}
    </svg>
  );
}
