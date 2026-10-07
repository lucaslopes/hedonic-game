import type { KeyboardEvent } from 'react';
import { adjacencyOf, blocksOf, EXAMPLE_GRAPH, EXAMPLE_POSITIONS, type SimpleGraph } from '../model';

export const BLOCK_COLORS = ['var(--block-0)', 'var(--block-1)', 'var(--block-2)', 'var(--block-3)'];

interface GraphGlyphProps {
  readonly cx: number;
  readonly cy: number;
  /** Distance from the centre to each vertex. */
  readonly radius: number;
  readonly graph?: SimpleGraph;
  readonly positions?: readonly (readonly [number, number])[];
  /** Canonical membership: colour vertices by community and draw community hulls. */
  readonly membership?: readonly number[] | null;
  /** Highlight the friends and strangers of this vertex. */
  readonly focus?: number | null;
  /** Emphasise one vertex (e.g. the mover of a walk step). */
  readonly accent?: number | null;
  readonly showLabels?: boolean;
  readonly vertexRadius?: number;
  /** Make vertices keyboard/pointer targets. */
  readonly onVertex?: (vertex: number) => void;
  readonly vertexLabel?: (vertex: number) => string;
}

/** The example network drawn at any size, as plain SVG (usable inside other SVGs). */
export function GraphGlyph({
  cx,
  cy,
  radius,
  graph = EXAMPLE_GRAPH,
  positions = EXAMPLE_POSITIONS,
  membership = null,
  focus = null,
  accent = null,
  showLabels = true,
  vertexRadius,
  onVertex,
  vertexLabel,
}: GraphGlyphProps) {
  const adj = adjacencyOf(graph);
  const r = vertexRadius ?? radius * 0.3;
  const at = (v: number) => [cx + positions[v][0] * radius, cy + positions[v][1] * radius] as const;
  const blocks = membership ? blocksOf(membership) : [];

  const vertexFill = (v: number) => {
    if (focus !== null) {
      if (v === focus) return 'var(--ink)';
      return adj[focus][v] ? 'var(--friend)' : 'var(--stranger)';
    }
    if (membership) return BLOCK_COLORS[membership[v] % BLOCK_COLORS.length];
    return 'var(--card)';
  };
  // Labels on filled vertices use the page colour, which contrasts in both themes.
  const textFill = focus !== null || membership ? 'var(--paper)' : 'var(--ink)';

  const onKey = (event: KeyboardEvent, v: number) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onVertex?.(v);
    }
  };

  return (
    <g>
      {/* Community hulls: a thick translucent stroke through the members. */}
      {blocks.map((block, label) =>
        block.length > 1 ? (
          <path
            key={`hull-${label}`}
            d={`${block.map((v, i) => `${i === 0 ? 'M' : 'L'}${at(v)[0]},${at(v)[1]}`).join(' ')}${block.length > 2 ? ' Z' : ''}`}
            fill={block.length > 2 ? BLOCK_COLORS[label % BLOCK_COLORS.length] : 'none'}
            fillOpacity={0.18}
            stroke={BLOCK_COLORS[label % BLOCK_COLORS.length]}
            strokeOpacity={0.3}
            strokeWidth={r * 2.9}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ) : null,
      )}
      {/* Stranger relations of the focus vertex, drawn as dashed "non-links". */}
      {focus !== null &&
        adj[focus].map((linked, v) =>
          !linked && v !== focus ? (
            <line
              key={`non-${v}`}
              x1={at(focus)[0]}
              y1={at(focus)[1]}
              x2={at(v)[0]}
              y2={at(v)[1]}
              stroke="var(--stranger)"
              strokeWidth={Math.max(1, radius * 0.035)}
              strokeDasharray={`${radius * 0.07} ${radius * 0.07}`}
              opacity={0.85}
            />
          ) : null,
        )}
      {graph.edges.map(([u, v]) => {
        const internal = membership ? membership[u] === membership[v] : true;
        const touchesFocus = focus !== null && (u === focus || v === focus);
        return (
          <line
            key={`e-${u}-${v}`}
            x1={at(u)[0]}
            y1={at(u)[1]}
            x2={at(v)[0]}
            y2={at(v)[1]}
            stroke={touchesFocus ? 'var(--friend)' : 'var(--ink-2)'}
            strokeWidth={Math.max(1.2, radius * (touchesFocus ? 0.06 : internal ? 0.045 : 0.03))}
            strokeDasharray={internal ? undefined : `${radius * 0.06} ${radius * 0.05}`}
            opacity={focus !== null && !touchesFocus ? 0.35 : internal ? 0.9 : 0.45}
          />
        );
      })}
      {Array.from({ length: graph.n }, (_, v) => {
        const [x, y] = at(v);
        const interactive = Boolean(onVertex);
        return (
          <g
            key={`v-${v}`}
            role={interactive ? 'button' : undefined}
            tabIndex={interactive ? 0 : undefined}
            aria-label={interactive ? vertexLabel?.(v) : undefined}
            aria-pressed={interactive ? focus === v : undefined}
            onClick={interactive ? () => onVertex?.(v) : undefined}
            onKeyDown={interactive ? (event) => onKey(event, v) : undefined}
            style={interactive ? { cursor: 'pointer' } : undefined}
            className={interactive ? 'graph-vertex' : undefined}
          >
            {accent === v && <circle cx={x} cy={y} r={r * 1.45} fill="none" stroke="var(--threshold)" strokeWidth={Math.max(2, r * 0.22)} />}
            {/* Shape redundancy: friends are circles, strangers get a square frame. */}
            {focus !== null && v !== focus && !adj[focus][v] ? (
              <rect x={x - r} y={y - r} width={r * 2} height={r * 2} rx={r * 0.25} fill={vertexFill(v)} stroke="var(--card)" strokeWidth={Math.max(1, r * 0.12)} />
            ) : (
              <circle cx={x} cy={y} r={r} fill={vertexFill(v)} stroke={membership || focus !== null ? 'var(--card)' : 'var(--ink-2)'} strokeWidth={Math.max(1, r * 0.12)} />
            )}
            {showLabels && (
              <text x={x} y={y} dy="0.35em" textAnchor="middle" fontSize={r * 1.05} fontWeight={700} fill={textFill} style={{ fontFamily: 'var(--font-body)' }}>
                {v}
              </text>
            )}
          </g>
        );
      })}
    </g>
  );
}
