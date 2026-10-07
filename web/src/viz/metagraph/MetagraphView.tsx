import { select } from 'd3-selection';
import 'd3-transition';
import { zoom, zoomIdentity, type ZoomBehavior } from 'd3-zoom';
import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { Icon } from '../../components/Icon';
import { formatFraction, formatNumber, isSink, nodePotential, type MetaNode, type Metagraph } from '../../model';
import { GraphGlyph } from '../GraphGlyph';
import type { MetagraphLayout } from '../metagraphLayout';
import { edgeVisual, type EdgeVisual, type MetagraphMode } from './edgeVisual';
import {
  cardSize,
  edgeGeometry,
  LINE_H,
  nodeLines,
  pathOf,
  pointOn,
  reverseGeometry,
  splitGeometry,
  type EdgeGeometry,
  type LabelMode,
  type Point,
  type Size,
} from './geometry';
import styles from './MetagraphView.module.css';

export type { MetagraphMode };

export interface WalkOverlay {
  /** Visited partitions, start first. */
  readonly path: readonly string[];
  /** Predicted continuation after the current partition. */
  readonly preview: readonly string[];
  readonly done: boolean;
  /** Changes on every step (drives the token animation). */
  readonly stepKey: number;
  readonly durationMs: number;
}

export interface MetagraphLabels {
  readonly figure: string;
  readonly nodeLabel: (node: MetaNode, phi: string, sink: boolean) => string;
  readonly zoomIn: string;
  readonly zoomOut: string;
  readonly zoomReset: string;
  readonly keyboardHint: string;
  readonly sink: string;
  readonly quality: string;
  readonly pairNotes?: readonly [string, string];
  /** Captions under the two halves of a frustrated edge in pair mode. */
  readonly pairEdge?: { readonly friends: string; readonly strangers: string };
}

interface MetagraphViewProps {
  readonly metagraph: Metagraph;
  readonly layout: MetagraphLayout;
  readonly gamma: number;
  readonly mode: MetagraphMode;
  readonly labels: MetagraphLabels;
  readonly reducedMotion: boolean;
  readonly selectedId?: string | null;
  readonly onSelectNode?: (id: string) => void;
  /** Called when the empty canvas is clicked (not a node, an edge, or the end of a pan). */
  readonly onClearSelection?: () => void;
  readonly selectedEdgeId?: string | null;
  readonly onSelectEdge?: (id: string) => void;
  /** Temporarily emphasised edge (e.g. a hovered row of the moves table). */
  readonly highlightEdgeId?: string | null;
  readonly walk?: WalkOverlay | null;
  readonly showSameLayer?: boolean;
  readonly showPotential?: boolean;
  readonly pair?: readonly [string, string];
  readonly zoomable?: boolean;
  /** 'auto' picks the more legible layout; the story forces the paper's left-to-right one on wide screens. */
  readonly orientation?: 'auto' | 'horizontal' | 'vertical';
  readonly className?: string;
}

type Orientation = 'horizontal' | 'vertical';

/** Largest text scale: 12px labels grow to at most 15px on roomy stages. */
const MAX_SCALE = 1.25;

interface Scene {
  readonly orientation: Orientation;
  readonly scale: number;
  readonly width: number;
  readonly height: number;
  readonly positions: ReadonlyMap<string, Point>;
  readonly sizes: ReadonlyMap<string, Size>;
  readonly lines: ReadonlyMap<string, readonly string[]>;
}

function buildScene(
  metagraph: Metagraph,
  layout: MetagraphLayout,
  W: number,
  H: number,
  forced: 'auto' | Orientation,
): Scene {
  const C = layout.columns.length;
  const measure = (orientation: Orientation) => {
    const lines = new Map<string, string[]>();
    const sizes = new Map<string, Size>();
    for (const node of metagraph.nodes) {
      const mode: LabelMode = orientation === 'vertical' && layout.columns[node.layer].length === 1 ? 'inline' : 'stacked';
      const text = nodeLines(node, mode);
      lines.set(node.id, text);
      sizes.set(node.id, cardSize(text));
    }
    const size = (id: string) => sizes.get(id) ?? { w: 0, h: 0 };
    const tallest = Math.max(...layout.columns.map((column) => column.length));
    // Within a layer, positions are proportional (the largest layer spans the
    // whole extent), so what matters is the gap between adjacent cards.
    const pitch = (extent: (id: string) => number, gap: number) =>
      Math.max(
        0,
        ...layout.columns.flatMap((column) => column.slice(1).map((id, i) => (extent(column[i]) + extent(id)) / 2 + gap)),
      );
    let requiredW: number;
    let requiredH: number;
    if (orientation === 'horizontal') {
      const colW = layout.columns.map((column) => Math.max(...column.map((id) => size(id).w)));
      const maxH = Math.max(...metagraph.nodes.map((node) => size(node.id).h));
      requiredW = colW.reduce((sum, w) => sum + w, 0) + (C - 1) * 46 + 28;
      requiredH = pitch((id) => size(id).h, 10) * (tallest - 1) + maxH + 28;
    } else {
      const rowH = layout.columns.map((column) => Math.max(...column.map((id) => size(id).h)));
      const maxW = Math.max(...metagraph.nodes.filter((n) => layout.columns[n.layer].length > 1).map((n) => size(n.id).w));
      const loneW = Math.max(...layout.columns.filter((column) => column.length === 1).flat().map((id) => size(id).w));
      requiredW = Math.max(pitch((id) => size(id).w, 8) * (tallest - 1) + maxW + 20, loneW + 20);
      requiredH = rowH.reduce((sum, h) => sum + h, 0) + (C - 1) * 30 + 30;
    }
    return { lines, sizes, scale: Math.min(W / requiredW, H / requiredH) };
  };
  const horizontal = measure('horizontal');
  const vertical = measure('vertical');
  // Prefer the paper's left-to-right layout unless it would be clearly less legible.
  const fit = (s: number) => Math.max(0.6, Math.min(MAX_SCALE, s));
  const orientation: Orientation =
    forced !== 'auto' ? forced : fit(horizontal.scale) >= fit(vertical.scale) * 0.8 ? 'horizontal' : 'vertical';
  const chosen = orientation === 'horizontal' ? horizontal : vertical;
  const scale = fit(chosen.scale);
  const width = W / scale;
  const height = H / scale;
  const size = (id: string) => chosen.sizes.get(id) ?? { w: 0, h: 0 };
  const positions = new Map<string, Point>();

  if (orientation === 'horizontal') {
    const edgeW = Math.max(...[...layout.columns[0], ...layout.columns[C - 1]].map((id) => size(id).w));
    const maxH = Math.max(...metagraph.nodes.map((node) => size(node.id).h));
    const marginX = edgeW / 2 + 16;
    const marginY = maxH / 2 + 14;
    for (const [id, point] of layout.points) {
      positions.set(id, {
        x: marginX + point.x * (width - 2 * marginX),
        y: marginY + point.y * (height - 2 * marginY),
      });
    }
  } else {
    const edgeH = Math.max(...[...layout.columns[0], ...layout.columns[C - 1]].map((id) => size(id).h));
    const shared = layout.columns.filter((column) => column.length > 1).flat();
    const maxW = Math.max(...shared.map((id) => size(id).w));
    const marginY = edgeH / 2 + 16;
    const marginX = maxW / 2 + 10;
    for (const [id, point] of layout.points) {
      positions.set(id, {
        x: marginX + point.y * (width - 2 * marginX),
        y: marginY + point.x * (height - 2 * marginY),
      });
    }
  }
  return { orientation, scale, width, height, positions, sizes: chosen.sizes, lines: chosen.lines };
}

const PAIR_SCALE = 1.3;

/** True for keyboard focus (older engines without :focus-visible count as pointer focus). */
function isKeyboardFocus(element: Element): boolean {
  try {
    return element.matches(':focus-visible');
  } catch {
    return false;
  }
}

export function MetagraphView({
  metagraph,
  layout,
  gamma,
  mode,
  labels,
  reducedMotion,
  selectedId = null,
  onSelectNode,
  onClearSelection,
  selectedEdgeId = null,
  onSelectEdge,
  highlightEdgeId = null,
  walk = null,
  showSameLayer = true,
  showPotential = false,
  pair,
  zoomable = false,
  orientation: forced = 'auto',
  className,
}: MetagraphViewProps) {
  const uid = useId().replace(/:/g, '');
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const tokenRef = useRef<SVGCircleElement>(null);
  const nodeRefs = useRef(new Map<string, SVGGElement>());
  const zoomBehavior = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [transform, setTransform] = useState('');
  const [focusId, setFocusId] = useState<string | null>(null);
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null);
  // Keyboard focus shows the same details card as a mouse hover.
  const [focusCard, setFocusCard] = useState<string | null>(null);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return undefined;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setBox({ w: Math.round(rect.width), h: Math.round(rect.height) });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const scene = useMemo(
    () => (box.w > 0 && box.h > 0 ? buildScene(metagraph, layout, box.w, box.h, forced) : null),
    [metagraph, layout, box.w, box.h, forced],
  );

  // Pan and zoom (explore mode). Wheel zoom needs Ctrl/⌘ and touch needs two
  // fingers, so the page keeps scrolling normally.
  useEffect(() => {
    const svg = svgRef.current;
    if (!zoomable || !svg) return undefined;
    const behavior = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.6, 4])
      .clickDistance(4)
      .filter((event: Event) => {
        if (event.type === 'wheel') return (event as WheelEvent).ctrlKey || (event as WheelEvent).metaKey;
        if (event.type.startsWith('touch')) return (event as TouchEvent).touches.length > 1;
        return !(event as MouseEvent).button;
      })
      .on('zoom', (event) => setTransform(event.transform.toString()));
    const selection = select(svg);
    selection.call(behavior).on('dblclick.zoom', null);
    zoomBehavior.current = behavior;
    return () => {
      selection.on('.zoom', null);
      zoomBehavior.current = null;
    };
  }, [zoomable]);

  const zoomBy = (factor: number | null) => {
    const svg = svgRef.current;
    const behavior = zoomBehavior.current;
    if (!svg || !behavior) return;
    const selection = select(svg).transition().duration(reducedMotion ? 0 : 280);
    if (factor === null) selection.call(behavior.transform, zoomIdentity);
    else selection.call(behavior.scaleBy, factor);
  };

  const pairIds = useMemo(
    () => pair ?? ([metagraph.grandCoalitionId, metagraph.edges[0]?.b ?? metagraph.grandCoalitionId] as const),
    [pair, metagraph],
  );
  const isPair = mode === 'pair';
  const visible = (id: string) => !isPair || pairIds.includes(id);
  const oriented = mode === 'oriented';

  // Where each node sits, including the enlarged pair layout.
  const placed = useMemo(() => {
    if (!scene) return null;
    const map = new Map<string, { p: Point; s: number }>();
    for (const [id, p] of scene.positions) map.set(id, { p, s: 1 });
    if (isPair) {
      map.set(pairIds[0], { p: { x: scene.width * 0.27, y: scene.height * 0.56 }, s: PAIR_SCALE });
      map.set(pairIds[1], { p: { x: scene.width * 0.73, y: scene.height * 0.56 }, s: PAIR_SCALE });
    }
    return map;
  }, [scene, isPair, pairIds]);

  // Geometry of every metaedge (a → b), recomputed only when the scene changes.
  const geometries = useMemo(() => {
    const map = new Map<string, EdgeGeometry>();
    if (!scene || !placed) return map;
    const scaled = (s: Size, k: number) => ({ w: s.w * k, h: s.h * k });
    for (const edge of metagraph.edges) {
      const a = placed.get(edge.a);
      const b = placed.get(edge.b);
      const sa = scene.sizes.get(edge.a);
      const sb = scene.sizes.get(edge.b);
      if (!a || !b || !sa || !sb) continue;
      let offset: Point | undefined;
      if (edge.sameLayer && !isPair) {
        const rowA = layout.points.get(edge.a)?.row ?? 0;
        const rowB = layout.points.get(edge.b)?.row ?? 0;
        const bulge = 24 + 15 * Math.abs(rowA - rowB);
        offset = scene.orientation === 'horizontal' ? { x: bulge, y: 0 } : { x: 0, y: bulge };
      }
      map.set(edge.id, edgeGeometry(a.p, scaled(sa, a.s), b.p, scaled(sb, b.s), offset));
    }
    return map;
  }, [scene, placed, metagraph, layout, isPair]);
  const geometryOf = (edgeId: string): EdgeGeometry | null => geometries.get(edgeId) ?? null;

  const oriented2 = (from: string, to: string): EdgeGeometry | null => {
    const edge = metagraph.edgeBetween(from, to);
    if (!edge) return null;
    const g = geometryOf(edge.id);
    if (!g) return null;
    return edge.a === from ? g : reverseGeometry(g);
  };

  // Animate a token along the latest walk step.
  const lastFrom = walk && walk.path.length >= 2 ? walk.path[walk.path.length - 2] : null;
  const lastTo = walk && walk.path.length >= 2 ? walk.path[walk.path.length - 1] : null;
  const tokenGeometry = lastFrom && lastTo ? oriented2(lastFrom, lastTo) : null;
  const stepKey = walk?.stepKey ?? 0;
  const duration = walk?.durationMs ?? 600;
  useEffect(() => {
    const token = tokenRef.current;
    if (!token) return undefined;
    if (!tokenGeometry || reducedMotion || stepKey === 0) {
      token.setAttribute('opacity', '0');
      return undefined;
    }
    let frame = 0;
    const start = performance.now();
    const total = Math.max(200, duration * 0.8);
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / total);
      const eased = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      const p = pointOn(tokenGeometry, eased);
      token.setAttribute('cx', p.x.toFixed(2));
      token.setAttribute('cy', p.y.toFixed(2));
      token.setAttribute('opacity', t < 1 ? '1' : '0');
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
    // Re-run only when a new step is taken.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stepKey, reducedMotion]);

  const walkEdges = new Set<string>();
  if (walk) {
    const route = [...walk.path, ...walk.preview];
    for (let i = 1; i < route.length; i += 1) {
      const edge = metagraph.edgeBetween(route[i - 1], route[i]);
      if (edge) walkEdges.add(edge.id);
    }
  }
  const walkRoute = walkEdges.size > 0;
  const currentId = walk ? walk.path[walk.path.length - 1] : null;
  const visited = new Set(walk?.path ?? []);
  const emphasis = hover?.id ?? (walk ? null : selectedId);

  const onNodeKey = (event: KeyboardEvent<SVGGElement>, id: string) => {
    const point = layout.points.get(id);
    if (!point || !scene) return;
    const along = scene.orientation === 'horizontal' ? ['ArrowUp', 'ArrowDown'] : ['ArrowLeft', 'ArrowRight'];
    const across = scene.orientation === 'horizontal' ? ['ArrowLeft', 'ArrowRight'] : ['ArrowUp', 'ArrowDown'];
    let target: string | undefined;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onSelectNode?.(id);
      return;
    }
    if (along.includes(event.key)) {
      const column = layout.columns[point.column];
      target = column[column.indexOf(id) + (event.key === along[0] ? -1 : 1)];
    } else if (across.includes(event.key)) {
      const column = layout.columns[point.column + (event.key === across[0] ? -1 : 1)];
      if (column) {
        target = [...column].sort(
          (x, y) => Math.abs((layout.points.get(x)?.y ?? 0) - point.y) - Math.abs((layout.points.get(y)?.y ?? 0) - point.y),
        )[0];
      }
    } else if (event.key === 'Home') target = metagraph.grandCoalitionId;
    else if (event.key === 'End') target = metagraph.singletonsId;
    if (target && visible(target)) {
      event.preventDefault();
      setFocusId(target);
      nodeRefs.current.get(target)?.focus();
    }
  };

  const tabStop = focusId && visible(focusId) ? focusId : selectedId && visible(selectedId) ? selectedId : pairIds[0];

  const onPointerEnter = (event: PointerEvent<SVGGElement>, id: string) => {
    if (event.pointerType !== 'mouse') return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    setHover({ id, x: event.clientX - rect.left, y: event.clientY - rect.top });
  };

  const renderEdge = (edgeId: string, visual: EdgeVisual, dimmed: boolean, highlighted: boolean) => {
    const edge = metagraph.edges.find((e) => e.id === edgeId);
    const g = geometryOf(edgeId);
    if (!edge || !g) return null;
    const weight = highlighted ? styles.edgeStrong : '';
    const hit = onSelectEdge ? (
      <path
        d={pathOf(g)}
        className={styles.edgeHit}
        onClick={(event) => {
          event.stopPropagation();
          onSelectEdge(edge.id);
        }}
      />
    ) : null;
    const common = `${styles.edge} ${dimmed ? styles.edgeDim : ''} ${weight} ${selectedEdgeId === edge.id ? styles.edgeSelected : ''}`;
    if (visual.style === 'neutral') {
      return (
        <g key={edge.id} className={`${common} ${styles.edgeNeutral}`}>
          <path d={pathOf(g)} />
          {hit}
        </g>
      );
    }
    if (visual.style === 'tie') {
      return (
        <g key={edge.id} className={`${common} ${styles.edgeTie}`}>
          <path d={pathOf(g)} />
          {hit}
        </g>
      );
    }
    if (visual.style === 'clear') {
      const toward = visual.head === edge.b ? g : reverseGeometry(g);
      return (
        <g key={edge.id} className={`${common} ${styles.edgeClear}`}>
          <path d={pathOf(toward)} markerEnd={`url(#${uid}-clear)`} />
          {hit}
        </g>
      );
    }
    // Frustrated: blue from the friend end to F, red from F to the stranger end.
    const fromFriend = visual.friendEnd === edge.a ? g : reverseGeometry(g);
    const [friendPart, strangerPart] = splitGeometry(fromFriend, visual.familiarity);
    const cursor = visual.cursor === null ? null : pointOn(fromFriend, visual.cursor);
    const split = pointOn(fromFriend, visual.familiarity);
    return (
      <g key={edge.id} className={`${common} ${visual.tie ? styles.edgeFrustratedTie : ''}`}>
        <path
          d={pathOf(friendPart)}
          className={styles.friendPart}
          markerStart={visual.friendHead ? `url(#${uid}-friend)` : undefined}
        />
        <path
          d={pathOf(strangerPart)}
          className={styles.strangerPart}
          markerEnd={visual.strangerHead ? `url(#${uid}-stranger)` : undefined}
        />
        {visual.cursor === null && <circle cx={split.x} cy={split.y} r={2.6} className={styles.splitDot} />}
        {cursor &&
          (visual.tie ? (
            <rect x={cursor.x - 4.5} y={cursor.y - 4.5} width={9} height={9} className={styles.cursorTie} transform={`rotate(45 ${cursor.x} ${cursor.y})`} />
          ) : visual.cursor !== null && visual.cursor < visual.familiarity ? (
            <circle cx={cursor.x} cy={cursor.y} r={4.6} className={styles.cursorFriend} />
          ) : (
            <rect x={cursor.x - 4} y={cursor.y - 4} width={8} height={8} className={styles.cursorStranger} />
          ))}
        {isPair && (
          <text x={split.x} y={split.y - 12} textAnchor="middle" className={styles.fLabel}>
            F = {formatFraction(visual.familiarity)}
          </text>
        )}
        {isPair &&
          labels.pairEdge &&
          (() => {
            // Caption each half under its midpoint, on staggered rows so short halves never collide.
            const friendMid = pointOn(friendPart, 0.5);
            const strangerMid = pointOn(strangerPart, 0.5);
            const friendOnLeft = fromFriend.p0.x <= fromFriend.p1.x;
            return (
              <>
                <text x={friendMid.x} y={friendMid.y + 17} textAnchor="middle" className={`${styles.halfLabel} friend-text`}>
                  {friendOnLeft ? `← ${labels.pairEdge.friends}` : `${labels.pairEdge.friends} →`}
                </text>
                <text x={strangerMid.x} y={strangerMid.y + 32} textAnchor="middle" className={`${styles.halfLabel} stranger-text`}>
                  {friendOnLeft ? `${labels.pairEdge.strangers} →` : `← ${labels.pairEdge.strangers}`}
                </text>
              </>
            );
          })()}
        {hit}
      </g>
    );
  };

  const cardId = hover?.id ?? focusCard;
  const hovered = cardId ? metagraph.node(cardId) : null;

  return (
    <div ref={containerRef} className={`${styles.container} ${className ?? ''}`}>
      <p id={`${uid}-hint`} className="visually-hidden">
        {labels.keyboardHint}
      </p>
      {scene && placed && (
        <svg
          ref={svgRef}
          className={`${styles.svg} ${zoomable ? styles.zoomable : ''}`}
          width={box.w}
          height={box.h}
          viewBox={`0 0 ${scene.width.toFixed(1)} ${scene.height.toFixed(1)}`}
          role="group"
          aria-label={labels.figure}
          aria-describedby={`${uid}-hint`}
          data-orientation={scene.orientation}
          onClick={
            onClearSelection
              ? (event) => {
                  if (!(event.target as Element).closest(`.${styles.node}`)) onClearSelection();
                }
              : undefined
          }
        >
          <defs>
            {[
              ['clear', 'var(--clear)'],
              ['friend', 'var(--friend)'],
              ['stranger', 'var(--stranger)'],
              ['walk', 'var(--threshold)'],
            ].map(([name, color]) => (
              <marker
                key={name}
                id={`${uid}-${name}`}
                viewBox="0 0 10 10"
                refX="8.6"
                refY="5"
                markerWidth={name === 'walk' ? 11 : 9}
                markerHeight={name === 'walk' ? 11 : 9}
                markerUnits="userSpaceOnUse"
                orient="auto-start-reverse"
              >
                <path d="M0 0.8 L10 5 L0 9.2 z" fill={color} />
              </marker>
            ))}
          </defs>
          <g transform={transform || undefined}>
            <g className={`${styles.edges} ${mode === 'nodes' ? '' : styles.edgesShown}`}>
              {metagraph.edges.map((edge) => {
                if (isPair) return null;
                if (edge.sameLayer && !showSameLayer) return null;
                const visual = edgeVisual(edge, mode, gamma);
                const incident = emphasis ? edge.a === emphasis || edge.b === emphasis : true;
                const dim = (emphasis !== null && !incident) || (walkRoute && !walkEdges.has(edge.id));
                return renderEdge(edge.id, visual, dim, highlightEdgeId === edge.id || (emphasis !== null && incident && !walk));
              })}
              {isPair &&
                (() => {
                  const edge = metagraph.edgeBetween(pairIds[0], pairIds[1]);
                  return edge ? renderEdge(edge.id, edgeVisual(edge, 'pair', gamma), false, true) : null;
                })()}
            </g>

            {walk && (
              <g className={styles.walk}>
                {walk.path.slice(1).map((to, i) => {
                  const g = oriented2(walk.path[i], to);
                  return g ? <path key={`w-${i}`} d={pathOf(g)} className={styles.walkEdge} markerEnd={`url(#${uid}-walk)`} /> : null;
                })}
                {[currentId, ...walk.preview].slice(0, -1).map((from, i) => {
                  const to = walk.preview[i];
                  const g = from && to ? oriented2(from, to) : null;
                  return g ? <path key={`p-${i}`} d={pathOf(g)} className={styles.previewEdge} markerEnd={`url(#${uid}-walk)`} /> : null;
                })}
                <circle ref={tokenRef} r={6} className={styles.token} opacity={0} />
              </g>
            )}

            <g>
              {metagraph.nodes.map((node) => {
                const place = placed.get(node.id);
                const size = scene.sizes.get(node.id);
                const lines = scene.lines.get(node.id) ?? [];
                if (!place || !size) return null;
                const shown = visible(node.id);
                const sink = oriented && isSink(metagraph, node.id, gamma);
                const phi = formatNumber(nodePotential(node, gamma));
                const category = node.isGrandCoalition ? styles.grand : node.isSingletons ? styles.singletons : styles.mid;
                const pairIndex = pairIds.indexOf(node.id);
                return (
                  <g
                    key={node.id}
                    ref={(element) => {
                      if (element) nodeRefs.current.set(node.id, element);
                      else nodeRefs.current.delete(node.id);
                    }}
                    className={`${styles.node} ${category} ${shown ? '' : styles.hidden} ${selectedId === node.id ? styles.selected : ''} ${
                      currentId === node.id ? styles.current : ''
                    } ${visited.has(node.id) && currentId !== node.id ? styles.visited : ''} ${
                      emphasis && emphasis !== node.id && !walk ? styles.nodeDim : ''
                    }`}
                    style={{ transform: `translate(${place.p.x}px, ${place.p.y}px) scale(${place.s})` }}
                    role={onSelectNode ? 'button' : 'img'}
                    tabIndex={onSelectNode && shown ? (tabStop === node.id ? 0 : -1) : undefined}
                    aria-hidden={shown ? undefined : true}
                    aria-pressed={onSelectNode ? selectedId === node.id : undefined}
                    aria-label={labels.nodeLabel(node, phi, sink)}
                    onClick={onSelectNode ? () => onSelectNode(node.id) : undefined}
                    onKeyDown={onSelectNode ? (event) => onNodeKey(event, node.id) : undefined}
                    onFocus={(event) => {
                      setFocusId(node.id);
                      setFocusCard(isKeyboardFocus(event.currentTarget) ? node.id : null);
                    }}
                    onBlur={() => setFocusCard(null)}
                    onPointerEnter={(event) => onPointerEnter(event, node.id)}
                    onPointerLeave={() => setHover(null)}
                  >
                    <rect className={styles.focusRing} x={-size.w / 2 - 5} y={-size.h / 2 - 5} width={size.w + 10} height={size.h + 10} rx={11} />
                    {sink && <rect className={styles.sinkRing} x={-size.w / 2 - 4} y={-size.h / 2 - 4} width={size.w + 8} height={size.h + 8} rx={10} />}
                    <rect className={styles.card} x={-size.w / 2} y={-size.h / 2} width={size.w} height={size.h} rx={7} />
                    {lines.map((line, i) => (
                      <text key={i} className={styles.label} x={0} y={-size.h / 2 + 6 + LINE_H * (i + 0.5)} dy="0.35em" textAnchor="middle">
                        {line}
                      </text>
                    ))}
                    {sink && (
                      <g className={styles.sinkBadge} transform={`translate(${size.w / 2} ${-size.h / 2})`}>
                        <circle r={7.5} />
                        <path d="M-3.4 -3.2 L0 0.6 L3.4 -3.2 M-3.6 3.2 H3.6" />
                      </g>
                    )}
                    {showPotential && !isPair && (
                      <text className={styles.phi} x={0} y={size.h / 2 + 11} textAnchor="middle">
                        Φ {phi}
                      </text>
                    )}
                    {isPair && pairIndex >= 0 && (
                      <g>
                        <GraphGlyph cx={0} cy={-size.h / 2 - 50} radius={26} membership={node.rgs} accent={3} showLabels vertexRadius={8.5} />
                        {labels.pairNotes && (
                          <text className={styles.pairNote} x={0} y={size.h / 2 + 16} textAnchor="middle">
                            {labels.pairNotes[pairIndex]}
                          </text>
                        )}
                      </g>
                    )}
                  </g>
                );
              })}
            </g>
          </g>
        </svg>
      )}

      {zoomable && (
        <div className={styles.zoomControls}>
          <button type="button" className="btn btn-small btn-icon" onClick={() => zoomBy(1.3)} aria-label={labels.zoomIn} title={labels.zoomIn}>
            <Icon name="zoomIn" />
          </button>
          <button type="button" className="btn btn-small btn-icon" onClick={() => zoomBy(1 / 1.3)} aria-label={labels.zoomOut} title={labels.zoomOut}>
            <Icon name="zoomOut" />
          </button>
          <button type="button" className="btn btn-small btn-icon" onClick={() => zoomBy(null)} aria-label={labels.zoomReset} title={labels.zoomReset}>
            <Icon name="fit" />
          </button>
        </div>
      )}

      {hovered && (
        <div
          className={styles.hoverCard}
          style={{
            left: hover ? Math.min(hover.x + 14, box.w - 190) : 8,
            top: hover ? Math.min(hover.y + 14, box.h - 150) : 8,
          }}
          aria-hidden="true"
        >
          <svg width={64} height={64} viewBox="0 0 64 64">
            <GraphGlyph cx={32} cy={32} radius={22} membership={hovered.rgs} vertexRadius={7.5} />
          </svg>
          <div>
            <p className={styles.hoverTitle}>{hovered.label}</p>
            <p className={styles.hoverLine}>
              {labels.quality} = {formatNumber(nodePotential(hovered, gamma))}
            </p>
            {oriented && isSink(metagraph, hovered.id, gamma) && <p className={styles.hoverSink}>{labels.sink}</p>}
          </div>
        </div>
      )}
    </div>
  );
}
