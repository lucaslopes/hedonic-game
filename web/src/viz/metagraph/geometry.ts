/** Geometry helpers for drawing the metagraph (all values in SVG user units). */

import type { MetaNode } from '../../model';

export interface Point {
  readonly x: number;
  readonly y: number;
}

export interface Size {
  readonly w: number;
  readonly h: number;
}

export const FONT_SIZE = 12;
export const CHAR_W = FONT_SIZE * 0.6; // monospace advance width
export const LINE_H = 15;
const PAD_X = 8;
const PAD_Y = 6;

export type LabelMode = 'stacked' | 'inline';

/** Text lines of a metanode: one community per line (as in the paper), or inline. */
export function nodeLines(node: MetaNode, mode: LabelMode): string[] {
  return mode === 'inline' ? [node.label] : node.blocks.map((block) => `{${block.join(',')}}`);
}

export function cardSize(lines: readonly string[]): Size {
  const longest = Math.max(...lines.map((line) => line.length));
  return { w: longest * CHAR_W + 2 * PAD_X, h: lines.length * LINE_H + 2 * PAD_Y };
}

export function lerp(a: Point, b: Point, t: number): Point {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

/** Where the ray from `center` toward `toward` leaves the (padded) card. */
export function rectBorderPoint(center: Point, size: Size, toward: Point, gap = 3): Point {
  const dx = toward.x - center.x;
  const dy = toward.y - center.y;
  if (dx === 0 && dy === 0) return center;
  const tx = dx === 0 ? Number.POSITIVE_INFINITY : (size.w / 2 + gap) / Math.abs(dx);
  const ty = dy === 0 ? Number.POSITIVE_INFINITY : (size.h / 2 + gap) / Math.abs(dy);
  const t = Math.min(tx, ty, 1);
  return { x: center.x + dx * t, y: center.y + dy * t };
}

export type EdgeGeometry =
  | { readonly kind: 'line'; readonly p0: Point; readonly p1: Point }
  | { readonly kind: 'quad'; readonly p0: Point; readonly c: Point; readonly p1: Point };

/** Straight edge between two cards, or a quadratic arc bulging by `offset`. */
export function edgeGeometry(pa: Point, sa: Size, pb: Point, sb: Size, offset?: Point): EdgeGeometry {
  if (offset) {
    const c = { x: (pa.x + pb.x) / 2 + offset.x, y: (pa.y + pb.y) / 2 + offset.y };
    return { kind: 'quad', p0: rectBorderPoint(pa, sa, c), c, p1: rectBorderPoint(pb, sb, c) };
  }
  return { kind: 'line', p0: rectBorderPoint(pa, sa, pb), p1: rectBorderPoint(pb, sb, pa) };
}

export function reverseGeometry(g: EdgeGeometry): EdgeGeometry {
  return g.kind === 'line' ? { kind: 'line', p0: g.p1, p1: g.p0 } : { kind: 'quad', p0: g.p1, c: g.c, p1: g.p0 };
}

export function pathOf(g: EdgeGeometry): string {
  const f = (n: number) => n.toFixed(2);
  return g.kind === 'line'
    ? `M${f(g.p0.x)} ${f(g.p0.y)} L${f(g.p1.x)} ${f(g.p1.y)}`
    : `M${f(g.p0.x)} ${f(g.p0.y)} Q${f(g.c.x)} ${f(g.c.y)} ${f(g.p1.x)} ${f(g.p1.y)}`;
}

export function pointOn(g: EdgeGeometry, t: number): Point {
  if (g.kind === 'line') return lerp(g.p0, g.p1, t);
  return lerp(lerp(g.p0, g.c, t), lerp(g.c, g.p1, t), t);
}

/** Split an edge at parameter t (de Casteljau for arcs). */
export function splitGeometry(g: EdgeGeometry, t: number): [EdgeGeometry, EdgeGeometry] {
  if (g.kind === 'line') {
    const m = lerp(g.p0, g.p1, t);
    return [
      { kind: 'line', p0: g.p0, p1: m },
      { kind: 'line', p0: m, p1: g.p1 },
    ];
  }
  const c1 = lerp(g.p0, g.c, t);
  const c2 = lerp(g.c, g.p1, t);
  const m = lerp(c1, c2, t);
  return [
    { kind: 'quad', p0: g.p0, c: c1, p1: m },
    { kind: 'quad', p0: m, c: c2, p1: g.p1 },
  ];
}
