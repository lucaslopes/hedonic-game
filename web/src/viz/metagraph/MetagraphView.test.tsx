// @vitest-environment jsdom
import { fireEvent, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LAYOUT, METAGRAPH } from '../../story/metagraphData';
import { renderWithProviders } from '../../test/render';
import { edgeVisual } from './edgeVisual';
import { MetagraphView, type MetagraphLabels } from './MetagraphView';

const labels: MetagraphLabels = {
  figure: 'Metagraph',
  nodeLabel: (node, phi, sink) => `${node.label} Φ=${phi}${sink ? ' sink' : ''}`,
  zoomIn: 'Zoom in',
  zoomOut: 'Zoom out',
  zoomReset: 'Fit',
  keyboardHint: 'Arrow keys move between partitions.',
  sink: 'Sink',
  quality: 'Φγ',
};

// jsdom has no layout: report a fixed stage size to the component.
class SizedObserver {
  constructor(private readonly callback: ResizeObserverCallback) {}
  observe() {
    this.callback([{ contentRect: { width: 900, height: 560 } } as ResizeObserverEntry], this as unknown as ResizeObserver);
  }
  unobserve() {}
  disconnect() {}
}

describe('MetagraphView', () => {
  const original = window.ResizeObserver;
  beforeEach(() => {
    window.ResizeObserver = SizedObserver as unknown as typeof ResizeObserver;
  });
  afterEach(() => {
    window.ResizeObserver = original;
  });

  it('renders every partition as a keyboard target with a single tab stop', () => {
    const onSelect = vi.fn();
    renderWithProviders(
      <MetagraphView metagraph={METAGRAPH} layout={LAYOUT} gamma={0.5} mode="oriented" labels={labels} reducedMotion onSelectNode={onSelect} />,
    );
    const nodes = screen.getAllByRole('button', { name: /Φ=/ });
    expect(nodes).toHaveLength(15);
    expect(nodes.filter((node) => node.getAttribute('tabindex') === '0')).toHaveLength(1);
    expect(screen.getByRole('button', { name: '{0,1,2}{3} Φ=1.50 sink' })).toBeTruthy();
  });

  it('moves focus with the keyboard and selects with Enter', () => {
    const onSelect = vi.fn();
    renderWithProviders(
      <MetagraphView metagraph={METAGRAPH} layout={LAYOUT} gamma={0.5} mode="oriented" labels={labels} reducedMotion onSelectNode={onSelect} />,
    );
    const grand = screen.getByRole('button', { name: /^\{0,1,2,3\}/ });
    grand.focus();
    fireEvent.keyDown(grand, { key: 'End' });
    const singletons = screen.getByRole('button', { name: /^\{0\}\{1\}\{2\}\{3\}/ });
    expect(document.activeElement).toBe(singletons);
    fireEvent.keyDown(singletons, { key: 'Enter' });
    expect(onSelect).toHaveBeenCalledWith(METAGRAPH.singletonsId);
  });

  it('shows only the two partitions of agent 3 in pair mode', () => {
    renderWithProviders(
      <MetagraphView
        metagraph={METAGRAPH}
        layout={LAYOUT}
        gamma={0.5}
        mode="pair"
        labels={{ ...labels, pairEdge: { friends: 'more friends', strangers: 'fewer strangers' } }}
        reducedMotion
        pair={['0000', '0001']}
        onSelectNode={() => undefined}
      />,
    );
    const visible = screen.getAllByRole('button', { name: /Φ=/ });
    expect(visible.map((node) => node.getAttribute('aria-label')?.split(' ')[0])).toEqual(['{0,1,2,3}', '{0,1,2}{3}']);
    expect(screen.getByText('F = 1/3')).toBeTruthy();
  });
});

describe('edge encoding', () => {
  const edge = METAGRAPH.edgeBetween('0000', '0001');
  if (!edge) throw new Error('missing edge');

  it('draws the paper view as a two-headed split edge', () => {
    expect(edgeVisual(edge, 'types', 0.5)).toMatchObject({ style: 'frustrated', friendHead: true, strangerHead: true, cursor: null });
  });

  it('keeps one arrowhead once γ is fixed, and none at the threshold', () => {
    expect(edgeVisual(edge, 'oriented', 0.5)).toMatchObject({ friendHead: false, strangerHead: true, cursor: 0.5, tie: false });
    expect(edgeVisual(edge, 'oriented', 0.2)).toMatchObject({ friendHead: true, strangerHead: false });
    expect(edgeVisual(edge, 'oriented', 1 / 3)).toMatchObject({ friendHead: false, strangerHead: false, tie: true });
  });

  it('draws indifferent moves as ties and untyped moves as neutral', () => {
    const tie = METAGRAPH.edges.find((e) => e.tradeoff.kind === 'indifferent');
    if (!tie) throw new Error('missing tie');
    expect(edgeVisual(tie, 'oriented', 0.5)).toEqual({ style: 'tie', reason: 'indifferent' });
    expect(edgeVisual(edge, 'edges', 0.5)).toEqual({ style: 'neutral' });
  });
});
