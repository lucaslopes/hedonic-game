// @vitest-environment jsdom
import { act, fireEvent, screen } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useWalk } from '../hooks/useWalk';
import { renderWithProviders } from '../test/render';
import { MetagraphView } from '../viz/metagraph/MetagraphView';
import { WalkStatus } from '../walk/WalkStatus';
import { DilemmaStage } from './DilemmaChapter';
import { DEFAULT_COUNTS, type Counts } from './dilemmaShared';
import { LAYOUT, METAGRAPH } from './metagraphData';

function Dilemma() {
  const [counts, setCounts] = useState<Counts>(DEFAULT_COUNTS);
  const [gamma, setGamma] = useState(0.5);
  return <DilemmaStage level={3} counts={counts} setCounts={setCounts} gamma={gamma} setGamma={setGamma} />;
}

describe('dilemma stage', () => {
  it('resets the counts and γ like the original prototype', () => {
    renderWithProviders(<Dilemma />);
    const reset = screen.getByRole('button', { name: 'Reset the scale' });
    expect(reset).toHaveProperty('disabled', true);
    fireEvent.click(screen.getByRole('button', { name: 'Add a friend to community B' }));
    fireEvent.click(screen.getByRole('button', { name: 'Set γ to 0.25' }));
    expect(screen.getByLabelText('2 friends in community B')).toBeTruthy();
    expect(reset).toHaveProperty('disabled', false);
    fireEvent.click(reset);
    expect(screen.getByLabelText('1 friend in community B')).toBeTruthy();
    expect(screen.getByRole('slider')).toHaveProperty('value', '0.5');
    expect(reset).toHaveProperty('disabled', true);
  });

  it('states the verdict and the Familiarity Index at level 3', () => {
    renderWithProviders(<Dilemma />);
    expect(screen.getByText('The agent chooses Community B: fewer strangers wins, because γ = 0.50 is above F = 1/3.')).toBeTruthy();
    expect(screen.getByText('Moving from A to B: Δd = −1, Δd̂ = −2. Familiarity Index: F = 1/3.')).toBeTruthy();
  });
});

function WalkPanel() {
  const walk = useWalk({ metagraph: METAGRAPH, gamma: 0.5, policy: 'best', seed: 1, start: '0000', speed: 'fast' });
  return (
    <>
      <WalkStatus metagraph={METAGRAPH} controller={walk} gamma={0.5} />
      <button type="button" onClick={walk.step}>
        advance
      </button>
    </>
  );
}

describe('walk status', () => {
  it('draws the current partition and describes the last move', () => {
    const { container } = renderWithProviders(<WalkPanel />);
    expect(container.querySelectorAll('svg circle').length).toBeGreaterThanOrEqual(4);
    act(() => fireEvent.click(screen.getByText('advance')));
    expect(screen.getByText(/Vertex 3 leaves to start its own community/)).toBeTruthy();
    expect(screen.getAllByText('Stable equilibrium reached').length).toBeGreaterThan(0);
  });
});

class SizedObserver {
  constructor(private readonly callback: ResizeObserverCallback) {}
  observe() {
    this.callback([{ contentRect: { width: 900, height: 560 } } as ResizeObserverEntry], this as unknown as ResizeObserver);
  }
  unobserve() {}
  disconnect() {}
}

describe('metagraph keyboard focus', () => {
  const original = window.ResizeObserver;
  beforeEach(() => {
    window.ResizeObserver = SizedObserver as unknown as typeof ResizeObserver;
    // jsdom cannot evaluate :focus-visible; treat every focus as keyboard focus here.
    vi.spyOn(Element.prototype, 'matches').mockImplementation(function matches(this: Element, selector: string) {
      return selector === ':focus-visible' ? true : false;
    });
  });
  afterEach(() => {
    window.ResizeObserver = original;
    vi.restoreAllMocks();
  });

  it('reveals partition details when a node receives keyboard focus', () => {
    const labels = {
      figure: 'Metagraph',
      nodeLabel: (node: { label: string }) => node.label,
      zoomIn: 'in',
      zoomOut: 'out',
      zoomReset: 'fit',
      keyboardHint: 'hint',
      sink: 'Sink',
      quality: 'Φγ',
    };
    renderWithProviders(
      <MetagraphView metagraph={METAGRAPH} layout={LAYOUT} gamma={0.5} mode="oriented" labels={labels} reducedMotion onSelectNode={() => undefined} />,
    );
    const node = screen.getByRole('button', { name: '{0,1,2}{3}' });
    act(() => node.focus());
    expect(screen.getByText('Φγ = 1.50')).toBeTruthy();
    expect(screen.getByText('Sink')).toBeTruthy();
    act(() => node.blur());
    expect(screen.queryByText('Φγ = 1.50')).toBeNull();
  });
});
