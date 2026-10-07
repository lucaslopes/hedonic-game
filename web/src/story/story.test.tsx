// @vitest-environment jsdom
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { dictionaries } from '../i18n';
import { useWalk } from '../hooks/useWalk';
import { decide, type WalkPolicy } from '../model';
import { renderWithProviders } from '../test/render';
import { DecisionTree } from '../viz/DecisionTree';
import { EXAMPLES, verdictText, winnerAt } from './dilemmaShared';
import { METAGRAPH } from './metagraphData';

const t = dictionaries.en;

describe('dilemma verdicts', () => {
  it('follows friends below F, strangers above F, and ties at F', () => {
    const counts = EXAMPLES.frustrated; // F = 1/3
    expect(verdictText(counts, 0.2, t)).toBe(
      'The agent chooses Community A: more friends wins, because γ = 0.20 is below F = 1/3.',
    );
    expect(verdictText(counts, 0.5, t)).toBe(
      'The agent chooses Community B: fewer strangers wins, because γ = 0.50 is above F = 1/3.',
    );
    expect(verdictText(counts, 1 / 3, t)).toBe('The scale is level: γ = F = 1/3, so the agent is indifferent.');
    expect(winnerAt(counts, 1 / 3)).toBeNull();
  });

  it('does not mention F before the story introduces it', () => {
    expect(verdictText(EXAMPLES.frustrated, 0.5, t, false)).toBe(
      'The agent chooses Community B: at γ = 0.50 shedding strangers outweighs losing friends.',
    );
  });

  it('explains endpoint ties of clear choices', () => {
    const fewerStrangers = { a: { friends: 1, strangers: 2 }, b: { friends: 1, strangers: 0 } };
    expect(verdictText(fewerStrangers, 0, t)).toContain('At γ = 0 strangers cost nothing');
    expect(verdictText(fewerStrangers, 0.4, t)).toContain('both instincts agree');
    const moreFriends = { a: { friends: 1, strangers: 1 }, b: { friends: 3, strangers: 1 } };
    expect(verdictText(moreFriends, 1, t)).toContain('At γ = 1 friends are worth nothing');
    expect(verdictText(EXAMPLES.identical, 0.5, t)).toBe(t.dilemma.verdictIndifferent);
  });
});

describe('DecisionTree', () => {
  const text = { nodes: t.tree.nodes, yes: t.common.yes, no: t.common.no, label: t.tree.figureLabel };

  it('marks the frustrated leaf and describes the path', () => {
    const result = decide(EXAMPLES.frustrated.a, EXAMPLES.frustrated.b);
    render(<DecisionTree result={result} text={text} />);
    const outline = screen.getByRole('list', { name: t.tree.figureLabel });
    const current = within(outline).getByText((_, element) => element?.getAttribute('aria-current') === 'true');
    expect(current.textContent).toContain('Frustrated');
    const img = screen.getByRole('img');
    expect(img.getAttribute('aria-label')).toContain('Frustrated');
    expect(img.getAttribute('aria-label')).toContain('No');
  });

  it('follows the clear branch for the ideal community', () => {
    const result = decide(EXAMPLES.ideal.a, EXAMPLES.ideal.b);
    render(<DecisionTree result={result} text={text} />);
    expect(screen.getByRole('img').getAttribute('aria-label')).toContain(t.tree.nodes.idealCommunity);
  });
});

function WalkProbe({ gamma, start, policy = 'best' }: { gamma: number; start: string; policy?: WalkPolicy }) {
  const walk = useWalk({ metagraph: METAGRAPH, gamma, policy, seed: 1, start, speed: 'fast' });
  return (
    <div>
      <p data-testid="status">{walk.state.status}</p>
      <p data-testid="current">{walk.state.current}</p>
      <p data-testid="preview">{walk.preview.join(',')}</p>
      <button type="button" onClick={walk.step}>
        step
      </button>
      <button type="button" onClick={walk.reset}>
        reset
      </button>
    </div>
  );
}

describe('useWalk', () => {
  it('steps from the singletons to the triangle-plus-loner sink at γ = 1/2', () => {
    renderWithProviders(<WalkProbe gamma={0.5} start="0123" />);
    expect(screen.getByTestId('preview').textContent).toBe('0012,0001');
    act(() => fireEvent.click(screen.getByText('step')));
    expect(screen.getByTestId('current').textContent).toBe('0012');
    act(() => fireEvent.click(screen.getByText('step')));
    expect(screen.getByTestId('current').textContent).toBe('0001');
    expect(screen.getByTestId('status').textContent).toBe('sink');
    act(() => fireEvent.click(screen.getByText('reset')));
    expect(screen.getByTestId('current').textContent).toBe('0123');
  });

  it('reports a start that is already stable as a sink', () => {
    renderWithProviders(<WalkProbe gamma={1 / 3} start="0000" />);
    expect(screen.getByTestId('status').textContent).toBe('sink');
    expect(screen.getByTestId('preview').textContent).toBe('');
  });

  it('restarts when γ changes', () => {
    const { rerender } = renderWithProviders(<WalkProbe gamma={0.5} start="0123" />);
    act(() => fireEvent.click(screen.getByText('step')));
    expect(screen.getByTestId('current').textContent).toBe('0012');
    rerender(<WalkProbe gamma={0.2} start="0123" />);
    expect(screen.getByTestId('current').textContent).toBe('0123');
    expect(screen.getByTestId('preview').textContent).toBe('0012,0001,0000');
  });
});
