// @vitest-environment jsdom
import { fireEvent, screen } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { renderWithProviders } from '../test/render';
import { ResolutionControl } from './ResolutionControl';

function Harness({ initial, onValue }: { initial: number; onValue: (g: number) => void }) {
  const [gamma, setGamma] = useState(initial);
  return (
    <ResolutionControl
      gamma={gamma}
      onChange={(g) => {
        setGamma(g);
        onValue(g);
      }}
      tradeoff={{ familiarity: 1 / 3, kind: 'frustrated' }}
    />
  );
}

describe('ResolutionControl', () => {
  it('shows the friend reward and stranger cost split', () => {
    renderWithProviders(<Harness initial={0.25} onValue={() => undefined} />);
    expect(screen.getByRole('slider')).toHaveProperty('value', '0.25');
    expect(screen.getByText('0.75')).toBeTruthy();
    expect(screen.getByRole('slider').getAttribute('aria-valuetext')).toBe(
      'γ = 0.25. Each friend is worth 0.75; each stranger costs 0.25.',
    );
  });

  it('steps through exact thresholds with the keyboard', () => {
    const onValue = vi.fn();
    renderWithProviders(<Harness initial={0.33} onValue={onValue} />);
    const slider = screen.getByRole('slider');
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(onValue).toHaveBeenLastCalledWith(1 / 3);
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(onValue).toHaveBeenLastCalledWith(0.34);
    fireEvent.keyDown(slider, { key: 'End' });
    expect(onValue).toHaveBeenLastCalledWith(1);
    fireEvent.keyDown(slider, { key: 'Home' });
    expect(onValue).toHaveBeenLastCalledWith(0);
  });

  it('jumps to the Familiarity Index and to presets', () => {
    const onValue = vi.fn();
    renderWithProviders(<Harness initial={0.5} onValue={onValue} />);
    fireEvent.click(screen.getByRole('button', { name: 'Set γ = F (1/3)' }));
    expect(onValue).toHaveBeenLastCalledWith(1 / 3);
    expect(screen.getByText('1/3 ≈ 0.333')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Set γ to 0.75' }));
    expect(onValue).toHaveBeenLastCalledWith(0.75);
  });

  it('accepts typed fractions and rejects out-of-range values', () => {
    const onValue = vi.fn();
    renderWithProviders(<Harness initial={0.5} onValue={onValue} />);
    const input = screen.getByRole('textbox', { name: 'γ value' });
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: '2/3' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onValue).toHaveBeenLastCalledWith(2 / 3);
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: '1.4' } });
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(screen.getByText('Enter a number between 0 and 1, like 0.4 or 1/3.')).toBeTruthy();
  });

  it('explains the extremes', () => {
    renderWithProviders(<ResolutionControl gamma={0} onChange={() => undefined} showExtremes />);
    expect(screen.getByRole('note').textContent).toContain('grand coalition');
  });
});
