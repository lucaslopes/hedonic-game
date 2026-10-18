// @vitest-environment jsdom
import { fireEvent, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { renderWithProviders } from '../test/render';
import { OverlapApp } from './OverlapApp';

describe('overlap page', () => {
  it('marks the overlap tab as the current page and links back to the disjoint story', () => {
    renderWithProviders(<OverlapApp />);
    const nav = screen.getByRole('navigation', { name: 'Primary' });
    expect(within(nav).getByRole('link', { name: 'Overlap' }).getAttribute('aria-current')).toBe('page');
    expect(within(nav).getByRole('link', { name: 'Disjoint' }).getAttribute('href')).toBe('/');
  });

  it('shows what vertex 2 sees in the shared-edge cover', () => {
    renderWithProviders(<OverlapApp />);
    expect(screen.getByText('Vertex 2 holds {A, B}, each at intensity 0.71: U = 1.91.')).toBeTruthy();
    expect(screen.getByText('{A, B} is a best response: the vertex stays.')).toBeTruthy();
    expect(screen.getByRole('button', { name: /Play the best response/ })).toHaveProperty('disabled', true);
  });

  it('lets vertex 0 take its best response below the end of its stable interval', () => {
    renderWithProviders(<OverlapApp />);
    fireEvent.click(screen.getAllByLabelText('Look through the eyes of vertex 0 (holds {A})')[1]);
    fireEvent.click(screen.getAllByLabelText('Set γ to 0.00')[0]);
    expect(screen.getByText('It would switch to {A, B} and gain +0.29.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /Play the best response/ }));
    expect(screen.getByText('Vertex 0 holds {A, B}, each at intensity 0.71: U = 2.71.')).toBeTruthy();
  });

  it('runs the dynamics to the cover the package returns', () => {
    renderWithProviders(<OverlapApp />);
    fireEvent.click(screen.getByRole('button', { name: 'Run to the end' }));
    expect(screen.getByText(/Equilibrium after \d+ visits and \d+ moves/)).toBeTruthy();
    const code = [...document.querySelectorAll('code.language-python')].map((element) => element.textContent ?? '');
    expect(code.some((text) => text.includes('seed=0)\ngraph.memberships  # [[0], [0], [0, 1], [0, 1], [1], [1]]'))).toBe(true);
  });
});
