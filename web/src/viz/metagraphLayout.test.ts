import { describe, expect, it } from 'vitest';
import { EXAMPLE_GRAPH, buildMetagraph } from '../model';
import { layoutMetagraph } from './metagraphLayout';

const mg = buildMetagraph(EXAMPLE_GRAPH);

describe('metagraph layout', () => {
  const layout = layoutMetagraph(mg);

  it('places every partition exactly once, in five columns from grand coalition to singletons', () => {
    expect(layout.points.size).toBe(15);
    expect(layout.columns.map((column) => column.length)).toEqual([1, 4, 3, 6, 1]);
    expect(layout.columns[0]).toEqual([mg.grandCoalitionId]);
    expect(layout.columns[4]).toEqual([mg.singletonsId]);
    expect(layout.points.get(mg.grandCoalitionId)?.x).toBe(0);
    expect(layout.points.get(mg.singletonsId)?.x).toBe(1);
  });

  it('keeps coordinates normalised and rows distinct within a column', () => {
    for (const column of layout.columns) {
      const ys = column.map((id) => layout.points.get(id)?.y ?? -1);
      ys.forEach((value) => {
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(1);
      });
      expect(new Set(ys).size).toBe(ys.length);
    }
  });

  it('is deterministic', () => {
    expect(layoutMetagraph(mg).columns).toEqual(layout.columns);
  });
});
