import { describe, expect, it } from 'vitest';
import { TREE_NODES, TREE_ROOT, decide } from '../decisionTree';
import { compareCommunities } from '../tradeoff';

describe('decision tree of Figure 2', () => {
  it('reaches every leaf through the paper’s questions', () => {
    expect(decide({ friends: 1, strangers: 1 }, { friends: 1, strangers: 1 })).toMatchObject({
      path: ['sameLinks', 'sameNonLinksWhenSameLinks', 'indifferent'],
      leaf: 'indifferent',
      preferred: null,
    });
    expect(decide({ friends: 1, strangers: 2 }, { friends: 1, strangers: 0 })).toMatchObject({
      path: ['sameLinks', 'sameNonLinksWhenSameLinks', 'fewerNonLinks'],
      preferred: 'B',
    });
    expect(decide({ friends: 3, strangers: 1 }, { friends: 1, strangers: 1 })).toMatchObject({
      path: ['sameLinks', 'sameNonLinks', 'moreLinks'],
      preferred: 'A',
    });
    expect(decide({ friends: 1, strangers: 2 }, { friends: 2, strangers: 0 })).toMatchObject({
      path: ['sameLinks', 'sameNonLinks', 'ideal', 'idealCommunity'],
      preferred: 'B',
    });
    expect(decide({ friends: 2, strangers: 2 }, { friends: 1, strangers: 0 })).toMatchObject({
      path: ['sameLinks', 'sameNonLinks', 'ideal', 'frustrated'],
      leaf: 'frustrated',
      preferred: null,
    });
  });

  it('records the answer followed at each question', () => {
    const { steps } = decide({ friends: 2, strangers: 2 }, { friends: 1, strangers: 0 });
    expect(steps).toEqual([
      { question: 'sameLinks', answer: 'no' },
      { question: 'sameNonLinks', answer: 'no' },
      { question: 'ideal', answer: 'no' },
    ]);
  });

  it('agrees with the Familiarity Index classification for every count', () => {
    for (let fa = 0; fa <= 4; fa += 1) {
      for (let sa = 0; sa <= 4; sa += 1) {
        for (let fb = 0; fb <= 4; fb += 1) {
          for (let sb = 0; sb <= 4; sb += 1) {
            const a = { friends: fa, strangers: sa };
            const b = { friends: fb, strangers: sb };
            const result = decide(a, b);
            const tradeoff = compareCommunities(a, b);
            const expectedKind =
              result.leaf === 'indifferent' ? 'indifferent' : result.leaf === 'frustrated' ? 'frustrated' : 'clear';
            expect(tradeoff.kind).toBe(expectedKind);
            expect(result.kind).toBe(expectedKind);
            if (tradeoff.kind === 'clear') {
              expect(result.preferred).toBe(tradeoff.paretoPreference === 'forward' ? 'B' : 'A');
            }
          }
        }
      }
    }
  });

  it('is a well-formed binary tree', () => {
    const visit = (id: keyof typeof TREE_NODES): number => {
      const node = TREE_NODES[id];
      if (node.kind === 'leaf') return 1;
      expect(node.yes && node.no).toBeTruthy();
      return visit(node.yes as keyof typeof TREE_NODES) + visit(node.no as keyof typeof TREE_NODES);
    };
    expect(visit(TREE_ROOT)).toBe(5);
  });
});
