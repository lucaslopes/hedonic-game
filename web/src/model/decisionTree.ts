/**
 * The decision tree of Figure 2 (SRC extended abstract): how an agent
 * compares two communities before any resolution value is chosen.
 *
 *   Same number of links in both communities?
 *   ├─ yes → Same number of non-links?
 *   │         ├─ yes → Indifferent between communities            (grey)
 *   │         └─ no  → Prefer community with fewer non-links     (purple)
 *   └─ no  → Same number of non-links?
 *             ├─ yes → Prefer community with more links          (purple)
 *             └─ no  → Is a community with more links also one with
 *                      fewer non-links?
 *                       ├─ yes → Prefer the ideal community      (purple)
 *                       └─ no  → Frustrated            (red, blue border)
 *
 * "Links" are friends (neighbours) and "non-links" are strangers.
 */

import { compareCommunities, type CommunityCounts, type TradeoffKind } from './tradeoff';

export type TreeQuestionId = 'sameLinks' | 'sameNonLinksWhenSameLinks' | 'sameNonLinks' | 'ideal';
export type TreeLeafId = 'indifferent' | 'fewerNonLinks' | 'moreLinks' | 'idealCommunity' | 'frustrated';
export type TreeNodeId = TreeQuestionId | TreeLeafId;

export interface TreeNode {
  readonly id: TreeNodeId;
  readonly kind: 'question' | 'leaf';
  /** Leaf colour class in the paper: grey, purple, or the red/blue frustrated box. */
  readonly tone?: 'neutral' | 'clear' | 'frustrated';
  readonly yes?: TreeNodeId;
  readonly no?: TreeNodeId;
}

export const TREE_ROOT: TreeQuestionId = 'sameLinks';

export const TREE_NODES: Readonly<Record<TreeNodeId, TreeNode>> = {
  sameLinks: { id: 'sameLinks', kind: 'question', yes: 'sameNonLinksWhenSameLinks', no: 'sameNonLinks' },
  sameNonLinksWhenSameLinks: {
    id: 'sameNonLinksWhenSameLinks',
    kind: 'question',
    yes: 'indifferent',
    no: 'fewerNonLinks',
  },
  sameNonLinks: { id: 'sameNonLinks', kind: 'question', yes: 'moreLinks', no: 'ideal' },
  ideal: { id: 'ideal', kind: 'question', yes: 'idealCommunity', no: 'frustrated' },
  indifferent: { id: 'indifferent', kind: 'leaf', tone: 'neutral' },
  fewerNonLinks: { id: 'fewerNonLinks', kind: 'leaf', tone: 'clear' },
  moreLinks: { id: 'moreLinks', kind: 'leaf', tone: 'clear' },
  idealCommunity: { id: 'idealCommunity', kind: 'leaf', tone: 'clear' },
  frustrated: { id: 'frustrated', kind: 'leaf', tone: 'frustrated' },
};

export interface DecisionStep {
  readonly question: TreeQuestionId;
  readonly answer: 'yes' | 'no';
}

export interface DecisionResult {
  /** Questions asked, in order, with the answer that was followed. */
  readonly steps: readonly DecisionStep[];
  /** Every node on the highlighted path, root first, leaf last. */
  readonly path: readonly TreeNodeId[];
  readonly leaf: TreeLeafId;
  /** Community preferred by the tree, when the choice is unambiguous. */
  readonly preferred: 'A' | 'B' | null;
  /** The equivalent trade-off classification of the move A → B. */
  readonly kind: TradeoffKind;
}

/** Walk the decision tree for an agent comparing communities A and B. */
export function decide(a: CommunityCounts, b: CommunityCounts): DecisionResult {
  const steps: DecisionStep[] = [];
  const sameLinks = a.friends === b.friends;
  const sameNonLinks = a.strangers === b.strangers;
  let leaf: TreeLeafId;
  let preferred: DecisionResult['preferred'] = null;

  steps.push({ question: 'sameLinks', answer: sameLinks ? 'yes' : 'no' });
  if (sameLinks) {
    steps.push({ question: 'sameNonLinksWhenSameLinks', answer: sameNonLinks ? 'yes' : 'no' });
    if (sameNonLinks) {
      leaf = 'indifferent';
    } else {
      leaf = 'fewerNonLinks';
      preferred = a.strangers < b.strangers ? 'A' : 'B';
    }
  } else {
    steps.push({ question: 'sameNonLinks', answer: sameNonLinks ? 'yes' : 'no' });
    if (sameNonLinks) {
      leaf = 'moreLinks';
      preferred = a.friends > b.friends ? 'A' : 'B';
    } else {
      const moreLinks: 'A' | 'B' = a.friends > b.friends ? 'A' : 'B';
      const fewerNonLinks: 'A' | 'B' = a.strangers < b.strangers ? 'A' : 'B';
      const ideal = moreLinks === fewerNonLinks;
      steps.push({ question: 'ideal', answer: ideal ? 'yes' : 'no' });
      leaf = ideal ? 'idealCommunity' : 'frustrated';
      preferred = ideal ? moreLinks : null;
    }
  }

  const path: TreeNodeId[] = steps.map((step) => step.question);
  path.push(leaf);
  return { steps, path, leaf, preferred, kind: compareCommunities(a, b).kind };
}
