import { useMemo } from 'react';
import { useT } from '../i18n';
import { buildMetagraph, EXAMPLE_GRAPH, frustrationThresholds } from '../model';
import { layoutMetagraph } from '../viz/metagraphLayout';
import type { MetagraphLabels, MetagraphMode } from '../viz/metagraph/MetagraphView';

/** The metagraph of the paper's four-vertex example, generated once. */
export const METAGRAPH = buildMetagraph(EXAMPLE_GRAPH);
export const LAYOUT = layoutMetagraph(METAGRAPH);
export const THRESHOLDS = frustrationThresholds(METAGRAPH);
/** Agent 3's dilemma: the grand coalition and {0,1,2}{3}. */
export const AGENT3_PAIR = [METAGRAPH.grandCoalitionId, '0001'] as const;

export function useMetagraphLabels(mode: MetagraphMode | 'walk' | 'explore'): MetagraphLabels {
  const t = useT();
  return useMemo(
    () => ({
      figure: t.metagraph.figureLabel(t.metagraph.modes[mode]),
      nodeLabel: (node, phi, sink) =>
        t.metagraph.nodeLabel(
          node.label,
          node.communityCount,
          node.sizes.join(', '),
          phi,
          sink,
          node.isGrandCoalition ? t.metagraph.kinds.grand : node.isSingletons ? t.metagraph.kinds.singletons : t.metagraph.kinds.partition,
        ),
      zoomIn: t.explore.zoomIn,
      zoomOut: t.explore.zoomOut,
      zoomReset: t.explore.zoomReset,
      keyboardHint: t.metagraph.keyboardHint,
      sink: t.metagraph.legend.sink,
      quality: 'Φγ',
      pairNotes: [t.metagraph.agent3.stay, t.metagraph.agent3.leave] as const,
      pairEdge: { friends: t.dilemma.moreFriends.toLowerCase(), strangers: t.dilemma.fewerStrangers.toLowerCase() },
    }),
    [t, mode],
  );
}
