import { useT } from '../i18n';
import {
  edgeState,
  formatBlocks,
  formatDelta,
  formatFraction,
  formatNumber,
  formatSigned,
  improvingMoves,
  isSink,
  nodePotential,
  scoredMoves,
  type Metagraph,
} from '../model';
import { GraphGlyph } from '../viz/GraphGlyph';
import { targetText } from '../walk/describe';
import styles from './Explore.module.css';

interface InspectorProps {
  readonly metagraph: Metagraph;
  readonly gamma: number;
  readonly selectedId: string | null;
  readonly onSelectNode: (id: string) => void;
  readonly selectedEdgeId: string | null;
  readonly onHighlightEdge: (id: string | null) => void;
}

export function Inspector({ metagraph, gamma, selectedId, onSelectNode, selectedEdgeId, onHighlightEdge }: InspectorProps) {
  const t = useT();
  const edge = selectedEdgeId ? metagraph.edges.find((e) => e.id === selectedEdgeId) : undefined;

  const edgePanel = edge ? (
    <section className={styles.edgePanel} aria-label={t.explore.edgeTitle}>
      <h4>{t.explore.edgeTitle}</h4>
      <p className={styles.edgeEnds}>
        <button type="button" className={styles.linkButton} onClick={() => onSelectNode(edge.a)}>
          <code>{metagraph.node(edge.a).label}</code>
        </button>
        <span aria-hidden="true">⇄</span>
        <button type="button" className={styles.linkButton} onClick={() => onSelectNode(edge.b)}>
          <code>{metagraph.node(edge.b).label}</code>
        </button>
      </p>
      <p>
        {t.explore.edgeMovers(edge.forward.map((m) => t.common.vertex(m.vertex)).join(` ${t.common.or} `))}
        {edge.forward.length > 1 && <span className="muted"> ({t.explore.edgeEither})</span>}
      </p>
      <p className="num">
        Δd = {formatDelta(edge.tradeoff.deltaFriends)}, Δd̂ = {formatDelta(edge.tradeoff.deltaStrangers)}
        {edge.tradeoff.familiarity !== null && <>, F = {formatFraction(edge.tradeoff.familiarity)}</>} ·{' '}
        <span className={`chip chip-${edge.tradeoff.kind === 'frustrated' ? 'stranger' : edge.tradeoff.kind === 'clear' ? 'clear' : 'tie'}`}>
          {t.explore.types[edge.tradeoff.kind]}
        </span>
      </p>
      {(() => {
        const state = edgeState(edge, gamma);
        return state.from && state.to ? (
          <p>
            γ = {formatNumber(gamma)}: <code>{metagraph.node(state.from).label}</code> → <code>{metagraph.node(state.to).label}</code> (
            {formatSigned(Math.abs(state.gain))})
          </p>
        ) : (
          <p>
            γ = {formatNumber(gamma)}: {t.metagraph.legend.tie}
          </p>
        );
      })()}
    </section>
  ) : null;

  if (!selectedId) {
    return (
      <div className={styles.inspector}>
        <p className={styles.empty}>{t.explore.nothingSelected}</p>
        {edgePanel}
      </div>
    );
  }

  const node = metagraph.node(selectedId);
  const sink = isSink(metagraph, node.id, gamma);
  const improving = improvingMoves(metagraph, node.id, gamma).length;
  const moves = scoredMoves(metagraph, node.id, gamma);

  return (
    <div className={styles.inspector}>
      <div className={styles.inspectorHead}>
        <svg width={96} height={96} viewBox="0 0 96 96" aria-hidden="true">
          <GraphGlyph cx={48} cy={48} radius={32} membership={node.rgs} vertexRadius={11} />
        </svg>
        <div>
          <p className={styles.nodeTitle}>
            <code>{node.label}</code>
          </p>
          <p className={styles.badges}>
            {node.isGrandCoalition && <span className="chip chip-friend">{t.metagraph.kinds.grand}</span>}
            {node.isSingletons && <span className="chip chip-stranger">{t.metagraph.kinds.singletons}</span>}
            {sink ? <span className={`chip ${styles.sinkChip}`}>{t.explore.isSink}</span> : <span className="chip">{t.explore.notSink(improving)}</span>}
          </p>
        </div>
      </div>
      <dl className={styles.facts}>
        <div>
          <dt>{t.explore.communities}</dt>
          <dd className="num">{node.communityCount}</dd>
        </div>
        <div>
          <dt>{t.explore.sizes}</dt>
          <dd className="num">{node.sizes.join(' + ')}</dd>
        </div>
        <div>
          <dt>{t.explore.quality}</dt>
          <dd className="num">{formatNumber(nodePotential(node, gamma))}</dd>
        </div>
        <div>
          <dt>{t.explore.internalLinks}</dt>
          <dd className="num">{node.internalEdges}</dd>
        </div>
        <div>
          <dt>{t.explore.internalNonLinks}</dt>
          <dd className="num">{node.internalNonEdges}</dd>
        </div>
        <div>
          <dt>{t.explore.layer}</dt>
          <dd className="num">
            {node.layer} ({node.distanceFromGrand} / {node.distanceFromSingletons})
          </dd>
        </div>
      </dl>
      <p className={styles.factNote}>
        {t.explore.layer}: ({t.explore.fromGrand.toLowerCase()} / {t.explore.toSingletons.toLowerCase()})
      </p>
      <h4 className={styles.movesTitle}>{t.explore.moves}</h4>
      <div className="table-scroll">
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">{t.explore.moveHeaders.move}</th>
              <th scope="col" className={styles.numeric}>
                {t.explore.moveHeaders.dd}
              </th>
              <th scope="col" className={styles.numeric}>
                {t.explore.moveHeaders.ds}
              </th>
              <th scope="col" className={styles.numeric}>
                {t.explore.moveHeaders.f}
              </th>
              <th scope="col">{t.explore.moveHeaders.type}</th>
              <th scope="col" className={styles.numeric}>
                {t.explore.moveHeaders.gain}
              </th>
            </tr>
          </thead>
          <tbody>
            {moves.map(({ move, gain }) => {
              const edgeId = metagraph.edgeBetween(move.from, move.to)?.id ?? null;
              const kind = metagraph.edgeBetween(move.from, move.to)?.tradeoff.kind ?? 'clear';
              const verdict = gain > 1e-9 ? 'improving' : gain < -1e-9 ? 'worse' : 'tie';
              const f = move.deltaFriends + move.deltaStrangers === 0 ? null : move.deltaFriends / (move.deltaFriends + move.deltaStrangers);
              return (
                <tr
                  key={`${move.vertex}-${move.to}`}
                  className={styles[verdict]}
                  onMouseEnter={() => onHighlightEdge(edgeId)}
                  onMouseLeave={() => onHighlightEdge(null)}
                >
                  <th scope="row">
                    <button
                      type="button"
                      className={styles.linkButton}
                      onClick={() => onSelectNode(move.to)}
                      onFocus={() => onHighlightEdge(edgeId)}
                      onBlur={() => onHighlightEdge(null)}
                      aria-label={`${t.explore.moveTo(move.vertex, targetText(move, t))}. ${t.explore.goTo(metagraph.node(move.to).label)}`}
                    >
                      {t.explore.moveTo(move.vertex, targetText(move, t))}
                    </button>
                  </th>
                  <td className={`${styles.numeric} num`}>{formatDelta(move.deltaFriends)}</td>
                  <td className={`${styles.numeric} num`}>{formatDelta(move.deltaStrangers)}</td>
                  <td className={`${styles.numeric} num`}>{f === null ? '—' : formatFraction(f)}</td>
                  <td>{t.explore.types[kind]}</td>
                  <td className={`${styles.numeric} num`}>
                    {formatSigned(gain)} <span className={styles.verdict}>{t.explore.verdicts[verdict]}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {edgePanel}
      <p className="visually-hidden">{formatBlocks(node.blocks)}</p>
    </div>
  );
}
