import type { WalkController } from '../hooks/useWalk';
import { useT } from '../i18n';
import { formatDelta, formatNumber, formatSigned, nodePotential, type Metagraph } from '../model';
import { GraphGlyph } from '../viz/GraphGlyph';
import { describeMove } from './describe';
import styles from './Walk.module.css';

interface WalkStatusProps {
  readonly metagraph: Metagraph;
  readonly controller: WalkController;
  readonly gamma: number;
}

export function WalkStatus({ metagraph, controller, gamma }: WalkStatusProps) {
  const t = useT();
  const { state } = controller;
  const current = metagraph.node(state.current);
  const last = state.steps[state.steps.length - 1];
  const sink = state.status === 'sink';
  const announcement = sink
    ? t.walk.announceSink(current.label)
    : last
      ? t.walk.announceStep(last.index, current.label, formatSigned(last.gain))
      : '';

  return (
    <div className={styles.status}>
      <div className={styles.statusHead}>
        {/* The current partition drawn on the network; the last mover is ringed. */}
        <svg className={styles.glyph} width={44} height={44} viewBox="0 0 44 44" aria-hidden="true">
          <GraphGlyph cx={22} cy={22} radius={14} membership={current.rgs} accent={last?.move.vertex ?? null} vertexRadius={5.2} />
        </svg>
        <span className={styles.statusLabel}>{t.walk.current}</span>
        <code className={styles.partition}>{current.label}</code>
        <span className="chip">{t.walk.stepCount(state.steps.length)}</span>
        <span className="chip chip-threshold num">
          {t.walk.quality} = {formatNumber(nodePotential(current, gamma))}
        </span>
        {sink && (
          <span className={`chip ${styles.sinkChip}`}>
            <svg viewBox="-8 -8 16 16" width="14" height="14" aria-hidden="true">
              <path d="M-3.4 -3.2 L0 0.6 L3.4 -3.2 M-3.6 3.2 H3.6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
            {t.walk.sinkTitle}
          </span>
        )}
      </div>
      {last ? (
        <p className={styles.lastMove}>
          <span className={styles.statusLabel}>{t.walk.lastMove}</span> {describeMove(last.move, t)}.{' '}
          <span className="num">{t.walk.moveDetail(formatDelta(last.move.deltaFriends), formatDelta(last.move.deltaStrangers), formatSigned(last.gain))}</span>{' '}
          <span className="muted">({t.walk.options(last.options)})</span>
        </p>
      ) : (
        <p className={styles.lastMove}>{sink ? t.walk.sinkSummary(current.label, t.walk.stepCount(0), formatNumber(nodePotential(current, gamma))) : t.walk.ready}</p>
      )}
      {controller.restarted && <p className={styles.notice}>{t.walk.restarted}</p>}
      <p className="visually-hidden" aria-live="polite" aria-atomic="true">
        {announcement}
      </p>
    </div>
  );
}
