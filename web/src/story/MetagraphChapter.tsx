import { useState } from 'react';
import { useMediaQuery, useSettings } from '../app/settings';
import { ResolutionControl } from '../components/ResolutionControl';
import { Switch } from '../components/Segmented';
import { useT } from '../i18n';
import { findSinks } from '../model';
import { MetagraphLegend } from '../viz/metagraph/Legend';
import { MetagraphView, type MetagraphMode } from '../viz/metagraph/MetagraphView';
import { AGENT3_PAIR, LAYOUT, METAGRAPH, THRESHOLDS, useMetagraphLabels } from './metagraphData';
import styles from './Metagraph.module.css';
import { Scrolly } from './Scrolly';

const MODES: readonly MetagraphMode[] = ['pair', 'nodes', 'edges', 'types', 'oriented', 'oriented'];

interface MetagraphChapterProps {
  readonly gamma: number;
  readonly setGamma: (gamma: number) => void;
}

const KIND_COUNTS = {
  clear: METAGRAPH.edges.filter((e) => e.tradeoff.kind === 'clear').length,
  frustrated: METAGRAPH.edges.filter((e) => e.tradeoff.kind === 'frustrated').length,
  ties: METAGRAPH.edges.filter((e) => e.tradeoff.kind === 'indifferent').length,
};

function MetagraphStage({ active, gamma, setGamma }: MetagraphChapterProps & { active: number }) {
  const t = useT();
  const { reducedMotion } = useSettings();
  const compact = useMediaQuery('(max-width: 999px)');
  const mode = MODES[Math.min(active, MODES.length - 1)];
  const labels = useMetagraphLabels(mode);
  const [showArcs, setShowArcs] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const sinks = mode === 'oriented' ? findSinks(METAGRAPH, gamma).map((id) => METAGRAPH.node(id).label) : [];
  const shownEdges = showArcs ? METAGRAPH.edges.length : METAGRAPH.edges.filter((e) => !e.sameLayer).length;

  return (
    <div className={styles.stage}>
      <div className={styles.head}>
        <span className="chip chip-threshold">{t.metagraph.modes[mode]}</span>
        <span className={styles.counts}>
          {mode === 'pair'
            ? t.metagraph.counts(2, 1)
            : mode === 'nodes'
              ? t.metagraph.counts(METAGRAPH.nodes.length, 0)
              : t.metagraph.counts(METAGRAPH.nodes.length, shownEdges)}
          {(mode === 'types' || mode === 'oriented') && ` · ${t.metagraph.kindCounts(KIND_COUNTS.clear, KIND_COUNTS.frustrated, KIND_COUNTS.ties)}`}
        </span>
      </div>
      <div className={styles.graph}>
        <MetagraphView
          metagraph={METAGRAPH}
          layout={LAYOUT}
          gamma={gamma}
          mode={mode}
          labels={labels}
          reducedMotion={reducedMotion}
          pair={AGENT3_PAIR}
          orientation={compact ? 'auto' : 'horizontal'}
          showSameLayer={showArcs}
          selectedId={selected}
          onSelectNode={(id) => setSelected((current) => (current === id ? null : id))}
        />
      </div>
      {mode !== 'pair' && <MetagraphLegend mode={mode} />}
      {mode === 'types' && (
        <div className={styles.note}>
          <Switch label={t.metagraph.showArcs} checked={showArcs} onChange={setShowArcs} />
          {!compact && <p>{t.metagraph.figureNote}</p>}
        </div>
      )}
      {mode === 'oriented' && (
        <div className={styles.controls}>
          <ResolutionControl gamma={gamma} onChange={setGamma} thresholds={THRESHOLDS} compact showPresets={active >= 5 || !compact} />
          <p className={styles.sinks} aria-live="polite">
            {t.metagraph.sinks(sinks.join(' · '))}
          </p>
        </div>
      )}
    </div>
  );
}

export function MetagraphChapter(props: MetagraphChapterProps) {
  const t = useT();
  const steps = t.metagraph.steps.map((step) => ({ title: step.title, body: step.body }));
  return (
    <Scrolly
      id="metagraph"
      kicker={t.metagraph.kicker}
      title={t.metagraph.title}
      steps={steps}
      stageLabel={`${t.a11y.stage}: ${t.chapters.metagraph}`}
      next={{ href: '#walk', label: t.chapters.walk }}
      stage={(active) => <MetagraphStage {...props} active={active} />}
    />
  );
}
