import { useState } from 'react';
import { useMediaQuery, useSettings } from '../app/settings';
import { ResolutionControl } from '../components/ResolutionControl';
import { useWalk, type WalkController, type WalkSpeed } from '../hooks/useWalk';
import { useT } from '../i18n';
import type { WalkPolicy } from '../model';
import { MetagraphLegend } from '../viz/metagraph/Legend';
import { MetagraphView } from '../viz/metagraph/MetagraphView';
import { PotentialChart } from '../walk/PotentialChart';
import { SinkCertificate } from '../walk/SinkCertificate';
import { WalkControls, type WalkOptions } from '../walk/WalkControls';
import { WalkStatus } from '../walk/WalkStatus';
import { LAYOUT, METAGRAPH, THRESHOLDS, useMetagraphLabels } from './metagraphData';
import styles from './Metagraph.module.css';
import { Scrolly } from './Scrolly';

interface WalkChapterProps {
  readonly gamma: number;
  readonly setGamma: (gamma: number) => void;
}

export function useWalkSetup(gamma: number) {
  const [policy, setPolicy] = useState<WalkPolicy>('best');
  const [seed, setSeed] = useState(7);
  const [speed, setSpeed] = useState<WalkSpeed>('normal');
  const [start, setStart] = useState(METAGRAPH.grandCoalitionId);
  const controller = useWalk({ metagraph: METAGRAPH, gamma, policy, seed, start, speed });
  const options: WalkOptions = { policy, setPolicy, seed, setSeed, speed, setSpeed, start, setStart };
  return { controller, options };
}

export function walkOverlay(controller: WalkController) {
  return {
    path: [controller.state.start, ...controller.state.steps.map((step) => step.to)],
    preview: controller.preview,
    done: controller.state.status !== 'running',
    stepKey: controller.state.steps.length,
    durationMs: controller.intervalMs,
  };
}

export function WalkChapter({ gamma, setGamma }: WalkChapterProps) {
  const t = useT();
  const { reducedMotion } = useSettings();
  const compact = useMediaQuery('(max-width: 999px)');
  const labels = useMetagraphLabels('walk');
  const { controller, options } = useWalkSetup(gamma);
  const sink = controller.state.status === 'sink';

  const steps = t.walk.steps.map((step, index) => {
    let extra = null;
    if (index === 1) extra = <PotentialChart metagraph={METAGRAPH} state={controller.state} gamma={gamma} />;
    if (index === 2 && sink) extra = <SinkCertificate metagraph={METAGRAPH} id={controller.state.current} gamma={gamma} steps={controller.state.steps.length} />;
    if (index === 3) {
      extra = (
        <div className={styles.cardControls}>
          <ResolutionControl gamma={gamma} onChange={setGamma} thresholds={THRESHOLDS} compact showPresets />
          <WalkControls metagraph={METAGRAPH} controller={controller} options={options} parts="options" />
        </div>
      );
    }
    return { title: step.title, body: step.body, extra };
  });

  return (
    <Scrolly
      id="walk"
      kicker={t.walk.kicker}
      title={t.walk.title}
      steps={steps}
      stageLabel={`${t.a11y.stage}: ${t.chapters.walk}`}
      next={{ href: '#explore', label: t.chapters.explore }}
      stage={() => (
        <div className={styles.stage}>
          <div className={styles.graph}>
            <MetagraphView
              metagraph={METAGRAPH}
              layout={LAYOUT}
              gamma={gamma}
              mode="oriented"
              labels={labels}
              reducedMotion={reducedMotion}
              walk={walkOverlay(controller)}
              orientation={compact ? 'auto' : 'horizontal'}
              selectedId={options.start}
              onSelectNode={(id) => {
                options.setStart(id);
                controller.reset();
              }}
              onClearSelection={() => {
                if (options.start === METAGRAPH.grandCoalitionId) return;
                options.setStart(METAGRAPH.grandCoalitionId);
                controller.reset();
              }}
            />
          </div>
          {!compact && <MetagraphLegend mode="walk" />}
          <div className={styles.panel}>
            <WalkStatus metagraph={METAGRAPH} controller={controller} gamma={gamma} />
            <WalkControls metagraph={METAGRAPH} controller={controller} options={options} parts="transport" />
            <p className={styles.counts}>{t.walk.disclaimer}</p>
          </div>
        </div>
      )}
    />
  );
}
