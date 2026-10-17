import { Icon } from '../components/Icon';
import { Segmented } from '../components/Segmented';
import type { WalkController, WalkSpeed } from '../hooks/useWalk';
import { useT } from '../i18n';
import { WALK_POLICIES, type Metagraph, type WalkPolicy } from '../model';
import styles from './Walk.module.css';

export interface WalkOptions {
  readonly policy: WalkPolicy;
  readonly setPolicy: (policy: WalkPolicy) => void;
  readonly seed: number;
  readonly setSeed: (seed: number) => void;
  readonly speed: WalkSpeed;
  readonly setSpeed: (speed: WalkSpeed) => void;
  readonly start: string;
  readonly setStart: (id: string) => void;
}

interface WalkControlsProps {
  readonly metagraph: Metagraph;
  readonly controller: WalkController;
  readonly options: WalkOptions;
  readonly selectedId?: string | null;
  /** Which controls to show: the transport buttons, the options, or both. */
  readonly parts?: 'all' | 'transport' | 'options';
}

export function WalkControls({ metagraph, controller, options, selectedId = null, parts = 'all' }: WalkControlsProps) {
  const t = useT();
  const done = controller.state.status !== 'running';
  const startChoices = [
    { value: metagraph.grandCoalitionId, label: t.walk.startGrand },
    { value: metagraph.singletonsId, label: t.walk.startSingletons },
  ];
  const customStart = [metagraph.grandCoalitionId, metagraph.singletonsId].includes(options.start) ? null : options.start;
  const selectable = selectedId && ![metagraph.grandCoalitionId, metagraph.singletonsId].includes(selectedId) ? selectedId : null;
  const extra = customStart ?? selectable;
  if (extra) startChoices.push({ value: extra, label: metagraph.node(extra).label });

  const transport = (
    <div className={styles.transport}>
      <button type="button" className="btn" onClick={controller.reset} disabled={controller.state.steps.length === 0 && !controller.playing}>
        <Icon name="reset" />
        {t.walk.reset}
      </button>
      <button type="button" className="btn" onClick={controller.step} disabled={done || controller.playing}>
        <Icon name="step" />
        {t.walk.stepButton}
      </button>
      {controller.playing ? (
        <button type="button" className="btn btn-primary" onClick={controller.pause}>
          <Icon name="pause" />
          {t.walk.pause}
        </button>
      ) : (
        <button type="button" className="btn btn-primary" onClick={controller.play} disabled={done}>
          <Icon name="play" />
          {t.walk.play}
        </button>
      )}
    </div>
  );

  const rule = (
    <div className={styles.option}>
      <Segmented
        legend={t.walk.rule}
        fill
        value={options.policy}
        onChange={options.setPolicy}
        options={WALK_POLICIES.map((policy) => ({ value: policy, label: t.walk.policies[policy], title: t.walk.policyHelp[policy] }))}
      />
      <p className={styles.help}>{t.walk.policyHelp[options.policy]}</p>
      {options.policy === 'random' && (
        <div className={styles.seed}>
          <span>
            {t.walk.seed}: <strong className="num">{options.seed}</strong>
          </span>
          <button type="button" className="btn btn-small" onClick={() => options.setSeed((options.seed * 48271 + 11) % 9973 || 1)}>
            <Icon name="shuffle" size={16} />
            {t.walk.newSeed}
          </button>
        </div>
      )}
    </div>
  );

  const speed = (
    <Segmented
      legend={t.walk.speed}
      value={options.speed}
      onChange={options.setSpeed}
      options={(['slow', 'normal', 'fast'] as const).map((value) => ({ value, label: t.walk.speeds[value] }))}
    />
  );

  const start = (
    <div className={styles.option}>
      <Segmented legend={t.walk.start} value={options.start} onChange={options.setStart} options={startChoices} />
      <p className={styles.help}>{t.walk.startHint}</p>
    </div>
  );

  if (parts === 'transport') return transport;
  return (
    <div className={styles.controls}>
      {parts === 'all' && transport}
      {start}
      {rule}
      {speed}
    </div>
  );
}
