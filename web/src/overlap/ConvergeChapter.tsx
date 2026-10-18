import { useEffect, useMemo, useState } from 'react';
import { useSettings } from '../app/settings';
import { CodeBlock } from '../components/CodeBlock';
import { Icon } from '../components/Icon';
import { Rich } from '../components/Rich';
import { Segmented } from '../components/Segmented';
import { useT } from '../i18n';
import { formatGamma, formatNumber, formatSigned } from '../model';
import {
  coverPotential,
  nashWindow,
  queuedVertices,
  relabelCover,
  runCover,
  stableIntervals,
  startCoverRun,
  stepCoverRun,
  type CoverRun,
} from '../model/overlap';
import { CoverGraph } from './CoverGraph';
import { CapControl, GammaControl, IsolationControl, type OverlapSettings } from './OverlapControls';
import { GAME, PRESETS, START_ORDER, intervalsText, rowText, type StartName } from './overlapData';
import styles from './Overlap.module.css';

const W = 560;
const H = 130;

function PotentialChart({ run }: { run: CoverRun }) {
  const t = useT();
  const start = coverPotential(GAME.adj, run.start, run.options.gamma);
  const points: [number, number][] = [[0, start], ...run.visits.map((visit) => [visit.index, visit.potentialAfter] as [number, number])];
  const span = Math.max(8, run.visits.length);
  const ys = points.map(([, y]) => y);
  const lo = Math.min(0, ...ys);
  const hi = Math.max(0.5, ...ys);
  const sx = (x: number) => 8 + (x / span) * (W - 16);
  const sy = (y: number) => 8 + ((hi - y) / (hi - lo || 1)) * (H - 26);
  let d = '';
  points.forEach(([x, y], i) => {
    if (i === 0) d = `M${sx(x)},${sy(y)}`;
    else d += ` H${sx(x)} V${sy(y)}`;
  });
  return (
    <figure className={styles.potential}>
      <figcaption className={styles.potentialTitle}>{t.overlap.converge.potential}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t.overlap.converge.potential}>
        <line x1={8} x2={W - 8} y1={sy(0)} y2={sy(0)} className={styles.axisZero} />
        <path d={d} className={styles.potentialLine} />
        {run.visits
          .filter((visit) => visit.moved)
          .map((visit) => (
            <circle key={visit.index} cx={sx(visit.index)} cy={sy(visit.potentialAfter)} r={2.8} className={styles.potentialDot} />
          ))}
        <text x={W - 8} y={H - 4} className={`${styles.potentialText} ${styles.end}`}>
          {t.overlap.converge.potentialAxis} {run.visits.length}
        </text>
        <text x={8} y={H - 4} className={styles.potentialText}>
          Φ = {formatNumber(points[points.length - 1][1])}
        </text>
      </svg>
    </figure>
  );
}

export function ConvergeChapter({ settings }: { settings: OverlapSettings }) {
  const t = useT();
  const c = t.overlap.converge;
  const { reducedMotion } = useSettings();
  const [start, setStart] = useState<StartName>('singletons');
  const [seed, setSeed] = useState(0);
  const [playing, setPlaying] = useState(false);
  const fresh = useMemo(
    () => startCoverRun(GAME, PRESETS[start], { gamma: settings.gamma, cap: settings.cap, allowIsolation: settings.isolation, seed }),
    [start, seed, settings.gamma, settings.cap, settings.isolation],
  );
  const [run, setRun] = useState<CoverRun>(fresh);
  const [source, setSource] = useState(fresh);
  if (source !== fresh) {
    // A setting changed: start again from the new first visiting order.
    setSource(fresh);
    setRun(fresh);
    setPlaying(false);
  }
  const done = run.status === 'equilibrium';

  const running = playing && !done;

  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setTimeout(() => setRun((current) => stepCoverRun(current)), reducedMotion ? 900 : 650);
    return () => window.clearTimeout(timer);
  }, [running, run, reducedMotion]);

  const last = run.visits.length ? run.visits[run.visits.length - 1] : null;
  const queue = queuedVertices(run);
  const result = done ? relabelCover(run.rows) : null;
  const nash = done ? nashWindow(stableIntervals(GAME, run.rows, { cap: settings.cap, allowIsolation: settings.isolation })) : [];
  const settingsText = c.settings(formatGamma(settings.gamma), settings.cap, settings.isolation);
  const code = result
    ? `graph.community_hedonic(resolution=${formatNumber(settings.gamma, 2)}, max_memberships=${settings.cap},\n                        allow_isolation=${settings.isolation ? 'True' : 'False'}, seed=${seed})\ngraph.memberships  # ${JSON.stringify(result).replaceAll(',', ', ')}`
    : '';

  return (
    <section id="converge" className={styles.section} aria-labelledby="converge-title">
      <div className="container">
        <header className={styles.header}>
          <p className="kicker">{c.kicker}</p>
          <h2 id="converge-title">{c.title}</h2>
          <Rich as="p" className="lede" text={c.lede} />
        </header>
        <div className={styles.twoColumn}>
          <div className={styles.card} role="region" aria-label={c.stage}>
            <div className={styles.runGraph}>
              <CoverGraph cover={run.rows} highlight={last?.vertex ?? null} queued={queue} />
            </div>
            <p className={styles.status}>
              <strong>{c.status(run.visits.length, run.moves)}</strong>
              <span className={styles.queueLine}>
                {c.queue}:{' '}
                {queue.length ? (
                  queue.map((v, i) => (
                    <span key={v} className={styles.queueItem} data-head={i === 0}>
                      {v}
                    </span>
                  ))
                ) : (
                  <span className="muted">{c.queueEmpty}</span>
                )}
              </span>
            </p>
            <div className={styles.transport}>
              <button type="button" className="btn btn-small" onClick={() => setRun(stepCoverRun(run))} disabled={done}>
                <Icon name="step" size={16} />
                {c.step}
              </button>
              <button type="button" className="btn btn-small btn-primary" onClick={() => setPlaying(!running)} disabled={done}>
                <Icon name={running ? 'pause' : 'play'} size={16} />
                {running ? c.pause : c.play}
              </button>
              <button
                type="button"
                className="btn btn-small"
                onClick={() => {
                  setPlaying(false);
                  setRun(runCover(run));
                }}
                disabled={done}
              >
                {c.finish}
              </button>
              <button
                type="button"
                className="btn btn-small"
                onClick={() => {
                  setPlaying(false);
                  setRun(fresh);
                }}
              >
                <Icon name="reset" size={16} />
                {c.reset}
              </button>
            </div>
            <div className={styles.log} aria-live="polite">
              {!last && !done && <p>{c.waiting}</p>}
              {last && (
                <p>
                  {last.rescan && <span className={styles.rescan}>{c.rescan} </span>}
                  {last.moved
                    ? c.moved(last.vertex, rowText(last.before), rowText(last.after), formatSigned(last.response.improvement, 3))
                    : c.stayed(last.vertex, rowText(last.before))}
                  {last.freshLabel !== null && ` ${c.fresh(rowText([last.freshLabel]))}`}
                </p>
              )}
              {done && (
                <p className={styles.equilibrium}>
                  <Icon name="check" size={16} /> {c.equilibrium(run.visits.length, run.moves)}{' '}
                  {nash.length ? c.window(intervalsText(nash) ?? '') : t.overlap.intervals.windowEmpty}
                </p>
              )}
            </div>
          </div>
          <div className={styles.card}>
            <div className={styles.inline}>
              <Segmented<StartName>
                legend={c.start}
                value={start}
                options={START_ORDER.map((name) => ({ value: name, label: c.startNames[name] }))}
                onChange={setStart}
              />
              <fieldset className={styles.seed}>
                <legend>{c.seed}</legend>
                <button type="button" className="btn btn-small btn-icon" onClick={() => setSeed(Math.max(0, seed - 1))} aria-label={c.seedDown} disabled={seed === 0}>
                  <Icon name="minus" size={16} />
                </button>
                <output className="num">{seed}</output>
                <button type="button" className="btn btn-small btn-icon" onClick={() => setSeed(seed + 1)} aria-label={c.seedUp}>
                  <Icon name="plus" size={16} />
                </button>
              </fieldset>
            </div>
            <GammaControl settings={settings} />
            <div className={styles.inline}>
              <CapControl settings={settings} minimum={1} />
              <IsolationControl settings={settings} />
            </div>
            <PotentialChart run={run} />
            {result && (
              <div className={styles.result}>
                <CodeBlock code={code} language="python" label={`${c.result} (${settingsText})`} />
              </div>
            )}
            {run.visits.length > 0 && (
              <details className={styles.visits}>
                <summary>
                  {c.log} ({run.visits.length})
                </summary>
                <ol>
                  {run.visits.map((visit) => (
                    <li key={visit.index} data-moved={visit.moved}>
                      <span className="num">#{visit.index}</span> {visit.vertex}:{' '}
                      {visit.moved ? (
                        <>
                          {rowText(visit.before)} → {rowText(visit.after)}{' '}
                          <span className="num">{formatSigned(visit.response.improvement, 3)}</span>
                        </>
                      ) : (
                        <span className="muted">{rowText(visit.before)}</span>
                      )}
                    </li>
                  ))}
                </ol>
              </details>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
