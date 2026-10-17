import { Icon } from '../components/Icon';
import { useT } from '../i18n';
import { formatNumber, formatSigned } from '../model';
import { FRESH, auditVertex, intensity, labelName, type BestResponse, type Cover, type LabelScore } from '../model/overlap';
import { Scrolly } from '../story/Scrolly';
import { CoverGraph } from './CoverGraph';
import { CapControl, GammaControl, IsolationControl, PresetControl, type OverlapSettings } from './OverlapControls';
import { GAME, labelColor, rowText, widestRow, type PresetName } from './overlapData';
import styles from './Overlap.module.css';

export interface CoverState {
  readonly cover: Cover;
  readonly preset: PresetName | 'edited';
  /** The preset the edits started from. */
  readonly base: PresetName;
  readonly setPreset: (preset: PresetName) => void;
  readonly focus: number;
  readonly setFocus: (v: number) => void;
  readonly response: BestResponse;
  readonly play: () => void;
}

interface SeesChapterProps {
  readonly settings: OverlapSettings;
  readonly state: CoverState;
}

/** A labelled colour swatch: the letter carries the meaning, the colour repeats it. */
export function LabelChip({ label }: { label: number }) {
  return (
    <span className={styles.labelChip} style={{ ['--label' as string]: labelColor(label) }}>
      {labelName(label)}
    </span>
  );
}

function holders(list: LabelScore['friends'], empty: string) {
  if (list.length === 0) return <span className="muted">{empty}</span>;
  return list.map((holder, index) => (
    <span key={holder.vertex} className={styles.holder}>
      {index > 0 && ', '}
      {holder.vertex} <span className="muted">({holder.f === 1 ? '1' : formatNumber(holder.f)})</span>
    </span>
  ));
}

export function ResponseTable({ response, gamma }: { response: BestResponse; gamma: number }) {
  const t = useT();
  const s = t.overlap.sees.table;
  const inBest = new Set(response.best.labels);
  const rows: (LabelScore | null)[] = [...response.ranked];
  if (response.best.kind === 'fresh') rows.unshift(null);
  return (
    <div className={`table-scroll ${styles.tableWrap}`}>
      <table className={styles.table}>
        <caption className="visually-hidden">{s.caption(response.vertex, formatNumber(gamma))}</caption>
        <thead>
          <tr>
            <th scope="col">{s.label}</th>
            <th scope="col" className="friend-text">
              {s.friends}
            </th>
            <th scope="col" className="stranger-text">
              {s.strangers}
            </th>
            <th scope="col" className="num">
              {s.support}
            </th>
            <th scope="col" className="num">
              {s.crowd}
            </th>
            <th scope="col" className="num">
              {s.gain}
            </th>
            <th scope="col">
              <span className="visually-hidden">{s.inBest}</span>
              <span aria-hidden="true">{s.best}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((score) =>
            score === null ? (
              <tr key="fresh" data-best="true">
                <th scope="row">
                  <span className={styles.freshChip}>∗</span> <span className={styles.tag}>{s.fresh}</span>
                </th>
                <td>{s.nobody}</td>
                <td>{s.nobody}</td>
                <td className="num">{formatNumber(0)}</td>
                <td className="num">{formatNumber(0)}</td>
                <td className="num">{formatNumber(0)}</td>
                <td>
                  <Icon name="check" size={16} />
                  <span className="visually-hidden">{s.inBest}</span>
                </td>
              </tr>
            ) : (
              <tr key={score.label} data-best={inBest.has(score.label)} data-own={score.own}>
                <th scope="row">
                  <LabelChip label={score.label} />
                  {score.own && <span className={styles.tag}>{score.alone ? s.alone : s.held}</span>}
                </th>
                <td>{holders(score.friends, s.nobody)}</td>
                <td>{holders(score.strangers, s.nobody)}</td>
                <td className="num">{formatNumber(score.support)}</td>
                <td className="num">{formatNumber(score.crowd)}</td>
                <td className={`num ${styles.gain}`} data-sign={score.gain > 1e-12 ? 'pos' : score.gain < -1e-12 ? 'neg' : 'zero'}>
                  {formatSigned(score.gain)}
                </td>
                <td>
                  {inBest.has(score.label) && (
                    <>
                      <Icon name="check" size={16} />
                      <span className="visually-hidden">{s.inBest}</span>
                    </>
                  )}
                </td>
              </tr>
            ),
          )}
        </tbody>
      </table>
    </div>
  );
}

export function ResponseSummary({ response }: { response: BestResponse }) {
  const t = useT();
  const k = response.row.length;
  const best = response.best.labels.map((label) => (label === FRESH ? '∗' : labelName(label)));
  const bestText = `{${best.join(', ')}}`;
  return (
    <div className={styles.summary}>
      <p className={styles.holds}>{t.overlap.sees.holds(response.vertex, rowText(response.row), k, formatNumber(intensity(k)), formatNumber(response.current))}</p>
      <div className={styles.prefixes}>
        <span className={styles.prefixLabel}>{t.overlap.sees.prefixes}</span>
        {response.positive === 0 ? (
          <span className="muted">{t.overlap.sees.noPositive}</span>
        ) : (
          response.prefixValues.slice(0, response.positive).map((value, index) => {
            const labels = response.ranked.slice(0, index + 1).map((c) => labelName(c.label));
            const chosen = response.best.kind === 'prefix' && response.best.labels.length === index + 1;
            return (
              <span key={index} className={styles.prefix} data-best={chosen}>
                {`{${labels.join(', ')}}`} <span className="num">{formatNumber(value)}</span>
              </span>
            );
          })
        )}
      </div>
      <p className={styles.verdict} data-moves={response.moves} aria-live="polite">
        {response.moves ? t.overlap.sees.move(bestText, formatSigned(response.improvement)) : t.overlap.sees.stay(rowText(response.row))}
      </p>
    </div>
  );
}

export function SeesChapter({ settings, state }: SeesChapterProps) {
  const t = useT();
  const minimum = widestRow(state.cover);
  // Snap points: where the selected vertex's set stops (or starts) being a best response.
  const cuts = auditVertex(GAME, state.cover, state.focus, { cap: settings.cap, allowIsolation: settings.isolation })
    .intervals.flat()
    .filter((cut) => cut > 1e-9 && cut < 1 - 1e-9);
  const extras = [
    <div key="preset" className={styles.cardControls}>
      <PresetControl value={state.preset} onChange={state.setPreset} />
    </div>,
    <div key="gamma" className={styles.cardControls}>
      <GammaControl settings={settings} thresholds={cuts} />
    </div>,
    <div key="cap" className={styles.cardControls}>
      <CapControl settings={settings} minimum={minimum} />
    </div>,
    <div key="iso" className={styles.cardControls}>
      <IsolationControl settings={settings} />
    </div>,
  ];
  const steps = t.overlap.sees.steps.map((step, index) => ({ title: step.title, body: step.body, extra: extras[index] }));

  return (
    <Scrolly
      id="sees"
      kicker={t.overlap.sees.kicker}
      title={t.overlap.sees.title}
      steps={steps}
      stageLabel={`${t.a11y.stage}: ${t.overlap.sees.stage}`}
      next={{ href: '#intervals', label: t.overlap.chapters.intervals }}
      stage={() => (
        <div className={styles.seesStage}>
          <div className={styles.graphBox}>
            <CoverGraph cover={state.cover} focus={state.focus} onVertex={state.setFocus} />
          </div>
          <div className={styles.seesPanel}>
            <ResponseSummary response={state.response} />
            <ResponseTable response={state.response} gamma={settings.gamma} />
            <div className={styles.actions}>
              <button type="button" className="btn btn-small btn-primary" onClick={state.play} disabled={!state.response.moves}>
                <Icon name="step" size={16} />
                {t.overlap.sees.apply}
              </button>
              {state.preset === 'edited' && (
                <button type="button" className="btn btn-small" onClick={() => state.setPreset(state.base)}>
                  <Icon name="reset" size={16} />
                  {t.overlap.sees.undo}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    />
  );
}
