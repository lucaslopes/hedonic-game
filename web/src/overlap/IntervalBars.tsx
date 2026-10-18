import { useT } from '../i18n';
import { intervalsText } from './overlapData';
import styles from './Overlap.module.css';

type Intervals = readonly (readonly [number, number])[];

function Track({ intervals, gamma, kind }: { intervals: Intervals; gamma: number; kind: 'vertex' | 'window' }) {
  return (
    <span className={styles.track} data-kind={kind} aria-hidden="true">
      {intervals.map(([lo, hi]) => (
        <span key={`${lo}-${hi}`} className={styles.segment} style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
      ))}
      <span className={styles.trackMarker} style={{ left: `${gamma * 100}%` }} />
    </span>
  );
}

interface IntervalBarsProps {
  readonly intervals: readonly Intervals[];
  readonly nash: Intervals;
  readonly gamma: number;
  readonly focus: number;
  readonly onFocus: (v: number) => void;
}

/** One stable-interval bar per vertex, then their intersection, the Nash window. */
export function IntervalBars({ intervals, nash, gamma, focus, onFocus }: IntervalBarsProps) {
  const t = useT();
  const s = t.overlap.intervals;
  return (
    <div className={styles.bars}>
      <p className={styles.barsTitle}>{s.all}</p>
      <ul className={styles.barList}>
        {intervals.map((list, v) => (
          <li key={v} className={styles.barRow} data-focus={v === focus}>
            <button type="button" className={styles.barVertex} onClick={() => onFocus(v)} aria-pressed={v === focus}>
              {v}
            </button>
            <Track intervals={list} gamma={gamma} kind="vertex" />
            <span className="visually-hidden">{s.intervalsOf(v, intervalsText(list) ?? s.never)}</span>
          </li>
        ))}
        <li className={`${styles.barRow} ${styles.windowRow}`}>
          <span className={styles.barVertex}>{s.window}</span>
          <Track intervals={nash} gamma={gamma} kind="window" />
        </li>
      </ul>
      <p className={styles.windowText} aria-live="polite">
        {nash.length ? s.windowText(intervalsText(nash) ?? '') : s.windowEmpty}
      </p>
    </div>
  );
}
