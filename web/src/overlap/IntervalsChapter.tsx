import { Rich } from '../components/Rich';
import { useT } from '../i18n';
import { formatNumber } from '../model';
import { nashWindow, stableIntervals } from '../model/overlap';
import { CoverGraph } from './CoverGraph';
import { IntervalBars } from './IntervalBars';
import { CapControl, GammaControl, IsolationControl, PresetControl, type OverlapSettings } from './OverlapControls';
import { GAME, intervalsText, widestRow } from './overlapData';
import type { CoverState } from './SeesChapter';
import { UtilityChart } from './UtilityChart';
import styles from './Overlap.module.css';

export function IntervalsChapter({ settings, state }: { settings: OverlapSettings; state: CoverState }) {
  const t = useT();
  const s = t.overlap.intervals;
  const options = { cap: settings.cap, allowIsolation: settings.isolation };
  const intervals = stableIntervals(GAME, state.cover, options);
  const nash = nashWindow(intervals);
  const regret = Math.max(0, state.response.best.value - state.response.current);

  return (
    <section id="intervals" className={styles.section} aria-labelledby="intervals-title">
      <div className="container">
        <header className={styles.header}>
          <p className="kicker">{s.kicker}</p>
          <h2 id="intervals-title">{s.title}</h2>
          <Rich as="p" className="lede" text={s.lede} />
        </header>
        <div className={styles.twoColumn}>
          <div className={styles.card}>
            <div className={styles.smallGraph}>
              <CoverGraph cover={state.cover} focus={state.focus} onVertex={state.setFocus} />
            </div>
            <PresetControl value={state.preset} onChange={state.setPreset} />
            <GammaControl settings={settings} />
            <div className={styles.inline}>
              <CapControl settings={settings} minimum={widestRow(state.cover)} />
              <IsolationControl settings={settings} />
            </div>
          </div>
          <div className={styles.card}>
            <UtilityChart
              cover={state.cover}
              vertex={state.focus}
              gamma={settings.gamma}
              cap={settings.cap}
              isolation={settings.isolation}
              onGamma={settings.setGamma}
            />
            <p className={styles.readout} aria-live="polite">
              {s.intervalsOf(state.focus, intervalsText(intervals[state.focus]) ?? s.never)} {s.regret(formatNumber(regret, 3))}
            </p>
            <IntervalBars intervals={intervals} nash={nash} gamma={settings.gamma} focus={state.focus} onFocus={state.setFocus} />
            <p className={styles.note}>{s.note}</p>
          </div>
        </div>
      </div>
    </section>
  );
}
