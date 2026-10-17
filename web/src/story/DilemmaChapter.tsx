import { useState } from 'react';
import { useMediaQuery, useSettings } from '../app/settings';
import { Icon } from '../components/Icon';
import { ResolutionControl } from '../components/ResolutionControl';
import { useT } from '../i18n';
import { communityUtility, compareCommunities, formatDelta, formatFraction, formatNumber } from '../model';
import { BalanceScale } from '../viz/BalanceScale';
import { CommunityCard, DEFAULT_COUNTS, ExampleChips, sameCounts, verdictText, winnerAt, type Counts } from './dilemmaShared';
import styles from './Dilemma.module.css';
import { Scrolly } from './Scrolly';

interface DilemmaChapterProps {
  readonly counts: Counts;
  readonly setCounts: (counts: Counts) => void;
  readonly gamma: number;
  readonly setGamma: (gamma: number) => void;
}

export function DilemmaStage({ level, counts, setCounts, gamma, setGamma }: DilemmaChapterProps & { level: 0 | 1 | 2 | 3 }) {
  const t = useT();
  const { reducedMotion } = useSettings();
  const compact = useMediaQuery('(max-width: 999px)');
  const tradeoff = compareCommunities(counts.a, counts.b);
  const winner = level >= 2 ? winnerAt(counts, gamma) : null;
  const verdict = level >= 2 ? verdictText(counts, gamma, t, level >= 3) : level === 1 ? t.dilemma.locked : '';

  let familiarityLine: string | null = null;
  if (level >= 3) {
    const deltas = t.dilemma.deltas(formatDelta(tradeoff.deltaFriends), formatDelta(tradeoff.deltaStrangers));
    if (tradeoff.kind === 'indifferent') familiarityLine = `${deltas} ${t.dilemma.fIndifferent}`;
    else if (tradeoff.familiarity === null) familiarityLine = `${deltas} ${t.dilemma.fUndefined}`;
    else if (tradeoff.kind === 'clear') familiarityLine = `${deltas} ${t.dilemma.fOutside(formatFraction(tradeoff.familiarity))}`;
    else familiarityLine = `${deltas} ${t.dilemma.familiarity}: F = ${formatFraction(tradeoff.familiarity)}.`;
  }

  const labels = {
    a: 'A',
    b: 'B',
    moreFriends: t.dilemma.moreFriends,
    fewerStrangers: t.dilemma.fewerStrangers,
    sameFriends: t.dilemma.sameFriends,
    sameStrangers: t.dilemma.sameStrangers,
  };

  const [announce, setAnnounce] = useState('');
  const pristine = sameCounts(counts, DEFAULT_COUNTS) && gamma === 0.5;
  const reset = () => {
    // Like the original prototype's reset: first example again and γ back to 1/2.
    setCounts(DEFAULT_COUNTS);
    setGamma(0.5);
    setAnnounce(t.dilemma.resetDone);
  };

  return (
    <div className={styles.stage}>
      <p className="visually-hidden" aria-live="polite">
        {pristine ? announce : ''}
      </p>
      <div className={styles.scaleWrap}>
        <BalanceScale a={counts.a} b={counts.b} gamma={gamma} level={level} reducedMotion={reducedMotion} labels={labels} />
        <p className="visually-hidden">
          {t.dilemma.scaleDescription(
            formatNumber(communityUtility(counts.a.friends, counts.a.strangers, gamma)),
            formatNumber(communityUtility(counts.b.friends, counts.b.strangers, gamma)),
            verdict,
          )}
        </p>
      </div>
      {level >= 1 && (
        <div className={styles.communities}>
          <CommunityCard side="A" counts={counts.a} onChange={(a) => setCounts({ ...counts, a })} gamma={gamma} showUtility={level >= 2} winner={winner === 'A'} compact={compact} />
          <CommunityCard side="B" counts={counts.b} onChange={(b) => setCounts({ ...counts, b })} gamma={gamma} showUtility={level >= 2} winner={winner === 'B'} compact={compact} />
        </div>
      )}
      {level >= 2 && (
        <div className={styles.dial}>
          <ResolutionControl
            gamma={gamma}
            onChange={setGamma}
            tradeoff={level >= 3 ? { familiarity: tradeoff.familiarity, kind: tradeoff.kind } : undefined}
            compact={compact}
            showPresets={!compact || level >= 3}
            showExtremes={!compact}
          />
        </div>
      )}
      {verdict && (
        <div className={`${styles.verdict} ${winner ? styles.verdictChoice : ''}`}>
          <div aria-live="polite">
            <p>{verdict}</p>
            {familiarityLine && <p className={styles.fLine}>{familiarityLine}</p>}
          </div>
          <button
            type="button"
            className={`btn btn-small btn-icon ${styles.reset}`}
            onClick={reset}
            disabled={pristine}
            aria-label={t.dilemma.reset}
            title={t.dilemma.reset}
          >
            <Icon name="reset" size={16} />
          </button>
        </div>
      )}
    </div>
  );
}

export function DilemmaChapter(props: DilemmaChapterProps) {
  const t = useT();
  const steps = t.dilemma.steps.map((step, index) => ({
    title: step.title,
    body: step.body,
    extra:
      index >= 1 ? (
        <ExampleChips ids={['frustrated', 'vertex3', 'clear', 'ideal', 'identical']} counts={props.counts} onPick={props.setCounts} />
      ) : undefined,
  }));
  return (
    <Scrolly
      id="dilemma"
      kicker={t.dilemma.kicker}
      title={t.dilemma.title}
      steps={steps}
      stageLabel={`${t.a11y.stage}: ${t.chapters.dilemma}`}
      next={{ href: '#tree', label: t.chapters.tree }}
      stage={(active) => <DilemmaStage {...props} level={Math.min(3, active) as 0 | 1 | 2 | 3} />}
    />
  );
}
