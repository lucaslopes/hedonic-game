import { Icon } from '../components/Icon';
import { InfoTip } from '../components/InfoTip';
import { Rich } from '../components/Rich';
import { useT, type Dictionary } from '../i18n';
import {
  communityUtility,
  compareCommunities,
  formatFraction,
  formatGamma,
  formatNumber,
  orientTradeoff,
  type CommunityCounts,
} from '../model';
import styles from './Dilemma.module.css';

export interface Counts {
  readonly a: CommunityCounts;
  readonly b: CommunityCounts;
}

export const MAX_COUNT = 6;

export type ExampleId = 'frustrated' | 'vertex3' | 'clear' | 'ideal' | 'identical';

export const EXAMPLES: Readonly<Record<ExampleId, Counts>> = {
  frustrated: { a: { friends: 2, strangers: 2 }, b: { friends: 1, strangers: 0 } },
  // Agent 3 in the grand coalition: stay with 1 friend and 2 strangers, or leave alone.
  vertex3: { a: { friends: 1, strangers: 2 }, b: { friends: 0, strangers: 0 } },
  clear: { a: { friends: 1, strangers: 1 }, b: { friends: 2, strangers: 1 } },
  ideal: { a: { friends: 1, strangers: 2 }, b: { friends: 2, strangers: 0 } },
  identical: { a: { friends: 1, strangers: 1 }, b: { friends: 1, strangers: 1 } },
};

export const DEFAULT_COUNTS = EXAMPLES.frustrated;

export function sameCounts(x: Counts, y: Counts): boolean {
  return (
    x.a.friends === y.a.friends && x.a.strangers === y.a.strangers && x.b.friends === y.b.friends && x.b.strangers === y.b.strangers
  );
}

export function ExampleChips({ ids, counts, onPick }: { ids: readonly ExampleId[]; counts: Counts; onPick: (c: Counts) => void }) {
  const t = useT();
  return (
    <div className={styles.examples} role="group" aria-label={t.dilemma.examples}>
      <span className={styles.examplesLabel}>{t.dilemma.examples}</span>
      {ids.map((id) => (
        <button key={id} type="button" className="btn btn-small" aria-pressed={sameCounts(counts, EXAMPLES[id])} onClick={() => onPick(EXAMPLES[id])}>
          {t.dilemma.exampleNames[id]}
        </button>
      ))}
    </div>
  );
}

function Stepper({
  value,
  onChange,
  kind,
  side,
  tone,
}: {
  value: number;
  onChange: (value: number) => void;
  kind: 'friends' | 'strangers';
  side: 'A' | 'B';
  tone: 'friend' | 'stranger';
}) {
  const t = useT();
  const word =
    kind === 'friends'
      ? value === 1
        ? t.common.friend
        : t.common.friends.toLowerCase()
      : value === 1
        ? t.common.strangerWord
        : t.common.strangers.toLowerCase();
  return (
    <div className={`${styles.stepper} ${styles[tone]}`}>
      <button
        type="button"
        className="btn btn-small btn-icon"
        aria-label={kind === 'friends' ? t.dilemma.removeFriend(side) : t.dilemma.removeStranger(side)}
        disabled={value <= 0}
        onClick={() => onChange(value - 1)}
      >
        <Icon name="minus" size={16} />
      </button>
      <span className={`${styles.stepperValue} num`} aria-live="polite" aria-label={t.dilemma.countLabel(value, word, side)}>
        {value}
      </span>
      <button
        type="button"
        className="btn btn-small btn-icon"
        aria-label={kind === 'friends' ? t.dilemma.addFriend(side) : t.dilemma.addStranger(side)}
        disabled={value >= MAX_COUNT}
        onClick={() => onChange(value + 1)}
      >
        <Icon name="plus" size={16} />
      </button>
    </div>
  );
}

export function CommunityCard({
  side,
  counts,
  onChange,
  gamma,
  showUtility,
  winner,
  compact = false,
}: {
  side: 'A' | 'B';
  counts: CommunityCounts;
  onChange: (counts: CommunityCounts) => void;
  gamma: number;
  showUtility: boolean;
  winner: boolean;
  compact?: boolean;
}) {
  const t = useT();
  const utility = communityUtility(counts.friends, counts.strangers, gamma);
  return (
    <div className={`${styles.community} ${winner ? styles.winner : ''} ${compact ? styles.communityCompact : ''}`}>
      <p className={styles.communityTitle}>
        <span className={styles.letter}>{side}</span>
        {!compact && t.dilemma.communityName(side)}
      </p>
      <div className={styles.row}>
        <span className={`${styles.rowLabel} friend-text`}>
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
            <circle cx="5" cy="5" r="5" fill="var(--friend)" />
          </svg>
          <span className={compact ? 'visually-hidden' : undefined}>{t.common.friends}</span>
          {!compact && <InfoTip label={`${t.common.friends}: ${t.common.info}`}>{t.dilemma.friendTip}</InfoTip>}
        </span>
        <Stepper value={counts.friends} onChange={(friends) => onChange({ ...counts, friends })} kind="friends" side={side} tone="friend" />
      </div>
      <div className={styles.row}>
        <span className={`${styles.rowLabel} stranger-text`}>
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
            <circle cx="5" cy="5" r="4" fill="var(--stranger-soft)" stroke="var(--stranger)" strokeWidth="1.6" />
          </svg>
          <span className={compact ? 'visually-hidden' : undefined}>{t.common.strangers}</span>
          {!compact && <InfoTip label={`${t.common.strangers}: ${t.common.info}`}>{t.dilemma.strangerTip}</InfoTip>}
        </span>
        <Stepper value={counts.strangers} onChange={(strangers) => onChange({ ...counts, strangers })} kind="strangers" side={side} tone="stranger" />
      </div>
      {showUtility && (
        <p className={`${styles.utility} num`}>
          <span className={styles.rowLabel}>
            {t.common.utility}
            {!compact && (
              <InfoTip label={`${t.common.utility}: ${t.common.info}`}>
                <Rich text={t.dilemma.utilityTip} />
              </InfoTip>
            )}
          </span>
          {compact ? (
            <strong>{formatNumber(utility)}</strong>
          ) : (
            <span>
              <span className="friend-text">{counts.friends}</span>×<span className="friend-text">{formatNumber(1 - gamma)}</span> −{' '}
              <span className="stranger-text">{counts.strangers}</span>×<span className="stranger-text">{formatNumber(gamma)}</span> ={' '}
              <strong>{formatNumber(utility)}</strong>
            </span>
          )}
        </p>
      )}
    </div>
  );
}

/**
 * Plain-language verdict of the agent at resolution γ. `withThreshold`
 * mentions the Familiarity Index (only once the story has introduced it).
 */
export function verdictText(counts: Counts, gamma: number, t: Dictionary, withThreshold = true): string {
  const tradeoff = compareCommunities(counts.a, counts.b);
  if (tradeoff.kind === 'indifferent') return t.dilemma.verdictIndifferent;
  const oriented = orientTradeoff(tradeoff, gamma);
  const name = (side: 'A' | 'B') => t.dilemma.communityName(side);
  if (oriented.orientation === 'tie') {
    if (tradeoff.kind === 'frustrated' && tradeoff.familiarity !== null) {
      return t.dilemma.verdictThreshold(formatFraction(tradeoff.familiarity));
    }
    const preferred = tradeoff.paretoPreference === 'forward' ? name('B') : name('A');
    return gamma === 0 ? t.dilemma.verdictEndpointZero(preferred) : t.dilemma.verdictEndpointOne(preferred);
  }
  const choice = oriented.orientation === 'forward' ? name('B') : name('A');
  const f = tradeoff.familiarity === null ? '' : formatFraction(tradeoff.familiarity);
  const g = formatGamma(gamma);
  const reason =
    oriented.reason === 'friends'
      ? withThreshold
        ? t.dilemma.verdictReason.friends(g, f)
        : t.dilemma.verdictReason.friendsPlain(g)
      : oriented.reason === 'strangers'
        ? withThreshold
          ? t.dilemma.verdictReason.strangers(g, f)
          : t.dilemma.verdictReason.strangersPlain(g)
        : t.dilemma.verdictReason.clear;
  return `${t.dilemma.verdictPrefers(choice)}: ${reason}`;
}

/** The winning side at γ, or null on a tie. */
export function winnerAt(counts: Counts, gamma: number): 'A' | 'B' | null {
  const oriented = orientTradeoff(compareCommunities(counts.a, counts.b), gamma);
  return oriented.orientation === 'forward' ? 'B' : oriented.orientation === 'backward' ? 'A' : null;
}
