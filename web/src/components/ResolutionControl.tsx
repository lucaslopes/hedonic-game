import { useId, useState, type CSSProperties, type KeyboardEvent } from 'react';
import { formatFraction, formatGamma, formatNumber, parseGamma, type TradeoffKind } from '../model';
import { useT } from '../i18n';
import { Icon } from './Icon';
import { InfoTip } from './InfoTip';
import { nextStop, pageStep, pointerValue } from './resolutionStops';
import styles from './ResolutionControl.module.css';

export interface TradeoffTrack {
  /** Familiarity Index of the trade-off shown on the track (null if undefined). */
  readonly familiarity: number | null;
  readonly kind: TradeoffKind;
}

interface ResolutionControlProps {
  readonly gamma: number;
  readonly onChange: (gamma: number) => void;
  /** Colour the track by a single trade-off: blue where friends win, red where strangers win. */
  readonly tradeoff?: TradeoffTrack;
  /** Threshold markers (e.g. every Familiarity Index of the metagraph). */
  readonly thresholds?: readonly number[];
  readonly compact?: boolean;
  readonly showPresets?: boolean;
  readonly showExtremes?: boolean;
  readonly label?: string;
}

const PRESETS = [0, 0.25, 0.5, 0.75, 1];

export function ResolutionControl({
  gamma,
  onChange,
  tradeoff,
  thresholds = [],
  compact = false,
  showPresets = true,
  showExtremes = false,
  label,
}: ResolutionControlProps) {
  const t = useT();
  const id = useId();
  const [snap, setSnap] = useState(true);
  const [draft, setDraft] = useState<string | null>(null);

  const frustratedF =
    tradeoff?.kind === 'frustrated' && tradeoff.familiarity !== null ? tradeoff.familiarity : null;
  const markers = frustratedF !== null ? [frustratedF] : [...thresholds];
  const reward = formatNumber(1 - gamma);
  const cost = formatNumber(gamma);
  const gammaText = formatGamma(gamma);

  const commit = (value: number) => onChange(Math.min(1, Math.max(0, value)));

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    let next: number | null = null;
    if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = nextStop(gamma, markers, 1);
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = nextStop(gamma, markers, -1);
    else if (event.key === 'PageUp') next = pageStep(gamma, 1);
    else if (event.key === 'PageDown') next = pageStep(gamma, -1);
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = 1;
    if (next !== null) {
      event.preventDefault();
      commit(next);
    }
  };

  const parsedDraft = draft === null ? null : parseGamma(draft);
  const draftInvalid = draft !== null && draft.trim() !== '' && parsedDraft === null;

  let trackStyle: CSSProperties | undefined;
  let trackClass = styles.track;
  if (tradeoff) {
    if (frustratedF !== null) {
      const at = `calc(var(--thumb) / 2 + ${frustratedF} * (100% - var(--thumb)))`;
      trackStyle = {
        background: `linear-gradient(90deg, var(--friend) 0 ${at}, var(--stranger) ${at} 100%)`,
      };
    } else {
      trackClass = `${styles.track} ${tradeoff.kind === 'indifferent' ? styles.trackTie : styles.trackClear}`;
    }
  }

  const trackCaption =
    tradeoff && frustratedF === null ? (tradeoff.kind === 'indifferent' ? t.resolution.indifferentTrack : t.resolution.clearTrack) : null;

  return (
    <div className={`${styles.control} ${compact ? styles.compact : ''}`} style={{ '--gamma': gamma } as CSSProperties}>
      <div className={styles.head}>
        {!compact && (
          <span className={styles.reward}>
            <span className={styles.headLabel}>{t.resolution.friendReward}</span>
            <span className="num">
              1 − γ = <strong>{reward}</strong>
            </span>
          </span>
        )}
        <span className={styles.value}>
          <label htmlFor={`${id}-range`} className={styles.valueLabel}>
            {label ?? t.resolution.label}
          </label>
          <output htmlFor={`${id}-range`} className={`${styles.valueOut} num`} aria-live="off">
            {gammaText}
          </output>
        </span>
        {!compact && (
          <span className={styles.cost}>
            <span className={styles.headLabel}>{t.resolution.strangerCost}</span>
            <span className="num">
              γ = <strong>{cost}</strong>
            </span>
          </span>
        )}
      </div>

      {!compact && (
        <div className={styles.split} aria-hidden="true" title={t.resolution.splitCaption}>
          <span className={styles.splitCost} style={{ flexGrow: Math.max(gamma, 0.0001) }}>
            {gamma >= 0.14 && <>γ</>}
          </span>
          <span className={styles.splitReward} style={{ flexGrow: Math.max(1 - gamma, 0.0001) }}>
            {gamma <= 0.86 && <>1 − γ</>}
          </span>
        </div>
      )}

      <div className={styles.trackWrap}>
        <div className={trackClass} style={trackStyle} aria-hidden="true" />
        {markers.map((marker) => (
          <span
            key={marker}
            className={`${styles.marker} ${Math.abs(marker - gamma) < 1e-9 ? styles.markerActive : ''}`}
            style={{ left: `calc(var(--thumb) / 2 + ${marker} * (100% - var(--thumb)))` }}
            aria-hidden="true"
          />
        ))}
        <input
          id={`${id}-range`}
          className={styles.range}
          type="range"
          min={0}
          max={1}
          step="any"
          value={gamma}
          aria-valuetext={t.resolution.valueText(gammaText, reward, cost)}
          onChange={(event) => commit(pointerValue(event.currentTarget.valueAsNumber, markers, snap))}
          onKeyDown={onKeyDown}
        />
      </div>

      <div className={styles.scale} aria-hidden="true">
        <span>0</span>
        {markers.map((marker) => (
          <span
            key={marker}
            className={styles.scaleMarker}
            style={{ left: `calc(var(--thumb) / 2 + ${marker} * (100% - var(--thumb)))` }}
          >
            {frustratedF !== null ? `F = ${formatFraction(marker)}` : formatFraction(marker)}
          </span>
        ))}
        <span>1</span>
      </div>
      {frustratedF !== null && !compact && (
        <div className={styles.regions} aria-hidden="true">
          <span className="friend-text">← {t.resolution.friendsRegion}</span>
          <span className="stranger-text">{t.resolution.strangersRegion} →</span>
        </div>
      )}
      {trackCaption && !compact && <p className={styles.trackCaption}>{trackCaption}</p>}

      {showPresets && (
        <div className={styles.tools}>
          <div className={styles.presets} role="group" aria-label={t.resolution.presets}>
            {PRESETS.map((value) => (
              <button
                key={value}
                type="button"
                className={`btn btn-small ${styles.preset}`}
                aria-pressed={Math.abs(gamma - value) < 1e-9}
                aria-label={t.resolution.preset(formatNumber(value))}
                onClick={() => commit(value)}
              >
                {value === 0.25 ? '¼' : value === 0.5 ? '½' : value === 0.75 ? '¾' : String(value)}
              </button>
            ))}
          </div>
          {markers.length > 0 && (
            <div className={styles.presets} role="group" aria-label={t.resolution.thresholds}>
              {markers.map((marker) => (
                <button
                  key={marker}
                  type="button"
                  className={`btn btn-small ${styles.threshold}`}
                  aria-pressed={Math.abs(gamma - marker) < 1e-9}
                  onClick={() => commit(marker)}
                >
                  {frustratedF !== null ? t.resolution.snapTo(formatFraction(marker)) : `γ = ${formatFraction(marker)}`}
                </button>
              ))}
            </div>
          )}
          {!compact && (
            <div className={styles.inputRow}>
              {markers.length > 0 && (
                <span className={styles.snap}>
                  <button
                    type="button"
                    className="btn btn-small btn-icon"
                    aria-pressed={snap}
                    aria-label={t.resolution.snap}
                    title={t.resolution.snap}
                    onClick={() => setSnap((value) => !value)}
                  >
                    <Icon name="magnet" size={16} />
                  </button>
                  <InfoTip label={t.resolution.snap}>{t.resolution.snapTip}</InfoTip>
                </span>
              )}
              <label className={styles.inputLabel}>
                <span className="visually-hidden">{t.resolution.input}</span>
                <input
                  className={`${styles.input} num`}
                  type="text"
                  inputMode="decimal"
                  autoComplete="off"
                  spellCheck={false}
                  value={draft ?? formatGamma(gamma).split(' ')[0]}
                  aria-invalid={draftInvalid}
                  aria-describedby={`${id}-help`}
                  onFocus={(event) => {
                    setDraft(event.currentTarget.value);
                    event.currentTarget.select();
                  }}
                  onChange={(event) => setDraft(event.currentTarget.value)}
                  onBlur={() => {
                    if (parsedDraft !== null) commit(parsedDraft);
                    setDraft(null);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && parsedDraft !== null) {
                      commit(parsedDraft);
                      setDraft(null);
                      event.currentTarget.blur();
                    } else if (event.key === 'Escape') {
                      setDraft(null);
                      event.currentTarget.blur();
                    }
                  }}
                />
              </label>
              <span id={`${id}-help`} className={styles.help}>
                {draftInvalid ? t.resolution.invalid : t.resolution.inputHelp}
              </span>
            </div>
          )}
        </div>
      )}

      {showExtremes && (gamma === 0 || gamma === 1) && (
        <p className={styles.extreme} role="note">
          {gamma === 0 ? t.resolution.extreme0 : t.resolution.extreme1}
        </p>
      )}
    </div>
  );
}
