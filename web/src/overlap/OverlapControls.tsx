import { ResolutionControl } from '../components/ResolutionControl';
import { Segmented, Switch } from '../components/Segmented';
import { useT } from '../i18n';
import { CAPS, PRESET_ORDER, type PresetName } from './overlapData';
import styles from './Overlap.module.css';

export interface OverlapSettings {
  readonly gamma: number;
  readonly setGamma: (gamma: number) => void;
  readonly cap: number;
  readonly setCap: (cap: number) => void;
  readonly isolation: boolean;
  readonly setIsolation: (isolation: boolean) => void;
}

export function GammaControl({ settings, thresholds }: { settings: OverlapSettings; thresholds?: readonly number[] }) {
  return <ResolutionControl gamma={settings.gamma} onChange={settings.setGamma} thresholds={thresholds} compact showPresets />;
}

/** M = 1…4; values below the widest row of the cover are disabled. */
export function CapControl({ settings, minimum }: { settings: OverlapSettings; minimum: number }) {
  const t = useT();
  const options = CAPS.filter((cap) => cap >= minimum).map((cap) => ({ value: String(cap), label: String(cap) }));
  return (
    <div className={styles.control}>
      <Segmented legend={t.overlap.controls.cap} value={String(settings.cap)} options={options} onChange={(value) => settings.setCap(Number(value))} />
      {minimum > 1 && <p className={styles.hint}>{t.overlap.controls.capTooSmall(minimum)}</p>}
    </div>
  );
}

export function IsolationControl({ settings }: { settings: OverlapSettings }) {
  const t = useT();
  return <Switch label={t.overlap.controls.isolation} checked={settings.isolation} onChange={settings.setIsolation} />;
}

export function PresetControl({ value, onChange }: { value: PresetName | 'edited'; onChange: (preset: PresetName) => void }) {
  const t = useT();
  const options = PRESET_ORDER.map((name) => ({
    value: name,
    label: t.overlap.controls.presetNames[name],
    title: t.overlap.controls.presetTitles[name],
  }));
  return (
    <div className={styles.control}>
      <Segmented<PresetName | 'edited'> legend={t.overlap.controls.presets} value={value} options={options} onChange={(name) => name !== 'edited' && onChange(name)} fill />
      {value === 'edited' && <p className={styles.hint}>{t.overlap.controls.edited}</p>}
    </div>
  );
}
