import { useId } from 'react';
import styles from './Segmented.module.css';

interface SegmentedProps<T extends string> {
  readonly legend: string;
  readonly value: T;
  readonly options: readonly { readonly value: T; readonly label: string; readonly title?: string }[];
  readonly onChange: (value: T) => void;
  readonly hideLegend?: boolean;
}

/** A pill-shaped radio group; native radios keep arrow-key navigation. */
export function Segmented<T extends string>({ legend, value, options, onChange, hideLegend }: SegmentedProps<T>) {
  const name = useId();
  return (
    <fieldset className={styles.group}>
      <legend className={hideLegend ? 'visually-hidden' : styles.legend}>{legend}</legend>
      <div className={styles.options}>
        {options.map((option) => (
          <label key={option.value} className={styles.option} title={option.title}>
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={option.value === value}
              onChange={() => onChange(option.value)}
            />
            <span>{option.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

interface SwitchProps {
  readonly label: string;
  readonly checked: boolean;
  readonly onChange: (checked: boolean) => void;
}

export function Switch({ label, checked, onChange }: SwitchProps) {
  return (
    <label className={styles.switch}>
      <input type="checkbox" role="switch" checked={checked} onChange={(event) => onChange(event.currentTarget.checked)} />
      <span className={styles.track} aria-hidden="true" />
      <span>{label}</span>
    </label>
  );
}
