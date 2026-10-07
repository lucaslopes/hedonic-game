import { useT } from '../i18n';
import { formatNumber, nodePotential, type Metagraph, type WalkState } from '../model';
import styles from './Walk.module.css';

interface PotentialChartProps {
  readonly metagraph: Metagraph;
  readonly state: WalkState;
  readonly gamma: number;
}

/** Φγ after each step: a strictly increasing staircase for every better-response walk. */
export function PotentialChart({ metagraph, state, gamma }: PotentialChartProps) {
  const t = useT();
  const values = [nodePotential(metagraph.node(state.start), gamma), ...state.steps.map((step) => step.potentialAfter)];
  const width = 320;
  const height = 132;
  const pad = { l: 38, r: 16, t: 14, b: 26 };
  const n = Math.max(3, values.length - 1);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const x = (i: number) => pad.l + (i / n) * (width - pad.l - pad.r);
  const y = (v: number) => height - pad.b - ((v - lo) / span) * (height - pad.t - pad.b);
  const d = values.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
  const description = t.walk.chartDescription(values.map((v) => formatNumber(v)).join(', '));

  return (
    <figure className={styles.chart}>
      <figcaption className={styles.chartTitle}>{t.walk.qualityChart}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={description}>
        <line x1={pad.l} x2={width - pad.r} y1={height - pad.b} y2={height - pad.b} className={styles.axis} />
        <line x1={pad.l} x2={pad.l} y1={pad.t} y2={height - pad.b} className={styles.axis} />
        <text x={pad.l - 6} y={y(hi)} dy="0.35em" textAnchor="end" className={styles.tick}>
          {formatNumber(hi)}
        </text>
        {hi !== lo && (
          <text x={pad.l - 6} y={y(lo)} dy="0.35em" textAnchor="end" className={styles.tick}>
            {formatNumber(lo)}
          </text>
        )}
        {values.map((_, i) => (
          <text key={`x-${i}`} x={x(i)} y={height - 8} textAnchor="middle" className={styles.tick}>
            {i}
          </text>
        ))}
        <path d={d} className={styles.line} />
        {values.map((v, i) => (
          <circle key={`p-${i}`} cx={x(i)} cy={y(v)} r={i === values.length - 1 ? 5 : 3.5} className={i === values.length - 1 ? styles.dotLast : styles.dot} />
        ))}
      </svg>
    </figure>
  );
}
