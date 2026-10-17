import { useT } from '../i18n';
import { formatBlocks, formatNumber, formatSigned, nodePotential, stabilityCertificate, type Metagraph } from '../model';
import { describeMove } from './describe';
import styles from './Walk.module.css';

interface SinkCertificateProps {
  readonly metagraph: Metagraph;
  readonly id: string;
  readonly gamma: number;
  readonly steps: number;
}

/** "Stable equilibrium reached": every vertex's best alternative has gain ≤ 0. */
export function SinkCertificate({ metagraph, id, gamma, steps }: SinkCertificateProps) {
  const t = useT();
  const node = metagraph.node(id);
  const rows = stabilityCertificate(metagraph, id, gamma);
  const anyTie = rows.some((row) => row.ties.length > 0);
  return (
    <section className={styles.certificate} aria-label={t.walk.sinkTitle}>
      <h4 className={styles.certificateTitle}>
        <svg viewBox="-8 -8 16 16" width="18" height="18" aria-hidden="true">
          <circle r="7.5" fill="var(--ink)" />
          <path d="M-3.4 -3.2 L0 0.6 L3.4 -3.2 M-3.6 3.2 H3.6" fill="none" stroke="var(--paper)" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        {t.walk.sinkTitle}
      </h4>
      <p className={styles.certificateSummary}>
        <code>{node.label}</code> · {t.walk.stepCount(steps)} · <span className="num">{t.walk.quality} = {formatNumber(nodePotential(node, gamma))}</span>
      </p>
      <p className={styles.certificateIntro}>{t.walk.certificateIntro}</p>
      <div className="table-scroll">
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">{t.walk.certificateHeaders.vertex}</th>
              <th scope="col">{t.walk.certificateHeaders.community}</th>
              <th scope="col">{t.walk.certificateHeaders.best}</th>
              <th scope="col" className={styles.numeric}>
                {t.walk.certificateHeaders.gain}
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.vertex}>
                <th scope="row">{row.vertex}</th>
                <td>
                  <code>{formatBlocks([row.community])}</code>
                </td>
                <td>{row.best ? describeMove(row.best.move, t) : t.walk.certificateNone}</td>
                <td className={`${styles.numeric} num ${row.best && Math.abs(row.best.gain) < 1e-9 ? styles.tie : ''}`}>
                  {row.best ? formatSigned(row.best.gain) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {anyTie && <p className={styles.help}>{t.walk.tieNote}</p>}
    </section>
  );
}
