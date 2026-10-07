import { useState } from 'react';
import { useSettings } from '../app/settings';
import { Icon } from '../components/Icon';
import { ResolutionControl } from '../components/ResolutionControl';
import { Switch } from '../components/Segmented';
import { useT } from '../i18n';
import { formatGamma, formatNumber, formatSigned, improvingMoves, isSink, nodePotential } from '../model';
import { MetagraphLegend } from '../viz/metagraph/Legend';
import { MetagraphView } from '../viz/metagraph/MetagraphView';
import { describeMove } from '../walk/describe';
import { PotentialChart } from '../walk/PotentialChart';
import { SinkCertificate } from '../walk/SinkCertificate';
import { WalkControls } from '../walk/WalkControls';
import { WalkStatus } from '../walk/WalkStatus';
import styles from './Explore.module.css';
import { Inspector } from './Inspector';
import { LAYOUT, METAGRAPH, THRESHOLDS, useMetagraphLabels } from './metagraphData';
import { useWalkSetup, walkOverlay } from './WalkChapter';

interface ExploreSectionProps {
  readonly gamma: number;
  readonly setGamma: (gamma: number) => void;
}

export function ExploreSection({ gamma, setGamma }: ExploreSectionProps) {
  const t = useT();
  const { reducedMotion } = useSettings();
  const labels = useMetagraphLabels('explore');
  const [selected, setSelected] = useState<string | null>('0001');
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);
  const [showArcs, setShowArcs] = useState(true);
  const [showPotential, setShowPotential] = useState(true);
  const [showWalk, setShowWalk] = useState(true);
  const [tableOpen, setTableOpen] = useState(false);
  const { controller, options } = useWalkSetup(gamma);
  const walking = showWalk && (controller.state.steps.length > 0 || controller.playing);

  return (
    <section id="explore" className={styles.section} aria-labelledby="explore-title">
      <div className="container">
        <header className={styles.header}>
          <p className="kicker">{t.explore.kicker}</p>
          <h2 id="explore-title">{t.explore.title}</h2>
          <p className="lede">{t.explore.lede}</p>
        </header>
        <div className={styles.grid}>
          <div className={styles.main}>
            <div className={styles.graphCard}>
              <div className={styles.graphArea}>
                <MetagraphView
                  metagraph={METAGRAPH}
                  layout={LAYOUT}
                  gamma={gamma}
                  mode="oriented"
                  labels={labels}
                  reducedMotion={reducedMotion}
                  selectedId={selected}
                  onSelectNode={(id) => {
                    setSelected(id);
                    setSelectedEdge(null);
                  }}
                  selectedEdgeId={selectedEdge}
                  onSelectEdge={setSelectedEdge}
                  highlightEdgeId={highlight}
                  walk={walking ? walkOverlay(controller) : null}
                  showSameLayer={showArcs}
                  showPotential={showPotential}
                  zoomable
                />
              </div>
              <MetagraphLegend mode={walking ? 'walk' : 'oriented'} />
            </div>
            <div className={styles.controlsCard}>
              <ResolutionControl gamma={gamma} onChange={setGamma} thresholds={THRESHOLDS} showExtremes />
              <div className={styles.switches} role="group" aria-label={t.explore.filters}>
                <Switch label={t.metagraph.showArcs} checked={showArcs} onChange={setShowArcs} />
                <Switch label={t.metagraph.showPotential} checked={showPotential} onChange={setShowPotential} />
                <Switch label={t.metagraph.legend.walk} checked={showWalk} onChange={setShowWalk} />
              </div>
            </div>
          </div>
          <aside className={styles.side} aria-label={t.explore.inspector}>
            <h3 className={styles.sideTitle}>{t.explore.inspector}</h3>
            <Inspector
              metagraph={METAGRAPH}
              gamma={gamma}
              selectedId={selected}
              onSelectNode={(id) => {
                setSelected(id);
                setSelectedEdge(null);
              }}
              selectedEdgeId={selectedEdge}
              onHighlightEdge={setHighlight}
            />
            <h3 className={styles.sideTitle}>{t.explore.walkTitle}</h3>
            <div className={styles.walkBox}>
              <WalkStatus metagraph={METAGRAPH} controller={controller} gamma={gamma} />
              <WalkControls metagraph={METAGRAPH} controller={controller} options={options} selectedId={selected} />
              {controller.state.steps.length > 0 && <PotentialChart metagraph={METAGRAPH} state={controller.state} gamma={gamma} />}
              {controller.state.status === 'sink' && (
                <SinkCertificate metagraph={METAGRAPH} id={controller.state.current} gamma={gamma} steps={controller.state.steps.length} />
              )}
              <p className={styles.disclaimer}>{t.walk.disclaimer}</p>
            </div>
          </aside>
        </div>

        <div className={styles.tableToggle}>
          <button type="button" className="btn" aria-expanded={tableOpen} aria-controls="metagraph-table" onClick={() => setTableOpen((open) => !open)}>
            <Icon name="table" />
            {tableOpen ? t.explore.hideTable : t.explore.showTable}
          </button>
        </div>
        {tableOpen && (
          <div id="metagraph-table" className="table-scroll">
            <table className={styles.fullTable}>
              <caption>{t.explore.tableCaption(formatGamma(gamma))}</caption>
              <thead>
                <tr>
                  <th scope="col">{t.explore.tableHeaders.partition}</th>
                  <th scope="col">{t.explore.tableHeaders.k}</th>
                  <th scope="col">{t.explore.tableHeaders.phi}</th>
                  <th scope="col">{t.explore.tableHeaders.status}</th>
                  <th scope="col">{t.explore.tableHeaders.out}</th>
                </tr>
              </thead>
              <tbody>
                {METAGRAPH.nodes.map((node) => {
                  const moves = improvingMoves(METAGRAPH, node.id, gamma);
                  return (
                    <tr key={node.id}>
                      <th scope="row">
                        <button
                          type="button"
                          className={styles.linkButton}
                          onClick={() => {
                            setSelected(node.id);
                            document.getElementById('explore')?.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth' });
                          }}
                        >
                          <code>{node.label}</code>
                        </button>
                      </th>
                      <td className="num">{node.communityCount}</td>
                      <td className="num">{formatNumber(nodePotential(node, gamma))}</td>
                      <td>{isSink(METAGRAPH, node.id, gamma) ? t.explore.isSink : t.explore.notSink(moves.length)}</td>
                      <td>
                        {moves.length === 0
                          ? '—'
                          : moves.map(({ move, gain }) => `${describeMove(move, t)} (${formatSigned(gain)})`).join('; ')}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
