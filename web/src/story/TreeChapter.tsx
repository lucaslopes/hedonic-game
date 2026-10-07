import { useT } from '../i18n';
import { compareCommunities, decide, formatFraction, formatGamma, orientTradeoff } from '../model';
import { DecisionTree } from '../viz/DecisionTree';
import { CommunityCard, ExampleChips, type Counts } from './dilemmaShared';
import styles from './Tree.module.css';
import { Scrolly } from './Scrolly';

interface TreeChapterProps {
  readonly counts: Counts;
  readonly setCounts: (counts: Counts) => void;
  readonly gamma: number;
  readonly setGamma: (gamma: number) => void;
}

function useTreeExplanation(counts: Counts, gamma: number) {
  const t = useT();
  const result = decide(counts.a, counts.b);
  const name = (side: 'A' | 'B') => t.dilemma.communityName(side);
  let explanation: string;
  switch (result.leaf) {
    case 'indifferent':
      explanation = t.tree.explain.indifferent;
      break;
    case 'fewerNonLinks':
      explanation = t.tree.explain.fewerNonLinks(name(result.preferred ?? 'A'));
      break;
    case 'moreLinks':
      explanation = t.tree.explain.moreLinks(name(result.preferred ?? 'A'));
      break;
    case 'idealCommunity':
      explanation = t.tree.explain.idealCommunity(name(result.preferred ?? 'A'));
      break;
    default: {
      const friendSide = counts.a.friends > counts.b.friends ? 'A' : 'B';
      explanation = t.tree.explain.frustrated(name(friendSide), name(friendSide === 'A' ? 'B' : 'A'));
    }
  }
  let resolution: string | null = null;
  const tradeoff = compareCommunities(counts.a, counts.b);
  if (result.leaf === 'frustrated' && tradeoff.familiarity !== null) {
    const oriented = orientTradeoff(tradeoff, gamma);
    resolution =
      oriented.orientation === 'tie'
        ? t.tree.resolvedTie(formatFraction(tradeoff.familiarity))
        : t.tree.resolvedBy(formatFraction(tradeoff.familiarity), formatGamma(gamma), name(oriented.orientation === 'forward' ? 'B' : 'A'));
  }
  return { result, explanation, resolution, tradeoff };
}

function TreeStage({ counts, setCounts, gamma }: TreeChapterProps) {
  const t = useT();
  const { result, explanation, resolution } = useTreeExplanation(counts, gamma);
  const text = { nodes: t.tree.nodes, yes: t.common.yes, no: t.common.no, label: t.tree.figureLabel };
  return (
    <div className={styles.stage}>
      <div className={styles.treeWrap}>
        <DecisionTree result={result} text={text} />
      </div>
      <p className={styles.wording}>{t.tree.wording}</p>
      <div className={styles.bottom}>
        <div className={styles.counts}>
          <CommunityCard side="A" counts={counts.a} onChange={(a) => setCounts({ ...counts, a })} gamma={gamma} showUtility={false} winner={result.preferred === 'A'} compact />
          <CommunityCard side="B" counts={counts.b} onChange={(b) => setCounts({ ...counts, b })} gamma={gamma} showUtility={false} winner={result.preferred === 'B'} compact />
        </div>
        <div className={`${styles.explain} ${result.leaf === 'frustrated' ? styles.explainFrustrated : result.leaf === 'indifferent' ? '' : styles.explainClear}`} aria-live="polite">
          <p className={styles.explainTitle}>{t.tree.yourComparison}</p>
          <p className={styles.explainText}>{explanation}</p>
          {resolution && <p className={styles.resolution}>{resolution}</p>}
        </div>
      </div>
    </div>
  );
}

export function TreeChapter(props: TreeChapterProps) {
  const t = useT();
  const chips = [
    <ExampleChips key="a" ids={['identical', 'clear', 'frustrated']} counts={props.counts} onPick={props.setCounts} />,
    <ExampleChips key="b" ids={['clear', 'ideal', 'identical']} counts={props.counts} onPick={props.setCounts} />,
    <ExampleChips key="c" ids={['frustrated', 'vertex3']} counts={props.counts} onPick={props.setCounts} />,
  ];
  const steps = t.tree.steps.map((step, index) => ({ title: step.title, body: step.body, extra: chips[index] }));
  return (
    <Scrolly
      id="tree"
      kicker={t.tree.kicker}
      title={t.tree.title}
      steps={steps}
      stageLabel={`${t.a11y.stage}: ${t.chapters.tree}`}
      next={{ href: '#metagraph', label: t.chapters.metagraph }}
      stage={() => <TreeStage {...props} />}
    />
  );
}
