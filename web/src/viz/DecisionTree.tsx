import { TREE_NODES, TREE_ROOT, type DecisionResult, type TreeNodeId } from '../model';
import styles from './DecisionTree.module.css';

export interface DecisionTreeText {
  readonly nodes: Readonly<Record<TreeNodeId, string>>;
  readonly yes: string;
  readonly no: string;
  readonly label: string;
}

interface DecisionTreeProps {
  readonly result: DecisionResult;
  readonly text: DecisionTreeText;
}

interface Placement {
  readonly x: number;
  readonly y: number;
  readonly w: number;
}

const PLACE: Readonly<Record<TreeNodeId, Placement>> = {
  sameLinks: { x: 500, y: 48, w: 300 },
  sameNonLinksWhenSameLinks: { x: 262, y: 158, w: 290 },
  sameNonLinks: { x: 738, y: 158, w: 290 },
  indifferent: { x: 118, y: 278, w: 206 },
  fewerNonLinks: { x: 356, y: 278, w: 222 },
  moreLinks: { x: 604, y: 278, w: 222 },
  ideal: { x: 846, y: 278, w: 288 },
  idealCommunity: { x: 744, y: 392, w: 200 },
  frustrated: { x: 922, y: 392, w: 140 },
};

const BOX_H = 58;
const CHAR_W = 7.6;

function wrap(text: string, width: number): string[] {
  const max = Math.max(8, Math.floor((width - 20) / CHAR_W));
  const lines: string[] = [];
  let line = '';
  for (const word of text.split(' ')) {
    if (line && `${line} ${word}`.length > max) {
      lines.push(line);
      line = word;
    } else {
      line = line ? `${line} ${word}` : word;
    }
  }
  if (line) lines.push(line);
  return lines.slice(0, 3);
}

function toneClass(id: TreeNodeId): string {
  const node = TREE_NODES[id];
  if (node.kind === 'question') return styles.question;
  return node.tone === 'frustrated' ? styles.frustrated : node.tone === 'clear' ? styles.clear : styles.neutral;
}

function edgesOf(): { from: TreeNodeId; to: TreeNodeId; answer: 'yes' | 'no' }[] {
  const out: { from: TreeNodeId; to: TreeNodeId; answer: 'yes' | 'no' }[] = [];
  const visit = (id: TreeNodeId) => {
    const node = TREE_NODES[id];
    if (node.yes) {
      out.push({ from: id, to: node.yes, answer: 'yes' });
      visit(node.yes);
    }
    if (node.no) {
      out.push({ from: id, to: node.no, answer: 'no' });
      visit(node.no);
    }
  };
  visit(TREE_ROOT);
  return out;
}

const EDGES = edgesOf();

function OutlineNode({ id, result, text, answer }: { id: TreeNodeId; result: DecisionResult; text: DecisionTreeText; answer?: 'yes' | 'no' }) {
  const node = TREE_NODES[id];
  const active = result.path.includes(id);
  return (
    <li className={active ? styles.onPath : styles.offPath}>
      <span className={`${styles.outlineBox} ${toneClass(id)}`} aria-current={result.leaf === id ? 'true' : undefined}>
        {answer && <span className={styles.answer}>{answer === 'yes' ? text.yes : text.no} →</span>} {text.nodes[id]}
      </span>
      {node.kind === 'question' && (
        <ul>
          {node.yes && <OutlineNode id={node.yes} result={result} text={text} answer="yes" />}
          {node.no && <OutlineNode id={node.no} result={result} text={text} answer="no" />}
        </ul>
      )}
    </li>
  );
}

export function DecisionTree({ result, text }: DecisionTreeProps) {
  const onPath = (id: TreeNodeId) => result.path.includes(id);
  const edgeOnPath = (from: TreeNodeId, to: TreeNodeId) => {
    const i = result.path.indexOf(from);
    return i >= 0 && result.path[i + 1] === to;
  };
  const pathDescription = result.steps
    .map((step) => `${text.nodes[step.question]} ${step.answer === 'yes' ? text.yes : text.no}`)
    .concat(text.nodes[result.leaf])
    .join(' → ');

  return (
    <figure className={styles.figure} aria-label={text.label}>
      <svg className={styles.svg} viewBox="0 0 1000 430" role="img" aria-label={`${text.label}: ${pathDescription}`}>
        <defs>
          <marker id="dt-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="var(--ink)" />
          </marker>
          <marker id="dt-arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="var(--rule-strong)" />
          </marker>
        </defs>
        {EDGES.map(({ from, to, answer }) => {
          const a = PLACE[from];
          const b = PLACE[to];
          const active = edgeOnPath(from, to);
          const x1 = a.x;
          const y1 = a.y + BOX_H / 2;
          const x2 = b.x;
          const y2 = b.y - BOX_H / 2 - 2;
          const mx = (x1 + x2) / 2;
          const my = (y1 + y2) / 2;
          return (
            <g key={`${from}-${to}`} className={active ? styles.edgeActive : styles.edge}>
              <path d={`M${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`} markerEnd={`url(#${active ? 'dt-arrow' : 'dt-arrow-muted'})`} />
              <text x={mx + (answer === 'yes' ? -10 : 10)} y={my} textAnchor={answer === 'yes' ? 'end' : 'start'} dy="0.35em" className={styles.edgeLabel}>
                {answer === 'yes' ? text.yes : text.no}
              </text>
            </g>
          );
        })}
        {(Object.keys(PLACE) as TreeNodeId[]).map((id) => {
          const p = PLACE[id];
          const lines = wrap(text.nodes[id], p.w);
          const frustrated = TREE_NODES[id].tone === 'frustrated';
          const h = frustrated ? 44 : BOX_H;
          return (
            <g key={id} className={`${styles.node} ${toneClass(id)} ${onPath(id) ? styles.nodeActive : styles.nodeIdle}`}>
              <rect x={p.x - p.w / 2} y={p.y - h / 2} width={p.w} height={h} rx={14} />
              {lines.map((line, i) => (
                <text key={i} x={p.x} y={p.y + (i - (lines.length - 1) / 2) * 17} dy="0.35em" textAnchor="middle">
                  {line}
                </text>
              ))}
            </g>
          );
        })}
      </svg>
      {/* Narrow screens: the same tree as an indented list (display: none on wide screens). */}
      <ul className={styles.outline} aria-label={text.label}>
        <OutlineNode id={TREE_ROOT} result={result} text={text} />
      </ul>
    </figure>
  );
}
