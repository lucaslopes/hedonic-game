import site from '../generated/site.json';
import { CodeBlock } from '../components/CodeBlock';
import { Icon } from '../components/Icon';
import { Rich } from '../components/Rich';
import { useT } from '../i18n';
import styles from '../story/Tool.module.css';

const EXAMPLE = `import igraph as ig
from hedonic import Game

# The graph of this page: two cliques of four that share the edge 2-3.
edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
         (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
graph = Game(ig.Graph(n=6, edges=edges))

# Overlapping cover: every vertex may hold up to max_memberships labels.
cover = graph.community_hedonic(resolution=0.5, max_memberships=2, seed=0)
print(graph.memberships)   # [[0], [0], [0, 1], [0, 1], [1], [1]]

# Exact audit of the cover: where is each vertex's set a best response?
intervals = graph.stable_intervals(max_memberships=2)
print(intervals[0])        # [(0.1715..., 1.0)], the lower end is 3 - 2*sqrt(2)
regret = graph.regrets([0.1, 0.5], max_memberships=2).regret
print(regret[0])           # about [0.122, 0]: below its interval, vertex 0 would move`;

export function OverlapTool() {
  const t = useT();
  const s = t.overlap.tool;
  const base = import.meta.env.BASE_URL;
  const docs = `${base}docs/`;
  const cards: [string, keyof typeof s.cards][] = [
    [`${docs}#overlapping-game`, 'overlap'],
    [`${docs}#community-hedonic`, 'method'],
    [`${docs}#game-api`, 'api'],
    [base, 'disjoint'],
  ];
  return (
    <section id="tool" className={styles.section} aria-labelledby="tool-title">
      <div className="container">
        <header className={styles.header}>
          <p className="kicker">{s.kicker}</p>
          <h2 id="tool-title">{s.title}</h2>
          <Rich as="p" className="lede" text={s.body} />
          <p className={styles.version}>
            <span className="chip">hedonic {site.packageVersion}</span>
            <span className="chip">Python {site.requiresPython}</span>
          </p>
        </header>
        <div className={styles.grid}>
          <div className={styles.code}>
            <CodeBlock code={EXAMPLE} language="python" label={t.tool.example} />
            <p className={styles.scope}>{s.scope}</p>
          </div>
          <ul className={styles.cards}>
            {cards.map(([href, key]) => (
              <li key={key}>
                <a className={styles.card} href={href}>
                  <span className={styles.cardTitle}>
                    {s.cards[key].title}
                    <Icon name="arrowRight" size={16} />
                  </span>
                  <Rich className={styles.cardBody} text={s.cards[key].body} />
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
