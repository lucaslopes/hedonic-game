import site from '../generated/site.json';
import { CodeBlock } from '../components/CodeBlock';
import { Icon } from '../components/Icon';
import { Rich } from '../components/Rich';
import { useT } from '../i18n';
import styles from './Tool.module.css';

const INSTALL = 'python -m pip install hedonic';

const EXAMPLE = `import igraph as ig
from hedonic import Game

graph = Game(ig.Graph.Famous("Zachary"))

# Disjoint partition: every vertex joins exactly one community.
partition = graph.community_hedonic(
    resolution=graph.density(),  # γ (the default is the graph density)
    max_memberships=1,           # 1 = disjoint; > 1 = overlapping cover
    n_iterations=-1,             # run to the native no-change stopping rule
)
print(partition.membership)`;

export function ToolSection() {
  const t = useT();
  const docs = `${import.meta.env.BASE_URL}docs/`;
  const cards: [string, keyof typeof t.tool.cards][] = [
    ['getting-started', 'start'],
    ['game-api', 'api'],
    ['community-hedonic', 'method'],
    ['familiarity-index', 'resolution'],
    ['better-response', 'dynamics'],
    ['references', 'references'],
  ];
  return (
    <section id="tool" className={styles.section} aria-labelledby="tool-title">
      <div className="container">
        <header className={styles.header}>
          <p className="kicker">{t.tool.kicker}</p>
          <h2 id="tool-title">{t.tool.title}</h2>
          <Rich as="p" className="lede" text={t.tool.body} />
          <p className={styles.version}>
            <span className="chip">hedonic {site.packageVersion}</span>
            {site.nativeDependency && <span className="chip">{site.nativeDependency}</span>}
            <span className="chip">Python {site.requiresPython}</span>
          </p>
        </header>
        <div className={styles.grid}>
          <div className={styles.code}>
            <CodeBlock code={INSTALL} language="bash" label={t.tool.install} />
            <CodeBlock code={EXAMPLE} language="python" label={t.tool.example} />
            <p className={styles.scope}>{t.tool.scope}</p>
          </div>
          <ul className={styles.cards}>
            {cards.map(([anchor, key]) => (
              <li key={anchor}>
                <a className={styles.card} href={`${docs}#${anchor}`}>
                  <span className={styles.cardTitle}>
                    {t.tool.cards[key].title}
                    <Icon name="arrowRight" size={16} />
                  </span>
                  <Rich className={styles.cardBody} text={t.tool.cards[key].body} />
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
