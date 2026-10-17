import { useState } from 'react';
import { Icon } from '../components/Icon';
import { useT } from '../i18n';
import { EXAMPLE_GRAPH, adjacencyOf, neighborsOf, nonNeighborsOf } from '../model';
import { GraphGlyph } from '../viz/GraphGlyph';
import styles from './Hero.module.css';

const ADJ = adjacencyOf(EXAMPLE_GRAPH);

export function Hero() {
  const t = useT();
  const [focus, setFocus] = useState(3);
  const list = (items: number[]) => (items.length ? items.join(', ') : t.hero.nobody);
  const chapters: [string, string][] = [
    ['#dilemma', t.chapters.dilemma],
    ['#tree', t.chapters.tree],
    ['#metagraph', t.chapters.metagraph],
    ['#walk', t.chapters.walk],
    ['#explore', t.chapters.explore],
    ['#tool', t.chapters.tool],
  ];
  return (
    <section id="top" className={styles.hero} aria-labelledby="hero-title">
      <div className={`container ${styles.grid}`}>
        <div className={styles.copy}>
          <p className="kicker">{t.hero.kicker}</p>
          <h1 id="hero-title">{t.hero.title}</h1>
          <p className="lede">{t.hero.lede}</p>
          <div className={styles.actions}>
            <a className="btn btn-primary" href="#dilemma">
              {t.hero.cta}
              <Icon name="arrowDown" size={16} />
            </a>
            <a className="btn" href={`${import.meta.env.BASE_URL}docs/`}>
              <Icon name="book" size={16} />
              {t.hero.docsCta}
            </a>
          </div>
          <nav className={styles.roadmap} aria-label={t.hero.roadmap}>
            <ol>
              {chapters.map(([href, label], index) => (
                <li key={href}>
                  <a href={href}>
                    <span className={styles.roadmapNumber}>{index + 1}</span>
                    {label}
                  </a>
                </li>
              ))}
            </ol>
          </nav>
        </div>
        <figure className={styles.figure}>
          <svg viewBox="-150 -150 300 300" className={styles.graph} role="group" aria-label={t.hero.figureCaption}>
            <GraphGlyph
              cx={0}
              cy={0}
              radius={100}
              focus={focus}
              vertexRadius={24}
              onVertex={setFocus}
              vertexLabel={(v) => t.hero.agentButton(v)}
            />
          </svg>
          <figcaption>
            <p className={styles.pick}>{t.hero.pickAgent}</p>
            <p className={styles.focus} aria-live="polite">
              {t.hero.focus(focus, list(neighborsOf(ADJ, focus)), list(nonNeighborsOf(ADJ, focus)))}
            </p>
            <p className={styles.legend}>
              <span className="chip chip-friend">
                <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
                  <circle cx="5" cy="5" r="5" fill="var(--friend)" />
                </svg>
                {t.common.friends}
              </span>
              <span className="chip chip-stranger">
                <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
                  <rect width="10" height="10" rx="2" fill="var(--stranger)" />
                </svg>
                {t.common.strangers}
              </span>
              <span className="muted">{t.hero.stats}</span>
            </p>
          </figcaption>
        </figure>
      </div>
    </section>
  );
}
