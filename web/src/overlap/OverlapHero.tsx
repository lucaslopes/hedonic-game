import { useState } from 'react';
import { Icon } from '../components/Icon';
import { useT } from '../i18n';
import styles from '../story/Hero.module.css';
import { CoverGraph } from './CoverGraph';
import { GAME, PRESETS, rowText } from './overlapData';
import local from './Overlap.module.css';

export function OverlapHero() {
  const t = useT();
  const [focus, setFocus] = useState(2);
  const cover = PRESETS.natural;
  const friends = GAME.neighbors[focus].slice().sort((a, b) => a - b);
  const strangers = cover.map((_, v) => v).filter((v) => v !== focus && !GAME.adj[focus][v]);
  const list = (items: number[]) => (items.length ? items.join(', ') : t.hero.nobody);
  const chapters: [string, string][] = [
    ['#sees', t.overlap.chapters.sees],
    ['#intervals', t.overlap.chapters.intervals],
    ['#converge', t.overlap.chapters.converge],
    ['#tool', t.overlap.chapters.tool],
  ];
  return (
    <section id="top" className={styles.hero} aria-labelledby="hero-title">
      <div className={`container ${styles.grid}`}>
        <div className={styles.copy}>
          <p className="kicker">{t.overlap.hero.kicker}</p>
          <h1 id="hero-title" className={local.heroTitle}>
            {t.overlap.hero.title}
          </h1>
          <p className="lede">{t.overlap.hero.lede}</p>
          <div className={styles.actions}>
            <a className="btn btn-primary" href="#sees">
              {t.overlap.hero.cta}
              <Icon name="arrowDown" size={16} />
            </a>
            <a className="btn" href={import.meta.env.BASE_URL}>
              {t.overlap.hero.disjointCta}
              <Icon name="arrowRight" size={16} />
            </a>
          </div>
          <nav className={styles.roadmap} aria-label={t.overlap.hero.roadmap}>
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
          <div className={local.heroGraph}>
            <CoverGraph cover={cover} focus={focus} onVertex={setFocus} />
          </div>
          <figcaption>
            <p className={styles.pick}>{t.overlap.hero.pickAgent}</p>
            <p className={styles.focus} aria-live="polite">
              {t.overlap.hero.focus(focus, rowText(cover[focus]), list(friends), list(strangers))}
            </p>
            <p className={styles.legend}>
              <span className="chip chip-friend">{t.common.friends}</span>
              <span className="chip chip-stranger">{t.common.strangers}</span>
              <span className="muted">{t.overlap.hero.stats}</span>
            </p>
            <p className={local.caption}>{t.overlap.hero.figureCaption}</p>
          </figcaption>
        </figure>
      </div>
    </section>
  );
}
