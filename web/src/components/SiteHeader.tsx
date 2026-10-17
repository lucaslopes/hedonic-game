import { useEffect, useRef } from 'react';
import site from '../generated/site.json';
import { useSettings, type MotionPreference, type ThemePreference } from '../app/settings';
import { useT } from '../i18n';
import { Icon } from './Icon';
import { LogoMark } from './LogoMark';
import styles from './SiteHeader.module.css';

const THEME_ORDER: ThemePreference[] = ['system', 'light', 'dark'];
const MOTION_ORDER: MotionPreference[] = ['system', 'reduce', 'allow'];

export function SiteHeader({ page }: { page: 'story' | 'docs' }) {
  const t = useT();
  const settings = useSettings();
  const progress = useRef<HTMLDivElement>(null);
  const base = import.meta.env.BASE_URL;

  useEffect(() => {
    if (page !== 'story') return undefined;
    let frame = 0;
    const update = () => {
      frame = 0;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const value = max > 0 ? Math.min(1, window.scrollY / max) : 0;
      progress.current?.style.setProperty('--progress', String(value));
    };
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };
    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      window.cancelAnimationFrame(frame);
    };
  }, [page]);

  const themeLabel = {
    system: t.header.themeSystem,
    light: t.header.themeLight,
    dark: t.header.themeDark,
  }[settings.themePreference];
  const motionLabel = {
    system: t.header.motionSystem,
    reduce: t.header.motionReduce,
    allow: t.header.motionAllow,
  }[settings.motionPreference];
  const cycle = <T,>(order: T[], current: T) => order[(order.indexOf(current) + 1) % order.length];

  return (
    <header className={styles.header}>
      <a className="skip-link" href="#main">
        {t.a11y.skip}
      </a>
      <div className={styles.inner}>
        <a className={styles.brand} href={page === 'story' ? '#top' : base} aria-label={t.header.home}>
          <LogoMark size={30} />
          <span className={styles.brandText}>Hedonic&nbsp;Game</span>
        </a>
        <nav className={styles.nav} aria-label="Primary">
          <a href={page === 'story' ? '#dilemma' : `${base}#dilemma`} className={styles.link} aria-current={page === 'story' ? 'page' : undefined}>
            {t.header.story}
          </a>
          <a href={page === 'story' ? '#explore' : `${base}#explore`} className={`${styles.link} ${styles.optional}`}>
            {t.header.explore}
          </a>
          <a href={`${base}docs/`} className={styles.link} aria-current={page === 'docs' ? 'page' : undefined}>
            {t.header.docs}
          </a>
        </nav>
        <div className={styles.tools}>
          <button
            type="button"
            className={`btn btn-small ${styles.lang}`}
            onClick={() => settings.setLang(settings.lang === 'en' ? 'pt' : 'en')}
            aria-label={t.header.switchLanguage}
            title={t.header.switchLanguage}
            lang={settings.lang === 'en' ? 'pt-BR' : 'en'}
          >
            {t.header.languageShort}
          </button>
          <button
            type="button"
            className="btn btn-small btn-icon"
            onClick={() => settings.setThemePreference(cycle(THEME_ORDER, settings.themePreference))}
            aria-label={themeLabel}
            title={themeLabel}
          >
            <Icon name={settings.themePreference === 'system' ? 'auto' : settings.themePreference === 'dark' ? 'moon' : 'sun'} />
          </button>
          <button
            type="button"
            className="btn btn-small btn-icon"
            onClick={() => settings.setMotionPreference(cycle(MOTION_ORDER, settings.motionPreference))}
            aria-label={motionLabel}
            title={motionLabel}
          >
            <Icon name={settings.reducedMotion ? 'still' : 'motion'} />
          </button>
          <a className={`btn btn-small btn-icon ${styles.optional}`} href={site.repositoryUrl} aria-label={t.header.github} title={t.header.github}>
            <Icon name="github" />
          </a>
        </div>
      </div>
      {page === 'story' && <div ref={progress} className={styles.progress} aria-hidden="true" />}
    </header>
  );
}
