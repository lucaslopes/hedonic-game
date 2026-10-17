import site from '../generated/site.json';
import { useT } from '../i18n';
import { LogoMark } from './LogoMark';
import { Rich } from './Rich';
import styles from './SiteFooter.module.css';

export function SiteFooter() {
  const t = useT();
  const base = import.meta.env.BASE_URL;
  return (
    <footer className={styles.footer}>
      <div className={`container ${styles.grid}`}>
        <div>
          <p className={styles.brand}>
            <LogoMark size={26} /> {t.footer.tagline}
          </p>
          <p className={styles.small}>{t.footer.basedOn}</p>
          <Rich as="p" className={styles.small} text={t.footer.disclaimer} />
          <p className={styles.small}>{t.footer.license}</p>
        </div>
        <nav aria-label={t.footer.links} className={styles.links}>
          <a href={`${base}#dilemma`}>{t.header.story}</a>
          <a href={`${base}#explore`}>{t.header.explore}</a>
          <a href={`${base}docs/`}>{t.header.docs}</a>
          <a href={`${base}docs/#references`}>{t.tool.cards.references.title}</a>
          <a href={site.repositoryUrl}>GitHub</a>
          <a href="https://arxiv.org/abs/2509.03834">arXiv:2509.03834</a>
        </nav>
      </div>
    </footer>
  );
}
