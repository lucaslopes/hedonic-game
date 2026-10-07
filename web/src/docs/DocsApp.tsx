import rawBundle from 'virtual:hedonic-docs';
import type { DocsBundle } from '../../plugins/docs-types';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useSettings } from '../app/settings';
import { Icon } from '../components/Icon';
import { Rich } from '../components/Rich';
import { SiteFooter } from '../components/SiteFooter';
import { SiteHeader } from '../components/SiteHeader';
import { useT } from '../i18n';
import styles from './Docs.module.css';

const bundle = rawBundle as DocsBundle;

/** Add copy buttons to the code blocks rendered from Markdown. */
function useCopyButtons(root: React.RefObject<HTMLElement | null>, labels: { copy: string; copied: string }) {
  useEffect(() => {
    const element = root.current;
    if (!element) return undefined;
    const buttons: HTMLButtonElement[] = [];
    element.querySelectorAll('pre:not(.api-signature)').forEach((pre) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'btn btn-small doc-copy';
      button.textContent = labels.copy;
      button.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(pre.querySelector('code')?.textContent ?? pre.textContent ?? '');
          button.textContent = labels.copied;
          window.setTimeout(() => (button.textContent = labels.copy), 1500);
        } catch {
          /* clipboard unavailable */
        }
      });
      pre.classList.add('doc-pre');
      pre.appendChild(button);
      buttons.push(button);
    });
    return () => buttons.forEach((button) => button.remove());
  }, [root, labels.copy, labels.copied]);
}

export function DocsApp() {
  const t = useT();
  const { lang, reducedMotion } = useSettings();
  const contentRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(bundle.pages[0]?.slug ?? '');

  useEffect(() => {
    document.title = t.meta.docsTitle;
  }, [t]);

  useCopyButtons(contentRef, { copy: t.common.copy, copied: t.common.copied });

  // Scrollspy: the page whose section crosses 30% of the viewport is active.
  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const line = window.innerHeight * 0.3;
      let current = bundle.pages[0]?.slug ?? '';
      for (const page of bundle.pages) {
        const element = document.getElementById(page.slug);
        if (element && element.getBoundingClientRect().top <= line) current = page.slug;
      }
      setActive((previous) => (previous === current ? previous : current));
    };
    const schedule = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };
    schedule();
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    return () => {
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
      window.cancelAnimationFrame(frame);
    };
  }, []);

  // Honour a deep link once the content exists.
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1));
    if (id) document.getElementById(id)?.scrollIntoView({ behavior: 'auto' });
  }, []);

  const groups = useMemo(() => {
    const map = new Map<string, typeof bundle.pages>();
    for (const page of bundle.pages) map.set(page.group, [...(map.get(page.group) ?? []), page]);
    return [...map.entries()];
  }, []);

  const activePage = bundle.pages.find((page) => page.slug === active);
  const editUrl = (path: string) => `${bundle.repository.url}/blob/${bundle.repository.ref}/${path}`;

  const nav = (
    <nav aria-label={t.docs.pages}>
      {groups.map(([group, pages]) => (
        <div key={group} className={styles.navGroup}>
          <p className={styles.navGroupTitle}>{group}</p>
          <ul>
            {pages.map((page) => (
              <li key={page.slug}>
                <a href={`#${page.slug}`} aria-current={page.slug === active ? 'location' : undefined}>
                  {page.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );

  return (
    <>
      <SiteHeader page="docs" />
      <main id="main" className={`container ${styles.layout}`}>
        <aside className={styles.sidebar}>
          <details className={styles.mobileNav}>
            <summary>
              <Icon name="menu" size={16} /> {t.docs.menu}
            </summary>
            {nav}
          </details>
          <div className={styles.desktopNav}>{nav}</div>
        </aside>
        <div className={styles.content} ref={contentRef}>
          <header className={styles.intro}>
            <p className="kicker">hedonic</p>
            <h1>{t.docs.title}</h1>
            <Rich as="p" className="lede" text={t.docs.intro} />
            <p className={styles.meta}>
              {bundle.packageVersion && (
                <span className="chip">{t.docs.version(bundle.packageVersion, bundle.nativeDependency ?? '')}</span>
              )}
              <a className="btn btn-small" href={import.meta.env.BASE_URL}>
                <Icon name="arrowRight" size={15} style={{ transform: 'rotate(180deg)' }} />
                {t.docs.back}
              </a>
            </p>
            {lang === 'pt' && t.docs.languageNote && (
              <p className={styles.langNote} lang="pt-BR">
                {t.docs.languageNote}
              </p>
            )}
          </header>
          <div lang="en">
            {bundle.pages.map((page) => (
              <section key={page.slug} id={page.slug} className={styles.page} aria-labelledby={`${page.slug}-title`}>
                <h2 id={`${page.slug}-title`} className={styles.pageTitle}>
                  <a href={`#${page.slug}`} className={styles.anchor}>
                    {page.title}
                  </a>
                </h2>
                {page.summary && <p className={styles.summary}>{page.summary}</p>}
                <div className="doc-prose" dangerouslySetInnerHTML={{ __html: page.html }} />
                <p className={styles.edit}>
                  <a href={editUrl(page.sourcePath)}>
                    {t.docs.edit} <Icon name="external" size={14} style={{ display: 'inline' }} />
                  </a>
                </p>
              </section>
            ))}
          </div>
          <p className={styles.generated}>{t.docs.generated(`${bundle.repository.ref}`)}</p>
        </div>
        <aside className={styles.toc} aria-label={t.docs.onThisPage}>
          {activePage && activePage.headings.length > 0 && (
            <>
              <p className={styles.navGroupTitle}>{t.docs.onThisPage}</p>
              <ul>
                {activePage.headings.map((heading) => (
                  <li key={heading.id} className={heading.level === 3 ? styles.tocSub : undefined}>
                    <a
                      href={`#${heading.id}`}
                      onClick={(event) => {
                        const target = document.getElementById(heading.id);
                        if (!target) return;
                        event.preventDefault();
                        target.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth' });
                        history.replaceState(null, '', `#${heading.id}`);
                      }}
                    >
                      {heading.text}
                    </a>
                  </li>
                ))}
              </ul>
            </>
          )}
        </aside>
      </main>
      <SiteFooter />
    </>
  );
}
