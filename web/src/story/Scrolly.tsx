import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Icon } from '../components/Icon';
import { Rich } from '../components/Rich';
import { useSettings } from '../app/settings';
import { useT } from '../i18n';
import styles from './Scrolly.module.css';

export interface ScrollyStep {
  readonly title: string;
  readonly body: string;
  /** Extra interactive content shown inside the step card. */
  readonly extra?: ReactNode;
}

interface ScrollyProps {
  readonly id: string;
  readonly kicker: string;
  readonly title: string;
  readonly steps: readonly ScrollyStep[];
  readonly stage: (active: number) => ReactNode;
  readonly stageLabel: string;
  readonly next?: { readonly href: string; readonly label: string };
  readonly onActiveChange?: (active: number) => void;
}

/**
 * Scrollytelling: a sticky stage beside (desktop) or above (mobile) a column
 * of step cards. The step crossing the trigger line drives the stage.
 */
export function Scrolly({ id, kicker, title, steps, stage, stageLabel, next, onActiveChange }: ScrollyProps) {
  const t = useT();
  const { reducedMotion } = useSettings();
  const [active, setActive] = useState(0);
  const stageRef = useRef<HTMLDivElement>(null);
  const stepRefs = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const viewport = window.innerHeight;
      const stageRect = stageRef.current?.getBoundingClientRect();
      const stacked = window.matchMedia('(max-width: 999px)').matches;
      const stageBottom = stacked && stageRect ? Math.max(0, stageRect.bottom) : 0;
      const trigger = stacked ? stageBottom + (viewport - stageBottom) * 0.4 : viewport * 0.52;
      let current = 0;
      stepRefs.current.forEach((element, index) => {
        if (element && element.getBoundingClientRect().top <= trigger) current = index;
      });
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
  }, [steps.length]);

  useEffect(() => {
    onActiveChange?.(active);
  }, [active, onActiveChange]);

  const goTo = (index: number) => {
    const target = stepRefs.current[index];
    if (!target) return;
    target.scrollIntoView({ block: 'center', behavior: reducedMotion ? 'auto' : 'smooth' });
    target.querySelector<HTMLElement>('h3')?.focus({ preventScroll: true });
  };

  return (
    <section id={id} className={styles.section} aria-labelledby={`${id}-title`}>
      <div className="container">
        <header className={styles.header}>
          <p className="kicker">{kicker}</p>
          <h2 id={`${id}-title`}>{title}</h2>
        </header>
        <div className={styles.grid}>
          <div ref={stageRef} className={styles.stage} role="region" aria-label={stageLabel}>
            <div className={styles.stageInner}>{stage(active)}</div>
          </div>
          <ol className={styles.steps}>
            {steps.map((step, index) => (
              <li
                key={step.title}
                ref={(element) => {
                  stepRefs.current[index] = element;
                }}
                className={styles.step}
                data-active={index === active}
              >
                <article className={styles.card} aria-current={index === active ? 'step' : undefined}>
                  <p className={styles.counter} aria-hidden="true">
                    {index + 1} / {steps.length}
                  </p>
                  <h3 tabIndex={-1}>{step.title}</h3>
                  <Rich as="p" text={step.body} />
                  {step.extra}
                  {index < steps.length - 1 ? (
                    <button type="button" className={`btn btn-small ${styles.next}`} onClick={() => goTo(index + 1)}>
                      {t.common.next}
                      <Icon name="arrowDown" size={16} />
                    </button>
                  ) : (
                    next && (
                      <a className={`btn btn-small ${styles.next}`} href={next.href}>
                        {t.common.continueTo(next.label)}
                        <Icon name="arrowDown" size={16} />
                      </a>
                    )
                  )}
                </article>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}
