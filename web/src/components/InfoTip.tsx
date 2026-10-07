import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { Icon } from './Icon';
import styles from './InfoTip.module.css';

interface InfoTipProps {
  /** Accessible name of the trigger button. */
  readonly label: string;
  readonly children: ReactNode;
}

/**
 * A small "i" button with an explanatory popover. It opens on hover, focus
 * or tap, closes on Escape, blur, scroll or a tap elsewhere, and its content
 * is always the button's accessible description.
 */
export function InfoTip({ label, children }: InfoTipProps) {
  const id = useId();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const closeTimer = useRef<number | undefined>(undefined);
  const [anchor, setAnchor] = useState<DOMRect | null>(null);

  const open = useCallback(() => {
    window.clearTimeout(closeTimer.current);
    if (buttonRef.current) setAnchor(buttonRef.current.getBoundingClientRect());
  }, []);
  const close = useCallback(() => {
    window.clearTimeout(closeTimer.current);
    setAnchor(null);
  }, []);
  const closeSoon = useCallback(() => {
    window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => setAnchor(null), 140);
  }, []);

  useEffect(() => {
    if (!anchor) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close();
    };
    const onPointer = (event: PointerEvent) => {
      if (!buttonRef.current?.contains(event.target as Node)) close();
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('pointerdown', onPointer);
    window.addEventListener('scroll', close, { passive: true, capture: true });
    window.addEventListener('resize', close);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onPointer);
      window.removeEventListener('scroll', close, { capture: true });
      window.removeEventListener('resize', close);
    };
  }, [anchor, close]);

  useEffect(() => () => window.clearTimeout(closeTimer.current), []);

  // Position the popover once it has a size; clamp it inside the viewport.
  const place = useCallback(
    (element: HTMLDivElement | null) => {
      if (!element || !anchor) return;
      const { width, height } = element.getBoundingClientRect();
      const margin = 12;
      const left = Math.max(margin, Math.min(anchor.left + anchor.width / 2 - width / 2, window.innerWidth - width - margin));
      let top = anchor.bottom + 8;
      if (top + height > window.innerHeight - margin) top = Math.max(margin, anchor.top - height - 8);
      element.style.left = `${left}px`;
      element.style.top = `${top}px`;
    },
    [anchor],
  );

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        className={styles.trigger}
        aria-label={label}
        aria-describedby={id}
        aria-expanded={anchor !== null}
        onMouseEnter={open}
        onMouseLeave={closeSoon}
        onFocus={open}
        onBlur={close}
        onClick={open}
      >
        <Icon name="info" size={15} />
      </button>
      {createPortal(
        <div
          id={id}
          role="tooltip"
          ref={place}
          hidden={anchor === null}
          className={styles.popover}
          onMouseEnter={() => window.clearTimeout(closeTimer.current)}
          onMouseLeave={closeSoon}
        >
          {children}
        </div>,
        document.body,
      )}
    </>
  );
}
