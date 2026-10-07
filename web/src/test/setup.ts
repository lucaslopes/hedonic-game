import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => cleanup());

// jsdom does not implement these browser APIs; the components only need
// inert stand-ins during tests.
if (typeof window !== 'undefined') {
  if (!window.matchMedia) {
    window.matchMedia = (query: string) =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        addListener: () => undefined,
        removeListener: () => undefined,
        dispatchEvent: () => false,
      }) as MediaQueryList;
  }
  class InertObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  }
  window.IntersectionObserver ??= InertObserver as unknown as typeof IntersectionObserver;
  window.ResizeObserver ??= InertObserver as unknown as typeof ResizeObserver;
  window.scrollTo ??= () => undefined;
  Element.prototype.scrollIntoView ??= () => undefined;
}
