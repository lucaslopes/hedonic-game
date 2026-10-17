import { createContext, useCallback, useContext, useEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from 'react';

export type Lang = 'en' | 'pt';
export type ThemePreference = 'system' | 'light' | 'dark';
export type MotionPreference = 'system' | 'reduce' | 'allow';

interface Settings {
  readonly lang: Lang;
  readonly setLang: (lang: Lang) => void;
  readonly themePreference: ThemePreference;
  readonly setThemePreference: (value: ThemePreference) => void;
  /** The theme actually applied. */
  readonly theme: 'light' | 'dark';
  readonly motionPreference: MotionPreference;
  readonly setMotionPreference: (value: MotionPreference) => void;
  /** True when animations should be skipped (setting or OS preference). */
  readonly reducedMotion: boolean;
}

const SettingsContext = createContext<Settings | null>(null);

const KEYS = { lang: 'hedonic:lang', theme: 'hedonic:theme', motion: 'hedonic:motion' } as const;

function readStored(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStored(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Storage can be unavailable (private mode, blocked site data); settings then last for the visit.
  }
}

function initialLang(): Lang {
  if (typeof window === 'undefined') return 'en';
  const fromUrl = new URLSearchParams(window.location.search).get('lang');
  if (fromUrl === 'pt' || fromUrl === 'pt-BR') return 'pt';
  if (fromUrl === 'en') return 'en';
  const stored = readStored(KEYS.lang);
  if (stored === 'pt' || stored === 'en') return stored;
  return navigator.language?.toLowerCase().startsWith('pt') ? 'pt' : 'en';
}

function initialChoice<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  const stored = typeof window === 'undefined' ? null : readStored(key);
  return allowed.includes(stored as T) ? (stored as T) : fallback;
}

export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const list = window.matchMedia(query);
      list.addEventListener('change', onChange);
      return () => list.removeEventListener('change', onChange);
    },
    [query],
  );
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => false,
  );
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);
  const [themePreference, setThemeState] = useState<ThemePreference>(() =>
    initialChoice(KEYS.theme, ['system', 'light', 'dark'], 'system'),
  );
  const [motionPreference, setMotionState] = useState<MotionPreference>(() =>
    initialChoice(KEYS.motion, ['system', 'reduce', 'allow'], 'system'),
  );
  const prefersDark = useMediaQuery('(prefers-color-scheme: dark)');
  const prefersReduced = useMediaQuery('(prefers-reduced-motion: reduce)');

  const theme = themePreference === 'system' ? (prefersDark ? 'dark' : 'light') : themePreference;
  const reducedMotion = motionPreference === 'reduce' || (motionPreference === 'system' && prefersReduced);

  useEffect(() => {
    const root = document.documentElement;
    root.lang = lang === 'pt' ? 'pt-BR' : 'en';
    root.dataset.theme = theme;
    if (motionPreference === 'system') delete root.dataset.motion;
    else root.dataset.motion = motionPreference;
  }, [lang, theme, motionPreference]);

  const value = useMemo<Settings>(
    () => ({
      lang,
      setLang: (next) => {
        setLangState(next);
        writeStored(KEYS.lang, next);
      },
      themePreference,
      setThemePreference: (next) => {
        setThemeState(next);
        writeStored(KEYS.theme, next);
      },
      theme,
      motionPreference,
      setMotionPreference: (next) => {
        setMotionState(next);
        writeStored(KEYS.motion, next);
      },
      reducedMotion,
    }),
    [lang, themePreference, theme, motionPreference, reducedMotion],
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings(): Settings {
  const settings = useContext(SettingsContext);
  if (!settings) throw new Error('useSettings must be used inside <SettingsProvider>');
  return settings;
}
