import { useSettings } from '../app/settings';
import { en, type Dictionary } from './en';
import { pt } from './pt';

export type { Dictionary };

export const dictionaries: Record<'en' | 'pt', Dictionary> = { en, pt };

/** The dictionary of the active language. */
export function useT(): Dictionary {
  return dictionaries[useSettings().lang];
}
