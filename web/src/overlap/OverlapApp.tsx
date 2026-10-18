import { useEffect, useState } from 'react';
import { SiteFooter } from '../components/SiteFooter';
import { SiteHeader } from '../components/SiteHeader';
import { useT } from '../i18n';
import { FRESH, bestResponse, type Cover } from '../model/overlap';
import { ConvergeChapter } from './ConvergeChapter';
import { IntervalsChapter } from './IntervalsChapter';
import type { OverlapSettings } from './OverlapControls';
import { OverlapHero } from './OverlapHero';
import { OverlapTool } from './OverlapTool';
import { GAME, PRESETS, widestRow, type PresetName } from './overlapData';
import type { CoverState } from './SeesChapter';
import { SeesChapter } from './SeesChapter';

/**
 * The overlap explainer. γ, the cap M and the label policy are shared by
 * every chapter; the first two chapters also share one editable cover and
 * the vertex whose view they show.
 */
export function OverlapApp() {
  const t = useT();
  const [gamma, setGamma] = useState(0.5);
  const [cap, setCapValue] = useState(2);
  const [isolation, setIsolation] = useState(false);
  const [cover, setCover] = useState<Cover>(PRESETS.natural);
  const [preset, setPresetName] = useState<PresetName | 'edited'>('natural');
  const [base, setBase] = useState<PresetName>('natural');
  const [focus, setFocus] = useState(2);

  useEffect(() => {
    document.title = t.overlap.title;
  }, [t]);

  // The cap never drops below the widest row of the cover being inspected.
  const effectiveCap = Math.max(cap, widestRow(cover));
  const settings: OverlapSettings = {
    gamma,
    setGamma,
    cap: effectiveCap,
    setCap: setCapValue,
    isolation,
    setIsolation,
  };
  const response = bestResponse(GAME, cover, focus, { gamma, cap: effectiveCap, allowIsolation: isolation });

  const setPreset = (name: PresetName) => {
    setCover(PRESETS[name]);
    setPresetName(name);
    setBase(name);
  };
  const play = () => {
    if (!response.moves) return;
    const used = cover.flat();
    const freshId = Math.max(...used) + 1;
    const row = response.best.labels.map((label) => (label === FRESH ? freshId : label)).sort((a, b) => a - b);
    setCover(cover.map((current, v) => (v === focus ? row : current)));
    setPresetName('edited');
  };
  const state: CoverState = { cover, preset, base, setPreset, focus, setFocus, response, play };

  return (
    <>
      <SiteHeader page="overlap" />
      <main id="main">
        <OverlapHero />
        <SeesChapter settings={settings} state={state} />
        <IntervalsChapter settings={settings} state={state} />
        <ConvergeChapter settings={{ ...settings, cap }} />
        <OverlapTool />
      </main>
      <SiteFooter page="overlap" />
    </>
  );
}
