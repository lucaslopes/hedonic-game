import { useEffect, useState } from 'react';
import { SiteFooter } from '../components/SiteFooter';
import { SiteHeader } from '../components/SiteHeader';
import { useT } from '../i18n';
import { DEFAULT_COUNTS, type Counts } from './dilemmaShared';
import { DilemmaChapter } from './DilemmaChapter';
import { ExploreSection } from './ExploreSection';
import { Hero } from './Hero';
import { MetagraphChapter } from './MetagraphChapter';
import { ToolSection } from './ToolSection';
import { TreeChapter } from './TreeChapter';
import { WalkChapter } from './WalkChapter';

/**
 * The explainer. One resolution γ is shared by every section so the whole
 * story responds to the same dial; the dilemma counts are shared by the
 * balance and the decision tree.
 */
export function StoryApp() {
  const t = useT();
  const [gamma, setGamma] = useState(0.5);
  const [counts, setCounts] = useState<Counts>(DEFAULT_COUNTS);

  useEffect(() => {
    document.title = t.meta.title;
  }, [t]);

  return (
    <>
      <SiteHeader page="story" />
      <main id="main">
        <Hero />
        <DilemmaChapter counts={counts} setCounts={setCounts} gamma={gamma} setGamma={setGamma} />
        <TreeChapter counts={counts} setCounts={setCounts} gamma={gamma} setGamma={setGamma} />
        <MetagraphChapter gamma={gamma} setGamma={setGamma} />
        <WalkChapter gamma={gamma} setGamma={setGamma} />
        <ExploreSection gamma={gamma} setGamma={setGamma} />
        <ToolSection />
      </main>
      <SiteFooter />
    </>
  );
}
