/** Shapes shared by the Markdown docs plugin and the docs page. */

export interface DocHeading {
  readonly id: string;
  readonly text: string;
  readonly level: 2 | 3;
}

export interface DocPage {
  /** Anchor of the page section, e.g. "game-api". */
  readonly slug: string;
  readonly title: string;
  readonly summary: string;
  readonly group: string;
  readonly order: number;
  /** Rendered HTML (math pre-rendered with KaTeX at build time). */
  readonly html: string;
  readonly headings: readonly DocHeading[];
  /** Repository path of the Markdown source, for "edit this page" links. */
  readonly sourcePath: string;
}

export interface DocsBundle {
  readonly pages: readonly DocPage[];
  readonly repository: { readonly url: string; readonly ref: string };
  readonly packageVersion: string | null;
  readonly nativeDependency: string | null;
}
