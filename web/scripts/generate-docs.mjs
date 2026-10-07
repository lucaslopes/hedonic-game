#!/usr/bin/env node
/**
 * `npm run docs:generate`
 *
 * 1. Extracts the public Python API (Game, community_hedonic, utils, the
 *    experiments CLI registry and the disjoint experiment presets) from the
 *    source files with web/scripts/extract_api.py → src/generated/api.json.
 * 2. Converts web/content/references.bib → src/generated/references.json.
 *
 * Both outputs are committed so `npm run dev` works without Python, and are
 * regenerated in CI before every Pages build. Re-run this script whenever the
 * Python API or the reference list changes.
 */

import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseBibtex, toReferences } from './bib.mjs';

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const generated = resolve(webRoot, 'src/generated');
mkdirSync(generated, { recursive: true });

function findPython() {
  const candidates = [process.env.PYTHON, 'python3', 'python'].filter(Boolean);
  for (const candidate of candidates) {
    const probe = spawnSync(candidate, ['-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'], { stdio: 'ignore' });
    if (probe.status === 0) return candidate;
  }
  return null;
}

const python = findPython();
if (!python) {
  console.error('docs:generate needs Python >= 3.9 on PATH (or set PYTHON=/path/to/python3).');
  process.exit(1);
}
const api = spawnSync(python, [resolve(webRoot, 'scripts/extract_api.py'), '--output', resolve(generated, 'api.json')], {
  stdio: 'inherit',
});
if (api.status !== 0) process.exit(api.status ?? 1);

// Small subset for the story page, so it does not bundle the whole API file.
const apiData = JSON.parse(readFileSync(resolve(generated, 'api.json'), 'utf8'));
const site = {
  repositoryUrl: apiData.repository.url,
  sourceRef: apiData.repository.ref,
  packageVersion: apiData.package.version,
  nativeDependency: apiData.package.nativeDependency,
  requiresPython: apiData.package.requiresPython,
};
writeFileSync(resolve(generated, 'site.json'), `${JSON.stringify(site, null, 2)}\n`);
console.log('site: wrote web/src/generated/site.json');

const bibPath = resolve(webRoot, 'content/references.bib');
const references = toReferences(parseBibtex(readFileSync(bibPath, 'utf8')));
writeFileSync(resolve(generated, 'references.json'), `${JSON.stringify({ schemaVersion: 1, source: 'web/content/references.bib', references }, null, 2)}\n`);
console.log(`references: wrote web/src/generated/references.json (${references.length} entries)`);
