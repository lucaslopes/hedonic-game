#!/usr/bin/env node
/**
 * `npm run docs:generate`
 *
 * 1. Writes the repository URL and source reference used by the header, the
 *    footer and the "edit this page" links → src/generated/site.json.
 * 2. Converts web/content/references.bib → src/generated/references.json.
 *
 * Both outputs are committed so `npm run dev` works without a generation
 * step, and are regenerated in CI before every Pages build. Re-run this
 * script whenever the reference list changes.
 *
 * Environment:
 *   GITHUB_REPOSITORY / GITHUB_SHA  set by GitHub Actions; used to pin links
 *                                   to the exact public commit.
 *   HEDONIC_DOCS_REF                override the git ref used in links.
 */

import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseBibtex, toReferences } from './bib.mjs';

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const generated = resolve(webRoot, 'src/generated');
mkdirSync(generated, { recursive: true });

function repositoryUrl() {
  if (process.env.GITHUB_REPOSITORY) return `https://github.com/${process.env.GITHUB_REPOSITORY}`;
  const remote = spawnSync('git', ['-C', webRoot, 'remote', 'get-url', 'origin'], { encoding: 'utf8' });
  const match = remote.status === 0 ? /github\.com[:/](.+?)(?:\.git)?$/.exec(remote.stdout.trim()) : null;
  return match ? `https://github.com/${match[1]}` : 'https://github.com/lucaslopes/hedonic-game';
}

const site = {
  repositoryUrl: repositoryUrl(),
  sourceRef: process.env.HEDONIC_DOCS_REF || process.env.GITHUB_SHA || 'main',
};
writeFileSync(resolve(generated, 'site.json'), `${JSON.stringify(site, null, 2)}\n`);
console.log('site: wrote web/src/generated/site.json');

const bibPath = resolve(webRoot, 'content/references.bib');
const references = toReferences(parseBibtex(readFileSync(bibPath, 'utf8')));
writeFileSync(resolve(generated, 'references.json'), `${JSON.stringify({ schemaVersion: 1, source: 'web/content/references.bib', references }, null, 2)}\n`);
console.log(`references: wrote web/src/generated/references.json (${references.length} entries)`);
