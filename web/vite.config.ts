/// <reference types="vitest/config" />
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { markdownDocs } from './plugins/markdown-docs.ts';

const root = fileURLToPath(new URL('.', import.meta.url));

/**
 * `BASE_PATH` is the public URL prefix. It defaults to "/" for local work;
 * the GitHub Pages workflow sets it to "/<repository>/" (from
 * actions/configure-pages), so every asset URL and internal link resolves
 * under the repository sub-path.
 */
function basePath(): string {
  const raw = process.env.BASE_PATH?.trim();
  if (!raw) return '/';
  const withLeading = raw.startsWith('/') ? raw : `/${raw}`;
  return withLeading.endsWith('/') ? withLeading : `${withLeading}/`;
}

export default defineConfig({
  base: basePath(),
  plugins: [react(), markdownDocs({ contentDir: resolve(root, 'content/docs'), apiFile: resolve(root, 'src/generated/api.json'), referencesFile: resolve(root, 'src/generated/references.json') })],
  build: {
    target: 'es2022',
    rolldownOptions: {
      input: {
        main: resolve(root, 'index.html'),
        docs: resolve(root, 'docs/index.html'),
      },
      output: {
        // Long-lived vendor chunks cache well across deployments of the content.
        codeSplitting: {
          groups: [
            { name: 'react', test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/ },
            { name: 'katex', test: /node_modules[\\/]katex[\\/]/ },
            { name: 'd3', test: /node_modules[\\/]d3-[a-z]+[\\/]/ },
          ],
        },
      },
    },
  },
  server: {
    port: 5173,
  },
  test: {
    environment: 'node', // component tests opt into jsdom with a docblock
    include: ['src/**/*.test.{ts,tsx}', 'plugins/**/*.test.ts', 'scripts/**/*.test.{js,mjs,ts}'],
    setupFiles: ['src/test/setup.ts'],
    css: false,
  },
});
