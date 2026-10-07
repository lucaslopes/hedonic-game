import '@fontsource-variable/inter';
import '@fontsource-variable/source-serif-4';
import '@fontsource-variable/jetbrains-mono';
import 'katex/dist/katex.min.css';
import './styles/tokens.css';
import './styles/base.css';
import './docs/docs.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { SettingsProvider } from './app/settings';
import { DocsApp } from './docs/DocsApp';

const root = document.getElementById('root');
if (!root) throw new Error('Missing #root element');

createRoot(root).render(
  <StrictMode>
    <SettingsProvider>
      <DocsApp />
    </SettingsProvider>
  </StrictMode>,
);
