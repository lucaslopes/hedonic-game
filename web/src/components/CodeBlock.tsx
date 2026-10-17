import { useState } from 'react';
import { useT } from '../i18n';
import { Icon } from './Icon';
import styles from './CodeBlock.module.css';

export function CodeBlock({ code, language, label }: { code: string; language: string; label?: string }) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };
  return (
    <div className={styles.block}>
      <div className={styles.bar}>
        <span className={styles.lang}>{label ?? language}</span>
        <button type="button" className="btn btn-small" onClick={copy} aria-live="polite">
          <Icon name={copied ? 'check' : 'copy'} size={15} />
          {copied ? t.common.copied : t.common.copy}
        </button>
      </div>
      <pre className={styles.pre}>
        <code className={`language-${language}`}>{code}</code>
      </pre>
    </div>
  );
}
