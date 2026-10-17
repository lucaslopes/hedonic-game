/** The four-agent network in two communities: the site's mark. */
export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg viewBox="0 0 32 32" width={size} height={size} aria-hidden="true" focusable="false">
      <path d="M16 5.5 L25 16 L16 26.5 Z" fill="var(--friend-soft)" stroke="var(--friend)" strokeWidth="1.2" strokeLinejoin="round" />
      <path d="M16 5.5 L7 16" stroke="var(--ink-3)" strokeWidth="1.6" strokeDasharray="2 2" />
      <circle cx="16" cy="5.5" r="3.6" fill="var(--friend)" />
      <circle cx="25" cy="16" r="3.6" fill="var(--friend)" />
      <circle cx="16" cy="26.5" r="3.6" fill="var(--friend)" />
      <circle cx="7" cy="16" r="3.6" fill="var(--stranger)" />
    </svg>
  );
}
