// The docs plugin serves this module; DocsApp casts it to DocsBundle
// (plugins/docs-types.ts), since ambient declarations cannot import relatively.
declare module 'virtual:hedonic-docs' {
  const bundle: unknown;
  export default bundle;
}
