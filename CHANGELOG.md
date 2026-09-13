# Changelog

## 0.1.1

- Update the native dependency to `lucas-igraph==1.0.0.4` and regenerate the
  dependency lock. The `Game` public API remains unchanged.
- Refresh the package README with installation and disjoint/overlapping examples.
- Test and build on public `main`; publish the exact successful Actions artifacts
  after tagging, with metadata, commit identity, and index hash verification.
- Preserve historical protocol locks and evidence identities at their original
  versions; a runtime dependency update does not reinterpret saved results.
