"""Private implementation modules backing :mod:`hedonic.Game`.

The public entry point intentionally remains ``hedonic.Game``.  Keeping these
implementation details in a package gives the facade a stable import path
while allowing the graph wrapper, membership state, metrics bridge, and
detector orchestration to evolve independently.
"""
