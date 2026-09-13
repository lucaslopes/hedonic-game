# Hedonic

`hedonic` is a Python library for hedonic community detection on
[`igraph`](https://igraph.org/) graphs. Its `Game` wrapper exposes one API for
both ordinary partitions and overlapping community covers.

The native Leiden implementation is provided by the `lucas-igraph` dependency;
Hedonic 0.1.1 installs `lucas-igraph==1.0.0.4` automatically.

## Installation

The released package supports Python 3.12 and newer:

```bash
python -m pip install hedonic
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv add hedonic
```

## Quick start

```python
import igraph as ig

from hedonic import Game

graph = Game(ig.Graph.Famous("Zachary"))

# Disjoint partition: one community per vertex.
partition = graph.community_hedonic(
    resolution=graph.density(),
    max_memberships=1,
    n_iterations=-1,
)
print(partition.membership)

# Overlapping cover: a vertex may belong to up to four communities.
cover = graph.community_hedonic(
    resolution=graph.density(),
    max_memberships=4,
    n_iterations=-1,
)
print(cover)
```

`community_hedonic` returns an `igraph.VertexClustering` when
`max_memberships=1` and an `igraph.VertexCover` when the cap is greater than
one. Overlapping warm starts accept a flat or per-vertex nested
`initial_membership`; disjoint warm starts use flat labels.

The native no-change stop is not a standalone mathematical certificate.
Audit the returned memberships independently when a certificate is required.

Useful options include:

- `local_move_only=True` for the hedonic local-moving phase, or `False` for
  the full Leiden refinement and aggregation pipeline;
- `n_iterations=-1` (the default) to run until the native no-change stopping
  condition;
- `allow_isolation`, `edge_weights`, `seed`, and `beta` to control the model;
- `resolution` to select the CPM resolution parameter.

For reproducible overlapping experiments, use an undirected, loopless graph
with finite weights and explicitly record the resolution, membership cap,
initialization, and seed used for each run.

## Experiments

Install the optional experiment dependencies to use the experiment drivers:

```bash
python -m pip install "hedonic[experiments]"
hedonic-exp list
hedonic-exp smoke
```

The `hedonic-exp` commands cover small smoke checks, synthetic disjoint
experiments, overlapping diagnostics, and reproducible benchmark pipelines.
Experiment data paths can be configured with the TOML files under `configs/`
or with the documented `HEDONIC_*_DIR` environment variables.

## Development

```bash
git clone https://github.com/lucaslopes/hedonic-game.git
cd hedonic-game
uv sync --extra experiments
uv run --with pytest pytest -q
uv build --no-sources
```

The public API is intentionally small:

```python
from hedonic import Game
```

Research manuscripts, private evidence, and manuscript-only configuration are
kept outside the public `main` publication boundary.

## Releases and publishing

Hedonic versions are independent from the four-component release identity of
the native `lucas-igraph` dependency. A dependency update can therefore be a
normal Hedonic patch release without changing the `community_hedonic` call
pattern.

Pushes to public `main` run tests and save wheel/sdist artifacts. After that
exact commit passes, a matching version tag publishes those saved artifacts
using `scripts/release.sh`; publication does not rebuild the package. See
[the release guide](https://github.com/lucaslopes/hedonic-game/blob/main/docs/releasing.md). Credentials must be supplied through the configured secret or a
hidden interactive environment variable; never put a PyPI token directly in a
shell command, README, commit, or issue.

## License

This project is distributed under the GNU General Public License, version 3 or
later. See [LICENSE](LICENSE) for the complete terms.
