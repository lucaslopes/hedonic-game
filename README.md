# Hedonic

A Python library for hedonic games.

## Installation

```bash
pip install hedonic
```

## Usage

```python
from hedonic import Game

# Create a new game
game = Game("My Hedonic Game")

# Add players
game.add_player("Alice")
game.add_player("Bob")
game.add_player("Charlie")

# Get all players
players = game.get_players()
print(players)  # ['Alice', 'Bob', 'Charlie']

# Print game info
print(game)  # Game: My Hedonic Game with 3 players
```

## Development

This project uses `uv` for dependency management and building.

To set up development environment:
```bash
uv venv
source .venv/bin/activate
uv sync
```

To build the package:
```bash
uv build
```

To publish to PyPI:
```bash
uv publish --token <your-pypi-token>
```

## GitHub Actions Publishing

This repository includes GitHub Actions workflows for automated publishing:

- **Automatic publishing** when you push version tags (e.g., `v0.0.1`)
- **Manual publishing** from the Actions tab
- **Secure authentication** using API tokens (with OIDC support planned)

### Quick Release

1. Update version in `pyproject.toml`
2. Commit and push your changes
3. Create and push a version tag:
   ```bash
   git tag v0.0.1
   git push origin v0.0.1
   ```
4. The workflow will automatically build and publish to TestPyPI

For detailed setup instructions, see [docs/OIDC_SETUP.md](docs/OIDC_SETUP.md).
