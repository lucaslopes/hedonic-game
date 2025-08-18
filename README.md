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
