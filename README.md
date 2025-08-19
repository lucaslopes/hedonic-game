# Hedonic

A Python library for hedonic games.

## Installation

```bash
pip install hedonic
```

## Usage

```python
import igraph as ig 
from hedonic import Game

g = ig.Graph.Famous("Zachary")  # sample graph: Zachary's karate club
h = Game(g)
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

- **TestPyPI workflow**: Currently active, publishes to TestPyPI on `v*` tags
- **PyPI workflow**: Currently disabled (can be re-enabled later)
- **Automatic publishing** when you push version tags
- **Manual publishing** from the Actions tab
- **Secure authentication** using OIDC for TestPyPI

### Quick Release

```bash
# Bump version and prepare release
./scripts/release.sh patch  # or minor/major

# Push everything (triggers TestPyPI workflow)
git push origin main && git push origin v0.0.2
```

### **Version Types**
- `patch`: `0.0.1` → `0.0.2` (bug fixes, small changes)
- `minor`: `0.0.1` → `0.1.0` (new features, backward compatible)  
- `major`: `0.0.1` → `1.0.0` (breaking changes)

### **Current Status**
- **TestPyPI**: ✅ Active - publishes on `v*` tags
- **PyPI**: ❌ Disabled - workflow file is commented out

### **Enabling PyPI Publishing Later**
When you're ready to publish to PyPI:
1. Rename `.github/workflows/publish-pypi.yml.disabled` to `publish-pypi.yml`
2. Add `PYPI_API_TOKEN` secret to your GitHub repository
3. Both workflows will then be active with different tag patterns

For detailed setup instructions, see [docs/OIDC_SETUP.md](docs/OIDC_SETUP.md).
