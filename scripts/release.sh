#!/bin/bash

# Release script for hedonic package
# Usage: ./scripts/release.sh [patch|minor|major]

set -e

# Check if version type is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [patch|minor|major]"
    echo "  patch: 0.0.1 -> 0.0.2"
    echo "  minor: 0.1.0 -> 0.2.0"
    echo "  major: 1.0.0 -> 2.0.0"
    echo ""
    echo "Note: Currently only TestPyPI publishing is enabled"
    echo "PyPI workflow is disabled and can be re-enabled later"
    exit 1
fi

VERSION_TYPE=$1

# Validate version type
if [[ ! "$VERSION_TYPE" =~ ^(patch|minor|major)$ ]]; then
    echo "Error: Version type must be patch, minor, or major"
    exit 1
fi

echo "Releasing $VERSION_TYPE version to TestPyPI..."

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
echo "Current version: $CURRENT_VERSION"

# Parse version components
IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

# Calculate new version
case $VERSION_TYPE in
    patch)
        NEW_PATCH=$((PATCH + 1))
        NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"
        ;;
    minor)
        NEW_MINOR=$((MINOR + 1))
        NEW_VERSION="$MAJOR.$NEW_MINOR.0"
        ;;
    major)
        NEW_MAJOR=$((MAJOR + 1))
        NEW_VERSION="$NEW_MAJOR.0.0"
        ;;
esac

echo "New version: $NEW_VERSION"

# Update pyproject.toml
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
else
    # Linux
    sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
fi

echo "Updated pyproject.toml"

# Build the package
echo "Building package..."
uv build

# Commit changes
git add pyproject.toml
git commit -m "Bump version to $NEW_VERSION"

# Create standard version tag
TAG="v$NEW_VERSION"
git tag "$TAG"
echo "Created tag: $TAG"

echo ""
echo "Release $NEW_VERSION prepared for TestPyPI!"
echo ""
echo "Next steps:"
echo "1. Review changes: git log --oneline -5"
echo "2. Push changes: git push origin main"
echo "3. Push tag: git push origin $TAG"
echo "4. Check GitHub Actions for automated publishing to TestPyPI"
echo ""
echo "Or push everything at once:"
echo "git push origin main && git push origin $TAG"
echo ""
echo "Tag format: $TAG"
echo "This will trigger: TestPyPI workflow only"
echo ""
echo "Note: PyPI workflow is currently disabled"
echo "To enable PyPI publishing later, uncomment .github/workflows/publish-pypi.yml.disabled"
