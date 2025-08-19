# Setting up OpenID Connect (OIDC) for PyPI Publishing

This guide explains how to set up secure authentication for publishing your Python packages to TestPyPI and PyPI using GitHub Actions with OpenID Connect (OIDC).

## What is OIDC?

OpenID Connect allows your GitHub Actions workflows to exchange short-lived tokens directly from PyPI, eliminating the need to store long-lived API tokens as secrets. This is a much more secure approach to authentication.

## Current Status

**Important**: Currently, only TestPyPI publishing is enabled and uses OIDC authentication. PyPI publishing is disabled but can be easily re-enabled later.

- **TestPyPI**: ✅ Active with OIDC (no secrets required)
- **PyPI**: ❌ Disabled (workflow file is commented out)

## Simplified Setup (Current)

### **TestPyPI Publishing (Active)**
- **Triggered by**: Standard version tags (e.g., `v0.0.1`, `v1.0.0`)
- **Workflow**: `.github/workflows/publish.yml`
- **Authentication**: OIDC (no secrets required)
- **Use case**: Testing releases before production

### **PyPI Publishing (Disabled)**
- **Status**: Currently disabled
- **File**: `.github/workflows/publish-pypi.yml.disabled`
- **To enable**: Rename file and add API token secret

## Current Setup

### 1. No Secrets Required for TestPyPI

Since TestPyPI uses OIDC, you don't need to add any secrets to GitHub. The workflow will work automatically.

### 2. Trigger Publishing

#### **TestPyPI Release (Current)**
```bash
./scripts/release.sh patch
git push origin main && git push origin v0.0.1
```

This creates a standard `v0.0.1` tag and triggers the TestPyPI workflow.

## Enabling PyPI Publishing Later

When you're ready to publish to PyPI:

### 1. Re-enable the PyPI Workflow
```bash
# Rename the disabled workflow file
mv .github/workflows/publish-pypi.yml.disabled .github/workflows/publish-pypi.yml
```

### 2. Get PyPI API Token
1. Go to [pypi.org](https://pypi.org) and create an account
2. Generate an API token in your account settings

### 3. Add Secret to GitHub
In your GitHub repository:
1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add: `PYPI_API_TOKEN` with your PyPI API token

### 4. Configure Tag Patterns
The PyPI workflow will then be active and you can use:
- **TestPyPI**: `test-v*` tags (e.g., `test-v0.0.1`)
- **PyPI**: `v*` tags (e.g., `v0.0.1`)

## Future OIDC Setup

When PyPI supports OIDC, you'll be able to:

1. **Remove the API token secret** from GitHub
2. **Update the PyPI workflow** to use OIDC authentication
3. **Configure trust relationships** with PyPI

### OIDC Benefits

- ✅ **No long-lived secrets** stored in GitHub
- ✅ **Automatic credential rotation** (tokens expire after each job)
- ✅ **Granular access control** through PyPI's trust policies
- ✅ **Better security** through short-lived authentication

## Workflow Files

- `.github/workflows/publish.yml` - Publishes to TestPyPI (OIDC, active)
- `.github/workflows/publish-pypi.yml.disabled` - PyPI workflow (disabled)

## Testing Your Setup

1. **Test TestPyPI publishing** (current):
   ```bash
   ./scripts/release.sh patch
   git push origin main && git push origin v0.0.1
   ```

2. **Check the Actions tab** to see the TestPyPI workflow running

3. **Verify on TestPyPI** that your package appears

4. **Test installation**:
   ```bash
   # From TestPyPI
   pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple hedonic
   ```

## Troubleshooting

- **Workflow not triggered**: Ensure you're pushing a tag that matches the pattern `v*`
- **Build failures**: Check that your `pyproject.toml` is properly configured
- **TestPyPI publishing fails**: Check that the workflow file is active and not commented out

## References

- [GitHub Actions OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [PyPI Publishing Guide](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd/)
- [TestPyPI Documentation](https://test.pypi.org/help/)
