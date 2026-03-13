"""
secrets_loader.py — Load project secrets from secrets.json with env var fallback.

Priority order (highest to lowest):
  1. secrets.json  — local dev file, never committed (gitignored)
  2. Environment variable  — used in deployed environments (Cloud Run, CI)
  3. Default value  — hard-coded last resort (never put real secrets here)

secrets.json values that begin with 'YOUR_' are treated as unfilled placeholders
and are silently skipped, so env vars take over automatically.

Usage
-----
  from secrets_loader import secret

  bucket    = secret("gcs_bucket",        env_var="GCS_BUCKET")
  flask_key = secret("flask_secret",      env_var="FLASK_SECRET", default="dev-only-unsafe")
  api_key   = secret("anthropic_api_key", env_var="ANTHROPIC_API_KEY")
"""

import json
import os
from pathlib import Path

_SECRETS: dict | None = None


def _load() -> dict:
    global _SECRETS
    if _SECRETS is not None:
        return _SECRETS
    path = Path(__file__).parent / "secrets.json"
    if path.exists():
        with open(path) as f:
            _SECRETS = json.load(f)
    else:
        _SECRETS = {}
    return _SECRETS


def secret(key: str, env_var: str = None, default=None):
    """
    Retrieve a secret value.

    Checks secrets.json first.  Values beginning with 'YOUR_' are treated as
    unfilled placeholders and are skipped.  Falls back to the named environment
    variable, then to `default`.
    """
    val = _load().get(key)
    if val is not None and not str(val).startswith("YOUR_"):
        return val
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val
    return default
