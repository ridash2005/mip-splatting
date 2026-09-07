#!/usr/bin/env python3
"""
Shared Kaggle API access for this repo's tooling.

Why this exists instead of the `kaggle` CLI / `kaggle.json`: this account's
API token is Kaggle's newer KGAT_-prefixed format, authenticated as a Bearer
token. The classic `kaggle` pip package's `kernels`/`datasets`/`competitions`
commands (kaggle/api/kaggle_api_extended.py) only do HTTP Basic Auth with a
{username, key} pair and reject KGAT_ tokens with a 401 regardless of
username — confirmed against the live API, not assumed. `kagglesdk` (also
installed as a dependency of `kaggle`) has a parallel, lower-level client
that supports Bearer auth via the KAGGLE_API_TOKEN environment variable
(kagglesdk/kaggle_http_client.py:_try_fill_auth) and exposes kernels
list/get/save/session-status/download-output — everything the ladder needs.

Token resolution order (never read from git, never printed):
  1. $KAGGLE_API_TOKEN environment variable
  2. C:\\Users\\<user>\\.kaggle\\kaggle_api_token (single line, local-only,
     gitignored) — the fallback used when a shell hasn't picked up a
     freshly-`setx`'d environment variable yet.

Usage:
    from kaggle_client import get_client
    with get_client() as client:
        client.kernels.kernels_api_client.save_kernel(request)
"""
import os

from kagglesdk import KaggleClient

_TOKEN_FILE = os.path.expanduser("~/.kaggle/kaggle_api_token")


def _resolve_token():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return token.strip()
    if os.path.exists(_TOKEN_FILE):
        with open(_TOKEN_FILE) as f:
            token = f.read().strip()
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token  # so KaggleClient() picks it up
            return token
    raise SystemExit(
        "No Kaggle API token found. Set $KAGGLE_API_TOKEN or put it (one line) "
        f"in {_TOKEN_FILE}. Get one from kaggle.com \u2192 Settings \u2192 API."
    )


def get_client() -> KaggleClient:
    _resolve_token()
    return KaggleClient()
