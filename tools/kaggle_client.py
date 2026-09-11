#!/usr/bin/env python3
r"""
Shared Kaggle API access for this repo's tooling, across several accounts.

Why this exists instead of the `kaggle` CLI / `kaggle.json`: these accounts'
API tokens are Kaggle's newer KGAT_-prefixed format, authenticated as a Bearer
token. The classic `kaggle` pip package's `kernels`/`datasets`/`competitions`
commands (kaggle/api/kaggle_api_extended.py) only do HTTP Basic Auth with a
{username, key} pair and reject KGAT_ tokens with a 401 regardless of
username -- confirmed against the live API, not assumed. `kagglesdk` (also
installed as a dependency of `kaggle`) has a parallel, lower-level client
that supports Bearer auth via the KAGGLE_API_TOKEN environment variable
(kagglesdk/kaggle_http_client.py:_try_fill_auth) and exposes kernels
list/get/save/session-status/download-output -- everything the ladder needs.

Several accounts, because the binding constraint on this project is compute,
not code: each Kaggle account carries its own 30 GPU-h weekly allowance on a
rolling seven-day window, and the 8 September 2026 exhaustion stalled the
ladder for days. Accounts are listed in `.env` (gitignored) and tried in
numeric order; when one reports its weekly quota exhausted, `mark_exhausted`
puts it in cooldown and the next one takes over. Cooldown is persisted in
`.kaggle_quota_state.json` so a rotation survives the driver being restarted
-- otherwise every `finish.py` run would re-push to the dead account first and
burn a poll cycle rediscovering that it is dead.

An account is a (username, token) PAIR and the two are never resolved
independently. A kernel is pushed to `<username>/<slug>` and authenticated
with `<token>`; pairing rickaryadas's username with ceoricky's token pushes
into an account the token cannot write to. This is why `username` lives in
`.env` beside its token rather than staying a command-line default.

Credential resolution order (never read from git, never printed):
  1. `.env` at the repo root -- KAGGLE_ACCOUNT_<n>_USERNAME / _TOKEN, n from 1
  2. $KAGGLE_API_TOKEN, paired with $KAGGLE_USERNAME (or the legacy default)
  3. C:\Users\<user>\.kaggle\kaggle_api_token (single line, local-only,
     gitignored) -- the fallback used when a shell hasn't picked up a
     freshly-`setx`'d environment variable yet.

Usage:
    from kaggle_client import get_client, active_account, mark_exhausted
    acct = active_account()
    with get_client(acct) as client:
        client.kernels.kernels_api_client.save_kernel(request)
"""
import json
import os
import time
from collections import namedtuple

from kagglesdk import KaggleClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENV_FILE = os.path.join(ROOT, ".env")
STATE_FILE = os.path.join(ROOT, ".kaggle_quota_state.json")
_TOKEN_FILE = os.path.expanduser("~/.kaggle/kaggle_api_token")

# The username the tooling used before `.env` existed, kept so a checkout with
# only $KAGGLE_API_TOKEN set behaves exactly as it did.
_LEGACY_USERNAME = "rickaryadas"

Account = namedtuple("Account", "username token")


class QuotaExhausted(RuntimeError):
    """Raised when Kaggle refuses a push because the weekly GPU quota is spent.

    Carries `username` so the caller knows which account to put in cooldown --
    by the time this propagates, the active account may already have rotated.
    """

    def __init__(self, username, detail=""):
        super().__init__(f"{username}: weekly GPU quota exhausted. {detail}".strip())
        self.username = username
        self.detail = detail


# Kaggle says this on a push once the account's 30 GPU-h week is spent. The
# first is the string this repo actually observed on 8 September 2026; the
# others are near-misses matched so a wording change does not silently turn a
# quota stall into a hard failure. Matching is case-insensitive.
QUOTA_MARKERS = (
    "maximum weekly gpu quota",
    "weekly gpu quota",
    "exceeded your gpu quota",
    "gpu quota exceeded",
)

# How long an exhausted account is skipped before it is tried again. The window
# is rolling rather than a weekly reset, so an account that ran out recovers
# gradually and is worth re-testing well before seven days are up; a push
# against a still-empty account is refused immediately and costs no quota.
COOLDOWN_SECONDS = int(float(os.environ.get("KAGGLE_QUOTA_COOLDOWN_HOURS", "6")) * 3600)


def is_quota_error(text):
    """True if `text` is Kaggle refusing a push for want of weekly GPU quota."""
    low = (text or "").lower()
    return any(m in low for m in QUOTA_MARKERS)


# ------------------------------------------------------------------ .env parsing
def _parse_env(path):
    """Minimal KEY=VALUE reader.

    Deliberately not python-dotenv: this repo pins a Python 3.10 environment
    whose dependency set is reproduced from requirements.txt, and a credential
    file is the last place to add a dependency for the sake of fifteen lines.
    Handles comments, blank lines, `export ` prefixes and quoted values; does
    not do interpolation, which a token never needs.
    """
    values = {}
    if not os.path.exists(path):
        return values
    quotes = ('"', "'")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in quotes:
                value = value[1:-1]
            values[key.strip()] = value
    return values


def accounts():
    """Every configured account, in rotation order, without touching cooldown.

    Numbering must be contiguous from 1: the scan stops at the first missing
    index, so deleting a middle block hides the ones after it rather than
    silently renumbering them.
    """
    env = _parse_env(ENV_FILE)
    found = []
    n = 1
    while True:
        username = env.get("KAGGLE_ACCOUNT_%d_USERNAME" % n, "").strip()
        token = env.get("KAGGLE_ACCOUNT_%d_TOKEN" % n, "").strip()
        if not username or not token:
            break
        found.append(Account(username, token))
        n += 1
    if found:
        return found

    # No .env: fall back to the single-account behaviour this tooling had before.
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token and os.path.exists(_TOKEN_FILE):
        with open(_TOKEN_FILE, encoding="utf-8") as f:
            token = f.read().strip()
    if token:
        return [Account(os.environ.get("KAGGLE_USERNAME", _LEGACY_USERNAME), token)]
    return []


# --------------------------------------------------------------- cooldown state
def _load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        # A missing or half-written state file must never stop a run: the worst
        # case is one wasted push against an account that is still exhausted.
        return {}


def _save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_FILE)


def mark_exhausted(username, when=None):
    """Put `username` in cooldown so the next lookup skips it."""
    state = _load_state()
    state[username] = {"exhausted_at": when or time.time()}
    _save_state(state)


def clear_exhausted(username=None):
    """Drop the cooldown for one account, or all of them."""
    state = _load_state()
    if username is None:
        state = {}
    else:
        state.pop(username, None)
    _save_state(state)


def cooldown_remaining(username, state=None):
    """Seconds until `username` is worth trying again; 0 if it is available."""
    state = _load_state() if state is None else state
    entry = state.get(username)
    if not entry:
        return 0
    elapsed = time.time() - entry.get("exhausted_at", 0)
    return max(0, COOLDOWN_SECONDS - elapsed)


def available_accounts():
    """Configured accounts that are not currently in quota cooldown."""
    state = _load_state()
    return [a for a in accounts() if cooldown_remaining(a.username, state) == 0]


def account_for(username):
    """The configured account owning `username`.

    Used when the target is fixed by something other than rotation -- fetching
    the log of a kernel that a particular account already ran, say. Rotating to
    a different token there would poll a kernel that account cannot see.
    """
    for a in accounts():
        if a.username == username:
            return a
    known = ", ".join(a.username for a in accounts()) or "none configured"
    raise SystemExit(
        "No token for Kaggle account %r in %s. Known: %s." % (username, ENV_FILE, known)
    )


def active_account(prefer=None):
    """The account to use now, or None if every one of them is in cooldown.

    `prefer` pins the choice to one username (and is honoured even in cooldown,
    since an explicit request is a deliberate override, not rotation).
    """
    if prefer:
        return account_for(prefer)
    ready = available_accounts()
    return ready[0] if ready else None


def next_account(after):
    """The next account to try after `after` has just been marked exhausted."""
    ready = [a for a in available_accounts() if a.username != after]
    return ready[0] if ready else None


def describe_accounts():
    """One line per account for logs -- usernames and cooldowns, never tokens."""
    state = _load_state()
    out = []
    for i, a in enumerate(accounts(), 1):
        left = cooldown_remaining(a.username, state)
        status = "ready" if left == 0 else "cooldown %.1f h" % (left / 3600)
        out.append("  %d. %s [%s...] %s" % (i, a.username, a.token[:9], status))
    return "\n".join(out) or "  (none configured)"


# -------------------------------------------------------------------- the client
def get_client(account=None) -> KaggleClient:
    """A KaggleClient authenticated as `account` (default: the active one).

    kagglesdk resolves KAGGLE_API_TOKEN once, when the session is opened
    (kaggle_http_client.py:_init_session -> _try_fill_auth), and caches the
    Bearer header on the session -- so switching accounts means a *new* client
    inside a new `with` block, not reassigning the environment variable under
    a live one. The variable is still set here because that is the only channel
    kagglesdk reads, and kaggle_push.py's direct `requests` call reads it too.
    """
    if account is None:
        account = active_account()
    if account is None:
        raise SystemExit(
            "Every configured Kaggle account is in quota cooldown:\n"
            + describe_accounts()
            + "\nAdd another account to .env, or wait for the rolling window."
        )
    if not account.token:
        raise SystemExit(
            "No Kaggle API token for %s. Add a KAGGLE_ACCOUNT_<n>_TOKEN block "
            "to %s, or set $KAGGLE_API_TOKEN. Get one from kaggle.com -> "
            "Settings -> API." % (account.username, ENV_FILE)
        )
    os.environ["KAGGLE_API_TOKEN"] = account.token
    return KaggleClient()
