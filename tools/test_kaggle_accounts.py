#!/usr/bin/env python3
"""
Verify multi-account Kaggle rotation: tools/kaggle_client.py.

The thing being protected here is a whole unattended run. `finish.py` pushes
~19 GPU-h of stages against a 30 GPU-h weekly allowance per account; when one
account runs dry the driver must move to the next and carry on, and it must not
move a stage onto an account that cannot see the kernel output that stage
mounts. Both behaviours are invisible until the quota actually runs out, which
is the worst possible moment to discover a typo in the rotation -- hence this.

Runs entirely offline against a temporary .env and state file. No live API call,
no quota spent.

    python tools/test_kaggle_accounts.py

Exit 0 means rotation, cooldown and account pinning behave as finish.py assumes.
"""
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaggle_client as kc  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

failures = []


def ok(desc, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {desc}" + (f" — {detail}" if detail else ""))
    if not passed:
        failures.append(desc)


ENV_TEXT = """\
# comment line, ignored
KAGGLE_ACCOUNT_1_USERNAME=alpha
KAGGLE_ACCOUNT_1_TOKEN=KGAT_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

export KAGGLE_ACCOUNT_2_USERNAME = beta
KAGGLE_ACCOUNT_2_TOKEN="KGAT_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

# a gap at 3 must hide 4, not silently renumber it
KAGGLE_ACCOUNT_4_USERNAME=delta
KAGGLE_ACCOUNT_4_TOKEN=KGAT_dddddddddddddddddddddddddddddddd
"""


def main():
    print("Kaggle account rotation, offline\n")
    tmp = tempfile.mkdtemp(prefix="kaggle_accounts_test_")
    # Point the module at a scratch .env / state file. The real credentials are
    # never read, so this test is safe to run on a machine with no .env at all.
    kc.ENV_FILE = os.path.join(tmp, ".env")
    kc.STATE_FILE = os.path.join(tmp, "state.json")
    with open(kc.ENV_FILE, "w", encoding="utf-8") as f:
        f.write(ENV_TEXT)

    try:
        # ---------------------------------------------------------- parsing
        got = kc.accounts()
        ok("two contiguous accounts parsed, gap at 3 truncates the list",
           [a.username for a in got] == ["alpha", "beta"],
           str([a.username for a in got]))
        ok("`export ` prefix and spaces around = are handled",
           any(a.username == "beta" for a in got))
        ok("quoted token is unquoted",
           next(a.token for a in got if a.username == "beta")
           == "KGAT_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

        # -------------------------------------------------------- rotation
        ok("first ready account is the active one",
           kc.active_account().username == "alpha")

        kc.mark_exhausted("alpha")
        ok("an exhausted account is skipped",
           kc.active_account().username == "beta")
        ok("rotation target after alpha is beta",
           kc.next_account("alpha").username == "beta")

        kc.mark_exhausted("beta")
        ok("every account exhausted yields None, not a stale account",
           kc.active_account() is None)
        ok("no rotation target left after the last account",
           kc.next_account("beta") is None)
        # This is the branch finish.py relies on to decide to wait rather than
        # to fail the stage, so it must raise a clean SystemExit with a message,
        # not an AttributeError from a None account.
        try:
            kc.get_client()
            ok("get_client refuses when all accounts are in cooldown", False,
               "returned a client")
        except SystemExit as e:
            ok("get_client refuses when all accounts are in cooldown",
               "cooldown" in str(e).lower())

        # -------------------------------------------------------- cooldown
        ok("cooldown is reported as remaining time",
           0 < kc.cooldown_remaining("alpha") <= kc.COOLDOWN_SECONDS)
        # An account marked exhausted a full cooldown ago is due for a retry:
        # Kaggle's window is rolling, so quota returns gradually.
        kc.mark_exhausted("alpha", when=time.time() - kc.COOLDOWN_SECONDS - 1)
        ok("cooldown expires and the account returns to rotation",
           kc.active_account().username == "alpha")
        kc.clear_exhausted()
        ok("clear_exhausted resets every account",
           [a.username for a in kc.available_accounts()] == ["alpha", "beta"])

        # --------------------------------------------------------- pinning
        ok("account_for returns the pinned account's own token",
           kc.account_for("beta").token.startswith("KGAT_bbbb"))
        kc.mark_exhausted("beta")
        ok("a pinned account is honoured even in cooldown (deliberate override)",
           kc.active_account(prefer="beta").username == "beta")
        # A stage pinned to an account that is not configured must fail loudly:
        # falling back to another token would push into an account the token
        # cannot write to, or mount a kernel output it cannot see.
        try:
            kc.account_for("gamma")
            ok("an unconfigured username is an error, not a silent fallback", False,
               "returned an account")
        except SystemExit as e:
            ok("an unconfigured username is an error, not a silent fallback",
               "gamma" in str(e))

        # ---------------------------------------------- quota-error matching
        ok("the string Kaggle returned on 8 Sep 2026 is recognised",
           kc.is_quota_error(
               "You have exceeded your Maximum weekly GPU quota of 30 hours."))
        ok("matching is case-insensitive",
           kc.is_quota_error("maximum weekly gpu quota"))
        ok("an unrelated failure is not mistaken for a quota stall",
           not kc.is_quota_error("push failed (404): kernel not found"))
        ok("empty output is not a quota stall", not kc.is_quota_error(""))

        # ------------------------------------------- tokens never in output
        # describe_accounts() goes into run logs that get pasted into reports.
        desc = kc.describe_accounts()
        ok("describe_accounts truncates tokens",
           "KGAT_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in desc
           and "KGAT_aaaa" in desc,
           desc.replace("\n", " | "))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + "; ".join(failures))
        return 1
    print("Rotation, cooldown and pinning behave as finish.py assumes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
