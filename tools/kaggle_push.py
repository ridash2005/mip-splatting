#!/usr/bin/env python3
"""
Push a script to run as a Kaggle kernel, poll it to completion, and save its
run log locally. Uses tools/kaggle_client.py — see that file for why this
goes through kagglesdk (Bearer/KGAT auth) instead of the `kaggle` CLI.

    python tools/kaggle_push.py kaggle/r0_smoke.py \
        --slug btp-r0-smoke --title "BTP R0 smoke: 3DGS vs Mip-Splatting (lego, 7k)" \
        --dataset nguyenhung1903/nerf-synthetic-dataset

Each `files` entry in a kernel's output has its own signed `url` for
download; `list_kernel_session_output`'s `log` field carries the full
stdout/stderr, which is what this tool saves and scans for a
`R0_SUMMARY_JSON=` line.

Accelerator note: `ApiSaveKernelRequest` carries only a boolean `enable_gpu`
(kagglesdk 0.1.28 has no accelerator field at all), and Kaggle allocates
whichever GPU is free — a P100 on one launch, 2xT4 on the next. 3DGS needs
compute capability 7.0+ (F15), so the kernel aborts in ~30 seconds on a P100
and prints `R0_ABORT=WRONG_ACCELERATOR`. `--retry-wrong-gpu N` relaunches on
exactly that marker, and on nothing else: a genuine failure is never retried.

Quota note: each account in `.env` has its own 30 GPU-h week. When Kaggle
refuses a push for want of quota, that account is put in cooldown and the push
is retried as the next account -- see tools/kaggle_client.py. `--username`
pins the run to one account and disables that rotation, which is what you want
when fetching the log of a kernel a specific account already ran.
"""
import argparse
import json
import re
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_client import (  # noqa: E402
    QuotaExhausted,
    accounts,
    active_account,
    describe_accounts,
    get_client,
    is_quota_error,
    mark_exhausted,
    next_account,
)
from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E402
    ApiSaveKernelRequest,
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)
from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus  # noqa: E402


# Kaggle allocates a Tesla P100 (compute capability 6.0) to every plain
# `enable_gpu: true` push, which 3DGS cannot run on (F15). The accelerator is
# chosen by a `machineShape` field on /api/v1/kernels/push -- confirmed against
# kagglesdk 0.1.37's own wire schema, whose docstring reads: "The machine shape
# to use for this session. Currently supported options: NvidiaTeslaT4,
# NvidiaTeslaP100, Tpu1VmV38."
#
# The installed clients cannot send it: kagglesdk 0.1.28's ApiSaveKernelRequest
# has no machine_shape field and rejects unknown attributes client-side, and
# both newer kagglesdk and `kaggle` 2.x require Python >= 3.11 (this machine has
# 3.10). So the same JSON body kagglesdk would have sent is posted directly with
# the field added.
#
# Verified, not assumed: a probe kernel pushed with machineShape=NvidiaTeslaT4
# reported `GPU 0: Tesla T4 / GPU 1: Tesla T4`, against P100 on eleven
# consecutive pushes without it. Note the API does NOT validate this field -- an
# unrecognised value returns HTTP 200 and silently yields the default P100 --
# so the choices below are restricted to the three the schema documents.
ACCELERATORS = ("NvidiaTeslaT4", "NvidiaTeslaP100", "Tpu1VmV38")
PUSH_URL = "https://www.kaggle.com/api/v1/kernels/push"


def push_with_accelerator(body, accelerator, username):
    """POST the save-kernel body with the `machineShape` field kagglesdk cannot set.

    Returns (ref, version, url). Raises QuotaExhausted when Kaggle refuses the
    push for want of weekly GPU quota, so the caller can rotate accounts; raises
    on any other non-2xx or an `error` in the reply. A quota refusal arrives
    both ways depending on the path Kaggle takes -- as a non-2xx body and as a
    200 with an `error` string -- so both are checked.

    Note the server does not validate machineShape, so a typo is not caught here
    -- it is caught by the kernel's own check 1, which aborts on anything below
    compute capability 7.0 and says which GPU it actually got.
    """
    body = dict(body, machineShape=accelerator)
    token = os.environ["KAGGLE_API_TOKEN"]
    r = requests.post(PUSH_URL, json=body,
                      headers={"Authorization": f"Bearer {token}",
                               "Content-Type": "application/json"}, timeout=120)
    if r.status_code >= 300:
        if is_quota_error(r.text):
            raise QuotaExhausted(username, r.text[:300])
        raise SystemExit(f"push failed ({r.status_code}): {r.text[:600]}")
    d = r.json()
    if d.get("error"):
        if is_quota_error(d["error"]):
            raise QuotaExhausted(username, str(d["error"])[:300])
        raise SystemExit(f"push failed: {d['error']}")
    return d.get("ref"), d.get("versionNumber"), d.get("url")


def apply_overrides(text, overrides):
    """Rewrite top-level `NAME = ...` constants in the kernel source before upload.

    Kaggle scripts take no arguments and no environment, so a rung kernel is
    parameterised by editing its own constants. Only a line that already exists
    at column 0 is rewritten -- an unknown name is an error, not a silently
    ignored typo that would run the wrong protocol for four hours.

    The value is coerced to the TYPE the constant already has. That matters:
    `--set PROTOCOLS=full` against `PROTOCOLS = ["full"]` must produce a
    one-element list, not the string "full" -- which iterates as 'f','u','l','l'
    and cost a session to discover.
    """
    for item in overrides or []:
        name, _, value = item.partition("=")
        name = name.strip()
        pattern = re.compile(rf"^{re.escape(name)} = (.*)$", re.M)
        m = pattern.search(text)
        if not m:
            raise SystemExit(f"--set {name}: no top-level `{name} = ...` in the kernel")
        current = m.group(1).strip()

        if value.startswith("["):
            literal = value
        elif current.startswith("["):
            # list-typed target: always a list, even for one element
            parts = [v.strip() for v in value.split(",") if v.strip()]
            literal = "[" + ", ".join(
                v if (v in ("True", "False") or v.lstrip("-").isdigit()) else repr(v)
                for v in parts) + "]"
        elif current[:1] in ("'", '"'):
            # string-typed target stays a string, even when the value looks
            # numeric: TABLE = "2" is used to build filenames, not arithmetic.
            literal = repr(value)
        elif value in ("True", "False") or value.lstrip("-").replace(".", "", 1).isdigit():
            literal = value
        else:
            literal = repr(value)

        text = pattern.sub(f"{name} = {literal}", text, count=1)
        print(f"  set {name} = {literal}   (was {current})")
    return text


def push(client, script_path, username, slug, title, datasets, session_timeout=None,
         accelerator=None, overrides=None, kernel_sources=None, gpu=True):
    with open(script_path, encoding="utf-8") as f:
        text = f.read()
    text = apply_overrides(text, overrides)

    # Kaggle keys a notebook by the slug it derives from the title, and ignores a
    # requested slug that does not match it. That is harmless on a first push and
    # fatal on a retry: the create succeeds, the SESSION is refused for the
    # two-concurrent-GPU-session cap, and the next attempt is a second create
    # against a title that is now taken -- 409, forever. Deriving the slug the
    # same way Kaggle does makes the retry an update of the notebook the first
    # attempt created, which is what it was always meant to be.
    derived = re.sub(r"[^a-z0-9]+", "-", (title or slug).lower()).strip("-")
    if derived and derived != slug:
        print(f"note: using slug {derived!r} derived from the title "
              f"(requested {slug!r}); Kaggle keys the notebook by this, so a "
              f"retry updates it instead of colliding with its own title")
        slug = derived

    req = ApiSaveKernelRequest()
    req.slug = f"{username}/{slug}"
    req.new_title = title
    req.text = text
    req.language = "python"
    req.kernel_type = "script"
    req.is_private = True
    # CPU sessions come from a separate pool, so a probe or a pure-NumPy job
    # need not wait behind the two-concurrent-GPU-session limit.
    req.enable_gpu = gpu
    req.enable_internet = True
    req.dataset_data_sources = datasets or []
    # A finished rung's output, mounted read-only under /kaggle/input. This is
    # how the instruments read trained point clouds without retraining them.
    req.kernel_data_sources = kernel_sources or []
    if session_timeout:
        req.session_timeout_seconds = session_timeout
    if accelerator:
        from kagglesdk.kaggle_object import KaggleObject
        ref, version, url = push_with_accelerator(KaggleObject.to_dict(req),
                                                  accelerator, username)
        print(f"pushed {ref} (version {version}, accelerator={accelerator}) -> {url}")

        class _R:  # same shape the caller reads off save_kernel's reply
            pass
        resp = _R(); resp.ref = ref; resp.version_number = version; resp.url = url
    else:
        resp = client.kernels.kernels_api_client.save_kernel(req)
        if resp.error:
            # Same rotation signal as the accelerator path above: a quota
            # refusal is a reason to change account, not to abort the ladder.
            if is_quota_error(resp.error):
                raise QuotaExhausted(username, str(resp.error)[:300])
            raise SystemExit(f"push failed: {resp.error}")
        print(f"pushed {resp.ref} (version {resp.version_number}) -> {resp.url}")
    # Kaggle derives the actual slug from new_title (must be "title, lowercased
    # with dashes") and silently ignores a mismatched requested slug — always
    # use what it actually assigned, from the tail of resp.ref, not our request.
    actual_slug = resp.ref.rsplit("/", 1)[-1]
    if actual_slug != slug:
        print(f"note: Kaggle assigned slug {actual_slug!r} (requested {slug!r})")
    return resp, actual_slug


def poll(client, username, slug, interval=30, timeout=6 * 3600):
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = username
    req.kernel_slug = slug
    start = time.time()
    last = None
    while True:
        resp = client.kernels.kernels_api_client.get_kernel_session_status(req)
        if resp.status != last:
            print(f"[{time.time() - start:6.0f}s] status: {resp.status.name}", flush=True)
            last = resp.status
        if resp.status in (KernelWorkerStatus.COMPLETE, KernelWorkerStatus.ERROR):
            if resp.failure_message:
                print("failure_message:", resp.failure_message)
            return resp.status
        if time.time() - start > timeout:
            raise SystemExit(f"timed out after {timeout}s waiting for the kernel to finish")
        time.sleep(interval)


def fetch_log(client, username, slug, out_dir):
    req = ApiListKernelSessionOutputRequest()
    req.user_name = username
    req.kernel_slug = slug
    resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(resp.log)
    print(f"wrote {log_path} ({len(resp.log)} bytes)")
    if resp.files:
        print("output files:", [f_.file_name for f_ in resp.files])
    return resp


WRONG_GPU_MARKER = "_ABORT=WRONG_ACCELERATOR"   # any rung: R0_, R1_, ...


def save_summary(log_resp, out_dir):
    """Pull the kernel's R0_SUMMARY_JSON line out of the log and save it.

    The log is Kaggle's JSON event stream, so the marker line can arrive either
    as raw text or split across events; the raw text is searched either way. The
    decoded log is written alongside it, because the event stream is unreadable.
    """
    raw = log_resp.log
    try:
        text = "".join(e["data"] for e in json.loads(raw))
    except Exception:
        text = raw
    decoded = os.path.join(out_dir, "log_decoded.txt")
    with open(decoded, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {decoded} ({len(text)} chars)")
    # Any rung's marker: R0_, R1_, R2_ ... Matching only R0_ silently dropped a
    # completed R2 run's summary on the floor.
    marker = re.compile(r"^R\d+_SUMMARY_JSON=")
    for line in text.splitlines():
        m = marker.match(line)
        if m:
            summary = json.loads(line[m.end():])
            path = os.path.join(out_dir, "summary.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"wrote {path}")
            return summary
    print("no R0_SUMMARY_JSON= line in the log — the run did not reach the end.")
    return None


def run_under(client, username, a):
    """One push/poll/fetch cycle under a single account. Returns an exit code.

    Raises QuotaExhausted rather than handling it, so the choice between
    rotating to another account and giving up stays in one place, in main().

    The wrong-GPU retry count restarts if the caller rotates accounts. That is
    deliberate: each such attempt aborts in ~30 s on the kernel's own check 1,
    so the quota cost is negligible, and a fresh account genuinely deserves its
    full allowance of attempts at drawing a compute-capability 7.0+ GPU.
    """
    out_dir = a.out or f"results/kaggle_runs/{a.slug}"
    if a.fetch_only:
        # For a run launched from the web UI -- the only way to choose the
        # accelerator. Nothing is pushed; the finished session's log and
        # summary are pulled down exactly as they would be after a push.
        status = poll(client, username, a.slug, timeout=a.poll_timeout)
        log_resp = fetch_log(client, username, a.slug, out_dir)
        save_summary(log_resp, out_dir)
        return 1 if status == KernelWorkerStatus.ERROR else 0

    for attempt in range(1, a.retry_wrong_gpu + 2):
        _, actual_slug = push(client, a.script, username, a.slug, a.title,
                              a.dataset, a.session_timeout, a.accelerator,
                              a.set, a.kernel_source, not a.no_gpu)
        if a.no_wait:
            return 0
        status = poll(client, username, actual_slug, timeout=a.poll_timeout)
        out_dir = a.out or f"results/kaggle_runs/{actual_slug}"
        log_resp = fetch_log(client, username, actual_slug, out_dir)

        if WRONG_GPU_MARKER in log_resp.log and attempt <= a.retry_wrong_gpu:
            print(f"attempt {attempt}: Kaggle allocated a sub-7.0 GPU; the kernel "
                  f"aborted per F15. Relaunching in {a.retry_delay}s "
                  f"({a.retry_wrong_gpu - attempt + 1} attempt(s) left).", flush=True)
            time.sleep(a.retry_delay)
            continue

        save_summary(log_resp, out_dir)
        if WRONG_GPU_MARKER in log_resp.log:
            sys.exit(f"gave up after {attempt} attempts: Kaggle never allocated a "
                     "compute-capability 7.0+ GPU. Raise --retry-wrong-gpu, or set "
                     "the notebook's accelerator to 'GPU T4 x2' in the web UI and "
                     "launch it there.")
        return 1 if status == KernelWorkerStatus.ERROR else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--username", default=None,
                     help="pin the run to one account configured in .env, which "
                          "also disables quota rotation. Default: the first "
                          "account that is not in quota cooldown. Required with "
                          "--fetch-only when the kernel belongs to an account "
                          "other than the first, since another account's token "
                          "cannot see it.")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", help="required unless --fetch-only; Kaggle derives "
                     "the slug from it (lowercased, dashed)")
    ap.add_argument("--dataset", action="append", default=[],
                     help="owner/slug of a Kaggle dataset to attach; repeatable")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-wait", action="store_true", help="push and exit without polling")
    ap.add_argument("--session-timeout", type=int, default=None)
    ap.add_argument("--no-gpu", action="store_true",
                     help="run on a CPU session, which is a separate pool from the "
                          "two concurrent GPU sessions an account may hold")
    ap.add_argument("--kernel-source", action="append", default=[],
                     metavar="USER/SLUG",
                     help="attach another kernel's output as an input, mounted "
                          "under /kaggle/input. Repeatable.")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                     help="override a top-level constant in the kernel source before "
                          "upload, e.g. --set LOAD_ALLRES=True --set RUNG=R2. "
                          "Repeatable. An unknown name is an error.")
    ap.add_argument("--accelerator", choices=ACCELERATORS,
                     help="machine shape to request. Without it Kaggle allocates a "
                          "P100 (CC 6.0), which fails F15. NvidiaTeslaT4 yields "
                          "Tesla T4 x2 (CC 7.5).")
    ap.add_argument("--fetch-only", action="store_true",
                     help="do not push; poll the existing kernel session and pull its "
                          "log and summary down. Use after launching from the web UI, "
                          "which is the only place the accelerator can be chosen.")
    ap.add_argument("--retry-wrong-gpu", type=int, default=0,
                     help="relaunch this many times if Kaggle allocates a GPU below "
                          "compute capability 7.0 (F15). Each such attempt aborts in "
                          "~30s, so the quota cost is negligible.")
    ap.add_argument("--retry-delay", type=int, default=60,
                     help="seconds between relaunch attempts")
    ap.add_argument("--poll-timeout", type=int, default=6 * 3600,
                     help="seconds to wait for the kernel to finish before giving up")
    a = ap.parse_args()
    if not a.fetch_only and not a.title:
        ap.error("--title is required unless --fetch-only")

    # --username pins the account and disables rotation; otherwise the first
    # account in .env with quota left is used. active_account() fails loudly on
    # a username that is not configured, rather than falling back to a token
    # that cannot write to it.
    if a.fetch_only and not a.username:
        # Fetching a log spends no GPU quota, so cooldown is irrelevant to it --
        # and letting rotation choose here would be actively wrong: a kernel's
        # output is visible only to the account that ran it, so a rotated fetch
        # would poll a kernel the token cannot see and look like a hung run.
        # Pick the first configured account deterministically, and say so.
        configured = accounts()
        if not configured:
            sys.exit("No Kaggle accounts configured. Add a KAGGLE_ACCOUNT_1_* "
                     "block to .env (see tools/kaggle_client.py).")
        account = configured[0]
        if len(configured) > 1:
            print(f"note: fetching as {account.username}, the first account in "
                  ".env. Pass --username if this kernel belongs to another "
                  "account -- its output is invisible to every other one.",
                  flush=True)
    else:
        account = active_account(a.username)
        if account is None:
            sys.exit("Maximum weekly GPU quota reached on every configured "
                     "account:\n" + describe_accounts()
                     + "\nAdd another account to .env, or wait for the window.")

    while True:
        print(f"account: {account.username}", flush=True)
        try:
            with get_client(account) as client:
                sys.exit(run_under(client, account.username, a))
        except QuotaExhausted as e:
            # Cooldown is persisted, so the next stage of the ladder -- and a
            # restarted driver -- starts on the account that still has quota
            # instead of rediscovering this one is dead.
            mark_exhausted(e.username)
            if a.username:
                sys.exit(f"Maximum weekly GPU quota reached on {e.username}, which "
                         "--username pinned this run to, so it cannot rotate. Wait "
                         "for that account's rolling window. finish.py pins a stage "
                         "only when it mounts a kernel output that account owns, "
                         "which no other account can see.")
            nxt = next_account(e.username)
            if nxt is None:
                sys.exit("Maximum weekly GPU quota reached on every configured "
                         "account:\n" + describe_accounts()
                         + "\nAdd another account to .env, or wait for the window.")
            print(f"{e.username}: weekly GPU quota exhausted -> switching to "
                  f"{nxt.username} and re-pushing.", flush=True)
            account = nxt


if __name__ == "__main__":
    main()
