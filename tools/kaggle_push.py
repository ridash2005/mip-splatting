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
"""
import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_client import get_client  # noqa: E402
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


def push_with_accelerator(body, accelerator):
    """POST the save-kernel body with the `machineShape` field kagglesdk cannot set.

    Returns (ref, version, url). Raises on a non-2xx or an `error` in the reply.
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
        raise SystemExit(f"push failed ({r.status_code}): {r.text[:600]}")
    d = r.json()
    if d.get("error"):
        raise SystemExit(f"push failed: {d['error']}")
    return d.get("ref"), d.get("versionNumber"), d.get("url")


def push(client, script_path, username, slug, title, datasets, session_timeout=None,
         accelerator=None):
    with open(script_path, encoding="utf-8") as f:
        text = f.read()
    req = ApiSaveKernelRequest()
    req.slug = f"{username}/{slug}"
    req.new_title = title
    req.text = text
    req.language = "python"
    req.kernel_type = "script"
    req.is_private = True
    req.enable_gpu = True
    req.enable_internet = True
    req.dataset_data_sources = datasets or []
    if session_timeout:
        req.session_timeout_seconds = session_timeout
    if accelerator:
        from kagglesdk.kaggle_object import KaggleObject
        ref, version, url = push_with_accelerator(KaggleObject.to_dict(req), accelerator)
        print(f"pushed {ref} (version {version}, accelerator={accelerator}) -> {url}")

        class _R:  # same shape the caller reads off save_kernel's reply
            pass
        resp = _R(); resp.ref = ref; resp.version_number = version; resp.url = url
    else:
        resp = client.kernels.kernels_api_client.save_kernel(req)
        if resp.error:
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


WRONG_GPU_MARKER = "R0_ABORT=WRONG_ACCELERATOR"


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
    for line in text.splitlines():
        if line.startswith("R0_SUMMARY_JSON="):
            summary = json.loads(line[len("R0_SUMMARY_JSON="):])
            path = os.path.join(out_dir, "summary.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"wrote {path}")
            return summary
    print("no R0_SUMMARY_JSON= line in the log — the run did not reach the end.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--username", default="rickaryadas")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", help="required unless --fetch-only; Kaggle derives "
                     "the slug from it (lowercased, dashed)")
    ap.add_argument("--dataset", action="append", default=[],
                     help="owner/slug of a Kaggle dataset to attach; repeatable")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-wait", action="store_true", help="push and exit without polling")
    ap.add_argument("--session-timeout", type=int, default=None)
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

    with get_client() as client:
        out_dir = a.out or f"results/kaggle_runs/{a.slug}"
        if a.fetch_only:
            # For a run launched from the web UI -- the only way to choose the
            # accelerator. Nothing is pushed; the finished session's log and
            # summary are pulled down exactly as they would be after a push.
            status = poll(client, a.username, a.slug, timeout=a.poll_timeout)
            log_resp = fetch_log(client, a.username, a.slug, out_dir)
            save_summary(log_resp, out_dir)
            sys.exit(1 if status == KernelWorkerStatus.ERROR else 0)
        for attempt in range(1, a.retry_wrong_gpu + 2):
            _, actual_slug = push(client, a.script, a.username, a.slug, a.title,
                                  a.dataset, a.session_timeout, a.accelerator)
            if a.no_wait:
                return
            status = poll(client, a.username, actual_slug, timeout=a.poll_timeout)
            out_dir = a.out or f"results/kaggle_runs/{actual_slug}"
            log_resp = fetch_log(client, a.username, actual_slug, out_dir)

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
            if status == KernelWorkerStatus.ERROR:
                sys.exit(1)
            return


if __name__ == "__main__":
    main()
