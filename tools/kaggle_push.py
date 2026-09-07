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

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_client import get_client  # noqa: E402
from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E402
    ApiSaveKernelRequest,
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)
from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus  # noqa: E402


def push(client, script_path, username, slug, title, datasets, session_timeout=None):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--username", default="rickaryadas")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--dataset", action="append", default=[],
                     help="owner/slug of a Kaggle dataset to attach; repeatable")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-wait", action="store_true", help="push and exit without polling")
    ap.add_argument("--session-timeout", type=int, default=None)
    ap.add_argument("--retry-wrong-gpu", type=int, default=0,
                     help="relaunch this many times if Kaggle allocates a GPU below "
                          "compute capability 7.0 (F15). Each such attempt aborts in "
                          "~30s, so the quota cost is negligible.")
    ap.add_argument("--retry-delay", type=int, default=60,
                     help="seconds between relaunch attempts")
    ap.add_argument("--poll-timeout", type=int, default=6 * 3600,
                     help="seconds to wait for the kernel to finish before giving up")
    a = ap.parse_args()

    with get_client() as client:
        out_dir = a.out or f"results/kaggle_runs/{a.slug}"
        for attempt in range(1, a.retry_wrong_gpu + 2):
            _, actual_slug = push(client, a.script, a.username, a.slug, a.title,
                                  a.dataset, a.session_timeout)
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

            for line in log_resp.log.splitlines():
                if line.startswith("R0_SUMMARY_JSON="):
                    summary = json.loads(line[len("R0_SUMMARY_JSON="):])
                    with open(os.path.join(out_dir, "summary.json"), "w") as f:
                        json.dump(summary, f, indent=2)
                    print(json.dumps(summary, indent=2))
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
