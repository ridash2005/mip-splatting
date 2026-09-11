#!/usr/bin/env python3
"""
Run a rung kernel on this machine (or a cluster node) instead of on Kaggle.

The kernels were written for Kaggle, but almost nothing in them is Kaggle: they
clone the repo at a pinned branch, build the rasteriser, train, render, score,
and print one summary line. Only three constants name Kaggle -- WORK, REPO and
the dataset mount -- and tools/kaggle_push.py already knows how to rewrite
top-level constants, so this reuses that rather than forking the kernels.

What this buys: the same code, the same pinned branches, the same assertions and
the same CSV rows, on any CUDA device with compute capability >= 7.0. The
provenance stays yours -- the commit hashes in every row are the same ones.

    python tools/run_local.py kaggle/blender_rung.py \
        --data /scratch/nerf_synthetic --work /scratch/btp/c1 \
        --out results/kaggle_runs/c1_verify \
        --set SCENES=lego --set ARMS_ENABLED=B \
        --set ARM_B_BRANCH=arm-b-3dgs-vanilla \
        --set ARM_B_EXTRA=--disable_2D_mip_compensation

On a cluster this is the body of the sbatch script; nothing else changes.
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load_pusher():
    """Reuse apply_overrides so the two paths cannot drift apart."""
    spec = importlib.util.spec_from_file_location(
        "kaggle_push", os.path.join(HERE, "kaggle_push.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["kaggle_push"] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="kernel under kaggle/")
    ap.add_argument("--data", required=True,
                    help="directory containing the scene folders "
                         "(the one whose child is lego/, chair/, ...)")
    ap.add_argument("--work", default=None,
                    help="scratch directory; defaults to a temp dir")
    ap.add_argument("--out", required=True, help="where to write log + summary")
    ap.add_argument("--repo", default=ROOT,
                    help="repo to clone the arms from; defaults to this "
                         "checkout, so the run needs no network")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pusher = _load_pusher()
    src_path = os.path.join(ROOT, a.script)
    text = open(src_path, encoding="utf-8").read()

    work = a.work or tempfile.mkdtemp(prefix="btp-local-")
    os.makedirs(work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    # The three constants that name Kaggle, plus whatever the caller set.
    # Forward slashes even on Windows: these land inside a Python string literal
    # in the generated kernel, where "C:\Users\..." is a truncated \U escape and
    # the file will not parse. Python and git both take '/' on either platform.
    def posix(p):
        return os.path.abspath(p).replace("\\", "/")

    overrides = [
        f"WORK={posix(work)}",
        f"REPO={posix(a.repo)}",
        f"DATA_MOUNT={posix(a.data)}",
        f"DATA_ROOT={posix(a.data)}",
    ] + a.set
    text = pusher.apply_overrides(text, overrides)

    run_path = os.path.join(work, "kernel.py")
    with open(run_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"kernel written to {run_path}", flush=True)
    if a.dry_run:
        return 0

    log_path = os.path.join(a.out, "log.txt")
    print(f"running; log -> {log_path}", flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen([sys.executable, "-u", run_path], cwd=work,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = p.wait()

    # The kernels end in one machine-readable line, named per rung.
    blob = open(log_path, encoding="utf-8", errors="replace").read()
    m = None
    for mm in re.finditer(r"^R\d+_SUMMARY_JSON=(.*)$", blob, re.M):
        m = mm
    if m:
        summary = json.loads(m.group(1))
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(f"wrote {a.out}/summary.json", flush=True)
    else:
        print("WARNING: no *_SUMMARY_JSON= line; the run did not finish",
              file=sys.stderr)

    produced = os.path.join(work, "results", "runs.csv")
    if os.path.exists(produced):
        shutil.copy(produced, os.path.join(a.out, "runs.csv"))
        print(f"wrote {a.out}/runs.csv", flush=True)

    return rc


if __name__ == "__main__":
    sys.exit(main())
