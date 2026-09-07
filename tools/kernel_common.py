#!/usr/bin/env python3
"""
Shared machinery for the Kaggle rung kernels.

kaggle/r0_smoke.py carries its own copy of these, deliberately: it is a frozen
record of the R0 run and its log is cited in the report, so it is not refactored
after the fact. Everything from R1 onward imports this module out of the cloned
repository instead.

A rung kernel does the accelerator check inline (it must abort before doing any
work), clones the repo, then imports this.
"""
import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback

_PROGRESS = re.compile(r"(it/s|s/it|\d+%\|)")


# ------------------------------------------------------------------- shell
def sh(cmd, cwd=None, check_rc=True, log_name=None, logdir=None,
       echo_progress_every=120):
    """Run a command, tee to logs/<name>.log, echo a throttled subset.

    Reads raw chunks and splits on CR as well as LF: train.py's tqdm bar redraws
    with bare carriage returns, so iterating the pipe by line would block for the
    whole run and then emit one enormous line.
    """
    print(f"+ {cmd}" + (f"   (cwd={cwd})" if cwd else ""), flush=True)
    buf, pending, last = [], "", [0.0]
    lf = None
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        lf = open(os.path.join(logdir, (log_name or "cmd") + ".log"), "a",
                  encoding="utf-8", errors="replace")
        lf.write(f"\n$ {cmd}\n")
    try:
        p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace",
                             bufsize=0)

        def emit(seg):
            if not seg.strip():
                return
            if _PROGRESS.search(seg):
                now = time.time()
                if now - last[0] < echo_progress_every:
                    return
                last[0] = now
            print(seg.rstrip(), flush=True)

        while True:
            chunk = p.stdout.read(4096)
            if not chunk:
                break
            buf.append(chunk)
            if lf:
                lf.write(chunk)
            pending += chunk
            parts = re.split(r"[\r\n]", pending)
            pending = parts.pop()
            for seg in parts:
                emit(seg)
        emit(pending)
        rc = p.wait()
    finally:
        if lf:
            lf.close()
    out = "".join(buf)
    if check_rc and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}\n--- tail ---\n{out[-3000:]}")
    return rc, out


def git_hash(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------- peak VRAM
class VramProbe(threading.Thread):
    """Sample nvidia-smi for a stage and keep the maximum.

    Measured from outside the training process: train.py never reports its own
    peak, and patching it to would be a code change beyond §3.
    """

    def __init__(self, gpu=None):
        super().__init__(daemon=True)
        self.peak_mb = 0
        self.gpu = gpu
        # NOT self._stop: threading.Thread._stop is an internal METHOD that
        # join() calls. Shadowing it makes join() raise "TypeError: 'Event'
        # object is not callable" — which cost a finished training run once.
        self._stop_evt = threading.Event()

    def run(self):
        q = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if self.gpu is not None:
            q += [f"--id={self.gpu}"]
        while not self._stop_evt.wait(3.0):
            try:
                out = subprocess.run(q, capture_output=True, text=True, timeout=10).stdout
                self.peak_mb = max([self.peak_mb] + [int(v) for v in out.split()
                                                     if v.strip().isdigit()])
            except Exception:
                pass

    def stop(self):
        # Instrumentation must never be able to lose a run.
        try:
            self._stop_evt.set()
            self.join(timeout=8)
        except Exception:
            traceback.print_exc()
        return self.peak_mb


# --------------------------------------------------------- build-time fixes
def patch_simple_knn_flt_max(repo_dir):
    """simple_knn.cu uses FLT_MAX without including <cfloat>.

    Older toolchains pulled <float.h> in transitively; CUDA 12.8 with GCC 13
    does not and the build fails outright. A compile fix, not an algorithm
    change — the file has no other edit.
    """
    path = os.path.join(repo_dir, "submodules", "simple-knn", "simple_knn.cu")
    with open(path) as f:
        source = f.read()
    assert "FLT_MAX" in source, f"unexpected simple_knn.cu — no FLT_MAX in {path}"
    if "#include <cfloat>" in source:
        print(f"simple-knn already includes <cfloat>: {path}", flush=True)
        return
    needle = "#include <vector>\n"
    if needle not in source:
        raise RuntimeError(f"cannot apply the <cfloat> build fix: {path}")
    with open(path, "w") as f:
        f.write(source.replace(needle, needle + "#include <cfloat>\n", 1))
    print(f"applied build fix: added #include <cfloat> to {path}", flush=True)


def neutralise_unused_open3d_import(repo_dir):
    """Fallback when open3d will not install. train.py imports it and never uses it.

    Proved here rather than assumed: `o3d`/`open3d` must appear exactly twice in
    CODE (the import line). Counting the raw file would let this function's own
    replacement comment satisfy the guard on a second call.
    """
    path = os.path.join(repo_dir, "train.py")
    marker = "# import open3d as o3d  # unused; neutralised by the rung kernel"
    with open(path) as f:
        source = f.read()
    if marker in source:
        print(f"open3d import already neutralised in {path}", flush=True)
        return True
    code = "\n".join(ln.split("#", 1)[0] for ln in source.splitlines())
    uses = len(re.findall(r"\bo3d\b", code)) + len(re.findall(r"\bopen3d\b", code))
    if uses != 2:
        raise RuntimeError(
            f"refusing to touch {path}: open3d/o3d referenced {uses} times in code, "
            "so the import is NOT unused and removing it would change behaviour")
    patched = source.replace("import open3d as o3d\n", marker + "\n", 1)
    if patched == source:
        raise RuntimeError(f"could not find the open3d import line in {path}")
    with open(path, "w") as f:
        f.write(patched)
    print(f"NOTE: neutralised the unused open3d import in {path}", flush=True)
    return True


def resolve_blender_dir(scene, mount="/kaggle/input/nerf-synthetic-dataset",
                        root="/kaggle/input"):
    """The dataset directory whose direct child is `scene`."""
    for cand in (os.path.join(mount, "nerf_synthetic"), mount):
        if os.path.isfile(os.path.join(cand, scene, "transforms_train.json")):
            return cand
    matches = []
    for directory, _, files in os.walk(root):
        if os.path.basename(directory) == scene and "transforms_train.json" in files:
            matches.append(os.path.dirname(directory))
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"could not find {scene}/transforms_train.json under {root}; "
                       f"matches={matches}")


def ply_stats(ply_path):
    """(count, max|filter_3D|) straight from the saved point cloud.

    save_ply writes filter_3D per Gaussian and load_ply reads it back, so
    render.py never recomputes it. The PLY is therefore the artefact that
    decides whether arm B really is 3DGS (F5).
    """
    import numpy as np
    from plyfile import PlyData
    v = PlyData.read(ply_path)["vertex"]
    return int(v.count), float(np.abs(np.asarray(v["filter_3D"])).max())


# ------------------------------------------------------------------- CSV
def append_rows(csv_path, schema, split, base):
    """Append one arm+scene's per-scale rows. Called as soon as they exist.

    Written per scene rather than once at the end: a session that dies must not
    also throw away the scenes that already finished.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=schema)
        if new:
            w.writeheader()
        for sc, v in split.items():
            row = {c: "" for c in schema}
            row.update(base)
            row.update({
                "test_scale": sc,
                "psnr": f"{v['PSNR']:.4f}",
                "ssim": f"{v['SSIM']:.5f}",
                "lpips": f"{v['LPIPS']:.5f}",
                "lpips_backbone": "vgg",
            })
            row["run_id"] = f"{base['run_id']}-{sc.replace('/', '')}"
            w.writerow(row)
    print(f"appended {len(split)} rows to {csv_path}", flush=True)


# --------------------------------------------------------------- hygiene
def prune_renders(model_dir, method, keep=6):
    """Delete the rendered PNGs once metrics have consumed them (§8).

    8 scenes x 4 scales x 200 views x 2 (render + gt) x 2 arms is several GB of
    PNGs, and /kaggle/working is ~20 GB and ships as the kernel's output.
    """
    d = os.path.join(model_dir, "test", method)
    if not os.path.isdir(d):
        return 0
    removed = 0
    for sub in os.listdir(d):
        p = os.path.join(d, sub)
        if not os.path.isdir(p):
            continue
        names = sorted(os.listdir(p))
        for fn in names[keep:]:
            os.remove(os.path.join(p, fn))
            removed += 1
    return removed


def du(path):
    try:
        out = subprocess.run(["du", "-sm", path], capture_output=True, text=True,
                             timeout=120).stdout.split()
        return int(out[0]) if out else -1
    except Exception:
        return -1
