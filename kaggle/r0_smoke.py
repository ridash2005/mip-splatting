#!/usr/bin/env python3
"""
R0 — smoke run (§7 R0 of docs/REPRODUCTION-PROMPT.md).

`lego`, 7000 iterations, both arms, end to end, plus the locally-verifiable
subset of §6's twelve pre-flight checks. Checks 5/6 (the numeric unit tests
against 33.3/33.4 dB) are calibrated to a 30000-iteration run (F11) — at
this rung's 7000 iterations they are recorded, honestly, as informational
only, never fudged to look like a pass (§14). Check 12's Hugging Face leg
needs a token this run doesn't have; it is marked skipped, not faked.

G1 requires "measured iterations/second recorded, replacing every estimate",
so this run times every stage and writes train_seconds / render_fps /
n_gaussians / model_mb / peak_vram_mb into results/runs.csv rather than
leaving those columns blank.

Runs inside a Kaggle kernel with a GPU attached and the
`nguyenhung1903/nerf-synthetic-dataset` dataset attached. Kaggle versions of
that dataset place the scene directory either directly in its mount or under
`nerf_synthetic`; both layouts are supported. Everything left under
/kaggle/working at exit becomes the kernel's downloadable output, so the run
prunes itself before finishing (§8 disk hygiene).

Every line this script asserts, it also prints — the calling process reads
the log back and does not need to re-derive anything. The final line is a
single JSON object prefixed R0_SUMMARY_JSON= for machine parsing.
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

WORK = "/kaggle/working"
REPO = "https://github.com/ridash2005/mip-splatting.git"
DATASET_MOUNT = "/kaggle/input/nerf-synthetic-dataset"
KAGGLE_INPUT = "/kaggle/input"
SCENE = "lego"
ITERS = 7000
RESUME_AT = 2000
LOGDIR = f"{WORK}/logs"
KEEP_QUALITATIVE = 8      # renders kept per arm after metrics, for the report

checks = []
timings = {}
notes = []

os.makedirs(LOGDIR, exist_ok=True)


def check(id_, desc, passed, detail=""):
    checks.append({"id": id_, "desc": desc, "passed": passed, "detail": str(detail)})
    tag = "PASS" if passed is True else ("SKIP" if passed is None else "FAIL")
    print(f"[check {id_}] {tag} — {desc} — {detail}", flush=True)


# --------------------------------------------------------------------- shell
_PROGRESS = re.compile(r"(it/s|s/it|\d+%\|)")


def sh(cmd, cwd=None, check_rc=True, log_name=None, echo_progress_every=60):
    """Run a command, tee its output to a file, echo a readable subset.

    Reads raw chunks and splits on CR *and* LF, because train.py's tqdm bar
    redraws with bare carriage returns: iterating the pipe by line would block
    for the whole run and then emit one multi-megabyte line. Progress redraws
    are throttled to one every `echo_progress_every` seconds so the Kaggle log
    stays readable; the complete stream still lands in logs/<name>.log, which
    ships as kernel output.
    """
    print(f"+ {cmd}" + (f"   (cwd={cwd})" if cwd else ""), flush=True)
    log_path = os.path.join(LOGDIR, (log_name or "cmd") + ".log")
    buf, pending, last_progress = [], "", [0.0]
    with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"\n$ {cmd}\n")
        p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace",
                             bufsize=0)

        def emit(seg):
            if not seg.strip():
                return
            if _PROGRESS.search(seg):
                now = time.time()
                if now - last_progress[0] < echo_progress_every:
                    return
                last_progress[0] = now
            print(seg.rstrip(), flush=True)

        while True:
            chunk = p.stdout.read(4096)
            if not chunk:
                break
            buf.append(chunk)
            lf.write(chunk)
            pending += chunk
            parts = re.split(r"[\r\n]", pending)
            pending = parts.pop()
            for seg in parts:
                emit(seg)
        emit(pending)
        rc = p.wait()
    out = "".join(buf)
    if check_rc and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}\n--- tail ---\n{out[-3000:]}")
    return rc, out


def timed(key, fn):
    t0 = time.time()
    try:
        return fn()
    finally:
        timings[key] = time.time() - t0
        print(f"[timing] {key}: {timings[key]:.1f}s", flush=True)


def git_hash(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------ peak VRAM probe
class VramProbe(threading.Thread):
    """Sample nvidia-smi for the duration of a stage and keep the maximum.

    Measured from outside the training process on purpose: train.py never
    reports its own peak, and patching it to do so would be a code change
    beyond §3.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mb = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(2.0):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                self.peak_mb = max([self.peak_mb] + [int(v) for v in out.split()
                                                     if v.strip().isdigit()])
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        self.join(timeout=6)
        return self.peak_mb


# ------------------------------------------------------------ build-time fixes
def patch_simple_knn_flt_max(repo_dir):
    """simple_knn.cu uses FLT_MAX without including <cfloat>.

    Older CUDA/GCC toolchains pulled <float.h> in transitively; CUDA 12.8 with
    GCC 13 does not, and the build fails outright. This is a compile fix, not a
    change to the extension's algorithm — the file has no other edit.
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
    """Last-resort fallback when open3d will not install on the kernel image.

    train.py imports open3d and never uses it. This is verified here, not
    assumed: the identifier must appear exactly once in the file (the import
    itself). Nothing in the optimiser, the model or the metrics touches it, so
    the import's only effect is to abort the run on an image without open3d.
    Applied only if `pip install open3d` fails, and always recorded in the run's
    notes so the report can state it.
    """
    path = os.path.join(repo_dir, "train.py")
    with open(path) as f:
        source = f.read()
    uses = len(re.findall(r"\bo3d\b", source)) + len(re.findall(r"\bopen3d\b", source))
    if uses != 2:      # 'open3d' and 'o3d' on the import line, nowhere else
        raise RuntimeError(
            f"refusing to touch {path}: open3d/o3d referenced {uses} times, "
            "so the import is NOT unused and removing it would change behaviour")
    with open(path, "w") as f:
        f.write(source.replace(
            "import open3d as o3d\n",
            "# import open3d as o3d  # unused; neutralised by kaggle/r0_smoke.py\n", 1))
    print(f"NOTE: neutralised the unused open3d import in {path}", flush=True)
    return True


def resolve_blender_dir():
    """Return the dataset directory whose direct child is the requested scene."""
    for candidate in (os.path.join(DATASET_MOUNT, "nerf_synthetic"), DATASET_MOUNT):
        if os.path.isfile(os.path.join(candidate, SCENE, "transforms_train.json")):
            print(f"using Blender dataset directory: {candidate}", flush=True)
            return candidate
    matches = []
    if os.path.isdir(KAGGLE_INPUT):
        for directory, _, files in os.walk(KAGGLE_INPUT):
            if os.path.basename(directory) == SCENE and "transforms_train.json" in files:
                matches.append(os.path.dirname(directory))
    if len(matches) == 1:
        print(f"using discovered Blender dataset directory: {matches[0]}", flush=True)
        return matches[0]
    raise RuntimeError(
        f"could not find {SCENE}/transforms_train.json under {KAGGLE_INPUT}. "
        "Expected <mount>/nerf_synthetic/<scene> or <mount>/<scene>; "
        f"recursive matches={matches}.")


def ply_filter_3d_stats(ply_path):
    """max |filter_3D| and the Gaussian count, read straight from the saved PLY.

    gaussian_model.save_ply writes filter_3D as a per-Gaussian property and
    load_ply reads it back, so render.py never recomputes it. The PLY is
    therefore the artefact that decides whether arm B really is 3DGS (F5):
    filter_3D must be identically zero there, and non-zero on arm A.
    """
    import numpy as np
    from plyfile import PlyData  # the same reader gaussian_model.load_ply uses
    v = PlyData.read(ply_path)["vertex"]
    return int(v.count), float(np.abs(np.asarray(v["filter_3D"])).max())


# =============================================================== check 1: GPU
sh("nvidia-smi", check_rc=False, log_name="env")
import torch  # noqa: E402  (after nvidia-smi so the driver line prints first)

cap = torch.cuda.get_device_capability()
gpu_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  capability {cap}  "
      f"gpu {gpu_name} x{n_gpu}")
check(1, "accelerator compute capability >= 7.0, abort on P100/6.0 (F15)",
      cap >= (7, 0), f"{gpu_name} x{n_gpu} {cap}")
if cap < (7, 0):
    # Machine-readable so tools/kaggle_push.py can tell "wrong GPU, retry" apart
    # from a real failure. The kernels API exposes only a boolean enable_gpu —
    # there is no accelerator field in kagglesdk 0.1.28 — and Kaggle hands out
    # whichever GPU is free, so the same script gets a P100 on one launch and
    # 2xT4 on the next. Retrying is the only lever, and it costs ~30s of quota.
    print("R0_ABORT=WRONG_ACCELERATOR", flush=True)
    raise SystemExit(
        f"ABORT: {gpu_name} has compute capability {cap}, below 7.0 — refusing "
        "to train (F15). Re-launch until Kaggle allocates a T4.")

blender_dir = resolve_blender_dir()

# =============================================================== clone the arms
sh(f"git clone --recursive -b main {REPO} {WORK}/armA", log_name="clone")
sh(f"git clone --recursive -b arm-b-3dgs-baseline {REPO} {WORK}/armB", log_name="clone")
armA_commit = git_hash(f"{WORK}/armA")
armB_commit = git_hash(f"{WORK}/armB")
print("armA commit", armA_commit, "armB commit", armB_commit)

# =============================================================== check 2: build
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
install_ok, open3d_ok = True, False
try:
    # requirements.txt minus open3d, which is installed separately below so its
    # failure can be handled rather than aborting the whole install step.
    sh("pip install -q ninja gputil lpips plyfile opencv-python", log_name="install")
    # open3d before the extensions: if it perturbs the numpy/torch stack we want
    # to find out before spending ten minutes compiling against that stack.
    rc, _ = sh("pip install -q open3d", check_rc=False, log_name="install")
    try:
        import open3d  # noqa: F401
        open3d_ok = True
        print("open3d", open3d.__version__, flush=True)
    except Exception:
        traceback.print_exc()
    import torch as _t
    assert _t.cuda.is_available(), "torch lost CUDA after the open3d install"
    print(f"post-install torch {_t.__version__} cuda_ok={_t.cuda.is_available()}", flush=True)

    if not open3d_ok:
        for d in (f"{WORK}/armA", f"{WORK}/armB"):
            neutralise_unused_open3d_import(d)
        notes.append("open3d unavailable on this image; its unused import in "
                     "train.py was neutralised (verified unused, 2 references)")

    for d in (f"{WORK}/armA", f"{WORK}/armB"):
        patch_simple_knn_flt_max(d)
    sh("nvcc --version", check_rc=False, log_name="env")
    sh("g++ --version", check_rc=False, log_name="env")
    # Arms A and B ship identical submodule sources (the §3 diff never touches
    # submodules/), so one build serves both. Built one at a time and verbose:
    # pip otherwise summarises a failed setup.py build as "see above for output"
    # with the real compiler error nowhere above it.
    sh(f"pip install -v {WORK}/armA/submodules/diff-gaussian-rasterization",
       log_name="build")
    sh(f"pip install -v {WORK}/armA/submodules/simple-knn", log_name="build")
    # Import everything train.py / render.py / metrics.py pull in, before the
    # data conversion rather than after it. `open3d` is checked separately above;
    # the rest are requirements.txt plus what the modules import transitively.
    # A missing one here costs five seconds; the same failure after
    # convert_blender_data.py costs several minutes of a metered session.
    missing = []
    for mod in ("GPUtil", "lpips", "plyfile", "cv2", "torchvision", "tqdm",
                "numpy", "PIL", "diff_gaussian_rasterization", "simple_knn._C"):
        try:
            __import__(mod)
        except Exception as e:
            missing.append(f"{mod} ({type(e).__name__})")
    if missing:
        raise RuntimeError("modules the training path needs are missing: "
                           + ", ".join(missing))
except Exception:
    install_ok = False
    traceback.print_exc()
check(2, "ninja/gputil/lpips/open3d + rasteriser & simple-knn built against installed torch",
      install_ok, f"open3d={'installed' if open3d_ok else 'unavailable, import neutralised'}")
if not install_ok:
    raise SystemExit("ABORT: extension build failed, nothing downstream is trustworthy.")

check(3, "commits pinned into this run", True,
      f"armA={armA_commit} armB={armB_commit} torch={torch.__version__} "
      f"cuda={torch.version.cuda}")

# =============================================================== check 4: diff
with open(f"{WORK}/armB/scene/gaussian_model.py") as f:
    armB_src = f.read()
with open(f"{WORK}/armA/scene/gaussian_model.py") as f:
    armA_src = f.read()
_, diff_out = sh(f"git -C {WORK}/armB diff origin/main..HEAD --stat",
                 check_rc=False, log_name="diff")
touched = {ln.split("|")[0].strip() for ln in diff_out.splitlines() if "|" in ln}
expected = {"arguments/__init__.py", "scene/gaussian_model.py", "render.py", "train.py"}
check(4, "§3 diff present on arm B, absent from main, and confined to its four files",
      "_disable_3D_filter" in armB_src and "torch.zeros_like(filter_3D" in armB_src
      and "_disable_3D_filter" not in armA_src and touched <= expected,
      f"files touched vs main: {sorted(touched)}")

# =============================================================== data + check 7
timed("convert_blender_data", lambda: sh(
    f"python {WORK}/armA/convert_blender_data.py --blender_dir {blender_dir} "
    f"--object_name {SCENE} --out_dir {WORK}/multi-scale", log_name="convert"))

with open(f"{WORK}/multi-scale/{SCENE}/metadata.json") as f:
    meta = json.load(f)


def scales_of(paths):
    return sorted({p.rsplit("_d", 1)[1].split(".")[0] for p in paths})


test_paths = meta["test"]["file_path"]
train_paths = meta["train"]["file_path"]
n_train_d0 = sum(1 for p in train_paths if p.endswith("d0.png"))
# F3 says the *loaded* training split is d0-only under the default
# load_allres=False — readMultiScale(..., only_highres) skips every non-d0 file
# at load time. metadata.json itself legitimately lists all four scales for
# train, so asserting on the file alone (as an earlier revision did) fails a
# correct pipeline. The real assertion is made below against the count train.py
# actually prints when it builds the Scene.
check(7, "multi-scale data written: 4 distinct test scales; metadata lists all "
         "4 train scales, of which only d0 may survive readMultiScale (F3)",
      scales_of(test_paths) == ["0", "1", "2", "3"]
      and scales_of(train_paths) == ["0", "1", "2", "3"] and n_train_d0 > 0,
      f"test={scales_of(test_paths)} n_test={len(test_paths)} "
      f"train={scales_of(train_paths)} n_train={len(train_paths)} d0={n_train_d0}")

# =============================================================== check 8: LPIPS
with open(f"{WORK}/armA/metrics.py") as f:
    metrics_src = f.read()
check(8, "LPIPS backbone is vgg, never alexnet (F9)",
      "lpips.LPIPS(net='vgg')" in metrics_src and "alex" not in metrics_src)


# =============================================================== train / score
def run_arm(arm_dir, out_dir, extra_flags, label, tag):
    stats = {}
    probe = VramProbe()
    probe.start()
    t0 = time.time()
    # --test_iterations -1: the default [7000, 30000] would run a full 800-view
    # test evaluation mid-training, which is pure cost at this rung (F16 uses
    # the same flag for the real scenes).
    _, train_out = sh(
        f"OMP_NUM_THREADS=4 python train.py -s {WORK}/multi-scale/{SCENE} -m {out_dir} "
        f"--eval --white_background --iterations {ITERS} --test_iterations -1 {extra_flags}",
        cwd=arm_dir, log_name=f"train_{tag}")
    stats["train_seconds"] = time.time() - t0
    stats["peak_vram_mb"] = probe.stop()
    stats["iters_per_second"] = ITERS / stats["train_seconds"]
    timings[f"train_{tag}"] = stats["train_seconds"]
    print(f"[timing] train_{tag}: {stats['train_seconds']:.1f}s "
          f"({stats['iters_per_second']:.2f} it/s), peak VRAM {stats['peak_vram_mb']} MB",
          flush=True)

    m = re.search(r"number of training images:\s*(\d+)", train_out)
    stats["n_train_images"] = int(m.group(1)) if m else None

    t0 = time.time()
    sh(f"OMP_NUM_THREADS=4 python render.py -m {out_dir} --skip_train",
       cwd=arm_dir, log_name=f"render_{tag}")
    stats["render_seconds"] = time.time() - t0

    method_dir = os.path.join(out_dir, "test", f"ours_{ITERS}")
    preds = [d for d in os.listdir(method_dir) if d.startswith("test_preds_")]
    n_rendered = len(os.listdir(os.path.join(method_dir, preds[0]))) if preds else 0
    stats["n_rendered"] = n_rendered
    stats["render_fps"] = n_rendered / stats["render_seconds"] if n_rendered else None
    print(f"[timing] render_{tag}: {stats['render_seconds']:.1f}s for {n_rendered} views "
          f"({stats['render_fps'] or 0:.2f} fps)", flush=True)

    ply = os.path.join(out_dir, "point_cloud", f"iteration_{ITERS}", "point_cloud.ply")
    stats["n_gaussians"], stats["filter_3D_max"] = ply_filter_3d_stats(ply)
    stats["model_mb"] = os.path.getsize(ply) / 1e6
    print(f"[model] {label}: {stats['n_gaussians']} gaussians, "
          f"{stats['model_mb']:.1f} MB, max|filter_3D| = {stats['filter_3D_max']:.6g}",
          flush=True)

    t0 = time.time()
    sh(f"OMP_NUM_THREADS=4 python metrics.py -m {out_dir}", cwd=arm_dir,
       log_name=f"metrics_{tag}")
    timings[f"metrics_{tag}"] = time.time() - t0

    results_path = os.path.join(out_dir, "results.json")
    ok = os.path.exists(results_path) and os.path.getsize(results_path) > 0
    check(10, f"metrics.py produced a non-empty results.json ({label}) — its bare "
              "`except:` makes a real failure print like a warning (F10)", ok, results_path)
    if not ok:
        raise RuntimeError(f"{label}: metrics.py silently failed — see {results_path} (F10)")
    with open(results_path) as f:
        return json.load(f), stats


out_armA = f"{WORK}/out_armA/{SCENE}"
out_armB = f"{WORK}/out_armB/{SCENE}"

# =============================================================== check 9: splitter
# Imported before the arms run so a broken splitter fails in seconds rather than
# after both arms have trained.
sys.path.insert(0, f"{WORK}/armA/tools")
import split_by_scale as sbs  # noqa: E402

now = datetime.now(timezone.utc).isoformat(timespec="seconds")
os.makedirs(f"{WORK}/results", exist_ok=True)
csv_path = f"{WORK}/results/runs.csv"
run_id = f"r0-{now.replace(':', '').replace('-', '')}"


def unpack(res):
    """Pull the pooled metrics and the method name out of results.json.

    metrics.py writes `json.dump(full_dict[scene_dir], ...)` — the value, not
    the enclosing dict — so the top-level key is the method directory name
    (`ours_7000`), not the scene path. Only one method directory exists per run.
    """
    methods = [k for k, v in res.items() if isinstance(v, dict) and "PSNR" in v]
    if len(methods) != 1:
        raise RuntimeError(f"expected exactly one method block in results.json, "
                           f"found {methods} (top-level keys {list(res)})")
    d = res[methods[0]]
    return d["PSNR"], d["SSIM"], d["LPIPS"], methods[0]


def verify_and_split(out_dir, method, pooled_psnr, label):
    res = sbs.split_scene(out_dir, f"{WORK}/multi-scale/{SCENE}", method)
    n_total = sum(v["n"] for v in res.values())
    weighted = sum(v["n"] * v["PSNR"] for v in res.values()) / n_total
    check(9, f"split_by_scale.py's count-weighted mean reproduces metrics.py's "
             f"pooled PSNR ({label}) — proves the render-index → scale mapping",
          abs(weighted - pooled_psnr) < 0.01,
          f"weighted={weighted:.4f} pooled={pooled_psnr:.4f} n_total={n_total} "
          f"n_per_scale={ {k: v['n'] for k, v in res.items()} }")
    for sc in ("1x", "1/2", "1/4", "1/8"):
        if sc in res:
            print(f"  {label} {sc:>4}  n={res[sc]['n']:<4} PSNR {res[sc]['PSNR']:7.3f}  "
                  f"SSIM {res[sc]['SSIM']:.4f}  LPIPS {res[sc]['LPIPS']:.4f}", flush=True)
    return res


def append_rows(arm_letter, split, commit, ks, disable, st):
    """Append this arm's four rows to results/runs.csv as soon as they exist.

    Written per arm rather than once at the end: getting a compute-capability
    7.0+ session is the scarce resource here, and a session that dies during
    arm B must not also throw away arm A.
    """
    note = (f"R0 smoke run, {ITERS} iters — plumbing check, not the R1 numeric target"
            + ("; " + "; ".join(notes) if notes else ""))
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sbs.RUNS_CSV_SCHEMA)
        if new:
            w.writeheader()
        for sc, v in split.items():
            row = {c: "" for c in sbs.RUNS_CSV_SCHEMA}
            row.update({
                "run_id": f"{run_id}-{arm_letter}-{sc.replace('/', '')}",
                "timestamp_utc": now,
                "method": "mip-splatting" if arm_letter == "A" else "3dgs",
                "arm": arm_letter,
                "impl_commit": commit,
                # The rasteriser lives in-repo, not as a separate submodule
                # pointer, so its commit is this repo's commit.
                "rasteriser_commit": commit,
                "dataset": "blender",
                "scene": SCENE,
                "resolution_flag": "-1",
                "load_allres": "False",
                "kernel_size": ks,
                "disable_3D_filter": disable,
                "train_scale": "1x",
                "test_scale": sc,
                "iterations": str(ITERS),
                # train.py calls safe_state(), which seeds torch/random to 0
                # unconditionally; there is no --seed flag to vary it.
                "seed": "0",
                "psnr": f"{v['PSNR']:.4f}",
                "ssim": f"{v['SSIM']:.5f}",
                "lpips": f"{v['LPIPS']:.5f}",
                "lpips_backbone": "vgg",
                "n_gaussians": str(st["n_gaussians"]),
                "model_mb": f"{st['model_mb']:.2f}",
                "peak_vram_mb": str(st["peak_vram_mb"]),
                "train_seconds": f"{st['train_seconds']:.1f}",
                "render_fps": f"{st['render_fps']:.3f}" if st["render_fps"] else "",
                "gpu_model": gpu_name,
                "platform": "kaggle",
                "level_claimed": "L3",
                "notes": note,
            })
            w.writerow(row)
    print(f"appended arm {arm_letter}'s {len(split)} rows to {csv_path}", flush=True)


def do_arm(arm_letter, arm_dir, out_dir, flags, label, tag, commit, ks, disable):
    res, st = run_arm(arm_dir, out_dir, flags, label, tag)
    psnr, ssim, lp, method = unpack(res)
    print(f"{label} pooled (all 4 test scales): PSNR {psnr:.3f}  SSIM {ssim:.4f}  "
          f"LPIPS {lp:.4f}", flush=True)
    split = verify_and_split(out_dir, method, psnr, label)
    append_rows(arm_letter, split, commit, ks, disable, st)
    return {"pooled": {"psnr": psnr, "ssim": ssim, "lpips": lp},
            "split": split, "stats": st, "method": method}


A = do_arm("A", f"{WORK}/armA", out_armA, "--kernel_size 0.1",
           "arm A / Mip-Splatting", "armA", armA_commit, "0.1", "False")
B = do_arm("B", f"{WORK}/armB", out_armB, "--kernel_size 0.3 --disable_3D_filter",
           "arm B / 3DGS", "armB", armB_commit, "0.3", "True")

statA, statB = A["stats"], B["stats"]
splitA, splitB = A["split"], B["split"]
psnrA, ssimA, lpipsA = (A["pooled"][k] for k in ("psnr", "ssim", "lpips"))
psnrB, ssimB, lpipsB = (B["pooled"][k] for k in ("psnr", "ssim", "lpips"))

# The F3 assertion that actually exercises the code path: with the default
# load_allres=False, Scene must be built from the d0 files alone.
check(7.1, "readMultiScale loaded the d0-only training split (F3, load_allres=False)",
      statA["n_train_images"] == n_train_d0 == statB["n_train_images"],
      f"armA loaded {statA['n_train_images']}, armB loaded {statB['n_train_images']}, "
      f"d0 files in metadata = {n_train_d0} (of {len(train_paths)} total)")

# §3's correctness argument, checked against the artefact rather than the source:
# arm B is 3DGS only if filter_3D is identically zero in the saved model.
check(4.1, "arm B's saved model has filter_3D ≡ 0 and arm A's does not (F5) — "
           "this, not the diff text, is what makes arm B exactly 3DGS",
      statB["filter_3D_max"] == 0.0 and statA["filter_3D_max"] > 0.0,
      f"max|filter_3D|: armA={statA['filter_3D_max']:.6g} armB={statB['filter_3D_max']:.6g}")


# §6 checks 5/6's 33.3/33.4 dB targets are calibrated to 30000 iterations (F11)
# and to the FULL-RESOLUTION column, not this pooled four-scale average. At
# ITERS=7000 they are recorded as measured, never scored against a target this
# run was never going to meet (§14).
check(5, "arm B (3DGS) unit test target 33.3±0.5 dB — informational at "
         f"{ITERS} iters, the real check is R1 @ 30k full-res", None,
      f"pooled {psnrB:.3f} dB; per-scale reported below")
check(6, "arm A (Mip-Splatting) sanity target 33.4±0.5 dB — informational at "
         f"{ITERS} iters, the real check is R1 @ 30k full-res", None,
      f"pooled {psnrA:.3f} dB; per-scale reported below")


def build_summary():
    return {
        "rung": "R0",
        "scene": SCENE,
        "iterations": ITERS,
        "gpu": gpu_name,
        "n_gpu": n_gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "open3d": open3d_ok,
        "armA_commit": armA_commit,
        "armB_commit": armB_commit,
        "pooled": {"A": A["pooled"], "B": B["pooled"]},
        "per_scale_A": {k: v["PSNR"] for k, v in splitA.items()},
        "per_scale_B": {k: v["PSNR"] for k, v in splitB.items()},
        "per_scale_A_full": {k: {m: v[m] for m in ("PSNR", "SSIM", "LPIPS", "n")}
                             for k, v in splitA.items()},
        "per_scale_B_full": {k: {m: v[m] for m in ("PSNR", "SSIM", "LPIPS", "n")}
                             for k, v in splitB.items()},
        "stats_A": statA,
        "stats_B": statB,
        "timings_seconds": dict(timings),
        "gpu_hours_used": sum(timings.values()) / 3600.0,
        "notes": list(notes),
        "checks": [dict(c) for c in checks],
        "all_checks_pass_or_skip": all(c["passed"] is not False for c in checks),
    }


# Both arms are complete and their rows are already in results/runs.csv. Write
# the summary now, before the two remaining stages — the resume test trains
# another 9 000 iterations and the session could hit its cap inside it. A later
# write overwrites this one with the fuller picture; a session that dies first
# still leaves a report-ready summary behind.
os.makedirs(f"{WORK}/results", exist_ok=True)
with open(f"{WORK}/results/r0_summary.json", "w") as f:
    json.dump(build_summary(), f, indent=2)
print(f"wrote interim {WORK}/results/r0_summary.json (both arms complete)", flush=True)

# =============================================================== check 9.1
# The whole reporting chain (splitter → CSV → tables → figures) on a synthetic
# fixture with known answers, so a wrong number downstream cannot hide behind a
# plausible-looking table.
rc_self, _ = sh(f"python {WORK}/armA/tools/selftest_pipeline.py", check_rc=False,
                log_name="selftest")
check(9.1, "tools/selftest_pipeline.py: reporting chain verified against a fixture "
           "with planted, known-in-advance answers", rc_self == 0, f"exit {rc_self}")

# =============================================================== check 11: resume
resume_dir = f"{WORK}/out_resume_test/{SCENE}"
resume_ok, resume_detail = False, ""
try:
    sh(f"OMP_NUM_THREADS=4 python train.py -s {WORK}/multi-scale/{SCENE} -m {resume_dir} "
       f"--eval --white_background --iterations {RESUME_AT} --test_iterations -1 "
       f"--checkpoint_iterations {RESUME_AT} --kernel_size 0.1",
       cwd=f"{WORK}/armA", log_name="resume")
    ckpt = f"{resume_dir}/chkpnt{RESUME_AT}.pth"
    ckpt_ok = os.path.exists(ckpt)
    sh(f"OMP_NUM_THREADS=4 python train.py -s {WORK}/multi-scale/{SCENE} -m {resume_dir} "
       f"--eval --white_background --iterations {ITERS} --test_iterations -1 "
       f"--start_checkpoint {ckpt} --kernel_size 0.1",
       cwd=f"{WORK}/armA", log_name="resume")
    final_ply = f"{resume_dir}/point_cloud/iteration_{ITERS}/point_cloud.ply"
    resume_ok = ckpt_ok and os.path.exists(final_ply)
    resume_detail = f"checkpoint at {RESUME_AT} -> final model at {ITERS}: {final_ply}"
except Exception as e:
    resume_detail = f"{type(e).__name__}: {e}"[:400]
    traceback.print_exc()
check(11, f"kill at {RESUME_AT}, restart from checkpoint, reach {ITERS} "
          "(scaled down from 30K for this rung)", resume_ok, resume_detail)

# =============================================================== check 12
check(12, "off-session persistence", None,
      "Kaggle leg satisfied — results/, logs/ and both models persist as kernel "
      "output and are pulled down by tools/kaggle_push.py. Hugging Face Hub leg "
      "skipped: no HF token is available to this run; not faked.")

# =============================================================== disk hygiene (§8)
# /kaggle/working is the kernel's output. Left alone it would ship both git
# clones, the regenerated multi-scale dataset and ~3200 PNGs.
def prune():
    freed = []
    for arm_out in (out_armA, out_armB, resume_dir):
        method_dir = os.path.join(arm_out, "test", f"ours_{ITERS}")
        if not os.path.isdir(method_dir):
            continue
        for sub in sorted(os.listdir(method_dir)):
            d = os.path.join(method_dir, sub)
            if not os.path.isdir(d):
                continue
            keep = sorted(os.listdir(d))[:KEEP_QUALITATIVE]
            for fn in os.listdir(d):
                if fn not in keep:
                    os.remove(os.path.join(d, fn))
            freed.append(f"{d} -> kept {len(keep)}")
    # Optimiser-state checkpoints are ~3x the model and exist only to prove
    # check 11; the final .ply is the artefact worth keeping.
    for arm_out in (out_armA, out_armB, resume_dir):
        if not os.path.isdir(arm_out):
            continue
        for fn in os.listdir(arm_out):
            if fn.startswith("chkpnt") and fn.endswith(".pth"):
                os.remove(os.path.join(arm_out, fn))
                freed.append(f"removed {arm_out}/{fn}")
    for path in (f"{WORK}/multi-scale", f"{WORK}/armA/.git", f"{WORK}/armB/.git",
                 f"{WORK}/armA/submodules", f"{WORK}/armB/submodules",
                 f"{WORK}/armA/assets", f"{WORK}/armB/assets",
                 f"{WORK}/armA/media", f"{WORK}/armB/media"):
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            freed.append(f"removed {path}")
    for line in freed:
        print("  prune:", line, flush=True)


try:
    prune()
    _, du = sh(f"du -sh {WORK} {WORK}/* 2>/dev/null | sort -h | tail -20",
               check_rc=False, log_name="prune")
except Exception:
    traceback.print_exc()

# =============================================================== summary
summary = build_summary()
with open(f"{WORK}/results/r0_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("R0_SUMMARY_JSON=" + json.dumps(summary))
if not summary["all_checks_pass_or_skip"]:
    raise SystemExit("one or more §6 checks FAILED — see the table above.")
