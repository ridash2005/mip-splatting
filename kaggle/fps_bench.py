#!/usr/bin/env python3
"""
C9 --- matched render speed for 3DGS, Mip-Splatting and B1.

Why this exists. Chapter 4 argues that B1 cannot cost anything at render time,
because Lambda is recomputed on the filter's own schedule during training and
never inside the render loop. That is an argument about code, and the thesis
should not be shipping arguments where it can ship measurements. The `render_fps`
column every other rung records is unusable for the purpose: it is
(views rendered) / (wall time of render.py), so it carries the process start,
the scene load, the PLY parse, PNG encoding and disk writes, which is why it
reads 8--10 against a published 180--300 and would be read as a catastrophic
regression by anyone who did not check what it measured.

What is measured here instead: the render call alone, on one scene, on one GPU,
in one session, with CUDA synchronised around each timed call and the first
`WARMUP` views discarded. The quantity the thesis defends is the RATIO between
the arms, so every shared cost is deliberately excluded from all three equally.

One rasteriser build serves all three arms. The C1 vanilla rasteriser differs
from this one only in whether the opacity compensation term is evaluated -- one
multiply per primitive, inside a kernel that is memory-bound on the sort -- so
the 3DGS row is produced here by 3DGS's band-limit semantics
(`filter_3D == 0`, `kernel_size = 0.3`, exact by Proposition 2) on the shared
build rather than by rebuilding CUDA twice. That is a statement about what is
being compared and it is recorded in the summary, not left implicit.

Models are mounted from the sessions that already trained them; nothing is
retrained. Ends with R9_SUMMARY_JSON= for the pusher.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
REPO = "https://github.com/ridash2005/mip-splatting.git"
DATA_MOUNT = "/kaggle/input/nerf-synthetic-dataset"
DATA_ROOT = "/kaggle/input"
BRANCH = "method-b1-b2"
LOGDIR = f"{WORK}/logs"
RESULTS = f"{WORK}/results"

# --- parameters, rewritten by `kaggle_push.py --set NAME=VALUE` --------------
RUNG = "C9"
ITERS = 30000
SCENE = "lego"
WARMUP = 20          # views discarded before timing starts
REPEATS = 3          # passes over the test set; the median pass is reported
# ----------------------------------------------------------------------------

METHOD_DIR = f"ours_{ITERS}"
SESSION_START = time.time()
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

subprocess.run("nvidia-smi", shell=True)
import torch  # noqa: E402

cap = torch.cuda.get_device_capability()
gpu_name, n_gpu = torch.cuda.get_device_name(), torch.cuda.device_count()
print(f"torch {torch.__version__} cuda {torch.version.cuda} cap {cap} {gpu_name} x{n_gpu}",
      flush=True)
if cap < (7, 0):
    print(f"{RUNG}_ABORT=WRONG_ACCELERATOR", flush=True)
    raise SystemExit(f"ABORT: {gpu_name} capability {cap} is below 7.0 (F15).")

subprocess.run(f"git clone --recursive -b {BRANCH} {REPO} {WORK}/repo", shell=True,
               check=True)
sys.path.insert(0, f"{WORK}/repo/tools")
import kernel_common as kc  # noqa: E402

commit = kc.git_hash(f"{WORK}/repo")
print("repo", commit, flush=True)

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
kc.sh("pip install -q ninja gputil lpips plyfile opencv-python", logdir=LOGDIR,
      log_name="install")
rc, _ = kc.sh("pip install -q open3d", check_rc=False, logdir=LOGDIR, log_name="install")
try:
    import open3d  # noqa: F401
except Exception:
    kc.neutralise_unused_open3d_import(f"{WORK}/repo")
kc.patch_simple_knn_flt_max(f"{WORK}/repo")
kc.sh(f"pip install -v {WORK}/repo/submodules/diff-gaussian-rasterization",
      logdir=LOGDIR, log_name="build")
kc.sh(f"pip install -v {WORK}/repo/submodules/simple-knn", logdir=LOGDIR,
      log_name="build")
print("build OK", flush=True)

# ---------------------------------------------------------------- the data
blender_dir = kc.resolve_blender_dir(SCENE, mount=DATA_MOUNT, root=DATA_ROOT)
print("blender dir:", blender_dir, flush=True)
if not os.path.exists(f"{WORK}/multi-scale/{SCENE}/metadata.json"):
    kc.sh(f"python {WORK}/repo/convert_blender_data.py --blender_dir {blender_dir} "
          f"--object_name {SCENE} --out_dir {WORK}/multi-scale",
          logdir=LOGDIR, log_name="convert")

# ------------------------------------------------------- the trained models
# Discovered rather than named: which sessions are mounted depends on what has
# finished, and a missing arm should cost a row, not the run.
WANT = {
    "3dgs": dict(needle=f"out_armB/{SCENE}", kernel_size=0.3, disable_3D=True,
                 fisher=False, label="3DGS"),
    "mip": dict(needle=f"out/mip_{SCENE}_full", kernel_size=0.1, disable_3D=False,
                fisher=False, label="Mip-Splatting"),
    "b1": dict(needle=f"out/b1_{SCENE}_full", kernel_size=0.1, disable_3D=False,
               fisher=True, label="B1 --- Fisher band-limit"),
}
found = {}
for root, dirs, files in os.walk(DATA_ROOT):
    if root.count("/") > 11:
        dirs[:] = []
        continue
    if "point_cloud.ply" not in files or f"iteration_{ITERS}" not in root:
        continue
    model_dir = os.path.dirname(os.path.dirname(root))   # .../<model>/point_cloud/iter_N
    norm = model_dir.replace("\\", "/")
    for key, spec in WANT.items():
        if key not in found and norm.endswith(spec["needle"]):
            found[key] = model_dir
print(f"models found: { {k: v for k, v in found.items()} }", flush=True)
if not found:
    raise SystemExit("ABORT: no trained model was mounted; nothing to benchmark.")

# ------------------------------------------------------------- the benchmark
sys.path.insert(0, f"{WORK}/repo")
os.chdir(f"{WORK}/repo")
from argparse import ArgumentParser  # noqa: E402

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402


def bench(key):
    """fps of the render call alone, for one arm's own trained model."""
    spec = WANT[key]
    parser = ArgumentParser()
    mp, pp = ModelParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    args.source_path = f"{WORK}/multi-scale/{SCENE}"
    args.model_path = found[key]
    args.eval = True
    args.white_background = True
    args.resolution = -1
    args.kernel_size = spec["kernel_size"]
    args.disable_3D_filter = spec["disable_3D"]
    args.use_fisher_filter = spec["fisher"]
    dataset, pipe = mp.extract(args), pp.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=ITERS, shuffle=False)
        gaussians.use_fisher_filter = spec["fisher"]
        if spec["fisher"]:
            # Sigma_filt is geometry, so render.py rebuilds it from the training
            # cameras. It is built ONCE, here, outside the timed loop -- which is
            # exactly the property being measured, not a way of flattering it.
            t = time.time()
            gaussians.compute_fisher_filter(scene.getTrainCameras(),
                                            dataset.fisher_s, dataset.fisher_beta)
            setup_s = time.time() - t
        else:
            setup_s = 0.0
        bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        views = scene.getTestCameras()
        n_gauss = gaussians.get_xyz.shape[0]

        for v in views[:WARMUP]:
            render(v, gaussians, pipe, bg, kernel_size=dataset.kernel_size)
        torch.cuda.synchronize()

        passes = []
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for v in views:
                render(v, gaussians, pipe, bg, kernel_size=dataset.kernel_size)
            torch.cuda.synchronize()
            passes.append(len(views) / (time.perf_counter() - t0))

    res = {"label": spec["label"], "model": found[key], "n_gaussians": int(n_gauss),
           "n_views": len(views), "fps_passes": passes,
           "fps": statistics.median(passes),
           "fps_spread": max(passes) - min(passes),
           "filter_setup_seconds": setup_s,
           "resolution": [int(views[0].image_height), int(views[0].image_width)]}
    print(f"[{key}] {res['fps']:.1f} fps  ({n_gauss} gaussians, "
          f"{res['resolution'][1]}x{res['resolution'][0]}, "
          f"spread {res['fps_spread']:.1f})", flush=True)
    return res


results, failed = {}, {}
for key in ("3dgs", "mip", "b1"):
    if key not in found:
        print(f"[{key}] not mounted -- skipped", flush=True)
        continue
    try:
        results[key] = bench(key)
    except Exception as e:
        failed[key] = f"{type(e).__name__}: {e}"[:600]
        print(f"[{key}] FAILED: {failed[key][:200]}", flush=True)
        traceback.print_exc()

ratios = {}
if "mip" in results:
    base = results["mip"]["fps"]
    ratios = {k: v["fps"] / base for k, v in results.items()}

summary = {
    "rung": RUNG, "scene": SCENE, "iterations": ITERS, "commit": commit,
    "gpu": gpu_name, "n_gpu": n_gpu, "warmup_views": WARMUP, "repeats": REPEATS,
    "results": results, "failed": failed,
    "ratio_vs_mip": ratios,
    "one_rasteriser_build": True,
    "note": ("Render call only: no scene load, no PNG encode, no disk write. "
             "All three arms share one rasteriser build; the 3DGS row is 3DGS's "
             "band-limit semantics (filter_3D == 0, kernel_size 0.3) rather than "
             "the C1 vanilla build, which differs only in the opacity "
             "compensation multiply."),
    "session_seconds": time.time() - SESSION_START,
    "gpu_hours_used": (time.time() - SESSION_START) / 3600.0,
    "complete": len(results) == 3,
}
with open(f"{RESULTS}/fps.json", "w") as f:
    json.dump(summary, f, indent=2)
print("R9_SUMMARY_JSON=" + json.dumps(summary))
if failed:
    raise SystemExit(f"{len(failed)} arm(s) failed.")
