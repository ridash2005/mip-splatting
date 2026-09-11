#!/usr/bin/env python3
"""
Blender rungs R1 / R2 / R5 (§7 of docs/REPRODUCTION-PROMPT.md).

One kernel, parameterised by the constants below, which tools/kaggle_push.py
rewrites at push time with --set:

  R1  STMT, the PRIMARY TARGET   RUNG=R1 LOAD_ALLRES=False
  R2  MTMT                       RUNG=R2 LOAD_ALLRES=True   (F3: one flag)
  R5  seed spread                RUNG=R5 SEEDS=[1,2] SCENES=lego,chair

Eight Blender scenes, 30 000 iterations, both arms. R1 fills Table 1 of the
thesis and is what gate G2 is defined on. R0 established the toolchain and
replaced §7's estimates with measurement: ~21 it/s on a T4, so 30 000 iterations
is ~24 min per scene per arm, and the two T4s halve the wall clock.

Structure follows R0's, with the parts R0 proved reused from tools/kernel_common
rather than copied. Two things matter more here than at R0 because this run is
four hours long:

  * Every scene's rows are appended to results/runs.csv and its summary written
    the moment that scene finishes. A session that hits its cap after six scenes
    yields six scenes, not nothing.
  * Renders are deleted as soon as metrics have consumed them (§8). Kept whole,
    8 scenes x 4 scales x 200 views x 2 x 2 arms would overrun /kaggle/working.

The final line is a single JSON object prefixed R1_SUMMARY_JSON= for parsing.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

WORK = "/kaggle/working"
REPO = "https://github.com/ridash2005/mip-splatting.git"
# Where the Blender data lives. The Kaggle defaults; tools/run_local.py
# overrides both so the same kernel runs on a cluster node unchanged.
DATA_MOUNT = "/kaggle/input/nerf-synthetic-dataset"
DATA_ROOT = "/kaggle/input"
LOGDIR = f"{WORK}/logs"
RESULTS = f"{WORK}/results"
# --- parameters, rewritten by `kaggle_push.py --set NAME=VALUE` ---------------
RUNG = "R1"
LOAD_ALLRES = False        # F3: STMT vs MTMT is this one flag
SEEDS = [0]                # R5 sweeps these; see tools/seeded_train.py.
                           # Seed 0 is a plain train.py run, so an R5 sweep
                           # anchors to the R1/R2 numbers rather than redrawing.
SCENES = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
ITERS = 30000
ARM_B_BRANCH = "arm-b-3dgs-baseline"   # "arm-b-3dgs-vanilla" drops the 2D Mip
                                       # opacity compensation as well (C1). The
                                       # branch decides what arm B *is*, so it
                                       # is recorded in every CSV row's notes.
ARM_B_EXTRA = ""                       # "--disable_2D_mip_compensation" with
                                       # the vanilla branch; the flag exists
                                       # nowhere else and would be rejected.
ARMS_ENABLED = ["A", "B"]              # narrow to save quota when only one arm
                                       # is in question.
# -----------------------------------------------------------------------------
METHOD = f"ours_{ITERS}"
KEEP_QUALITATIVE = 6

SESSION_START = time.time()
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# ============================================================ accelerator (F15)
# Inline and before anything else: the kernels API cannot choose the GPU, so a
# session can start on a P100 that 3DGS cannot run on. See kaggle/README.md.
subprocess.run("nvidia-smi", shell=True)
import torch  # noqa: E402

cap = torch.cuda.get_device_capability()
gpu_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  capability {cap}  "
      f"gpu {gpu_name} x{n_gpu}", flush=True)
if cap < (7, 0):
    print("R1_ABORT=WRONG_ACCELERATOR", flush=True)   # marker name kept for the pusher
    raise SystemExit(f"ABORT: {gpu_name} has compute capability {cap}, below 7.0 (F15).")

# ============================================================ clone + build
subprocess.run(f"git clone --recursive -b main {REPO} {WORK}/armA", shell=True, check=True)
subprocess.run(f"git clone --recursive -b {ARM_B_BRANCH} {REPO} {WORK}/armB",
               shell=True, check=True)
sys.path.insert(0, f"{WORK}/armA/tools")
import kernel_common as kc          # noqa: E402
import split_by_scale as sbs        # noqa: E402

armA_commit, armB_commit = kc.git_hash(f"{WORK}/armA"), kc.git_hash(f"{WORK}/armB")
print("armA", armA_commit, "armB", armB_commit, flush=True)

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
kc.sh("pip install -q ninja gputil lpips plyfile opencv-python", logdir=LOGDIR,
      log_name="install")
rc, _ = kc.sh("pip install -q open3d", check_rc=False, logdir=LOGDIR, log_name="install")
open3d_ok = True
try:
    import open3d  # noqa: F401
except Exception:
    open3d_ok = False
    for d in (f"{WORK}/armA", f"{WORK}/armB"):
        kc.neutralise_unused_open3d_import(d)
for d in (f"{WORK}/armA", f"{WORK}/armB"):
    kc.patch_simple_knn_flt_max(d)
# Which arm's rasteriser gets installed. Normally they are byte-identical, so
# armA's is built once and both arms import it. C1 breaks that: the 2D Mip
# opacity compensation lives in CUDA, so the vanilla arm needs *its own* build.
# Only one `diff_gaussian_rasterization` can be installed per session, so the
# two cannot run together under that branch.
RASTERISER_ARM = "armB" if ARM_B_EXTRA else "armA"
if ARM_B_EXTRA and ARMS_ENABLED != ["B"]:
    raise SystemExit(
        "ABORT: a vanilla arm B needs its own rasteriser build, and only one "
        "can be installed per session. Run it with ARMS_ENABLED=B; arm A's "
        "numbers are already measured and are not re-run.")
print(f"installing the rasteriser from {RASTERISER_ARM}", flush=True)
kc.sh(f"pip install -v {WORK}/{RASTERISER_ARM}/submodules/diff-gaussian-rasterization",
      logdir=LOGDIR, log_name="build")
kc.sh(f"pip install -v {WORK}/armA/submodules/simple-knn", logdir=LOGDIR, log_name="build")
for mod in ("GPUtil", "lpips", "plyfile", "cv2", "torchvision", "tqdm", "numpy", "PIL",
            "diff_gaussian_rasterization", "simple_knn._C"):
    __import__(mod)
print(f"build OK (open3d={'yes' if open3d_ok else 'neutralised'})", flush=True)

# §3 must still be exactly the §3 diff, checked every run.
_, diff_out = kc.sh(f"git -C {WORK}/armB diff origin/main...HEAD --stat",
                    check_rc=False, logdir=LOGDIR, log_name="diff")
touched = {ln.split("|")[0].strip() for ln in diff_out.splitlines() if "|" in ln}
expected = {"arguments/__init__.py", "scene/gaussian_model.py", "render.py", "train.py"}
if ARM_B_EXTRA:
    # C1 removes the 2D Mip opacity compensation. That term is computed in CUDA,
    # so the diff necessarily reaches the rasteriser. Enumerated rather than
    # waved through by prefix, so the check still catches anything else.
    expected |= {
        "gaussian_renderer/__init__.py",
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu",
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.h",
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h",
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h",
        "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu",
        "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py",
        "submodules/diff-gaussian-rasterization/rasterize_points.cu",
        "submodules/diff-gaussian-rasterization/rasterize_points.h",
    }
if not touched <= expected:
    raise SystemExit(f"ABORT: arm B introduces changes beyond §3: {sorted(touched)}")
print(f"§3 diff confined to {sorted(touched)}", flush=True)

# The switch is only real if the build actually carries it. Checking the source
# would prove nothing about the binary that just got compiled.
if ARM_B_EXTRA:
    import diff_gaussian_rasterization as _dgr
    if "mip_compensation" not in _dgr.GaussianRasterizationSettings._fields:
        raise SystemExit(
            "ABORT: the installed rasteriser has no mip_compensation field, so "
            f"the build came from the wrong tree (expected {RASTERISER_ARM}).")
    print("rasteriser carries mip_compensation: the C1 switch is live", flush=True)

# ============================================================ data
blender_dir = kc.resolve_blender_dir(SCENES[0], mount=DATA_MOUNT, root=DATA_ROOT)
print("blender dir:", blender_dir, flush=True)
t0 = time.time()
for scene in SCENES:
    if os.path.exists(f"{WORK}/multi-scale/{scene}/metadata.json"):
        continue
    kc.sh(f"python {WORK}/armA/convert_blender_data.py --blender_dir {blender_dir} "
          f"--object_name {scene} --out_dir {WORK}/multi-scale",
          logdir=LOGDIR, log_name="convert")
convert_seconds = time.time() - t0
print(f"[timing] convert all {len(SCENES)} scenes: {convert_seconds:.0f}s", flush=True)

# ============================================================ the run
now = datetime.now(timezone.utc).isoformat(timespec="seconds")
run_id = f"r1-{now.replace(':', '').replace('-', '')}"
csv_path = f"{RESULTS}/runs.csv"
ARMS = {
    "A": dict(dir=f"{WORK}/armA", flags="--kernel_size 0.1", method="mip-splatting",
              commit=armA_commit, kernel_size="0.1", disable="False"),
    "B": dict(dir=f"{WORK}/armB",
              flags=f"--kernel_size 0.3 --disable_3D_filter {ARM_B_EXTRA}".strip(),
              method="3dgs", commit=armB_commit, kernel_size="0.3", disable="True"),
}
done, failed = {}, {}


def run_one(gpu, arm, scene, seed):
    """train -> render -> metrics -> split -> CSV -> prune, for one (arm, scene)."""
    cfg = ARMS[arm]
    suffix = "" if seed == 0 else f"_s{seed}"
    out = f"{WORK}/out_arm{arm}/{scene}{suffix}"
    tag = f"{arm}_{scene}{suffix}"
    env = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}"
    if seed:
        env += f" BTP_SEED={seed} BTP_KEEP_INIT=1"
    entry = "tools/seeded_train.py" if seed else "train.py"
    protocol_flags = "--load_allres" if LOAD_ALLRES else ""
    port = 6209 + int(gpu)
    st = {}
    probe = kc.VramProbe(gpu=gpu)
    probe.start()
    t = time.time()
    src = scene_src(scene, seed)
    kc.sh(f"{env} python {entry} -s {src} -m {out} --eval "
          f"--white_background --iterations {ITERS} --test_iterations -1 --port {port} "
          f"{protocol_flags} {cfg['flags']}", cwd=cfg["dir"], logdir=LOGDIR,
          log_name=f"train_{tag}")
    st["train_seconds"] = time.time() - t
    st["peak_vram_mb"] = probe.stop()
    st["iters_per_second"] = ITERS / st["train_seconds"]

    t = time.time()
    kc.sh(f"{env} python render.py -m {out} --skip_train {protocol_flags}", cwd=cfg["dir"],
          logdir=LOGDIR, log_name=f"render_{tag}")
    st["render_seconds"] = time.time() - t
    mdir = os.path.join(out, "test", METHOD)
    preds = [d for d in os.listdir(mdir) if d.startswith("test_preds_")]
    n_rendered = len(os.listdir(os.path.join(mdir, preds[0]))) if preds else 0
    st["n_rendered"] = n_rendered
    st["render_fps"] = n_rendered / st["render_seconds"] if n_rendered else None

    ply = os.path.join(out, "point_cloud", f"iteration_{ITERS}", "point_cloud.ply")
    st["n_gaussians"], st["filter_3D_max"] = kc.ply_stats(ply)
    st["model_mb"] = os.path.getsize(ply) / 1e6

    t = time.time()
    kc.sh(f"{env} python metrics.py -m {out}", cwd=cfg["dir"], logdir=LOGDIR,
          log_name=f"metrics_{tag}")
    st["metrics_seconds"] = time.time() - t

    rj = os.path.join(out, "results.json")
    if not (os.path.exists(rj) and os.path.getsize(rj) > 0):
        # metrics.py wraps every scene in a bare `except:` (F10) — a hard failure
        # prints like a warning, so the file is asserted, never the console.
        raise RuntimeError(f"{tag}: metrics.py failed silently, no {rj} (F10)")

    split = sbs.split_scene(out, src, METHOD)
    n_tot = sum(v["n"] for v in split.values())
    weighted = sum(v["n"] * v["PSNR"] for v in split.values()) / n_tot
    pooled = sbs.method_block(json.load(open(rj)), METHOD)["PSNR"]
    if abs(weighted - pooled) > 0.01:
        raise RuntimeError(f"{tag}: splitter disagrees with metrics.py "
                           f"({weighted:.4f} vs {pooled:.4f}) — index mapping is wrong")
    st["pooled_psnr"] = pooled

    kc.append_rows(csv_path, sbs.RUNS_CSV_SCHEMA, split, {
        "run_id": f"{run_id}-{arm}-{scene}{suffix}", "timestamp_utc": now,
        "method": cfg["method"], "arm": arm, "impl_commit": cfg["commit"],
        "rasteriser_commit": cfg["commit"], "dataset": "blender", "scene": scene,
        "resolution_flag": "-1", "load_allres": str(LOAD_ALLRES),
        "kernel_size": cfg["kernel_size"], "disable_3D_filter": cfg["disable"],
        "train_scale": "1x" if not LOAD_ALLRES else "multi",
        "iterations": str(ITERS), "seed": str(seed),
        "n_gaussians": str(st["n_gaussians"]), "model_mb": f"{st['model_mb']:.2f}",
        "peak_vram_mb": str(st["peak_vram_mb"]),
        "train_seconds": f"{st['train_seconds']:.1f}",
        "render_fps": f"{st['render_fps']:.3f}" if st["render_fps"] else "",
        "gpu_model": gpu_name, "platform": "kaggle", "level_claimed": "L1",
        "notes": f"{RUNG} Blender {'MTMT' if LOAD_ALLRES else 'STMT'}, {ITERS} iters, "
                 f"{len(SCENES)}-scene sweep, seed {seed}",
    })
    removed = kc.prune_renders(out, METHOD, keep=KEEP_QUALITATIVE)
    print(f"[{tag}] done: pooled {pooled:.3f} dB, {st['n_gaussians']} gaussians, "
          f"{st['train_seconds']:.0f}s train, pruned {removed} PNGs, "
          f"working dir {kc.du(WORK)} MB", flush=True)
    return {"split": {k: {m: v[m] for m in ('PSNR', 'SSIM', 'LPIPS', 'n')}
                      for k, v in split.items()}, "stats": st}


def checkpoint_summary():
    with open(f"{RESULTS}/r1_summary.json", "w") as f:
        json.dump(build_summary(), f, indent=2)


def build_summary():
    return {
        "rung": RUNG, "protocol": "MTMT" if LOAD_ALLRES else "STMT",
        "load_allres": LOAD_ALLRES, "seeds": SEEDS,
        "scenes": SCENES, "iterations": ITERS,
        "gpu": gpu_name, "n_gpu": n_gpu, "torch": torch.__version__,
        "cuda": torch.version.cuda, "open3d": open3d_ok,
        "armA_commit": armA_commit, "armB_commit": armB_commit,
        "convert_seconds": convert_seconds,
        "done": done, "failed": failed,
        "session_seconds": time.time() - SESSION_START,
        "gpu_hours_used": (time.time() - SESSION_START) / 3600.0,
        "complete": len(done) == 2 * len(SCENES) * len(SEEDS),
    }


# Two T4s, one (arm, scene) job each. The shipped scripts use a GPUtil poller
# (F13); a fixed round-robin over a known job list is equivalent here and cannot
# wedge if GPUtil misreports a busy GPU as free.
import threading  # noqa: E402


def scene_src(scene, seed):
    """Source directory for one (scene, seed).

    Seed 0 uses the shared directory. A non-zero seed gets its own, holding
    symlinks to the same images and its own metadata.json and points3d.ply, so
    two seeds never contend for one initial point cloud and the two arms of a
    seed share exactly one.
    """
    base = f"{WORK}/multi-scale/{scene}"
    if seed == 0:
        return base
    return f"{WORK}/multi-scale/{scene}__s{seed}"


def pregenerate_inits():
    """Draw each non-zero seed's initial point cloud once, before dispatch.

    readMultiScaleNerfSyntheticInfo creates points3d.ply lazily on first load and
    caches it. Two workers starting together would both create it. Doing it here,
    serially, removes the race and makes each seed's initialisation an explicit,
    logged artefact rather than a side effect of whichever job started first.
    The draw below is the loader's own, reproduced so no images need loading.
    """
    sys.path.insert(0, f"{WORK}/armA")
    from scene.dataset_readers import storePly      # noqa: E402
    from utils.sh_utils import SH2RGB               # noqa: E402
    import numpy as np                              # noqa: E402

    for seed in SEEDS:
        if seed == 0:
            continue
        for scene in SCENES:
            src = scene_src(scene, seed)
            os.makedirs(src, exist_ok=True)
            base = f"{WORK}/multi-scale/{scene}"
            for entry in os.listdir(base):
                if entry in ("points3d.ply",):
                    continue
                dst = os.path.join(src, entry)
                if os.path.exists(dst):
                    continue
                s_ = os.path.join(base, entry)
                if os.path.isdir(s_):
                    os.symlink(s_, dst)
                else:
                    shutil.copy(s_, dst)
            ply = os.path.join(src, "points3d.ply")
            if os.path.exists(ply):
                continue
            np.random.seed(seed)
            num_pts = 100_000
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            storePly(ply, xyz, SH2RGB(shs) * 255)
            print(f"  seed {seed}: drew {num_pts} initial points for {scene}", flush=True)


if any(sd != 0 for sd in SEEDS):
    import shutil  # noqa: E402
    pregenerate_inits()


jobs = [(arm, scene, seed) for seed in SEEDS for scene in SCENES
        for arm in ARMS_ENABLED]
lock = threading.Lock()
queue_idx = [0]


def worker(gpu):
    while True:
        with lock:
            if queue_idx[0] >= len(jobs):
                return
            arm, scene, seed = jobs[queue_idx[0]]
            queue_idx[0] += 1
        key = f"{arm}/{scene}" + ("" if seed == 0 else f"/s{seed}")
        try:
            done[key] = run_one(gpu, arm, scene, seed)
        except Exception as e:
            failed[key] = f"{type(e).__name__}: {e}"[:600]
            traceback.print_exc()
        with lock:
            checkpoint_summary()   # every scene, so a killed session keeps its work


threads = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, n_gpu))]
for t in threads:
    t.start()
for t in threads:
    t.join()

# ============================================================ summary
summary = build_summary()
# The §3 claim, checked on the artefacts rather than the diff text: arm B is
# 3DGS only if filter_3D is identically zero in every saved model.
fmax = {k: v["stats"]["filter_3D_max"] for k, v in done.items()}
summary["filter_3D_max"] = fmax
summary["armB_all_zero"] = all(v == 0.0 for k, v in fmax.items() if k.startswith("B/"))
summary["armA_all_nonzero"] = all(v > 0.0 for k, v in fmax.items() if k.startswith("A/"))
with open(f"{RESULTS}/r1_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

for path in (f"{WORK}/multi-scale", f"{WORK}/armA/.git", f"{WORK}/armB/.git",
             f"{WORK}/armA/submodules", f"{WORK}/armB/submodules",
             f"{WORK}/armA/assets", f"{WORK}/armB/assets",
             f"{WORK}/armA/media", f"{WORK}/armB/media"):
    if os.path.exists(path):
        __import__("shutil").rmtree(path, ignore_errors=True)
print(f"working dir after prune: {kc.du(WORK)} MB", flush=True)

print(f"completed {len(done)}/{2 * len(SCENES) * len(SEEDS)} jobs; failed: {list(failed)}", flush=True)
print("R1_SUMMARY_JSON=" + json.dumps(summary))   # marker name kept for the pusher
if failed:
    raise SystemExit(f"{len(failed)} job(s) failed — see the table above.")
