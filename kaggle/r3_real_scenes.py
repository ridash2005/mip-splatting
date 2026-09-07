#!/usr/bin/env python3
"""
R3 — the real scenes that fit 16 GB (§7 R3 of docs/REPRODUCTION-PROMPT.md).

Mip-NeRF 360 indoor at -i images_2, Tanks & Temples and Deep Blending at their
defaults, per F16. Both arms, 30 000 iterations, every-8th test split -- which
applies HERE and not to Blender (F12): llffhold=8 lives only in
readColmapSceneInfo.

Differences from the Blender rungs, all forced by the data:

  * COLMAP scenes, so no convert step and no multi-scale metadata. There is one
    test resolution, so tools/split_by_scale.py does not apply and metrics.py's
    single pooled number IS the result. The CSV records test_scale as the
    resolution actually used.
  * The published references are protocol-specific and NOT comparable across
    subsets. Mip-NeRF 360's all-9 average cannot be set against a 4-indoor
    subset, and the report says so rather than quoting it.

§7's exit rule for this rung is explicit: "On OOM: record it and move on. Do not
reduce resolution, iterations or Gaussian count to make a scene fit." A failing
scene is therefore recorded with its error and the sweep continues; nothing is
retried smaller.

Ends with a single line prefixed R3_SUMMARY_JSON= for the pusher.
"""
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

WORK = "/kaggle/working"
REPO = "https://github.com/ridash2005/mip-splatting.git"
LOGDIR = f"{WORK}/logs"
RESULTS = f"{WORK}/results"

# --- parameters, rewritten by `kaggle_push.py --set NAME=VALUE` --------------
RUNG = "R3"
ITERS = 30000
# (scene, dataset tag, -i images flag). F16: outdoor images_4, indoor images_2,
# T&T / DB default. The five outdoor 360 scenes are R4, not R3.
JOBS = [
    ("room", "mipnerf360", "images_2"),
    ("counter", "mipnerf360", "images_2"),
    ("kitchen", "mipnerf360", "images_2"),
    ("bonsai", "mipnerf360", "images_2"),
    ("truck", "tandt", ""),
    ("train", "tandt", ""),
    ("drjohnson", "db", ""),
    ("playroom", "db", ""),
]
# ----------------------------------------------------------------------------

METHOD = f"ours_{ITERS}"
SESSION_START = time.time()
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

subprocess.run("nvidia-smi", shell=True)
import torch  # noqa: E402

cap = torch.cuda.get_device_capability()
gpu_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  capability {cap}  "
      f"gpu {gpu_name} x{n_gpu}", flush=True)
if cap < (7, 0):
    print("R3_ABORT=WRONG_ACCELERATOR", flush=True)
    raise SystemExit(f"ABORT: {gpu_name} capability {cap}, below 7.0 (F15).")

subprocess.run(f"git clone --recursive -b main {REPO} {WORK}/armA", shell=True, check=True)
subprocess.run(f"git clone --recursive -b arm-b-3dgs-baseline {REPO} {WORK}/armB",
               shell=True, check=True)
sys.path.insert(0, f"{WORK}/armA/tools")
import kernel_common as kc        # noqa: E402
import split_by_scale as sbs      # noqa: E402

armA_commit, armB_commit = kc.git_hash(f"{WORK}/armA"), kc.git_hash(f"{WORK}/armB")

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
kc.sh("pip install -q ninja gputil lpips plyfile opencv-python", logdir=LOGDIR,
      log_name="install")
kc.sh("pip install -q open3d", check_rc=False, logdir=LOGDIR, log_name="install")
open3d_ok = True
try:
    import open3d  # noqa: F401
except Exception:
    open3d_ok = False
    for d in (f"{WORK}/armA", f"{WORK}/armB"):
        kc.neutralise_unused_open3d_import(d)
for d in (f"{WORK}/armA", f"{WORK}/armB"):
    kc.patch_simple_knn_flt_max(d)
kc.sh(f"pip install -v {WORK}/armA/submodules/diff-gaussian-rasterization",
      logdir=LOGDIR, log_name="build")
kc.sh(f"pip install -v {WORK}/armA/submodules/simple-knn", logdir=LOGDIR, log_name="build")
for mod in ("GPUtil", "lpips", "plyfile", "cv2", "torchvision", "tqdm", "numpy", "PIL",
            "diff_gaussian_rasterization", "simple_knn._C"):
    __import__(mod)
print(f"build OK (open3d={'yes' if open3d_ok else 'neutralised'})", flush=True)


def find_scene(scene):
    """The COLMAP directory for `scene`, wherever Kaggle mounted it."""
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count("/") > 8:
            dirs[:] = []
            continue
        if os.path.basename(root) == scene and "sparse" in dirs and "images" in dirs:
            return root
    return None


sources = {}
for scene, tag, _ in JOBS:
    p = find_scene(scene)
    sources[scene] = p
    print(f"  {tag}/{scene}: {p or 'NOT FOUND'}", flush=True)

now = datetime.now(timezone.utc).isoformat(timespec="seconds")
run_id = f"r3-{now.replace(':', '').replace('-', '')}"
csv_path = f"{RESULTS}/runs.csv"
ARMS = {
    "A": dict(dir=f"{WORK}/armA", flags="--kernel_size 0.1", method="mip-splatting",
              commit=armA_commit, kernel_size="0.1", disable="False"),
    "B": dict(dir=f"{WORK}/armB", flags="--kernel_size 0.3 --disable_3D_filter",
              method="3dgs", commit=armB_commit, kernel_size="0.3", disable="True"),
}
done, failed = {}, {}
lock = threading.Lock()


def run_one(gpu, arm, scene, tag, images):
    cfg = ARMS[arm]
    src = sources[scene]
    if not src:
        raise RuntimeError(f"scene directory for {scene} not found under /kaggle/input")
    out = f"{WORK}/out_arm{arm}/{scene}"
    t_ = f"{arm}_{scene}"
    env = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}"
    img = f"-i {images}" if images else ""
    st = {}
    probe = kc.VramProbe(gpu=gpu)
    probe.start()
    t = time.time()
    kc.sh(f"{env} python train.py -s {src} -m {out} --eval {img} --iterations {ITERS} "
          f"--test_iterations -1 --port {6209 + int(gpu)} --quiet {cfg['flags']}",
          cwd=cfg["dir"], logdir=LOGDIR, log_name=f"train_{t_}")
    st["train_seconds"] = time.time() - t
    st["peak_vram_mb"] = probe.stop()
    st["iters_per_second"] = ITERS / st["train_seconds"]

    t = time.time()
    kc.sh(f"{env} python render.py -m {out} --skip_train {img}", cwd=cfg["dir"],
          logdir=LOGDIR, log_name=f"render_{t_}")
    st["render_seconds"] = time.time() - t
    mdir = os.path.join(out, "test", METHOD)
    preds = [d for d in os.listdir(mdir) if d.startswith("test_preds_")]
    n_rendered = len(os.listdir(os.path.join(mdir, preds[0]))) if preds else 0
    st["n_rendered"] = n_rendered
    st["render_fps"] = n_rendered / st["render_seconds"] if n_rendered else None

    ply = os.path.join(out, "point_cloud", f"iteration_{ITERS}", "point_cloud.ply")
    st["n_gaussians"], st["filter_3D_max"] = kc.ply_stats(ply)
    st["model_mb"] = os.path.getsize(ply) / 1e6

    kc.sh(f"{env} python metrics.py -m {out}", cwd=cfg["dir"], logdir=LOGDIR,
          log_name=f"metrics_{t_}")
    rj = os.path.join(out, "results.json")
    if not (os.path.exists(rj) and os.path.getsize(rj) > 0):
        raise RuntimeError(f"{t_}: metrics.py failed silently, no {rj} (F10)")
    block = sbs.method_block(json.load(open(rj)), METHOD)
    st.update({"PSNR": block["PSNR"], "SSIM": block["SSIM"], "LPIPS": block["LPIPS"]})

    # One test resolution here, so no per-scale split: metrics.py's pooled number
    # is the measurement, and test_scale records the resolution it was taken at.
    with lock:
        kc.append_rows(csv_path, sbs.RUNS_CSV_SCHEMA,
                       {images or "default": {"PSNR": block["PSNR"], "SSIM": block["SSIM"],
                                              "LPIPS": block["LPIPS"]}},
                       {"run_id": f"{run_id}-{arm}-{scene}", "timestamp_utc": now,
                        "method": cfg["method"], "arm": arm, "impl_commit": cfg["commit"],
                        "rasteriser_commit": cfg["commit"], "dataset": tag, "scene": scene,
                        "resolution_flag": images or "default", "load_allres": "False",
                        "kernel_size": cfg["kernel_size"], "disable_3D_filter": cfg["disable"],
                        "train_scale": "native", "iterations": str(ITERS), "seed": "0",
                        "n_gaussians": str(st["n_gaussians"]),
                        "model_mb": f"{st['model_mb']:.2f}",
                        "peak_vram_mb": str(st["peak_vram_mb"]),
                        "train_seconds": f"{st['train_seconds']:.1f}",
                        "render_fps": f"{st['render_fps']:.3f}" if st["render_fps"] else "",
                        "gpu_model": gpu_name, "platform": "kaggle", "level_claimed": "L2",
                        "notes": f"R3 real scene, {tag}, {images or 'default'} res, "
                                 f"{ITERS} iters, every-8th test split (F12)"})
    kc.prune_renders(out, METHOD, keep=6)
    print(f"[{t_}] PSNR {block['PSNR']:.3f} SSIM {block['SSIM']:.4f} "
          f"LPIPS {block['LPIPS']:.4f} | {st['n_gaussians']} gaussians, "
          f"{st['peak_vram_mb']} MB peak, {st['train_seconds']:.0f}s | "
          f"working dir {kc.du(WORK)} MB", flush=True)
    return st


def build_summary():
    return {"rung": RUNG, "iterations": ITERS, "gpu": gpu_name, "n_gpu": n_gpu,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "armA_commit": armA_commit, "armB_commit": armB_commit,
            "jobs": [list(j) for j in JOBS], "sources": sources,
            "done": done, "failed": failed,
            "session_seconds": time.time() - SESSION_START,
            "gpu_hours_used": (time.time() - SESSION_START) / 3600.0,
            "complete": len(done) == 2 * len(JOBS)}


queue = [(arm, s, t, i) for (s, t, i) in JOBS for arm in ("A", "B")]
qi = [0]


def worker(gpu):
    while True:
        with lock:
            if qi[0] >= len(queue):
                return
            arm, scene, tag, images = queue[qi[0]]
            qi[0] += 1
        key = f"{arm}/{scene}"
        try:
            done[key] = run_one(gpu, arm, scene, tag, images)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            # §7: record an OOM, never work around it. No retry at lower
            # resolution, fewer iterations or a smaller Gaussian budget.
            failed[key] = ("OUT OF MEMORY -- recorded, not worked around: " + msg[:500]
                           if "out of memory" in msg.lower() else msg[:600])
            print(f"[{key}] FAILED: {failed[key][:200]}", flush=True)
            traceback.print_exc()
        with lock:
            with open(f"{RESULTS}/r3_summary.json", "w") as f:
                json.dump(build_summary(), f, indent=2)


threads = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, n_gpu))]
for t in threads:
    t.start()
for t in threads:
    t.join()

summary = build_summary()
summary["oom"] = [k for k, v in failed.items() if "OUT OF MEMORY" in v]
with open(f"{RESULTS}/r3_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
for path in (f"{WORK}/armA/.git", f"{WORK}/armB/.git", f"{WORK}/armA/submodules",
             f"{WORK}/armB/submodules", f"{WORK}/armA/assets", f"{WORK}/armB/assets",
             f"{WORK}/armA/media", f"{WORK}/armB/media"):
    if os.path.exists(path):
        __import__("shutil").rmtree(path, ignore_errors=True)
print(f"completed {len(done)}/{2 * len(JOBS)}; failed {list(failed)}; "
      f"OOM {summary['oom']}", flush=True)
print("R3_SUMMARY_JSON=" + json.dumps(summary))
