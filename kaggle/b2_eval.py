#!/usr/bin/env python3
"""
B2 evaluation — Table 4, angular identifiability of spherical harmonics.

Post-hoc on models trained already: the criterion needs primitive positions, the
training camera directions and the SH coefficients, all of which the saved PLY
and the dataset carry. Nothing is retrained, so this costs a render and a metrics
pass per configuration rather than half an hour of optimisation.

Three configurations per scene:

  full        the trained model as it stands, degree 3 everywhere. The control.
  b2          degree l kept only where lambda_min(A_k(l)) > tau (Equation 5.2),
              everything above masked to zero.
  magnitude   the obvious baseline: drop the smallest non-DC coefficients until
              the SAME fraction remains as B2 kept. A criterion that cannot beat
              magnitude pruning at matched size is not worth the machinery, so
              this comparison is the point of the table rather than a courtesy.

Masking is done by rewriting the PLY, so render.py and metrics.py run unmodified
and the measurement path is the one every other rung used.

Ends with R8_SUMMARY_JSON= for the pusher.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
REPO = "https://github.com/ridash2005/mip-splatting.git"
# Where the Blender data lives. The Kaggle defaults; tools/run_local.py
# overrides both so the same kernel runs on a cluster node unchanged.
DATA_MOUNT = "/kaggle/input/nerf-synthetic-dataset"
DATA_ROOT = "/kaggle/input"
BRANCH = "method-b1-b2"
LOGDIR = f"{WORK}/logs"
RESULTS = f"{WORK}/results"

# --- parameters, rewritten by `kaggle_push.py --set NAME=VALUE` --------------
RUNG = "R8"
ITERS = 30000
SCENES = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
TAUS = [0.01]
SWEEP_SCENE = "lego"          # the tau sweep runs on this one scene
SWEEP_TAUS = [0.003, 0.03, 0.1]
# ----------------------------------------------------------------------------

METHOD_DIR = f"ours_{ITERS}"
SESSION_START = time.time()
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

subprocess.run("nvidia-smi", shell=True)
import torch  # noqa: E402

cap = torch.cuda.get_device_capability()
gpu_name, n_gpu = torch.cuda.get_device_name(), torch.cuda.device_count()
print(f"torch {torch.__version__} cap {cap} {gpu_name} x{n_gpu}", flush=True)
if cap < (7, 0):
    print("R8_ABORT=WRONG_ACCELERATOR", flush=True)
    raise SystemExit(f"ABORT: {gpu_name} capability {cap}, below 7.0 (F15).")

subprocess.run(f"git clone --recursive -b {BRANCH} {REPO} {WORK}/repo", shell=True,
               check=True)
sys.path.insert(0, f"{WORK}/repo/tools")
import kernel_common as kc          # noqa: E402
import split_by_scale as sbs        # noqa: E402
import sh_identifiability as shid   # noqa: E402
import instruments as inst          # noqa: E402
import numpy as np                  # noqa: E402

commit = kc.git_hash(f"{WORK}/repo")
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
kc.sh("pip install -q ninja gputil lpips plyfile opencv-python", logdir=LOGDIR,
      log_name="install")
kc.sh("pip install -q open3d", check_rc=False, logdir=LOGDIR, log_name="install")
try:
    import open3d  # noqa: F401
except Exception:
    kc.neutralise_unused_open3d_import(f"{WORK}/repo")
kc.patch_simple_knn_flt_max(f"{WORK}/repo")
kc.sh(f"pip install -v {WORK}/repo/submodules/diff-gaussian-rasterization",
      logdir=LOGDIR, log_name="build")
kc.sh(f"pip install -v {WORK}/repo/submodules/simple-knn", logdir=LOGDIR, log_name="build")

blender_dir = kc.resolve_blender_dir(SCENES[0], mount=DATA_MOUNT, root=DATA_ROOT)
for scene in SCENES:
    if not os.path.exists(f"{WORK}/multi-scale/{scene}/metadata.json"):
        kc.sh(f"python {WORK}/repo/convert_blender_data.py --blender_dir {blender_dir} "
              f"--object_name {scene} --out_dir {WORK}/multi-scale",
              logdir=LOGDIR, log_name="convert")

# The trained models come from R1's arm A, attached as a kernel data source.
srcs = {}
for root, dirs, files in os.walk("/kaggle/input"):
    if root.count("/") > 9:
        dirs[:] = []
        continue
    if "point_cloud.ply" in files and f"iteration_{ITERS}" in root and "out_armA" in root:
        scene = root.split("/")[-3]
        srcs[scene] = root
print(f"found {len(srcs)} trained models: {sorted(srcs)}", flush=True)

SH_REST_COLS = None


def load_ply_arrays(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    names = list(v.data.dtype.names)
    return v, names


def write_masked_ply(src_ply, dst_ply, keep_mask):
    """Copy the PLY with non-DC SH zeroed where keep_mask is False.

    keep_mask is [N, 15] over the per-channel non-DC coefficients; the PLY
    stores them as f_rest_0..44 laid out channel-major (all 15 of channel 0,
    then channel 1, then 2), which is what gaussian_model.save_ply writes after
    its transpose(1,2).flatten.
    """
    from plyfile import PlyData
    ply = PlyData.read(src_ply)
    v = ply["vertex"]
    names = list(v.data.dtype.names)
    rest = sorted([n for n in names if n.startswith("f_rest_")],
                  key=lambda s: int(s.split("_")[-1]))
    assert len(rest) == 45, f"expected 45 f_rest columns, found {len(rest)}"
    data = v.data.copy()
    for ch in range(3):
        for j in range(15):
            col = rest[ch * 15 + j]
            data[col] = np.where(keep_mask[:, j], data[col], 0.0)
    ply["vertex"].data = data
    os.makedirs(os.path.dirname(dst_ply), exist_ok=True)
    PlyData([ply["vertex"]], text=False).write(dst_ply)


def b2_mask(scene, tau):
    """keep_mask [N,15] from Equation 5.2, plus the diagnostics."""
    ply = os.path.join(srcs[scene], "point_cloud.ply")
    data = inst.load_ply(ply)
    c2w, _ = inst.load_cameras(f"{WORK}/multi-scale/{scene}/metadata.json")
    blocks, n_seen = shid.angular_gram(data["xyz"], c2w)
    lm = shid.l_max(blocks, n_seen, tau)
    keep = np.zeros((len(lm), 15), dtype=bool)
    for l, sl in shid.DEG_SLICES.items():
        if l == 0:
            continue
        lo, hi = sl.start - 1, sl.stop - 1        # non-DC index space
        keep[:, lo:hi] = (lm >= l)[:, None]
    return keep, lm, n_seen


def magnitude_mask(scene, target_frac):
    """Drop the smallest non-DC coefficients until target_frac remains.

    Magnitude is taken across the three colour channels together, so a
    coefficient is kept or dropped for the primitive as a whole -- the same
    granularity B2 masks at, which is what makes the sizes comparable.
    """
    from plyfile import PlyData
    v = PlyData.read(os.path.join(srcs[scene], "point_cloud.ply"))["vertex"]
    names = list(v.data.dtype.names)
    rest = sorted([n for n in names if n.startswith("f_rest_")],
                  key=lambda s: int(s.split("_")[-1]))
    n = v.count
    mag = np.zeros((n, 15))
    for ch in range(3):
        for j in range(15):
            mag[:, j] += np.asarray(v[rest[ch * 15 + j]]).astype(np.float64) ** 2
    flat = mag.reshape(-1)
    k = int(round(target_frac * flat.size))
    if k <= 0:
        return np.zeros((n, 15), dtype=bool)
    thresh = np.partition(flat, -k)[-k]
    return (mag >= thresh)


def evaluate(scene, tag, keep_mask):
    """Write the masked model, render it and score it, using the shipped path."""
    out = f"{WORK}/out/{scene}_{tag}"
    dst = os.path.join(out, "point_cloud", f"iteration_{ITERS}", "point_cloud.ply")
    write_masked_ply(os.path.join(srcs[scene], "point_cloud.ply"), dst, keep_mask)
    # render.py reads cfg_args for the resolution and background flags.
    shutil.copy(os.path.join(os.path.dirname(os.path.dirname(srcs[scene])), "cfg_args"),
                os.path.join(out, "cfg_args"))
    env = "OMP_NUM_THREADS=4"
    kc.sh(f"{env} python render.py -m {out} --skip_train -s {WORK}/multi-scale/{scene}",
          cwd=f"{WORK}/repo", logdir=LOGDIR, log_name=f"render_{scene}_{tag}")
    kc.sh(f"{env} python metrics.py -m {out}", cwd=f"{WORK}/repo", logdir=LOGDIR,
          log_name=f"metrics_{scene}_{tag}")
    rj = os.path.join(out, "results.json")
    if not (os.path.exists(rj) and os.path.getsize(rj) > 0):
        raise RuntimeError(f"{scene}/{tag}: metrics.py failed silently (F10)")
    block = sbs.method_block(json.load(open(rj)), METHOD_DIR)
    split = sbs.split_scene(out, f"{WORK}/multi-scale/{scene}", METHOD_DIR)
    mb = os.path.getsize(dst) / 1e6
    kc.prune_renders(out, METHOD_DIR, keep=3)
    return {"PSNR": block["PSNR"], "SSIM": block["SSIM"], "LPIPS": block["LPIPS"],
            "model_mb": mb, "retained": float(keep_mask.mean()),
            "per_scale": {k: v["PSNR"] for k, v in split.items()}}


results, failed = {}, {}


def do(scene, tau, with_sweep_tag=""):
    try:
        keep, lm, n_seen = b2_mask(scene, tau)
        frac = float(keep.mean())
        key = f"{scene}/b2/tau{tau}{with_sweep_tag}"
        r = evaluate(scene, f"b2_{str(tau).replace('.', 'p')}", keep)
        r.update({"tau": tau, "mean_l_max": float(lm.mean()),
                  "l_max_hist": {int(l): int((lm == l).sum()) for l in range(4)},
                  "mean_views": float(n_seen.mean())})
        results[key] = r
        print(f"[{key}] PSNR {r['PSNR']:.3f} | {r['model_mb']:.1f} MB | "
              f"non-DC kept {frac:.1%} | mean l_max {r['mean_l_max']:.2f}", flush=True)

        mkey = f"{scene}/magnitude/tau{tau}{with_sweep_tag}"
        mmask = magnitude_mask(scene, frac)
        rm = evaluate(scene, f"mag_{str(tau).replace('.', 'p')}", mmask)
        rm["matched_to_tau"] = tau
        results[mkey] = rm
        print(f"[{mkey}] PSNR {rm['PSNR']:.3f} | {rm['model_mb']:.1f} MB | "
              f"non-DC kept {rm['retained']:.1%}", flush=True)
    except Exception as e:
        failed[f"{scene}/tau{tau}"] = f"{type(e).__name__}: {e}"[:400]
        traceback.print_exc()
    with open(f"{RESULTS}/b2.json", "w") as f:
        json.dump({"results": results, "failed": failed}, f, indent=2)


for scene in SCENES:
    if scene not in srcs:
        failed[scene] = "no trained model found under /kaggle/input"
        continue
    # The control: the unmasked model, scored through the identical path so the
    # comparison is not confounded by how it was measured.
    try:
        results[f"{scene}/full"] = evaluate(scene, "full",
                                            np.ones((inst.load_ply(
                                                os.path.join(srcs[scene],
                                                             "point_cloud.ply")
                                            )["xyz"].shape[0], 15), dtype=bool))
        print(f"[{scene}/full] PSNR {results[f'{scene}/full']['PSNR']:.3f} | "
              f"{results[f'{scene}/full']['model_mb']:.1f} MB", flush=True)
    except Exception as e:
        failed[f"{scene}/full"] = f"{type(e).__name__}: {e}"[:400]
        traceback.print_exc()
    for tau in TAUS:
        do(scene, tau)

if SWEEP_SCENE in srcs:
    for tau in SWEEP_TAUS:
        do(SWEEP_SCENE, tau, "_sweep")

summary = {"rung": RUNG, "iterations": ITERS, "scenes": SCENES, "taus": TAUS,
           "sweep_scene": SWEEP_SCENE, "sweep_taus": SWEEP_TAUS, "commit": commit,
           "gpu": gpu_name, "results": results, "failed": failed,
           "session_seconds": time.time() - SESSION_START,
           "gpu_hours_used": (time.time() - SESSION_START) / 3600.0}
with open(f"{RESULTS}/b2.json", "w") as f:
    json.dump(summary, f, indent=2)
for path in (f"{WORK}/multi-scale", f"{WORK}/repo/.git", f"{WORK}/repo/submodules"):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
print(f"done {len(results)}, failed {list(failed)}", flush=True)
print("R8_SUMMARY_JSON=" + json.dumps(summary))
