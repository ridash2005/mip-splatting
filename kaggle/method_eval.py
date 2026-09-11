#!/usr/bin/env python3
"""
B1 evaluation — Table 2 (standard benchmark) and Table 3 (the stress suite).

  TABLE = "2"   8 Blender scenes, full orbit, method B1.
                Prediction, recorded before the run: PARITY with Mip-Splatting,
                because I-1 measured the Nyquist floor binding on 100% of
                primitives there and Proposition 2 then makes the two filters
                identical. A gain here would be a bug, not a result.

  TABLE = "3"   2 scenes x 5 capture protocols x {Mip-Splatting, B1}. Training
                cameras are restricted to the protocol's subset; the TEST set is
                left whole, so the only thing that varies is the geometry the
                reconstruction had available. This is where §4.6 predicts the
                method must help, and it is the thesis's central claim.

Both arms of the comparison run in this same session on the same data with the
same seed, so a paired per-scene delta is meaningful. Mip-Splatting is re-run
inside Table 3 rather than reused from R1 because the camera subset changes the
training set, and a control must see the same one.

Ends with R7_SUMMARY_JSON= for the pusher.
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
# Where the Blender data lives. The Kaggle defaults; tools/run_local.py
# overrides both so the same kernel runs on a cluster node unchanged.
DATA_MOUNT = "/kaggle/input/nerf-synthetic-dataset"
DATA_ROOT = "/kaggle/input"
BRANCH = "method-b1-b2"
LOGDIR = f"{WORK}/logs"
RESULTS = f"{WORK}/results"

# --- parameters, rewritten by `kaggle_push.py --set NAME=VALUE` --------------
RUNG = "R7"
TABLE = "2"
ITERS = 30000
SCENES = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
PROTOCOLS = ["full"]
METHODS = ["b1"]                # "mip" | "b1"
# ----------------------------------------------------------------------------

METHOD_FLAGS = {
    "mip": "--kernel_size 0.1",
    "b1": "--kernel_size 0.1 --use_fisher_filter",
}
METHOD_NAME = {"mip": "mip-splatting", "b1": "b1-fisher"}

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
    print("R7_ABORT=WRONG_ACCELERATOR", flush=True)
    raise SystemExit(f"ABORT: {gpu_name} capability {cap}, below 7.0 (F15).")

subprocess.run(f"git clone --recursive -b {BRANCH} {REPO} {WORK}/repo", shell=True,
               check=True)
sys.path.insert(0, f"{WORK}/repo/tools")
import kernel_common as kc        # noqa: E402
import split_by_scale as sbs      # noqa: E402
import camera_protocols as cp     # noqa: E402
import instruments as inst        # noqa: E402

commit = kc.git_hash(f"{WORK}/repo")
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
kc.sh("pip install -q ninja gputil lpips plyfile opencv-python", logdir=LOGDIR,
      log_name="install")
kc.sh("pip install -q open3d", check_rc=False, logdir=LOGDIR, log_name="install")
open3d_ok = True
try:
    import open3d  # noqa: F401
except Exception:
    open3d_ok = False
    kc.neutralise_unused_open3d_import(f"{WORK}/repo")
kc.patch_simple_knn_flt_max(f"{WORK}/repo")
kc.sh(f"pip install -v {WORK}/repo/submodules/diff-gaussian-rasterization",
      logdir=LOGDIR, log_name="build")
kc.sh(f"pip install -v {WORK}/repo/submodules/simple-knn", logdir=LOGDIR, log_name="build")
for m in ("GPUtil", "lpips", "plyfile", "cv2", "torchvision", "tqdm", "numpy", "PIL",
          "diff_gaussian_rasterization", "simple_knn._C"):
    __import__(m)

# Propositions 1 and 2, verified inside the session that uses them. If the exact
# reduction does not hold here, nothing measured afterwards means anything.
rc_prop, _ = kc.sh(f"python {WORK}/repo/tools/test_fisher_filter.py", check_rc=False,
                   logdir=LOGDIR, log_name="propositions")
print(f"Propositions 1 and 2 unit test: {'PASS' if rc_prop == 0 else 'FAIL'}", flush=True)
if rc_prop != 0:
    raise SystemExit("ABORT: the exact-reduction test failed; no result would be valid.")

blender_dir = kc.resolve_blender_dir(SCENES[0], mount=DATA_MOUNT, root=DATA_ROOT)
t0 = time.time()
for scene in SCENES:
    if not os.path.exists(f"{WORK}/multi-scale/{scene}/metadata.json"):
        kc.sh(f"python {WORK}/repo/convert_blender_data.py --blender_dir {blender_dir} "
              f"--object_name {scene} --out_dir {WORK}/multi-scale",
              logdir=LOGDIR, log_name="convert")
convert_seconds = time.time() - t0

# The protocols, built once per scene from that scene's own cameras and written
# out so the subsets are versioned artefacts rather than a run-time accident.
protocol_spec = {}
os.makedirs(f"{WORK}/results/protocols", exist_ok=True)
for scene in SCENES:
    c2w, focal = inst.load_cameras(f"{WORK}/multi-scale/{scene}/metadata.json")
    # readMultiScale keeps only d0 files for the training split under the default
    # load_allres=False, so the protocol indexes that same d0 subsequence.
    meta = json.load(open(f"{WORK}/multi-scale/{scene}/metadata.json"))["train"]
    d0 = [i for i, p in enumerate(meta["file_path"]) if p.endswith("d0.png")]
    spec = cp.build(c2w[d0], [focal[i] for i in d0], seed=0)
    protocol_spec[scene] = spec
    with open(f"{WORK}/results/protocols/{scene}.json", "w") as f:
        json.dump(spec, f, indent=2)
    print(f"  {scene}: " + ", ".join(
        f"{k} n={spec[k]['n']} span={spec[k]['span_deg']:.0f}deg" for k in PROTOCOLS),
        flush=True)

now = datetime.now(timezone.utc).isoformat(timespec="seconds")
run_id = f"{RUNG.lower()}-t{TABLE}-{now.replace(':', '').replace('-', '')}"
csv_path = f"{RESULTS}/runs.csv"
done, failed = {}, {}
lock = threading.Lock()


def run_one(gpu, method, scene, proto):
    tag = f"{method}_{scene}_{proto}"
    out = f"{WORK}/out/{tag}"
    env = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}"
    spec = protocol_spec[scene][proto]
    subset = json.dumps(spec["idx"])
    sub_flag = "" if proto == "full" else f"--camera_subset '{subset}'"
    st = {"protocol": proto, "n_train_cameras": spec["n"], "span_deg": spec["span_deg"]}

    probe = kc.VramProbe(gpu=gpu)
    probe.start()
    t = time.time()
    _, tr_out = kc.sh(
        f"{env} python train.py -s {WORK}/multi-scale/{scene} -m {out} --eval "
        f"--white_background --iterations {ITERS} --test_iterations -1 "
        f"--port {6209 + int(gpu)} {sub_flag} {METHOD_FLAGS[method]}",
        cwd=f"{WORK}/repo", logdir=LOGDIR, log_name=f"train_{tag}")
    st["train_seconds"] = time.time() - t
    st["peak_vram_mb"] = probe.stop()

    # The B1 diagnostic §6.6 asks for: what fraction of primitives sits above the
    # floor, and how anisotropic the filter actually became. Printed by the
    # filter itself during training, so it is read back rather than recomputed.
    for line in tr_out.splitlines()[::-1]:
        if line.startswith("Fisher filter:"):
            try:
                st["frac_above_floor"] = float(line.split("%")[0].split()[-1]) / 100.0
                st["mean_anisotropy"] = float(line.split("anisotropy")[1].split("x")[0])
            except Exception:
                pass
            break

    t = time.time()
    kc.sh(f"{env} python render.py -m {out} --skip_train {sub_flag}",
          cwd=f"{WORK}/repo", logdir=LOGDIR, log_name=f"render_{tag}")
    st["render_seconds"] = time.time() - t
    mdir = os.path.join(out, "test", METHOD_DIR)
    preds = [d for d in os.listdir(mdir) if d.startswith("test_preds_")]
    n_rendered = len(os.listdir(os.path.join(mdir, preds[0]))) if preds else 0
    st["render_fps"] = n_rendered / st["render_seconds"] if n_rendered else None

    ply = os.path.join(out, "point_cloud", f"iteration_{ITERS}", "point_cloud.ply")
    st["n_gaussians"], _ = kc.ply_stats(ply)
    st["model_mb"] = os.path.getsize(ply) / 1e6

    kc.sh(f"{env} python metrics.py -m {out}", cwd=f"{WORK}/repo", logdir=LOGDIR,
          log_name=f"metrics_{tag}")
    rj = os.path.join(out, "results.json")
    if not (os.path.exists(rj) and os.path.getsize(rj) > 0):
        raise RuntimeError(f"{tag}: metrics.py failed silently (F10)")

    split = sbs.split_scene(out, f"{WORK}/multi-scale/{scene}", METHOD_DIR)
    n_tot = sum(v["n"] for v in split.values())
    weighted = sum(v["n"] * v["PSNR"] for v in split.values()) / n_tot
    pooled = sbs.method_block(json.load(open(rj)), METHOD_DIR)["PSNR"]
    if abs(weighted - pooled) > 0.01:
        raise RuntimeError(f"{tag}: splitter disagrees with metrics.py")
    st["pooled_psnr"] = pooled
    st["split"] = {k: {m: v[m] for m in ("PSNR", "SSIM", "LPIPS", "n")}
                   for k, v in split.items()}

    with lock:
        kc.append_rows(csv_path, sbs.RUNS_CSV_SCHEMA, split, {
            "run_id": f"{run_id}-{tag}", "timestamp_utc": now,
            "method": METHOD_NAME[method], "arm": "C" if method == "b1" else "A",
            "impl_commit": commit, "rasteriser_commit": commit, "dataset": "blender",
            "scene": scene, "resolution_flag": "-1", "load_allres": "False",
            "kernel_size": "0.1", "disable_3D_filter": "False",
            "train_scale": f"1x/{proto}", "iterations": str(ITERS), "seed": "0",
            "n_gaussians": str(st["n_gaussians"]), "model_mb": f"{st['model_mb']:.2f}",
            "peak_vram_mb": str(st["peak_vram_mb"]),
            "train_seconds": f"{st['train_seconds']:.1f}",
            "render_fps": f"{st['render_fps']:.3f}" if st["render_fps"] else "",
            "gpu_model": gpu_name, "platform": "kaggle",
            "level_claimed": "L3" if proto == "full" else "L2",
            "notes": f"{RUNG} table{TABLE} {METHOD_NAME[method]} protocol={proto} "
                     f"n_cam={spec['n']} span={spec['span_deg']:.0f}deg {ITERS} iters",
        })
    kc.prune_renders(out, METHOD_DIR, keep=4)
    print(f"[{tag}] pooled {pooled:.3f} dB | {st['n_gaussians']} gaussians | "
          f"above floor {st.get('frac_above_floor', float('nan')):.1%} | "
          f"aniso {st.get('mean_anisotropy', float('nan')):.2f}x | "
          f"{st['train_seconds']:.0f}s | wd {kc.du(WORK)} MB", flush=True)
    return st


def build_summary():
    return {"rung": RUNG, "table": TABLE, "iterations": ITERS, "scenes": SCENES,
            "protocols": PROTOCOLS, "methods": METHODS, "commit": commit,
            "gpu": gpu_name, "n_gpu": n_gpu, "propositions_pass": rc_prop == 0,
            "convert_seconds": convert_seconds,
            "protocol_spec": {s: {p: {k: v[k] for k in ("n", "span_deg")}
                                  for p, v in sp.items() if p in PROTOCOLS}
                              for s, sp in protocol_spec.items()},
            "done": done, "failed": failed,
            "session_seconds": time.time() - SESSION_START,
            "gpu_hours_used": (time.time() - SESSION_START) / 3600.0,
            "complete": len(done) == len(SCENES) * len(PROTOCOLS) * len(METHODS)}


jobs = [(m, s, p) for s in SCENES for p in PROTOCOLS for m in METHODS]
qi = [0]


def worker(gpu):
    while True:
        with lock:
            if qi[0] >= len(jobs):
                return
            m, s, p = jobs[qi[0]]
            qi[0] += 1
        key = f"{m}/{s}/{p}"
        try:
            done[key] = run_one(gpu, m, s, p)
        except Exception as e:
            failed[key] = f"{type(e).__name__}: {e}"[:600]
            print(f"[{key}] FAILED: {failed[key][:200]}", flush=True)
            traceback.print_exc()
        with lock:
            with open(f"{RESULTS}/method_summary.json", "w") as f:
                json.dump(build_summary(), f, indent=2)


threads = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, n_gpu))]
for t in threads:
    t.start()
for t in threads:
    t.join()

summary = build_summary()
with open(f"{RESULTS}/method_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
for path in (f"{WORK}/multi-scale", f"{WORK}/repo/.git", f"{WORK}/repo/submodules",
             f"{WORK}/repo/assets", f"{WORK}/repo/media"):
    if os.path.exists(path):
        __import__("shutil").rmtree(path, ignore_errors=True)
print(f"completed {len(done)}/{len(jobs)}; failed {list(failed)}", flush=True)
print("R7_SUMMARY_JSON=" + json.dumps(summary))
if failed:
    raise SystemExit(f"{len(failed)} job(s) failed.")
