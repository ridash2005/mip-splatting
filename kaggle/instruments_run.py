#!/usr/bin/env python3
"""
I-1 (floor occupancy) and I-2 (Lambda conditioning) over every trained scene.

Attaches a finished rung's kernel output for the point clouds and the Blender
dataset for the camera JSONs, and retrains nothing: both instruments need only
primitive positions and capture geometry (see tools/instruments.py). Pure NumPy,
no CUDA extension build, so this runs in minutes rather than hours.

  RUNG_KERNEL   the kernel whose output carries out_arm*/<scene>/point_cloud/...
  ITER          the iteration whose point cloud to read

Decides gates G3 (does the Nyquist floor bind everywhere? then the method IS the
baseline and the project pivots) and G4 (does lambda_3/lambda_1 behave as
Equation 4.8 predicts on real scenes?).

Ends with a single line prefixed R9_SUMMARY_JSON= for the pusher.
"""
import glob
import json
import os
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
REPO = "https://github.com/ridash2005/mip-splatting.git"
RESULTS = f"{WORK}/results"

# --- parameters, rewritten by `kaggle_push.py --set NAME=VALUE` --------------
RUNG = "R9"
RUNG_KERNEL = "btp-r2-blender-mtmt"
ITER = 30000
SCENES = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
ARMS = ["A", "B"]
# ----------------------------------------------------------------------------

SESSION_START = time.time()
os.makedirs(RESULTS, exist_ok=True)

subprocess.run(f"git clone -q {REPO} {WORK}/repo", shell=True, check=True)
sys.path.insert(0, f"{WORK}/repo/tools")
subprocess.run("pip install -q plyfile", shell=True, check=True)

import numpy as np              # noqa: E402
import instruments as inst      # noqa: E402


def find(pattern):
    hits = glob.glob(pattern, recursive=True)
    return sorted(hits)


# The attached kernel output and the dataset both mount under /kaggle/input, but
# the exact directory name Kaggle assigns is not worth guessing -- search for the
# artefacts instead, and say what was found.
print("mounted inputs:", os.listdir("/kaggle/input"), flush=True)
plys = find(f"/kaggle/input/**/out_arm*/*/point_cloud/iteration_{ITER}/point_cloud.ply")
print(f"found {len(plys)} point clouds", flush=True)
for p in plys[:4]:
    print("  ", p, flush=True)

cams = {}
for scene in SCENES:
    for cand in find(f"/kaggle/input/**/{scene}/transforms_train.json"):
        cams[scene] = cand
        break
print(f"found camera JSON for {len(cams)}/{len(SCENES)} scenes", flush=True)

results, failed = {}, {}
for ply in plys:
    parts = ply.split("/")
    try:
        arm = [q for q in parts if q.startswith("out_arm")][0].replace("out_arm", "")
        scene = parts[parts.index("point_cloud") - 1]
    except Exception:
        failed[ply] = "could not parse arm/scene from path"
        continue
    if scene not in cams or arm not in ARMS:
        failed[f"{arm}/{scene}"] = "no camera JSON" if scene not in cams else "arm skipped"
        continue
    key = f"{arm}/{scene}"
    try:
        t = time.time()
        data = inst.load_ply(ply)
        c2w, focal = inst.load_cameras(cams[scene])
        lam, d_min, n_seen, _ = inst.fisher(data["xyz"], c2w, focal)
        lam_min, lam_max = lam[:, 0], lam[:, 2]

        unseen = n_seen == 0
        f_k = inst.SQRT_POINT_TWO * np.where(unseen, np.nan, d_min) / focal.max()
        with np.errstate(divide="ignore", invalid="ignore"):
            est_max = 1.0 / np.where(lam_min > 0, lam_min, np.nan)
            est_min = 1.0 / np.where(lam_max > 0, lam_max, np.nan)
            aniso = np.where(lam_max > 0, lam_min / lam_max, np.nan)
        valid = (~unseen) & np.isfinite(f_k) & np.isfinite(est_max)
        fk2 = f_k[valid] ** 2

        def q(x, ps=(1, 5, 25, 50, 75, 95, 99)):
            x = x[np.isfinite(x)]
            return {f"p{p}": float(np.percentile(x, p)) for p in ps} if len(x) else {}

        rec = {
            "n_primitives": int(len(data["xyz"])), "n_cameras": int(len(c2w)),
            "n_unseen": int(unseen.sum()), "frac_unseen": float(unseen.mean()),
            "I1_frac_est_exceeds_floor_worst_dir": float(np.mean(est_max[valid] > fk2)),
            "I1_frac_est_exceeds_floor_all_dirs": float(np.mean(est_min[valid] > fk2)),
            "I1_floor_fk_q": q(f_k[valid]),
            "I1_ratio_est_over_floor_q": q(np.sqrt(est_max[valid]) / f_k[valid]),
            "I2_lambda3_over_lambda1_q": q(aniso[valid]),
            "I2_median_sigma_ratio": float(
                np.nanmedian(1.0 / np.sqrt(np.maximum(aniso[valid], 1e-30)))),
            "seconds": time.time() - t,
        }
        if "filter_3D" in data:
            f3 = data["filter_3D"]
            rec["filter_3D_max"] = float(np.abs(f3).max())
            rec["filter_3D_q"] = q(f3)
        results[key] = rec
        print(f"[{key}] {rec['n_primitives']} prims | I-1 exceeds floor "
              f"{rec['I1_frac_est_exceeds_floor_worst_dir']:.1%} (worst dir), "
              f"{rec['I1_frac_est_exceeds_floor_all_dirs']:.1%} (all dirs) | "
              f"I-2 median sigmaD/sigmaL {rec['I2_median_sigma_ratio']:.2f}x | "
              f"{rec['seconds']:.0f}s", flush=True)
        with open(f"{RESULTS}/instruments.json", "w") as f:
            json.dump({"results": results, "failed": failed}, f, indent=2)
    except Exception as e:
        failed[key] = f"{type(e).__name__}: {e}"[:400]
        traceback.print_exc()

summary = {
    "rung": RUNG, "source_kernel": RUNG_KERNEL, "iteration": ITER,
    "results": results, "failed": failed,
    "n_done": len(results),
    "session_seconds": time.time() - SESSION_START,
}
armA = [v for k, v in results.items() if k.startswith("A/")]
if armA:
    summary["G3_median_frac_exceeding_floor_worst_dir"] = float(
        np.median([v["I1_frac_est_exceeds_floor_worst_dir"] for v in armA]))
    summary["G4_median_sigma_ratio"] = float(
        np.median([v["I2_median_sigma_ratio"] for v in armA]))
with open(f"{RESULTS}/instruments.json", "w") as f:
    json.dump(summary, f, indent=2)
print("R9_SUMMARY_JSON=" + json.dumps(summary))
if failed:
    print(f"note: {len(failed)} item(s) failed: {list(failed)[:6]}", flush=True)
