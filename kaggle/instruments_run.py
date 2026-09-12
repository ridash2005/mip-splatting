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
import math
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
ARMS = ["A"]               # I-1/I-2 are geometry; one arm per scene suffices
PROTOCOLS = ["full", "arc", "cone", "pencil", "mixed", "grazing"]
# ----------------------------------------------------------------------------

SESSION_START = time.time()
os.makedirs(RESULTS, exist_ok=True)

# Pinned to the SAME branch kaggle/method_eval.py trains on. Cloning the default
# branch instead is how the instruments and the stress suite came to describe
# different captures: `main` still carries the width-based `arc` and `cone`
# (20 cameras at 81 degrees, 3 at 30), while method-b1-b2 sizes them by camera
# count (25 at 93, 10 at 54). Both are defensible protocols; pairing conditioning
# measured on one with PSNR measured on the other is not, and nothing in either
# kernel could have noticed, because each was self-consistent.
BRANCH = "method-b1-b2"
subprocess.run(f"git clone -q -b {BRANCH} {REPO} {WORK}/repo", shell=True, check=True)
sys.path.insert(0, f"{WORK}/repo/tools")
subprocess.run("pip install -q plyfile", shell=True, check=True)

import numpy as np              # noqa: E402
import instruments as inst      # noqa: E402
import camera_protocols as cp   # noqa: E402


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

results, failed, skipped = {}, {}, []
for ply in plys:
    parts = ply.split("/")
    try:
        arm = [q for q in parts if q.startswith("out_arm")][0].replace("out_arm", "")
        scene = parts[parts.index("point_cloud") - 1]
    except Exception:
        failed[ply] = "could not parse arm/scene from path"
        continue
    if arm not in ARMS:
        skipped.append(f"{arm}/{scene}")      # deliberate, not a failure
        continue
    if scene not in cams:
        failed[f"{arm}/{scene}"] = "no camera JSON found under /kaggle/input"
        continue
    try:
        data = inst.load_ply(ply)
        c2w_all, focal_all = inst.load_cameras(cams[scene])
    except Exception as e:
        failed[f"{arm}/{scene}"] = f"load: {type(e).__name__}: {e}"[:300]
        continue

    # The stress suite (§6.4): subsets of the SAME cameras, so scene content is
    # fixed and only the capture geometry varies. Costs no training -- I-1 and
    # I-2 are functions of geometry alone.
    spec = cp.build(c2w_all, focal_all, seed=0)

    for proto in PROTOCOLS:
        sub = spec[proto]
        idx = np.array(sub["idx"], dtype=int)
        c2w = c2w_all[idx]
        focal = np.array(sub.get("focal", focal_all), dtype=float)[idx]
        key = f"{arm}/{scene}/{proto}"
        try:
            t = time.time()
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
                "protocol": proto,
                "n_primitives": int(len(data["xyz"])), "n_cameras": int(len(c2w)),
                "span_deg": sub["span_deg"],
                "predicted_sigma_ratio": float(
                    1.0 / max(math.sin(math.radians(sub["span_deg"]) / 2), 1e-9)),
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
            print(f"[{key}] n_cam={rec['n_cameras']:3d} span={rec['span_deg']:5.1f}deg | "
                  f"I-1 exceeds floor {rec['I1_frac_est_exceeds_floor_worst_dir']:6.1%} "
                  f"(worst) {rec['I1_frac_est_exceeds_floor_all_dirs']:6.1%} (all) | "
                  f"I-2 sigmaD/sigmaL {rec['I2_median_sigma_ratio']:6.2f}x "
                  f"(pred {rec['predicted_sigma_ratio']:6.2f}x) | "
                  f"{rec['seconds']:.0f}s", flush=True)
            with open(f"{RESULTS}/instruments.json", "w") as f:
                json.dump({"results": results, "failed": failed}, f, indent=2)
        except Exception as e:
            failed[key] = f"{type(e).__name__}: {e}"[:400]
            traceback.print_exc()

summary = {
    "rung": RUNG, "source_kernel": RUNG_KERNEL, "iteration": ITER,
    "results": results, "failed": failed, "skipped": skipped,
    "n_done": len(results),
    "session_seconds": time.time() - SESSION_START,
}
by_proto = {}
for proto in PROTOCOLS:
    vs = [v for v in results.values() if v.get("protocol") == proto]
    if not vs:
        continue
    by_proto[proto] = {
        "n_scenes": len(vs),
        "median_span_deg": float(np.median([v["span_deg"] for v in vs])),
        "median_frac_exceeding_floor_worst_dir": float(
            np.median([v["I1_frac_est_exceeds_floor_worst_dir"] for v in vs])),
        "median_frac_exceeding_floor_all_dirs": float(
            np.median([v["I1_frac_est_exceeds_floor_all_dirs"] for v in vs])),
        "median_sigma_ratio_measured": float(
            np.median([v["I2_median_sigma_ratio"] for v in vs])),
        "median_sigma_ratio_predicted": float(
            np.median([v["predicted_sigma_ratio"] for v in vs])),
        "median_ratio_est_over_floor_p50": float(
            np.median([v["I1_ratio_est_over_floor_q"].get("p50", float("nan"))
                       for v in vs])),
    }
summary["by_protocol"] = by_proto
if "full" in by_proto:
    summary["G3_median_frac_exceeding_floor_worst_dir"] = \
        by_proto["full"]["median_frac_exceeding_floor_worst_dir"]
    summary["G4_median_sigma_ratio"] = by_proto["full"]["median_sigma_ratio_measured"]

print("\nprotocol summary (median over scenes)")
print(f"{'protocol':>9} {'span':>7} {'I-1 worst':>10} {'I-1 all':>8} "
      f"{'sD/sL meas':>11} {'pred':>7} {'est/floor':>10}")
for proto, v in by_proto.items():
    print(f"{proto:>9} {v['median_span_deg']:6.1f}d "
          f"{v['median_frac_exceeding_floor_worst_dir']:10.2%} "
          f"{v['median_frac_exceeding_floor_all_dirs']:8.2%} "
          f"{v['median_sigma_ratio_measured']:11.2f} "
          f"{v['median_sigma_ratio_predicted']:7.2f} "
          f"{v['median_ratio_est_over_floor_p50']:10.3f}", flush=True)
with open(f"{RESULTS}/instruments.json", "w") as f:
    json.dump(summary, f, indent=2)
print("R9_SUMMARY_JSON=" + json.dumps(summary))
if failed:
    print(f"note: {len(failed)} item(s) failed: {list(failed)[:6]}", flush=True)
