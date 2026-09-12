#!/usr/bin/env python3
"""
I-1 (floor occupancy) and I-2 (Lambda conditioning), computed from a trained
point cloud and the capture geometry.

Both instruments are read-only with respect to the optimiser, so attaching them
leaves the reproduction a reproduction. Neither needs the images, the rasteriser
or a training run: everything they need is the primitive positions (from the
saved PLY) and the camera poses and focal lengths (from the dataset's own JSON).
That is what makes them cheap enough to run on every scene.

I-1 -- THE KILL-SHOT.  Equation 4.4 floors the per-direction variance at
Mip-Splatting's f_k^2. If s^2/lambda_i <= f_k^2 for essentially every primitive
and every direction, Proposition 2 applies everywhere, the proposed method IS
the baseline, and the project must pivot. This is gate G3, and it is decidable
before a line of method code exists.

I-2 -- CONDITIONING.  Equation 4.8 predicts lambda_3/lambda_1 = sin^2(theta/2)
for two symmetric views, and a regime table over capture geometries. Measuring
the real distribution of lambda_3/lambda_1 on a trained scene turns that
prediction into evidence. This is gate G4.

    python tools/instruments.py --ply <point_cloud.ply> --cameras <transforms.json> \
        --scene lego --arm A --out results/instruments

The one trap this handles explicitly: compute_3D_filter assigns primitives that
NO camera sees `distance[valid].max()`. Those are an artefact of the
implementation, not a measurement of the capture, and left in they form a
spurious right tail that reads as the floor binding -- the exact wrong answer at
G3. They are identified and reported separately, never silently included.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Mip-Splatting's constant. s * sigma_pix = sqrt(0.2) by Proposition 1, so the
# estimation term and the floor are expressed in the same units by construction.
POINT_TWO = 0.2
SQRT_POINT_TWO = math.sqrt(POINT_TWO)


def load_ply(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    out = {"xyz": xyz.astype(np.float64)}
    if "filter_3D" in v.data.dtype.names:
        out["filter_3D"] = np.asarray(v["filter_3D"]).astype(np.float64)
    scales = [n for n in v.data.dtype.names if n.startswith("scale_")]
    if scales:
        out["scale"] = np.stack([np.asarray(v[n]) for n in sorted(scales)],
                                axis=1).astype(np.float64)
    return out


def load_cameras(path):
    """Camera-to-world matrices and focal lengths, in pixels.

    Accepts the Blender `transforms_*.json` (one global camera_angle_x) and the
    multi-scale `metadata.json` (per-frame focal, which is why the original 3DGS
    loader cannot read it -- F2).
    """
    with open(path) as f:
        meta = json.load(f)

    if "frames" in meta:                       # transforms_*.json
        c2w = np.array([fr["transform_matrix"] for fr in meta["frames"]], dtype=np.float64)
        # Width is not in the file; 800 is the Blender synthetic convention and
        # only sets the pixel scale, which cancels in lambda_3/lambda_1.
        w = meta.get("w", 800)
        focal = np.full(len(c2w), 0.5 * w / math.tan(0.5 * meta["camera_angle_x"]))
        return c2w, focal

    split = meta.get("train", meta.get("test"))
    c2w = np.array(split["cam2world"], dtype=np.float64)
    focal = np.array(split["focal"], dtype=np.float64)
    return c2w, focal


def fisher(xyz, c2w, focal, chunk=20000):
    """Accumulate Lambda per primitive, and the per-view depth statistics.

    Lambda_k = sum_n (J_n W_n)^T Sigma_pix^-1 (J_n W_n), with sigma_pix folded
    into the sqrt(0.2) calibration of Proposition 1 rather than carried
    separately.

    Visibility weight w_nk is taken as binary in-front-of-camera, which is the
    handbook's stated starting point (W4: "binary visibility weights"). Using the
    rasteriser's blending contributions instead is a refinement listed there as
    an upgrade, not a correction: it can only remove views, so it lowers
    lambda_min and can only make the estimation term LARGER relative to the
    floor. A floor that binds under binary visibility therefore binds a fortiori
    under contribution weighting, which is the direction that matters for G3.
    """
    n_pts = xyz.shape[0]
    lam = np.zeros((n_pts, 3), dtype=np.float64)
    d_min = np.full(n_pts, np.inf)
    n_seen = np.zeros(n_pts, dtype=np.int64)
    mean_dir = np.zeros((n_pts, 3), dtype=np.float64)

    R = c2w[:, :3, :3]
    C = c2w[:, :3, 3]

    for s in range(0, n_pts, chunk):
        p = xyz[s:s + chunk]                              # (m,3)
        acc = np.zeros((p.shape[0], 3, 3))
        for i in range(len(c2w)):
            # Blender/OpenGL cameras look down -z; the camera-frame depth of a
            # visible point is therefore positive after this negation.
            rel = p - C[i]
            cam = rel @ R[i]
            z = -cam[:, 2]
            vis = z > 1e-3
            if not vis.any():
                continue
            zz = np.where(vis, z, 1.0)

            # A single view's information is rank 2: (f/z)^2 on the two
            # directions transverse to the ray, exactly zero along it
            # (Appendix B.1). Build it as (f/z)^2 (I - dd^T) in world axes,
            # where d is the unit viewing direction.
            d = rel / np.linalg.norm(rel, axis=1, keepdims=True)
            w = ((focal[i] / zz) ** 2) * vis
            acc += w[:, None, None] * (np.eye(3)[None] - d[:, :, None] * d[:, None, :])

            mean_dir[s:s + chunk] += d * vis[:, None]
            n_seen[s:s + chunk] += vis
            d_min[s:s + chunk] = np.minimum(d_min[s:s + chunk], np.where(vis, z, np.inf))

        lam[s:s + chunk] = np.linalg.eigvalsh(acc)        # ascending
    return lam, d_min, n_seen, mean_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", default="results/instruments")
    ap.add_argument("--s", type=float, default=1.0, help="confidence scaling s")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ply = load_ply(a.ply)
    c2w, focal = load_cameras(a.cameras)
    xyz = ply["xyz"]
    print(f"{a.scene}/{a.arm}: {len(xyz)} primitives, {len(c2w)} cameras, "
          f"focal {focal.min():.1f}-{focal.max():.1f} px", flush=True)

    lam, d_min, n_seen, mean_dir = fisher(xyz, c2w, focal)
    lam_min, lam_mid, lam_max = lam[:, 0], lam[:, 1], lam[:, 2]

    unseen = n_seen == 0
    # Mip-Splatting's own floor, from the primitive's nearest depth and the
    # largest focal over cameras -- the two taken independently, as the source
    # does (F6), which is the mixed-camera defect of section 3.3.
    f_k = SQRT_POINT_TWO * np.where(unseen, np.nan, d_min) / focal.max()

    # The estimation term is (s * sigma_pix)^2 / mu_i, and Proposition 1 fixes
    # (s * sigma_pix)^2 = 0.2 -- the same constant the floor is built from. Both
    # sides of the comparison below therefore carry it, and dropping it from the
    # estimation side alone (which this did) inflates that side by a factor of 5,
    # or sqrt(5) = 2.24 in sigma. scene/fisher_filter.py has always applied it;
    # this is the over-determination between "s defaults to 1" and
    # "s * sigma_pix = sqrt(0.2)" that section 4.4 now states explicitly.
    with np.errstate(divide="ignore", invalid="ignore"):
        est_max = (a.s ** 2) * POINT_TWO / np.where(lam_min > 0, lam_min, np.nan)
        est_min = (a.s ** 2) * POINT_TWO / np.where(lam_max > 0, lam_max, np.nan)
        aniso = np.where(lam_max > 0, lam_min / lam_max, np.nan)        # lambda_3/lambda_1

    valid = (~unseen) & np.isfinite(f_k) & np.isfinite(est_max)
    fk2 = f_k[valid] ** 2

    # I-1: does the estimation term ever exceed the floor?
    exceeds_any = float(np.mean(est_max[valid] > fk2))
    exceeds_all = float(np.mean(est_min[valid] > fk2))

    def q(x, ps=(1, 5, 25, 50, 75, 95, 99)):
        x = x[np.isfinite(x)]
        return {f"p{p}": float(np.percentile(x, p)) for p in ps} if len(x) else {}

    result = {
        "scene": a.scene, "arm": a.arm, "s": a.s,
        "n_primitives": int(len(xyz)),
        "n_cameras": int(len(c2w)),
        "n_unseen_by_any_camera": int(unseen.sum()),
        "frac_unseen": float(unseen.mean()),
        "I1": {
            "frac_estimation_exceeds_floor_worst_direction": exceeds_any,
            "frac_estimation_exceeds_floor_all_directions": exceeds_all,
            "floor_fk_quantiles": q(f_k[valid]),
            "est_sigma_worst_quantiles": q(np.sqrt(est_max[valid])),
            "ratio_est_over_floor_quantiles": q(np.sqrt(est_max[valid]) / f_k[valid]),
        },
        "I2": {
            "lambda3_over_lambda1_quantiles": q(aniso[valid]),
            "median_anisotropy_sigma_ratio": float(
                np.nanmedian(1.0 / np.sqrt(np.maximum(aniso[valid], 1e-30)))),
        },
    }
    if "filter_3D" in ply:
        f3 = ply["filter_3D"]
        result["filter_3D"] = {
            "max": float(np.abs(f3).max()), "mean": float(f3.mean()),
            "quantiles": q(f3),
            # compute_3D_filter gives unseen primitives distance[valid].max();
            # flagged, never silently folded into the histogram.
            "n_at_unseen_fallback": int(unseen.sum()),
        }

    path = os.path.join(a.out, f"{a.scene}_{a.arm}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  I-1 estimation exceeds floor: {exceeds_any:.1%} of primitives "
          f"(worst direction), {exceeds_all:.1%} (all directions)")
    print(f"  I-2 median sigma_depth/sigma_lat: "
          f"{result['I2']['median_anisotropy_sigma_ratio']:.2f}x")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
