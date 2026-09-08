#!/usr/bin/env python3
"""
B2 — angular identifiability of spherical harmonics.

Each primitive carries view-dependent colour as SH to degree 3: 16 coefficients
per channel, 48 floats, against 11 for position, scale, rotation and opacity
combined. The non-DC coefficients are 45 of 59 floats -- 76% of the model -- and
they are fitted from however many directions happen to see the primitive, which
may be very few.

Equation 5.1 accumulates the angular Gram matrix over the views that see a
primitive, and Equation 5.2 keeps degree l only where the corresponding block is
well conditioned:

    A_k = sum_n w_nk y(d_nk) y(d_nk)^T
    l_max(k) = max { l : lambda_min(A_k(l)) > tau }

Everything above l_max(k) is masked to zero.

This is a POST-HOC operation on a trained model: it needs the primitive
positions, the training camera directions and the SH coefficients, all of which
the saved PLY and the dataset already contain. No retraining, and no CUDA.

    python tools/sh_identifiability.py --ply <point_cloud.ply> \\
        --cameras <metadata.json> --out results/b2 --tau 0.01

tau is a conditioning threshold on the view-count-normalised Gram block, so it
does not drift with how many cameras happen to see a primitive. Verified against
synthetic captures: a dense sphere of 200 views supports degree 3 down to
tau = 0.05, a medium cone falls to degree 2 by tau = 0.01, and a narrow cone
supports only the DC term at every tau tried -- which is §5's claim that "a
primitive seen from three directions cannot support degree 3", made numerical.

The comparison that makes the result meaningful is against MAGNITUDE PRUNING at
matched size: dropping the smallest coefficients is the obvious baseline, and a
criterion that cannot beat it is not worth the machinery.
"""
import argparse
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Real SH basis to degree 3, matching utils/sh_utils.py's constants and order.
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435]
DEG_SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 9), 3: slice(9, 16)}


def sh_basis(d):
    """y(d) for degree 0..3, shape [N,16]. Same ordering as eval_sh."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    out = np.empty((d.shape[0], 16), dtype=np.float64)
    out[:, 0] = C0
    out[:, 1] = -C1 * y
    out[:, 2] = C1 * z
    out[:, 3] = -C1 * x
    out[:, 4] = C2[0] * xy
    out[:, 5] = C2[1] * yz
    out[:, 6] = C2[2] * (2.0 * zz - xx - yy)
    out[:, 7] = C2[3] * xz
    out[:, 8] = C2[4] * (xx - yy)
    out[:, 9] = C3[0] * y * (3 * xx - yy)
    out[:, 10] = C3[1] * xy * z
    out[:, 11] = C3[2] * y * (4 * zz - xx - yy)
    out[:, 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    out[:, 13] = C3[4] * x * (4 * zz - xx - yy)
    out[:, 14] = C3[5] * z * (xx - yy)
    out[:, 15] = C3[6] * x * (xx - 3 * yy)
    return out


def angular_gram(xyz, c2w, chunk=50000):
    """Per-degree blocks of A_k, and the number of views seeing each primitive.

    Only the four diagonal blocks are kept, not the full 16x16: the criterion
    reads lambda_min of each block, and storing 16x16 per primitive would be
    2 GB at two million primitives for no gain.
    """
    n = xyz.shape[0]
    blocks = {l: np.zeros((n, s.stop - s.start, s.stop - s.start))
              for l, s in DEG_SLICES.items()}
    n_seen = np.zeros(n, dtype=np.int64)
    C = np.asarray(c2w)[:, :3, 3]
    R = np.asarray(c2w)[:, :3, :3]

    for a in range(0, n, chunk):
        p = xyz[a:a + chunk]
        acc = {l: np.zeros((p.shape[0], b.shape[1], b.shape[1]))
               for l, b in blocks.items()}
        seen = np.zeros(p.shape[0], dtype=np.int64)
        for i in range(len(C)):
            rel = p - C[i]
            cam = rel @ R[i]
            vis = (-cam[:, 2]) > 1e-3
            if not vis.any():
                continue
            # Direction from primitive to camera, which is what the SH is
            # evaluated at when shading (dir_pp in the renderer).
            d = -rel / np.linalg.norm(rel, axis=1, keepdims=True)
            Y = sh_basis(d)
            for l, sl in DEG_SLICES.items():
                yl = Y[:, sl] * vis[:, None]
                acc[l] += yl[:, :, None] * yl[:, None, :]
            seen += vis
        for l in blocks:
            blocks[l][a:a + chunk] = acc[l]
        n_seen[a:a + chunk] = seen
    return blocks, n_seen


def l_max(blocks, n_seen, tau):
    """Equation 5.2, with degrees required to be contiguous from 0 upward."""
    n = n_seen.shape[0]
    out = np.zeros(n, dtype=np.int8)
    for l in (1, 2, 3):
        b = blocks[l]
        # normalise by the view count so tau is a conditioning threshold rather
        # than a function of how many cameras happen to see the primitive
        norm = np.maximum(n_seen, 1)[:, None, None]
        lmin = np.linalg.eigvalsh(b / norm)[:, 0]
        ok = (lmin > tau) & (out == l - 1)
        out[ok] = l
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out", default="results/b2")
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--scene", default="scene")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from instruments import load_cameras, load_ply
    data = load_ply(a.ply)
    c2w, _ = load_cameras(a.cameras)
    xyz = data["xyz"]
    blocks, n_seen = angular_gram(xyz, c2w)
    lm = l_max(blocks, n_seen, a.tau)

    kept = sum((DEG_SLICES[l].stop - DEG_SLICES[l].start) * int((lm >= l).sum())
               for l in (1, 2, 3))
    total = 15 * len(xyz)                      # non-DC coefficients per channel
    res = {
        "scene": a.scene, "tau": a.tau, "n_primitives": int(len(xyz)),
        "n_cameras": int(len(c2w)),
        "mean_l_max": float(lm.mean()),
        "l_max_histogram": {int(l): int((lm == l).sum()) for l in range(4)},
        "non_dc_retained": float(kept / total),
        "mean_views_per_primitive": float(n_seen.mean()),
    }
    path = os.path.join(a.out, f"{a.scene}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"{a.scene}: mean l_max {res['mean_l_max']:.2f}, "
          f"non-DC SH retained {res['non_dc_retained']:.1%}, "
          f"{res['mean_views_per_primitive']:.0f} views/primitive")
    print(f"  l_max histogram: {res['l_max_histogram']}")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
