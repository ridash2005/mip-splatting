#!/usr/bin/env python3
"""
Verify tools/instruments.py against the closed form the thesis derives.

Appendix B.2 claims lambda_3/lambda_1 = sin^2(theta/2) exactly for two symmetric
views, and Table B.2 quotes agreement to six significant figures at
theta = 2, 6, 15, 45, 90 degrees. That table is an assertion until something
recomputes it. This does, from the same accumulation code the instruments run on
real scenes -- so a pass validates the implementation AND independently confirms
the appendix.

    python tools/test_instruments.py

Exit 0 means Equation 4.8 and the code that measures it agree.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from instruments import fisher  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

failures = []


def ok(desc, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {desc}" + (f" — {detail}" if detail else ""))
    if not passed:
        failures.append(desc)


def two_symmetric_cameras(theta_deg, z=4.0, focal=1000.0):
    """Two cameras at distance z, symmetric about the origin, parallax theta.

    Built as camera-to-world matrices in the Blender convention the loader uses
    (camera looks down its own -z), so the test exercises the same axis handling
    as the real path rather than a simplified one.
    """
    th = math.radians(theta_deg)
    mats = []
    for sign in (-1, +1):
        a = sign * th / 2
        # Camera centre on a circle of radius z in the x--z plane.
        C = np.array([z * math.sin(a), 0.0, z * math.cos(a)])
        fwd = -C / np.linalg.norm(C)              # points at the origin
        zc = -fwd                                 # camera +z is backwards
        up = np.array([0.0, 1.0, 0.0])
        xc = np.cross(up, zc); xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = xc, yc, zc, C
        mats.append(m)
    return np.array(mats), np.full(2, focal)


def main():
    print("Equation 4.8 against tools/instruments.py, two symmetric views\n")
    print(f"{'theta':>7} {'l3/l1 measured':>16} {'sin^2(th/2)':>13} "
          f"{'sD/sL measured':>15} {'1/sin(th/2)':>12}")
    print("-" * 70)
    worst_rel = 0.0
    for theta in (2, 6, 15, 45, 90):
        c2w, focal = two_symmetric_cameras(theta)
        lam, d_min, n_seen, _ = fisher(np.zeros((1, 3)), c2w, focal)
        l1, l2, l3 = sorted(lam[0], reverse=True)      # l1 largest
        ratio = l3 / l1
        want = math.sin(math.radians(theta) / 2) ** 2
        sd_sl = 1.0 / math.sqrt(ratio) if ratio > 0 else float("inf")
        want_sd = 1.0 / math.sin(math.radians(theta) / 2)
        print(f"{theta:>6}° {ratio:16.6f} {want:13.6f} {sd_sl:15.3f} {want_sd:12.3f}")
        worst_rel = max(worst_rel, abs(ratio - want) / want)

        ok(f"lambda3/lambda1 = sin^2(theta/2) at {theta}°",
           abs(ratio - want) <= 1e-9 + 1e-6 * want,
           f"rel err {abs(ratio - want) / want:.2e}")
        ok(f"both cameras see the point at {theta}°", int(n_seen[0]) == 2,
           f"n_seen={int(n_seen[0])}")
        ok(f"nearest depth recovered at {theta}°", abs(d_min[0] - 4.0) < 1e-9,
           f"d_min={d_min[0]:.6f}")

    print()
    ok("agreement to at least six significant figures across all angles",
       worst_rel < 1e-6, f"worst relative error {worst_rel:.2e}")

    # A single view must carry exactly zero information along its own ray
    # (Appendix B.1) -- the rank claim the whole method rests on.
    c2w, focal = two_symmetric_cameras(0.001)
    lam, _, _, _ = fisher(np.zeros((1, 3)), c2w[:1], focal[:1])
    ev = sorted(lam[0])
    ok("a single view has rank 2: smallest eigenvalue is exactly zero",
       abs(ev[0]) < 1e-12 * max(ev[2], 1.0), f"eigenvalues {['%.4g' % e for e in ev]}")
    ok("a single view's two non-zero eigenvalues are equal (isotropic laterally)",
       abs(ev[1] - ev[2]) <= 1e-9 * ev[2], f"{ev[1]:.6g} vs {ev[2]:.6g}")
    ok("single-view lateral eigenvalue is (f/z)^2",
       abs(ev[2] - (1000.0 / 4.0) ** 2) < 1e-6 * (1000.0 / 4.0) ** 2,
       f"{ev[2]:.6g} vs {(1000.0/4.0)**2:.6g}")

    # Proposition 1, as a constraint that pins the constant on BOTH sides of I-1.
    # At one view, on axis, the estimation term must EQUAL Mip-Splatting's floor
    # -- that identity is what Proposition 1 asserts, and it fixes
    # (s * sigma_pix)^2 = 0.2. The instrument omitted the 0.2 from the estimation
    # side while the floor carried it, so the term came out 5x too large and I-1
    # disagreed with the filter's own training diagnostic on the arc and the cone.
    # Nothing raised; both numbers looked reasonable.
    print()
    import instruments as _inst
    f, z = 1111.0, 4.0
    c2w_one = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z], [0, 0, 0, 1]]],
                       dtype=float)
    lam1, d_min1, _, _ = _inst.fisher(np.zeros((1, 3)), c2w_one, np.array([f]))
    est = _inst.POINT_TWO / lam1[0].max()
    fk2 = (_inst.SQRT_POINT_TWO * d_min1[0] / f) ** 2
    ok("Proposition 1: at one view on axis the estimation term IS the floor",
       abs(est - fk2) <= 1e-9 * fk2,
       f"{est:.6e} vs {fk2:.6e}; without the 0.2 it would be 5x the floor")

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + "; ".join(failures))
        return 1
    print("Equation 4.8 and its implementation agree; Appendix B.2 reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
