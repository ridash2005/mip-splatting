#!/usr/bin/env python3
"""
The closed-form claims §4 and Appendix B make about Lambda, checked numerically.

Every number the thesis quotes for M-4, M-5 and M-6 is printed by this script and
by nothing else. Run it and the chapter's arithmetic is reproduced:

    python tools/test_geometry_claims.py

Three claims, in the order the thesis makes them.

M-4  The rank-1 shortcut (f/z)^2 (I - d d^T) is EXACT ONLY ON-AXIS. Off-axis the
     two non-zero eigenvalues of the true per-view information separate, and they
     do so by an exact law -- lambda_2/lambda_1 = cos^2(phi), phi the off-axis
     angle. So perspective contributes anisotropy on its own, with no parallax at
     all, and a scalar filter throws that away before parallax is even considered.

M-5  Two views subtending theta give Lambda proportional to 2I - d1 d1^T - d2 d2^T,
     whose spectrum is exactly {2, 2cos^2(theta/2), 2sin^2(theta/2)}. The filter is
     therefore TRIAXIAL: the in-plane lateral direction is degraded too, not just
     depth. NOTE the ordering caveat this script exists to catch -- see below.

M-6  alpha <- alpha sqrt(|Sigma| / |Sigma + Sigma_filt|) preserves total mass for
     ANY positive-definite Sigma_filt, so Mip-Splatting's opacity compensation
     carries over to the anisotropic case unchanged.

Exit status is non-zero if any claim fails its tolerance, so this is a test, not a
demo -- the rung kernels run it the same way they run test_fisher_filter.py.
"""
import argparse
import json
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILURES = []

# What the thesis quotes. Written to JSON by --json and read by
# tools/make_tex.py, so no closed-form number is ever typed into the document.
RESULTS = {}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# The true per-view information, with no rank-1 approximation anywhere.
# --------------------------------------------------------------------------
def view_information(p_cam, f=1.0):
    """J^T J for a pinhole camera observing a point at camera-frame p_cam.

    u = f x/z, v = f y/z, so J = (f/z) [[1, 0, -x/z], [0, 1, -y/z]] and the
    per-view Fisher information in the point's position is J^T J. This is the
    quantity Eq. 4.5 approximates by (f/z)^2 (I - d d^T).
    """
    x, y, z = p_cam
    J = (f / z) * np.array([[1.0, 0.0, -x / z],
                            [0.0, 1.0, -y / z]])
    return J.T @ J


def rank1_shortcut(p_cam, f=1.0):
    """(f/z)^2 (I - d d^T) -- Eq. 4.5's approximation, in the same frame."""
    z = p_cam[2]
    d = p_cam / np.linalg.norm(p_cam)
    return (f / z) ** 2 * (np.eye(3) - np.outer(d, d))


# --------------------------------------------------------------------------
# M-4
# --------------------------------------------------------------------------
def m4():
    print("\nM-4  the rank-1 shortcut is exact only on-axis")
    print("-" * 62)

    # (a) On-axis it is exact.
    p = np.array([0.0, 0.0, 4.0])
    err = np.abs(view_information(p) - rank1_shortcut(p)).max()
    check("on-axis: J^T J == (f/z)^2 (I - d d^T)", err < 1e-12, f"max|diff| = {err:.2e}")

    # (b) The shortcut is rank-1-deficient BY CONSTRUCTION in the sense that its
    #     two non-zero eigenvalues are always EQUAL: I - d d^T is a projector, so
    #     its spectrum is {1, 1, 0} whatever d is. The true information does not
    #     behave that way off-axis. That is the whole of M-4.
    d = np.array([0.3, -0.7, 2.0]); d /= np.linalg.norm(d)
    lam_short = np.linalg.eigvalsh(np.eye(3) - np.outer(d, d))[::-1]
    check("shortcut spectrum is always {1, 1, 0}",
          abs(lam_short[0] - lam_short[1]) < 1e-12 and abs(lam_short[2]) < 1e-12,
          f"eigs = {np.round(lam_short, 12)}")

    # (c) The exact law. Derivation, in full, because the thesis quotes it:
    #       J^T J = (f/z)^2 B^T B  with  B = [[1, 0, -a1], [0, 1, -a2]],
    #       a = (x/z, y/z).  B B^T = [[1+a1^2, a1 a2], [a1 a2, 1+a2^2]] has
    #       trace 2+|a|^2 and determinant 1+|a|^2, so its eigenvalues are
    #       1+|a|^2 and 1.  Hence
    #                 lambda_2 / lambda_1 = 1 / (1 + |a|^2) = cos^2(phi),
    #       phi = atan|a| being the angle off the optical axis. Verified against a
    #       numeric eigendecomposition over random poses below.
    print("\n  exact law:  lambda_2/lambda_1 = cos^2(phi),  phi = off-axis angle")
    print(f"  {'phi (deg)':>10}  {'measured l2/l1':>15}  {'cos^2(phi)':>12}  {'|diff|':>10}")

    rng = np.random.default_rng(0)
    worst = 0.0
    sampled = []
    for _ in range(4):
        # A random point in front of the camera, off-axis by construction.
        p = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5),
                      rng.uniform(1.5, 4.0)])
        lam = np.linalg.eigvalsh(view_information(p))[::-1]
        ratio = lam[1] / lam[0]
        phi = np.arctan2(np.hypot(p[0], p[1]), p[2])
        pred = np.cos(phi) ** 2
        worst = max(worst, abs(ratio - pred))
        sampled.append((float(np.degrees(phi)), float(ratio), float(pred)))
        print(f"  {np.degrees(phi):10.2f}  {ratio:15.6f}  {pred:12.6f}  "
              f"{abs(ratio - pred):10.2e}")

    # And over a dense sweep, so the law is checked rather than sampled.
    for _ in range(2000):
        p = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.5, 6.0)])
        lam = np.linalg.eigvalsh(view_information(p))[::-1]
        phi = np.arctan2(np.hypot(p[0], p[1]), p[2])
        worst = max(worst, abs(lam[1] / lam[0] - np.cos(phi) ** 2))
    check("lambda_2/lambda_1 == cos^2(phi) over 2000 random poses",
          worst < 1e-9, f"max|diff| = {worst:.2e}")

    # (d) The consequence the thesis actually needs: off-axis, the shortcut is
    #     wrong by a factor that is not a scalar -- it is a different SHAPE.
    p = np.array([1.2, -0.8, 2.5])
    rel = (np.abs(view_information(p) - rank1_shortcut(p)).max()
           / np.abs(view_information(p)).max())
    check("off-axis: shortcut differs from the truth", rel > 0.05,
          f"relative max|diff| = {rel:.3f}")
    print(f"\n  four sampled poses, lambda_2/lambda_1 = "
          f"{', '.join(f'{r[1]:.3f}' for r in sorted(sampled))}")
    RESULTS["m4"] = {"poses": sorted(sampled, key=lambda t: t[0]),
                     "law_max_abs_err": float(worst), "n_sweep": 2000,
                     "offaxis_rel_diff": float(rel)}
    return sorted(sampled)


# --------------------------------------------------------------------------
# M-5
# --------------------------------------------------------------------------
def m5():
    print("\nM-5  two views: the exact spectrum, and the ordering caveat")
    print("-" * 62)

    def two_view(theta):
        """Lambda for two unit rays symmetric about x, subtending theta."""
        h = theta / 2.0
        d1 = np.array([np.cos(h), np.sin(h), 0.0])
        d2 = np.array([np.cos(h), -np.sin(h), 0.0])
        return 2 * np.eye(3) - np.outer(d1, d1) - np.outer(d2, d2)

    print(f"  {'theta':>7}  {'measured spectrum (sorted desc)':>36}  "
          f"{'predicted set':>30}  {'|diff|':>9}")
    worst = 0.0
    spectrum = []
    for deg in (2, 6, 15, 45, 90, 120, 179):
        th = np.radians(deg)
        lam = np.linalg.eigvalsh(two_view(th))[::-1]
        pred = np.sort([2.0, 2 * np.cos(th / 2) ** 2, 2 * np.sin(th / 2) ** 2])[::-1]
        diff = np.abs(lam - pred).max()
        worst = max(worst, diff)
        spectrum.append({"theta_deg": deg, "measured": [float(x) for x in lam],
                         "predicted": [float(x) for x in pred], "abs_err": float(diff),
                         "true_ratio": float(lam[2] / lam[0]),
                         "naive": float(np.sin(th / 2) ** 2),
                         "corrected": float(min(np.sin(th / 2) ** 2,
                                                np.cos(th / 2) ** 2))})
        print(f"  {deg:5d} deg  {np.array2string(lam, precision=6):>36}  "
              f"{np.array2string(pred, precision=6):>30}  {diff:9.1e}")
    check("spectrum == {2, 2cos^2(theta/2), 2sin^2(theta/2)}", worst < 1e-10,
          f"max|diff| = {worst:.1e}")

    # Triaxial, not merely elongated: at a generic theta all three differ.
    lam = np.linalg.eigvalsh(two_view(np.radians(45)))[::-1]
    check("filter is triaxial (three distinct eigenvalues at 45 deg)",
          abs(lam[0] - lam[1]) > 1e-3 and abs(lam[1] - lam[2]) > 1e-3,
          f"eigs = {np.round(lam, 6)}")

    # ---- the caveat this script exists to catch -------------------------
    # lambda_3/lambda_1 = sin^2(theta/2) is the SMALLEST-over-LARGEST ratio only
    # while sin^2(theta/2) <= cos^2(theta/2), i.e. only for theta <= 90 deg. Past
    # 90 deg the depth direction is better constrained than the in-plane lateral
    # one and the two swap, so the correct statement for all theta is
    #        lambda_min / lambda_max = min(sin^2(theta/2), cos^2(theta/2)).
    # Quoting sin^2(theta/2) at theta = 120 deg gives 0.750 where the true ratio
    # is 0.250 -- a factor of three, and in the optimistic direction.
    print("\n  ordering: sin^2(theta/2) is lambda_min/lambda_max only for theta <= 90 deg")
    print(f"  {'theta':>7}  {'true l_min/l_max':>17}  {'sin^2(t/2)':>11}  "
          f"{'min(sin^2,cos^2)':>17}")
    ok = True
    for deg in (2, 6, 15, 45, 90, 120, 179):
        th = np.radians(deg)
        lam = np.linalg.eigvalsh(two_view(th))[::-1]
        true = lam[2] / lam[0]
        naive = np.sin(th / 2) ** 2
        corrected = min(np.sin(th / 2) ** 2, np.cos(th / 2) ** 2)
        ok &= abs(true - corrected) < 1e-10
        flag = "  <-- naive form wrong here" if abs(true - naive) > 1e-6 else ""
        print(f"  {deg:5d} deg  {true:17.6f}  {naive:11.6f}  {corrected:17.6f}{flag}")
    check("lambda_min/lambda_max == min(sin^2, cos^2) for all theta", ok)
    RESULTS["m5"] = {"spectrum": spectrum, "max_abs_err": float(worst),
                     "triaxial_at_45": [float(x) for x in
                                        np.linalg.eigvalsh(two_view(np.radians(45)))[::-1]]}


# --------------------------------------------------------------------------
# M-5b -- what the STRESS PROTOCOLS actually predict
# --------------------------------------------------------------------------
def cap_ratio(alpha):
    """lambda_min/lambda_max of Lambda for views uniform on a cap of half-angle alpha.

    Closed form. With directions d uniform on a spherical cap of half-angle
    alpha about +z, cos t is uniform on [cos alpha, 1], so

        E[d_z^2] = (1 + c + c^2)/3,          c = cos alpha,
        E[d_x^2] = E[d_y^2] = (1 - E[d_z^2])/2.

    The unit-weight observability matrix is Lambda = N (I - E[d d^T]), diagonal
    in this frame, so

        lambda_depth = N (1 - E[d_z^2]),   lambda_lat = N (1 + E[d_z^2])/2,

    and the ratio below follows. It is exact, not a small-angle expansion; the
    Monte-Carlo check in m5b() confirms it against sampled directions.
    """
    c = np.cos(alpha)
    e = (1.0 + c + c * c) / 3.0
    return 2.0 * (1.0 - e) / (1.0 + e)


def m5b():
    """The two-view law does not transfer to a distribution of cameras.

    Section 6.4's regime table reads the two-view formula off a protocol's
    measured angular SPAN, which is only defensible when the protocol really is
    two views. A cone of ten cameras spanning 20 degrees is a distribution, and
    its Lambda is the SUM over that distribution -- a different number.

    The direction of the error matters and is easy to get backwards. For a cap
    of half-angle alpha the ratio is asymptotically HALF the two-view value
    (alpha^2/2 against sin^2 alpha ~ alpha^2), because the cap's cameras are
    spread over the interval rather than sitting at its two extremes. A smaller
    lambda_3/lambda_1 is a WORSE-conditioned capture, so the two-view form
    UNDERSTATES a cap's depth anisotropy -- by sqrt(2) in sigma_D/sigma_L. That
    is the direction that matters: the measured anisotropy exceeding the two-view
    prediction is the model working, not the model failing.
    """
    print("\nM-5b  the multi-view prediction the protocols need")
    print("-" * 62)
    print("  cameras spread over a spherical cap, primitive at the centre.")
    print("  closed form:  l_min/l_max = 2(1-E)/(1+E),  E = (1+c+c^2)/3, c = cos(span/2)")
    print(f"  {'span':>8}  {'cap (exact)':>12}  {'cap (MC)':>10}  "
          f"{'two-view':>10}  {'cap sD/sL':>10}  {'2view sD/sL':>12}")

    rng = np.random.default_rng(0)
    cap = []
    worst_mc = 0.0
    for span_deg in (6, 20, 41, 60, 80, 93, 120, 168, 179):
        alpha = np.radians(span_deg) / 2.0
        exact = cap_ratio(alpha)
        # Monte-Carlo on the same cap, so the closed form is checked not asserted.
        n = 200000
        cos_t = 1 - rng.random(n) * (1 - np.cos(alpha))
        sin_t = np.sqrt(1 - cos_t ** 2)
        phi = rng.random(n) * 2 * np.pi
        d = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
        lam_mat = n * np.eye(3) - d.T @ d
        lam = np.linalg.eigvalsh(lam_mat)[::-1]
        mc = lam[2] / lam[0]
        worst_mc = max(worst_mc, abs(mc - exact) / exact)
        naive = np.sin(alpha) ** 2
        cap.append({"span_deg": span_deg,
                    "half_deg": float(np.degrees(alpha)),
                    "cap_ratio": float(exact),
                    "cap_ratio_mc": float(mc),
                    "two_view": float(naive),
                    "understatement": float(exact / max(naive, 1e-12)),
                    "cap_sigma_ratio": float(1.0 / np.sqrt(max(exact, 1e-12))),
                    "two_view_sigma_ratio": float(1.0 / np.sqrt(max(naive, 1e-12)))})
        print(f"  {span_deg:5d} deg  {exact:12.6f}  {mc:10.6f}  {naive:10.6f}  "
              f"{1 / np.sqrt(exact):10.2f}x  {1 / np.sqrt(naive):11.2f}x")
    check("cap closed form matches Monte-Carlo over 9 spans", worst_mc < 5e-3,
          f"max relative error = {worst_mc:.2e}")

    small = cap_ratio(np.radians(0.5)) / np.sin(np.radians(0.5)) ** 2
    check("cap ratio -> half the two-view value as the span closes",
          abs(small - 0.5) < 1e-3, f"cap/two-view at 1 deg span = {small:.6f}")

    print("\n  The two-view form UNDERSTATES the depth anisotropy of a cap: its")
    print("  l_min/l_max is about twice the cap's, so sigma_D/sigma_L is sqrt(2)")
    print("  too small. Section 6.4's table must quote the cap prediction, and the")
    print("  instruments' measured sigma_D/sigma_L must be compared to THAT.")
    RESULTS["m5b"] = {"cap": cap, "n_samples": 200000,
                      "mc_max_rel_err": float(worst_mc),
                      "small_span_limit": float(small)}


# --------------------------------------------------------------------------
# M-6
# --------------------------------------------------------------------------
def m6():
    print("\nM-6  the mass-preserving opacity compensation, for any PD Sigma_filt")
    print("-" * 62)

    def mass(alpha, cov):
        """Integral of alpha * exp(-1/2 x^T cov^-1 x) over R^3."""
        return alpha * (2 * np.pi) ** 1.5 * np.sqrt(np.linalg.det(cov))

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        def rand_pd():
            A = rng.normal(size=(3, 3))
            return A @ A.T + 1e-3 * np.eye(3)

        Sigma, Sigma_f = rand_pd(), rand_pd()
        alpha = float(rng.uniform(0.05, 1.0))
        alpha_new = alpha * np.sqrt(np.linalg.det(Sigma)
                                    / np.linalg.det(Sigma + Sigma_f))
        rel = abs(mass(alpha_new, Sigma + Sigma_f) - mass(alpha, Sigma)) / mass(alpha, Sigma)
        worst = max(worst, rel)
    check("mass preserved for 500 random anisotropic Sigma_filt", worst < 1e-12,
          f"max relative error = {worst:.2e}")

    # And the isotropic case Mip-Splatting actually ships is the special case.
    Sigma = np.diag([0.4, 0.1, 0.02])
    f2 = 0.05
    iso = alpha_iso = 1.0
    a_aniso = iso * np.sqrt(np.linalg.det(Sigma) / np.linalg.det(Sigma + f2 * np.eye(3)))
    a_mip = alpha_iso * np.sqrt(np.prod(np.diag(Sigma)) / np.prod(np.diag(Sigma) + f2))
    check("reduces to Mip-Splatting's scalar form when Sigma_filt = f^2 I",
          abs(a_aniso - a_mip) < 1e-14, f"|diff| = {abs(a_aniso - a_mip):.2e}")
    RESULTS["m6"] = {"mass_max_rel_err": float(worst), "n_trials": 500,
                     "iso_abs_diff": float(abs(a_aniso - a_mip))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write every quoted number here")
    args = ap.parse_args()
    print("=" * 62)
    print("Closed-form claims of §4 and Appendix B, verified numerically")
    print("=" * 62)
    m4()
    m5()
    m5b()
    m6()
    print("\n" + "=" * 62)
    if args.json:
        RESULTS["failures"] = FAILURES
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(RESULTS, fh, indent=2)
        print(f"wrote {args.json}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} claim(s): {', '.join(FAILURES)}")
        sys.exit(1)
    print("All closed-form claims verified.")
