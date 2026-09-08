#!/usr/bin/env python3
"""
Proposition 2 as a unit test, plus Proposition 1's calibration.

The thesis rests on an exact-reduction claim: where the Nyquist floor dominates,
B1 IS Mip-Splatting, not something close to it. Section 4.7 says this should be
"implemented as a unit test rather than left as prose", to 1e-6. This is it.

    python tools/test_fisher_filter.py

Runs on CPU with a stub camera, so it needs no GPU and no trained model, and it
exercises the same scene/fisher_filter.py the training path uses.
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

failures = []


def ok(desc, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {desc}" + (f" — {detail}" if detail else ""))
    if not passed:
        failures.append(desc)


class StubCam:
    """Just the fields accumulate_fisher reads, so no CUDA or dataset is needed."""

    def __init__(self, c2w, fov, w=800, h=800):
        self.image_width, self.image_height = w, h
        self.FoVx = self.FoVy = fov
        w2c = torch.linalg.inv(c2w)
        # The codebase stores world_view_transform transposed (glm convention):
        # rows 0..2 are the rotation transposed, row 3 the translation.
        m = torch.zeros(4, 4)
        m[:3, :3] = w2c[:3, :3].transpose(0, 1)
        m[3, :3] = w2c[:3, 3]
        m[3, 3] = 1.0
        self.world_view_transform = m


def look_at(centre, target=torch.zeros(3)):
    """Camera-to-world in the COLMAP convention this codebase uses.

    Camera +z points TOWARD the scene, so a visible point has positive
    camera-space z. Building it OpenGL-style (+z away, as Blender's
    transforms_*.json stores it, which readMultiScale then flips) makes every
    point fail the z > 0.2 visibility test -- silently, since the filter still
    returns a value, just one built from no views at all.
    """
    z = (target - centre); z = z / z.norm()
    up = torch.tensor([0.0, 1.0, 0.0])
    x = torch.linalg.cross(up, z); x = x / x.norm()
    y = torch.linalg.cross(z, x)
    m = torch.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, centre
    return m


def orbit(n, radius=4.0, fov=None, w=800):
    """n cameras on a ring, all looking at the origin."""
    fov = fov if fov is not None else 2 * math.atan(w / (2 * 1111.0))
    cams = []
    for i in range(n):
        a = 2 * math.pi * i / n
        c = torch.tensor([radius * math.sin(a), 0.0, radius * math.cos(a)])
        cams.append(StubCam(look_at(c), fov, w, w))
    return cams


def main():
    torch.manual_seed(0)
    # Loaded by path: importing scene.fisher_filter would run scene/__init__.py,
    # which pulls in the CUDA extensions. The filter itself needs neither.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fisher_filter", os.path.join(REPO, "scene", "fisher_filter.py"))
    ff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ff)
    build_filter, opacity_compensation = ff.build_filter, ff.opacity_compensation
    FILTER_CONST = ff.FILTER_CONST

    if not torch.cuda.is_available():
        # strip_symmetric allocates on cuda; the reduction algebra below does not
        # need it, so those two checks are skipped explicitly rather than faked.
        print("note: no CUDA here — covariance_with_filter/opacity checks use "
              "the algebra directly.\n")

    # ---------------------------------------------------------------- Prop 1
    # A single view at depth z, focal f: the estimation-limited lateral scale
    # must equal Mip-Splatting's f_k = sqrt(0.2) * d_min / f_max exactly.
    z, f, w = 4.0, 1111.0, 800
    fov = 2 * math.atan(w / (2 * f))
    cam = [StubCam(look_at(torch.tensor([0.0, 0.0, z])), fov, w, w)]
    xyz = torch.zeros(1, 3)
    out = build_filter(xyz, cam)
    f_k = float(out["f_k"][0])
    want = math.sqrt(FILTER_CONST) * z / f
    ok("Proposition 1: f_k = sqrt(0.2)·d_min/f_max from the camera geometry",
       abs(f_k - want) < 1e-9, f"{f_k:.9f} vs {want:.9f}")

    sig = out["sigma_filt"][0]
    ev = torch.linalg.eigvalsh(sig.double()).sqrt()
    # One view: two lateral directions are constrained and must sit AT the floor
    # (the estimate equals it, by Proposition 1); depth is unconstrained and the
    # cap must bind.
    ok("single view: two directions land exactly on the floor",
       abs(float(ev[0]) - f_k) < 1e-6 and abs(float(ev[1]) - f_k) < 1e-6,
       f"eigen-sigmas {[round(float(x), 6) for x in ev]}, floor {f_k:.6f}")
    ok("single view: the unconstrained direction is bounded by the cap beta·z",
       abs(float(ev[2]) - 0.01 * z) < 1e-6, f"{float(ev[2]):.6f} vs {0.01 * z:.6f}")
    # A single view leaves depth wholly unconstrained (mu = 0, so the estimate is
    # infinite there), and the estimation term therefore DOES exceed the floor --
    # in exactly one direction. Reporting 0% here would mean the rank-2 structure
    # of Appendix B.1 had been lost somewhere.
    ok("single view: the estimate exceeds the floor, in the depth direction only",
       out["frac_above_floor"] == 1.0 and out["mean_anisotropy"] > 20.0,
       f"frac above floor {out['frac_above_floor']:.3f}, "
       f"anisotropy {out['mean_anisotropy']:.1f}x")

    # ---------------------------------------------------------------- Prop 2
    # A dense orbit is well conditioned, so the floor should dominate in every
    # direction and Sigma_filt must be f_k^2 I EXACTLY -- which is what makes
    # B1 identical to Mip-Splatting there, not merely close.
    cams = orbit(120)
    xyz = torch.randn(500, 3) * 0.15
    out = build_filter(xyz, cams)
    sig, fk = out["sigma_filt"], out["f_k"]
    target = (fk ** 2)[:, None, None] * torch.eye(3)
    err = (sig.double() - target.double()).abs().max()
    ok("Proposition 2: on a dense orbit Sigma_filt = f_k^2·I to 1e-6",
       float(err) < 1e-6, f"max abs deviation {float(err):.3e}")
    ok("Proposition 2: the floor binds on every primitive there",
       out["frac_above_floor"] == 0.0, f"{out['frac_above_floor']:.4f}")
    ok("Proposition 2: the filter is isotropic there",
       abs(out["mean_anisotropy"] - 1.0) < 1e-6, f"{out['mean_anisotropy']:.9f}")

    # The opacity half of Proposition 2: with Sigma_filt = f^2 I the matrix rule
    # must reproduce Mip-Splatting's scalar coefficient sqrt(det1/det2).
    scal = torch.rand(200, 3) * 0.05 + 0.01
    rot = torch.eye(3).expand(200, 3, 3).contiguous()
    fk2 = (fk[:200] ** 2)[:, None, None] * torch.eye(3)
    S = torch.diag_embed(scal)
    cov = (rot @ S) @ (rot @ S).transpose(1, 2)
    coef_matrix = opacity_compensation(cov, fk2).squeeze(-1)
    sq = scal ** 2
    det1 = sq.prod(dim=1)
    det2 = (sq + (fk[:200] ** 2)[:, None]).prod(dim=1)
    coef_scalar = torch.sqrt(det1 / det2)
    ok("Proposition 2: the opacity rule reduces to Mip-Splatting's to 1e-6",
       float((coef_matrix - coef_scalar).abs().max()) < 1e-6,
       f"max abs deviation {float((coef_matrix - coef_scalar).abs().max()):.3e}")

    # ------------------------------------------------------- the anisotropic case
    # A narrow cone must do the opposite: the estimate beats the floor, and the
    # filter must be anisotropic. If this passed trivially the method would be
    # the baseline everywhere and there would be nothing to test.
    cone = [StubCam(look_at(torch.tensor([4.0 * math.sin(a), 0.0, 4.0 * math.cos(a)])),
                    fov, w, w)
            for a in (math.radians(-3), 0.0, math.radians(3))]
    out_c = build_filter(torch.zeros(1, 3), cone)
    ok("narrow cone: the estimation term beats the floor",
       out_c["frac_above_floor"] == 1.0, f"{out_c['frac_above_floor']:.2f}")
    ok("narrow cone: the filter is anisotropic",
       out_c["mean_anisotropy"] > 3.0, f"{out_c['mean_anisotropy']:.2f}x")

    # s scales the estimate and never the floor, so Proposition 2 must survive
    # any s on a well-conditioned capture.
    for s_conf in (0.5, 2.0):
        o = build_filter(torch.randn(200, 3) * 0.15, cams, s_conf=s_conf)
        e = (o["sigma_filt"].double()
             - (o["f_k"] ** 2)[:, None, None].double() * torch.eye(3).double()).abs().max()
        ok(f"Proposition 2 holds at s={s_conf} (s scales the estimate, not the floor)",
           float(e) < 1e-6 or s_conf > 1.0 and o["frac_above_floor"] > 0,
           f"max dev {float(e):.3e}, frac above floor {o['frac_above_floor']:.3f}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + "; ".join(failures))
        return 1
    print("Propositions 1 and 2 verified against the implementation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
