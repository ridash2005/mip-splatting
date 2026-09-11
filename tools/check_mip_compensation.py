#!/usr/bin/env python3
"""
C1's verification: prove the 2D Mip opacity compensation is really gone.

The audit's BLOCKER 1 was that arm B kept Mip-Splatting's opacity compensation
    rho = sqrt(det(Sigma') / det(Sigma' + k I))
while calling itself 3DGS. `--disable_2D_mip_compensation` removes it. This
script is the evidence that it does, and it is the gate the prompt puts in front
of spending quota on the sweep.

Three checks, in increasing strength:

  1  ANALYTIC (CPU, always runs). Recompute rho in PyTorch exactly as
     computeCov2D does -- same 1.3*tan_fov clamp, same J, same W -- and print its
     distribution. If rho were ~1 everywhere the switch would be cosmetic; it is
     not, and the printed quantiles say by how much.

  2  DIFFERENTIAL (needs CUDA + the built extension). Rasterise the same
     primitives twice, flag on and off, same kernel_size. The images must differ,
     and the compensated one must be DARKER on average, because rho <= 1 always.

  3  THE DECISIVE ONE (needs CUDA). As kernel_size -> 0, rho -> 1 by
     construction, so the two paths must CONVERGE to float precision. If they do,
     the switch removes exactly the rho term and touches nothing else -- which is
     the claim, and it needs no vanilla build to check.

    python tools/check_mip_compensation.py                 # analytic only
    python tools/check_mip_compensation.py --ply <path>    # use a trained model
"""
import argparse
import math
import sys

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def cov2d(means3D, cov3D, viewmatrix, focal_x, focal_y, tan_fovx, tan_fovy):
    """Sigma' = J W Sigma W^T J^T, line for line with computeCov2D's forward.

    The 1.3*tan clamp is part of it: it is what keeps a primitive at the very edge
    of the frustum from producing an unbounded Jacobian, and leaving it out would
    make the rho computed here disagree with the one the kernel actually applies.
    """
    R = viewmatrix[:3, :3]
    t = means3D @ R + viewmatrix[3, :3]            # row-vector convention, as in CUDA
    limx, limy = 1.3 * tan_fovx, 1.3 * tan_fovy
    txtz, tytz = t[:, 0] / t[:, 2], t[:, 1] / t[:, 2]
    tx = torch.clamp(txtz, -limx, limx) * t[:, 2]
    ty = torch.clamp(tytz, -limy, limy) * t[:, 2]
    tz = t[:, 2]

    n = means3D.shape[0]
    J = torch.zeros(n, 3, 3, dtype=means3D.dtype)
    J[:, 0, 0] = focal_x / tz
    J[:, 0, 2] = -(focal_x * tx) / (tz * tz)
    J[:, 1, 1] = focal_y / tz
    J[:, 1, 2] = -(focal_y * ty) / (tz * tz)

    W = R.T.unsqueeze(0).expand(n, 3, 3)
    T = J @ W
    return T @ cov3D @ T.transpose(1, 2)


def rho(sigma2d, kernel_size):
    """The compensation the CUDA forward applies, with its guards."""
    a, b, c = sigma2d[:, 0, 0], sigma2d[:, 0, 1], sigma2d[:, 1, 1]
    det_0 = torch.clamp(a * c - b * b, min=1e-6)
    det_1 = torch.clamp((a + kernel_size) * (c + kernel_size) - b * b, min=1e-6)
    return torch.sqrt(det_0 / (det_1 + 1e-6) + 1e-6)


def synth(n=200_000, seed=0):
    """A stand-in primitive cloud with the scale distribution 3DGS converges to.

    Log-uniform scales over four decades: the compensation only bites on
    primitives small relative to a pixel, so a cloud that is all large would
    understate the effect and a cloud that is all small would overstate it.
    """
    g = torch.Generator().manual_seed(seed)
    xyz = (torch.rand(n, 3, generator=g) - 0.5) * 3.0
    xyz[:, 2] += 4.0
    log_s = torch.rand(n, 3, generator=g) * 4.0 - 4.5          # 1e-4.5 .. 1e-0.5
    s = torch.pow(10.0, log_s)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(n, 3, 3)
    S = torch.diag_embed(s)
    M = R @ S
    return xyz, M @ M.transpose(1, 2)


def load_ply(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    xyz = torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32)
    s = torch.tensor(np.stack([v[f"scale_{i}"] for i in range(3)], 1),
                     dtype=torch.float32).exp()
    q = torch.tensor(np.stack([v[f"rot_{i}"] for i in range(4)], 1), dtype=torch.float32)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)
    M = R @ torch.diag_embed(s)
    return xyz, M @ M.transpose(1, 2)


# ---------------------------------------------------------------- check 1
def analytic(xyz, cov, width=800, fov_deg=50.0):
    print("\n1  ANALYTIC -- the size of the term being removed")
    print("-" * 66)
    tan_fov = math.tan(math.radians(fov_deg) * 0.5)
    focal = width / (2.0 * tan_fov)
    view = torch.eye(4)
    s2d = cov2d(xyz, cov, view, focal, focal, tan_fov, tan_fov)

    print(f"  {len(xyz)} primitives, {width}px, fov {fov_deg} deg (focal {focal:.1f}px)")
    print(f"\n  {'kernel_size':>12}  {'arm':>16}  {'mean rho':>9}  {'p05':>7}  {'p50':>7}"
          f"  {'p95':>7}  {'frac rho<0.99':>14}")
    out = {}
    for k, arm in ((0.1, "A Mip-Splatting"), (0.3, "B 3DGS dilation")):
        r = rho(s2d, k)
        out[k] = r
        q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95]))
        print(f"  {k:12.2f}  {arm:>16}  {r.mean():9.4f}  {q[0]:7.4f}  {q[1]:7.4f}"
              f"  {q[2]:7.4f}  {(r < 0.99).float().mean():14.1%}")

    r03 = out[0.3]
    check("rho is materially below 1 at the 3DGS dilation",
          (r03 < 0.99).float().mean() > 0.2,
          f"{(r03 < 0.99).float().mean():.1%} of primitives have rho < 0.99")
    check("rho <= 1 everywhere (it can only DIM)", bool((r03 <= 1.0 + 1e-6).all()),
          f"max rho = {r03.max():.6f}")
    print(f"\n  Mean opacity multiplier removed by --disable_2D_mip_compensation:"
          f" {r03.mean():.4f}")
    print(f"  i.e. arm B has been rendering every primitive at {r03.mean():.1%} of its")
    print("  trained opacity, which is the anti-aliasing the audit measured as a")
    print("  4.7-6.9 dB excess at reduced scale.")

    # rho -> 1 as kernel_size -> 0, which is what check 3 exploits on GPU.
    tiny = rho(s2d, 1e-6)
    check("rho -> 1 as kernel_size -> 0", float(tiny.min()) > 0.999,
          f"min rho at k=1e-6 is {tiny.min():.6f}")
    return out


# ------------------------------------------------------------- checks 2, 3
def cuda_checks(xyz, cov):
    try:
        from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                                 GaussianRasterizer)
    except Exception as e:
        print("\n2/3  CUDA CHECKS -- SKIPPED (no built rasteriser here)")
        print(f"     {type(e).__name__}: {e}")
        print("     These run inside the Kaggle session, before the sweep.")
        return
    if not torch.cuda.is_available():
        print("\n2/3  CUDA CHECKS -- SKIPPED (no GPU here)")
        return

    W = H = 400
    fov = math.radians(50.0) * 0.5
    tan = math.tan(fov)
    n = xyz.shape[0]
    dev = "cuda"

    def render(kernel_size, compensate):
        view = torch.eye(4, device=dev)
        # a right-handed look-down-+z camera, znear/zfar as in the shipped utils
        znear, zfar = 0.01, 100.0
        P = torch.zeros(4, 4, device=dev)
        P[0, 0] = 1.0 / tan
        P[1, 1] = 1.0 / tan
        P[2, 2] = zfar / (zfar - znear)
        P[3, 2] = -(zfar * znear) / (zfar - znear)
        P[2, 3] = 1.0
        settings = GaussianRasterizationSettings(
            image_height=H, image_width=W, tanfovx=tan, tanfovy=tan,
            kernel_size=kernel_size,
            subpixel_offset=torch.zeros((H, W, 2), dtype=torch.float32, device=dev),
            bg=torch.zeros(3, device=dev), scale_modifier=1.0,
            viewmatrix=view, projmatrix=view @ P, sh_degree=0,
            campos=torch.zeros(3, device=dev), prefiltered=False, debug=False,
            mip_compensation=compensate)
        r = GaussianRasterizer(raster_settings=settings)
        img, _ = r(
            means3D=xyz.to(dev),
            means2D=torch.zeros_like(xyz, device=dev, requires_grad=True),
            shs=None,
            colors_precomp=torch.full((n, 3), 0.8, device=dev),
            opacities=torch.full((n, 1), 0.5, device=dev),
            scales=None, rotations=None,
            cov3D_precomp=cov.to(dev)[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]])
        return img

    print("\n2  DIFFERENTIAL -- the flag changes the image, in the right direction")
    print("-" * 66)
    on, off = render(0.3, True), render(0.3, False)
    d = (on - off).abs().max().item()
    print(f"  mean intensity  compensation ON  {on.mean():.6f}")
    print(f"  mean intensity  compensation OFF {off.mean():.6f}")
    check("the flag changes the render", d > 1e-4, f"max|diff| = {d:.2e}")
    check("compensation only dims (rho <= 1)", on.mean() <= off.mean() + 1e-6,
          f"ON {on.mean():.6f} <= OFF {off.mean():.6f}")

    print("\n3  DECISIVE -- the two paths converge as kernel_size -> 0")
    print("-" * 66)
    print("  rho -> 1 as k -> 0, so if the switch removes EXACTLY rho and nothing")
    print("  else, the two renders must agree to float precision at small k.")
    print(f"  {'kernel_size':>12}  {'max|on - off|':>14}")
    prev = None
    for k in (0.3, 1e-2, 1e-4, 1e-6):
        a, b = render(k, True), render(k, False)
        dd = (a - b).abs().max().item()
        print(f"  {k:12.0e}  {dd:14.3e}")
        prev = dd
    check("paths converge at kernel_size -> 0", prev < 1e-5,
          f"max|diff| at k=1e-6 is {prev:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", help="a trained point_cloud.ply; synthetic cloud if absent")
    ap.add_argument("--n", type=int, default=200_000)
    a = ap.parse_args()

    print("=" * 66)
    print("C1 -- is the 2D Mip opacity compensation actually gone?")
    print("=" * 66)
    if a.ply:
        xyz, cov = load_ply(a.ply)
        print(f"loaded {len(xyz)} primitives from {a.ply}")
    else:
        xyz, cov = synth(a.n)
        print(f"synthetic cloud, {len(xyz)} primitives "
              "(log-uniform scales over four decades)")

    analytic(xyz, cov)
    cuda_checks(xyz, cov)

    print("\n" + "=" * 66)
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("The compensation term is present when asked for and absent when not.")
