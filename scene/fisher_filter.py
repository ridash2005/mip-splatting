#
# B1 — the sampling-geometry band-limit.
#
# Implements Equations 4.3-4.5 of the thesis: accumulate the sampling Fisher
# information per primitive, invert it to a per-direction position uncertainty,
# floor that at Mip-Splatting's own scalar value and cap it at a scene-scale
# fraction, and add the resulting MATRIX to the covariance instead of a scalar.
#
# No CUDA change is needed. The rasteriser already accepts a precomputed 3x3
# covariance per primitive (cov3D_precomp, reached through
# pipe.compute_cov3D_python), so the whole method lives in PyTorch.
#
# Calibration. Proposition 1 fixes s * sigma_pix = sqrt(0.2). Writing
# M_k = sum_n w_nk (J_n W_n)^T (J_n W_n) WITHOUT the pixel-noise factor,
#
#     Lambda_k = M_k / sigma_pix^2   =>   s^2 / lambda_i = s^2 sigma_pix^2 / mu_i
#                                                        = 0.2 / mu_i
#
# where mu_i are the eigenvalues of M_k. So the estimation term is 0.2/mu_i and
# needs no separate sigma_pix. A single view at depth z with focal f gives
# mu = (f/z)^2, hence sqrt(0.2)*z/f -- exactly Mip-Splatting's f_k with
# z = d_min and f = f_max. The identity of Proposition 1 is therefore built in
# rather than asserted.
#
import torch

from utils.general_utils import strip_symmetric

# Mip-Splatting's constant, and the calibration of Proposition 1.
FILTER_CONST = 0.2


@torch.no_grad()
def accumulate_fisher(xyz, cameras, chunk=200000):
    """M_k = sum_n w_nk (J_n W_n)^T (J_n W_n), and each primitive's nearest depth.

    w_nk is binary visibility -- in front of the camera and inside a 1.15x screen
    margin, matching compute_3D_filter's own test (F6) so the two filters see the
    same view set and Proposition 2 is exact rather than approximate.

    Returns (M, d_min, n_seen, f_max) with M of shape [N,3,3].
    """
    n = xyz.shape[0]
    M = torch.zeros((n, 3, 3), dtype=torch.float32, device=xyz.device)
    d_min = torch.full((n,), float("inf"), dtype=torch.float32, device=xyz.device)
    n_seen = torch.zeros(n, dtype=torch.int32, device=xyz.device)
    f_max = 0.0

    for cam in cameras:
        W = cam.world_view_transform[:3, :3].transpose(0, 1).to(xyz.device)  # world->cam
        t = cam.world_view_transform[3, :3].to(xyz.device)
        focal_x = cam.image_width / (2.0 * torch.tan(torch.tensor(cam.FoVx * 0.5)))
        focal_y = cam.image_height / (2.0 * torch.tan(torch.tensor(cam.FoVy * 0.5)))
        f_max = max(f_max, float(focal_x))

        for s in range(0, n, chunk):
            p = xyz[s:s + chunk]
            cam_p = p @ W.transpose(0, 1) + t
            z = cam_p[:, 2]
            valid = z > 0.2
            if not bool(valid.any()):
                continue
            # The same 1.15x margin compute_3D_filter uses, so both filters are
            # built from the same views.
            zc = torch.clamp(z, min=0.001)
            # compute_3D_filter's own test, in pixels: x = x/z*f + W/2 must lie in
            # [-0.15W, 1.15W], i.e. |x/z*f| <= 0.65W. Written the same way here so
            # the two filters see exactly the same view set.
            px = cam_p[:, 0] / zc * focal_x + cam.image_width / 2.0
            py = cam_p[:, 1] / zc * focal_y + cam.image_height / 2.0
            valid = (valid
                     & (px >= -0.15 * cam.image_width)
                     & (px <= 1.15 * cam.image_width)
                     & (py >= -0.15 * cam.image_height)
                     & (py <= 1.15 * cam.image_height))
            if not bool(valid.any()):
                continue

            # J = d(u,v)/d(x,y,z) for a pinhole, in camera coordinates.
            # Its null direction is the viewing ray, so J has rank 2 exactly
            # (Appendix B.1) -- computed here rather than approximated by a
            # projector, so off-axis primitives are handled correctly.
            iz = 1.0 / zc
            J = torch.zeros((p.shape[0], 2, 3), dtype=torch.float32, device=xyz.device)
            J[:, 0, 0] = focal_x * iz
            J[:, 0, 2] = -focal_x * cam_p[:, 0] * iz * iz
            J[:, 1, 1] = focal_y * iz
            J[:, 1, 2] = -focal_y * cam_p[:, 1] * iz * iz
            A = J @ W                                    # [m,2,3], world frame
            contrib = A.transpose(1, 2) @ A              # [m,3,3]
            M[s:s + chunk] += contrib * valid[:, None, None].float()

            d_min[s:s + chunk] = torch.where(valid, torch.minimum(d_min[s:s + chunk], z),
                                             d_min[s:s + chunk])
            n_seen[s:s + chunk] += valid.int()
    return M, d_min, n_seen, f_max



def _batched_eigh(A):
    """Symmetric eigendecomposition of a batch of 3x3 matrices.

    torch.linalg.eigh on CUDA fails outright for batched float64 on this stack:

        cusolver error: CUSOLVER_STATUS_INVALID_VALUE, when calling
        cusolverDnXsyevBatched_bufferSize(... CUDA_R_64F ...)

    so float32 is used on GPU and float64 on CPU (where the unit tests run and
    accuracy is free). The precision loss does not touch Proposition 2: when the
    three eigenvalues are equal the filter is V diag(c,c,c) V^T = c*V V^T = c*I
    for ANY orthogonal V, so the reduction is exact regardless of how accurately
    the eigenvectors are resolved.

    NaNs are scrubbed first. A single non-finite entry makes cuSOLVER fail with
    the message above rather than with anything naming the real cause, so it is
    worth ruling out explicitly.
    """
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = 0.5 * (A + A.transpose(-1, -2))          # enforce exact symmetry
    if A.is_cuda:
        try:
            mu, V = torch.linalg.eigh(A.float())
            return mu.double(), V.double()
        except Exception:
            mu, V = torch.linalg.eigh(A.double().cpu())
            return mu.to(A.device), V.to(A.device)
    mu, V = torch.linalg.eigh(A.double())
    return mu, V


@torch.no_grad()
def build_filter(xyz, cameras, s_conf=1.0, beta=0.01, floor=None):
    """Sigma_filt per primitive, plus the diagnostics §6.5 asks for.

    Returns a dict with:
      sigma_filt   [N,3,3]  the matrix added to the covariance
      f_k          [N]      Mip-Splatting's scalar floor, for comparison
      frac_above   scalar   fraction of primitives where the estimation term
                            beats the floor in at least one direction -- the I-1
                            diagnostic, logged from inside training
      mean_aniso   scalar   mean sigma_max/sigma_min of Sigma_filt
    """
    M, d_min, n_seen, f_max = accumulate_fisher(xyz, cameras)
    unseen = n_seen == 0

    # compute_3D_filter gives primitives no camera sees distance[valid].max().
    # Matched here so the floor is identical for both methods, and flagged.
    if bool(unseen.any()):
        seen_max = d_min[~unseen].max() if bool((~unseen).any()) else torch.tensor(1.0)
        d_min = torch.where(unseen, seen_max, d_min)

    f_k = (FILTER_CONST ** 0.5) * d_min / max(f_max, 1e-9)          # [N]
    if floor is not None:
        # Mip-Splatting's own floor, passed in verbatim (see §4.4).
        f_k = floor.to(f_k.device).reshape(-1)
    fk2 = (f_k ** 2)[:, None]                                        # [N,1]

    # eigh on a symmetric [N,3,3], eigenvalues ascending. Chunked: at two million
    # primitives the float64 copy alone is ~150 MB, and this runs beside training.
    cap2 = ((beta * d_min) ** 2)[:, None].double()
    fk2d = fk2.double()
    s2 = torch.empty((n_pts := xyz.shape[0], 3), dtype=torch.float64, device=xyz.device)
    sigma_filt = torch.empty((n_pts, 3, 3), dtype=torch.float32, device=xyz.device)
    est2_all = torch.empty((n_pts, 3), dtype=torch.float64, device=xyz.device)

    for a in range(0, n_pts, 65536):
        b = min(a + 65536, n_pts)
        mu, V = _batched_eigh(M[a:b])
        mu = mu.clamp_min(0.0)
        # Equation 4.4. The estimation term is s^2/lambda_i = s^2 * 0.2 / mu_i by
        # the calibration above; mu_i = 0 is an unconstrained direction and gives
        # +inf, which the cap then bounds. That is exactly what beta is for:
        # without it an unconstrained direction produces an unbounded primitive.
        est2 = torch.where(mu > 0, (s_conf ** 2) * FILTER_CONST / mu.clamp_min(1e-30),
                           torch.full_like(mu, float("inf")))
        est2_all[a:b] = est2
        # The floor is Mip-Splatting's own f_k^2, unscaled -- s scales the
        # estimate, never the floor, so Proposition 2 holds for any s.
        blk = torch.minimum(torch.maximum(fk2d[a:b], est2), cap2[a:b])
        blk = torch.where(unseen[a:b, None], fk2d[a:b].expand_as(blk), blk)
        s2[a:b] = blk
        sigma_filt[a:b] = (V @ torch.diag_embed(blk) @ V.transpose(1, 2)).float()

    above = (est2_all > fk2d)
    frac_above = float((above.any(dim=1) & ~unseen).float().mean())
    smax = s2.max(dim=1).values.sqrt()
    smin = s2.min(dim=1).values.sqrt().clamp_min(1e-30)
    # Proposition 2's hypothesis, evaluated: is the filter isotropic on EVERY
    # primitive? Where it is, B1 and Mip-Splatting are the same filter and can
    # take the same code path.
    spread = (s2.max(dim=1).values - s2.min(dim=1).values).abs()
    isotropic = bool((spread <= 1e-12 * s2.max(dim=1).values.clamp_min(1e-30)).all())

    return {
        "sigma_filt": sigma_filt,
        "sigma_iso": s2.max(dim=1).values.sqrt().float(),   # the common sigma
        "isotropic": isotropic,
        "f_k": f_k,
        "frac_above_floor": frac_above,
        "mean_anisotropy": float((smax / smin).mean()),
        "frac_unseen": float(unseen.float().mean()),
        "frac_capped": float((s2 >= cap2 - 1e-30).any(dim=1).float().mean()),
    }


def covariance_with_filter(scaling, rotation_matrix, sigma_filt, scaling_modifier=1.0):
    """Sigma_k + Sigma_filt, as the 6-vector upper triangle the rasteriser wants."""
    S = torch.diag_embed(scaling * scaling_modifier)
    L = rotation_matrix @ S
    cov = L @ L.transpose(1, 2)
    return strip_symmetric(cov + sigma_filt), cov


def opacity_compensation(cov, sigma_filt, scaling=None, eps=1e-300):
    """alpha <- alpha * sqrt(|Sigma| / |Sigma + Sigma_filt|).

    Mip-Splatting's rule, written for a matrix filter instead of a scalar one.
    With Sigma_filt = f^2 I it reduces to their expression exactly, which is the
    opacity half of Proposition 2.

    eps is a guard against dividing by zero and NOTHING ELSE. The first version
    used 1e-12, which is a physically meaningful magnitude here: det(Sigma) is
    prod(s^2), and a primitive with s = 0.005 has det = 1.6e-14. Both
    determinants clamped to the same floor, their ratio became exactly 1, and
    the compensation silently switched itself off for most of the model --
    B1 rendered with uncompensated opacity while every baseline did not.

    det(Sigma) is taken from the scale vector when it is available, because
    prod(s^2) is exact whereas the determinant of a float32 3x3 loses accuracy
    badly for the anisotropic primitives this method exists to handle.
    """
    covd = cov.double()
    det1 = ((scaling.double() ** 2).prod(dim=1) if scaling is not None
            else torch.linalg.det(covd))
    det2 = torch.linalg.det(covd + sigma_filt.double())
    return torch.sqrt(det1.clamp_min(eps) / det2.clamp_min(eps)).float()[..., None]
