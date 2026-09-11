#!/usr/bin/env python3
r"""
The camera-subset branch of scene/__init__.py, checked without data or a GPU.

This exists because of a silent failure, and silent failures are the ones worth a
test. `camera_protocols.build` writes `mixed` as every camera with a seeded
random half downsampled 2--4x, and records the resulting focal lengths beside the
index list. The runner passed only `idx`, and the loader dropped everything else
from a dict spec -- so the protocol trained all 100 cameras at their original
intrinsics, which is the full orbit. Nothing raised. The row would have been
reported as a measurement of the mixed-camera defect while being a duplicate of
the control, and a near-zero delta does not distinguish "no effect" from "no
experiment".

What is asserted here:

  * a plain index list still restricts the training set and nothing else;
  * a dict spec carrying `focal` downsamples the cameras whose focal changed,
    and leaves the rest byte-identical;
  * the FIELD OF VIEW does not move. Downsampling by k scales the focal and the
    image dimensions by 1/k together, so what changes is the pixel footprint.
    Overriding the field of view instead -- the obvious first implementation --
    points the camera somewhere else and renders a different scene;
  * `mixed` really does change about half the cameras, so the protocol is not
    quietly equal to `full`.

    python tools/test_camera_subset.py
"""
import math
import os
import sys
from collections import namedtuple

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import camera_protocols as cp                      # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILURES = []
W = H = 800
F0 = 1111.0

CameraInfo = namedtuple(
    "CameraInfo", "uid R T FovY FovX image image_path image_name width height")


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def cameras(n):
    return [CameraInfo(i, None, None, focal2fov(F0, H), focal2fov(F0, W),
                       Image.new("RGB", (W, H)), "", f"c{i}", W, H)
            for i in range(n)]


def apply_subset(cams, spec):
    """scene/__init__.py's branch, in the same order, on plain CameraInfos."""
    focal = None
    if isinstance(spec, dict):
        idx, focal = spec["idx"], spec.get("focal")
    else:
        idx = spec
    picked = [cams[i] for i in idx]
    changed = 0
    if focal:
        assert len(focal) == len(idx)
        for j, cam in enumerate(picked):
            k = fov2focal(cam.FovX, cam.width) / float(focal[j])
            if abs(k - 1.0) < 1e-6:
                continue
            w, h = max(1, round(cam.width / k)), max(1, round(cam.height / k))
            picked[j] = cam._replace(
                image=cam.image.resize((w, h), Image.LANCZOS), width=w, height=h)
            changed += 1
    return picked, changed


def main():
    print("=" * 62)
    print("camera_subset: index lists, focal overrides, and the field of view")
    print("=" * 62)

    print("\na plain index list restricts the training set and nothing else")
    cams = cameras(10)
    picked, changed = apply_subset(cams, [0, 3, 7])
    check("three of ten cameras selected", len(picked) == 3,
          f"{[c.image_name for c in picked]}")
    check("no camera was modified", changed == 0 and
          all(c.width == W and c.height == H for c in picked))

    print("\na dict spec with `focal` downsamples, and only where the focal moved")
    cams = cameras(6)
    req = [F0, F0 / 2, F0, F0 / 4, F0 / 3, F0]
    picked, changed = apply_subset(cams, {"idx": list(range(6)), "focal": req})
    check("exactly the three reduced cameras changed", changed == 3,
          f"changed={changed}")
    check("the unchanged cameras keep their original size",
          all(picked[i].width == W for i in (0, 2, 5)))
    sizes = [c.width for c in picked]
    check("sizes follow the requested factors", sizes == [800, 400, 800, 200, 267, 800],
          f"{sizes}")

    print("\nthe field of view does not move -- only the pixel footprint")
    worst = 0.0
    for c in picked:
        worst = max(worst, abs(math.degrees(c.FovX) - math.degrees(focal2fov(F0, W))))
    check("FovX identical on every camera", worst < 1e-9,
          f"max drift {worst:.1e} deg")
    for c, f in zip(picked, req):
        eff = fov2focal(c.FovX, c.width)
        if abs(eff - f) / f > 0.005:
            check(f"effective focal on {c.image_name} matches the request", False,
                  f"{eff:.1f} against {f:.1f}")
            break
    else:
        check("effective focal within 0.5 % of the request on every camera", True,
              "the residual is integer pixel dimensions: 800/3 is 266.67")
    check("the image really was resampled",
          all(c.image.size == (c.width, c.height) for c in picked))

    print("\n`mixed` is not the full orbit under another name")
    rng = np.random.default_rng(0)
    n = 100
    az = rng.random(n) * 2 * math.pi
    c2w = np.zeros((n, 4, 4))
    for i in range(n):
        c2w[i] = np.eye(4)
        c2w[i][:3, 3] = [4 * math.sin(az[i]), 0.3 * rng.standard_normal(),
                         4 * math.cos(az[i])]
    spec = cp.build(c2w, [F0] * n, seed=0)
    mixed = spec["mixed"]
    check("the builder records a focal list for `mixed`", bool(mixed.get("focal")))
    check("`full` carries no focal override", not spec["full"].get("focal"))
    picked, changed = apply_subset(cameras(n), mixed)
    frac = changed / n
    check("about half the cameras are downsampled", 0.3 < frac < 0.7,
          f"{changed} of {n}")
    check("every camera is kept -- `mixed` degrades intrinsics, not coverage",
          len(picked) == n)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("A protocol that changes intrinsics is instantiated, not just named.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
