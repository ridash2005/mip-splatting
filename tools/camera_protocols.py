#!/usr/bin/env python3
"""
The stress suite: camera subsets of a capture already downloaded.

Section 6.4 predicts that the method's advantage is a function of parallax, so
testing it needs captures of varying conditioning. Building those as SUBSETS of
an existing camera set costs no new data and no GPU time, and keeps the scene
content fixed so the only thing that varies is the geometry.

Protocols, each a seeded, reproducible index list:

  full        every camera. The control. Proposition 2 predicts parity here.
  arc         half the orbit -- a contiguous azimuth wedge.
  cone        a narrow angular cone sized by camera COUNT (a tenth of them).
  pencil      a narrow angular cone sized by WIDTH (20 degrees). The two are
              both kept because they land in different regimes: on the Blender
              captures `cone` comes out at 36-54 degrees, where the Nyquist
              floor still binds on every primitive and B1 reduces to
              Mip-Splatting exactly, while `pencil` reaches ~15 degrees on four
              cameras, which is the regime Equation 4.10 is about. Reporting
              both is what turns "where does the estimation term overtake the
              floor" from an assertion into a measurement.
  mixed       every camera, a random half at reduced focal length. This is the
              only protocol that changes intrinsics rather than pose, and it
              targets the mixed-camera defect of §3.3: f_k takes d_min and
              f_max from DIFFERENT cameras, so the quotient corresponds to no
              physical pixel footprint.
  grazing     high-incidence views only -- the lowest-elevation cameras.

    python tools/camera_protocols.py --cameras transforms_train.json --out protocols/

The angular span of each subset is measured and written alongside it, so the
regime table is checked against the geometry that actually resulted rather than
against a nominal angle asserted in advance.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROTOCOLS = ("full", "arc", "cone", "pencil", "mixed", "grazing")


def camera_centres(c2w):
    return np.asarray(c2w)[:, :3, 3]


def spherical(centres, origin=None):
    """Azimuth and elevation of each camera about the capture centroid."""
    o = centres.mean(axis=0) if origin is None else origin
    d = centres - o
    r = np.linalg.norm(d, axis=1)
    az = np.arctan2(d[:, 0], d[:, 2])
    el = np.arcsin(np.clip(d[:, 1] / np.maximum(r, 1e-12), -1, 1))
    return az, el, r


def angular_span(centres, idx, origin=None):
    """Max pairwise angle subtended at the capture centre, in degrees.

    This is the geometric quantity Equation 4.8 is written in terms of, so
    reporting it lets the predicted 1/sin(theta/2) be checked against the subset
    that was actually built.
    """
    o = centres.mean(axis=0) if origin is None else origin
    d = centres[idx] - o
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    g = np.clip(d @ d.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(g.min())))


def build(c2w, focal, seed=0):
    """Index lists (and focal overrides) for each protocol."""
    rng = np.random.default_rng(seed)
    centres = camera_centres(c2w)
    az, el, _ = spherical(centres)
    n = len(centres)
    out = {}

    out["full"] = {"idx": list(range(n))}

    # arc and cone are defined by ANGULAR EXTENT, not by a camera count. Taking
    # "half the cameras by azimuth" looks like a one-sided arc and is not: on a
    # full orbit it still spans ~180 degrees, because the half nearest -180 and
    # the half nearest +180 are the same set. Selecting a contiguous wedge of a
    # stated width is the thing §6.4 actually describes, and it degrades the
    # conditioning as intended.
    d = centres - centres.mean(axis=0)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    anchor_i = int(rng.integers(n))

    def nearest(k):
        """The k cameras closest in DIRECTION to the anchor.

        Contiguous by construction, and sized by count rather than by a fixed
        angular width. Width alone is not usable for a trainable subset: on the
        Blender captures a 20-degree wedge catches three cameras, and three views
        is a sparse-reconstruction experiment rather than a conditioning one. The
        angular span that results is measured and reported, so the conditioning
        is a measurement either way.
        """
        cos = np.clip(d @ d[anchor_i], -1.0, 1.0)
        return sorted(int(i) for i in np.argsort(-cos)[:max(2, min(k, n))])

    def wedge(width_deg, min_n=3):
        """Every camera within `width_deg` of the anchor, as a real 3D angle.

        Sized by ANGULAR WIDTH, which is the variable Equation 4.10 is written
        in. `nearest` was introduced to avoid a three-camera training set, and
        that objection is sound as far as it goes -- but the count-based cone
        lands at 36-54 degrees on the Blender captures, and there the Nyquist
        floor still binds on every primitive, so B1 reduces to Mip-Splatting
        exactly and the protocol cannot test the claim it was built for. Both
        are kept: `cone` for a trainable subset, this for the regime.
        """
        cos = np.clip(d @ d[anchor_i], -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        idx = np.where(ang <= width_deg / 2.0)[0]
        if len(idx) < min_n:                       # widen until it is trainable
            idx = np.argsort(ang)[:min_n]
        return sorted(int(i) for i in idx)

    out["arc"] = {"idx": nearest(max(2, n // 4))}
    out["cone"] = {"idx": nearest(max(2, n // 10))}
    # The narrow regime. Named separately so the count-based cone keeps its
    # meaning and the two can be reported side by side: the crossover between
    # them is where the estimation term overtakes the floor, and that crossover
    # is a measurement this suite can make rather than assert.
    out["pencil"] = {"idx": wedge(20.0)}

    # Mixed focal: same poses, a random half downsampled 2-4x.
    half = rng.permutation(n)[: n // 2]
    factors = rng.choice([2.0, 3.0, 4.0], size=len(half))
    f = np.array(focal, dtype=float).copy()
    f[half] = f[half] / factors
    out["mixed"] = {"idx": list(range(n)), "focal": [float(x) for x in f]}

    # Grazing: the lowest-elevation half, i.e. the most oblique views.
    out["grazing"] = {"idx": sorted(int(i) for i in np.argsort(el)[: max(2, n // 2)])}

    for name, spec in out.items():
        spec["n"] = len(spec["idx"])
        spec["span_deg"] = angular_span(centres, spec["idx"])
        spec["seed"] = seed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out", default="protocols")
    ap.add_argument("--scene", default="scene")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from instruments import load_cameras
    c2w, focal = load_cameras(a.cameras)
    spec = build(c2w, focal, a.seed)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.scene}.json")
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"{a.scene}: {len(c2w)} cameras")
    for name in PROTOCOLS:
        s = spec[name]
        print(f"  {name:8} n={s['n']:4d}  angular span {s['span_deg']:6.1f} deg"
              f"  -> 1/sin(span/2) = "
              f"{1.0/max(math.sin(math.radians(s['span_deg'])/2), 1e-9):6.2f}x")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
