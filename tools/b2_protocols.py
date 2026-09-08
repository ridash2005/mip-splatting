#!/usr/bin/env python3
"""
B2 across the capture protocols, and across tau.

On the standard Blender captures every primitive is seen from ~100 well-spread
directions, so Equation 5.2 keeps degree 3 everywhere and B2 masks nothing. That
is the correct answer, and it is the same shape as I-1's: on a well-conditioned
capture the proposed criterion reduces to the baseline. It also means the full
orbit cannot show what B2 does, in exactly the way it could not show what B1
does.

So the criterion is evaluated over the same five capture protocols the stress
suite uses. This needs no GPU and no retraining -- Equation 5.1 depends on the
primitive positions and the training view directions, both of which are already
on disk.

    python tools/b2_protocols.py --plys <dir> --cameras <dir> --out results/b2_protocols

Reports, per protocol: mean l_max, the fraction of non-DC coefficients retained,
and the mean number of views per primitive that produced it.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import camera_protocols as cp        # noqa: E402
import instruments as inst           # noqa: E402
import sh_identifiability as shid    # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def evaluate(xyz, c2w, tau, criterion="cond"):
    blocks, n_seen = shid.angular_gram(xyz, c2w)
    lm = shid.l_max(blocks, n_seen, tau, criterion)
    kept = sum((shid.DEG_SLICES[l].stop - shid.DEG_SLICES[l].start)
               * int((lm >= l).sum()) for l in (1, 2, 3))
    return {
        "mean_l_max": float(lm.mean()),
        "non_dc_retained": float(kept / (15 * len(lm))),
        "l_max_hist": {int(l): int((lm == l).sum()) for l in range(4)},
        "mean_views": float(n_seen.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plys", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out", default="results/b2_protocols")
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--taus", default="0.001,0.003,0.01,0.03,0.1",
                    help="tau sweep, run on the first scene only")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--criterion", choices=("cond", "abs"), default="cond")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    scenes = ([s for s in a.scenes.split(",") if s] or
              sorted(f[:-4] for f in os.listdir(a.plys) if f.endswith(".ply")))
    results, sweep = {}, {}

    for scene in scenes:
        ply = os.path.join(a.plys, f"{scene}.ply")
        cam = os.path.join(a.cameras, f"{scene}.json")
        if not (os.path.exists(ply) and os.path.exists(cam)):
            print(f"skip {scene}: missing inputs")
            continue
        xyz = inst.load_ply(ply)["xyz"]
        c2w_all, focal_all = inst.load_cameras(cam)
        spec = cp.build(c2w_all, focal_all, seed=0)
        print(f"\n{scene}: {len(xyz)} primitives, {len(c2w_all)} cameras")
        for proto in cp.PROTOCOLS:
            idx = np.array(spec[proto]["idx"], dtype=int)
            r = evaluate(xyz, c2w_all[idx], a.tau, a.criterion)
            r.update({"protocol": proto, "criterion": a.criterion, "n_cameras": len(idx),
                      "span_deg": spec[proto]["span_deg"], "scene": scene})
            results[f"{scene}/{proto}"] = r
            print(f"  {proto:8} n_cam={len(idx):4d} span={r['span_deg']:5.0f}deg "
                  f"views/prim={r['mean_views']:6.1f}  "
                  f"mean l_max={r['mean_l_max']:.2f}  "
                  f"non-DC retained={r['non_dc_retained']:6.1%}")
            with open(os.path.join(a.out, "protocols.json"), "w") as f:
                json.dump(results, f, indent=2)

        if scene == scenes[0]:
            print(f"  tau sweep on {scene} (cone protocol):")
            idx = np.array(spec["cone"]["idx"], dtype=int)
            for t in [float(x) for x in a.taus.split(",")]:
                r = evaluate(xyz, c2w_all[idx], t, a.criterion)
                sweep[str(t)] = r
                print(f"    tau={t:<7} mean l_max={r['mean_l_max']:.2f}  "
                      f"non-DC retained={r['non_dc_retained']:6.1%}")
            with open(os.path.join(a.out, "tau_sweep.json"), "w") as f:
                json.dump(sweep, f, indent=2)

    print(f"\nwrote {a.out}/protocols.json and tau_sweep.json")


if __name__ == "__main__":
    main()
