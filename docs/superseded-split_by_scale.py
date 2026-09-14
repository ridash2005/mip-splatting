#!/usr/bin/env python3
"""
Split Mip-Splatting multi-scale metrics into the four per-scale columns.

WHY THIS EXISTS
---------------
The shipped pipeline cannot produce Mip-Splatting's Table 1/Table 2 by itself.

  render.py   writes  test/ours_{iter}/test_preds_{r}/{idx:05d}.png
                 and  test/ours_{iter}/gt_{r}/{idx:05d}.png
              where {r} is dataset.resolution (the -r flag), NOT the per-image scale.
              With the default -r -1 all four scales land in ONE directory.

  metrics.py  averages every image in that directory into a single PSNR/SSIM/LPIPS
              and writes results.json (one number) and per_view.json (per image).

So `metrics.py` alone gives you one number where the paper reports four.

HOW THE MAPPING IS RECOVERED
----------------------------
scene/dataset_readers.py::readMultiScale iterates metadata.json[split]["file_path"]
in order and appends one CameraInfo per entry with uid=idx. render.py's render_set
then does `for idx, view in enumerate(views)` and saves '{0:05d}'.format(idx).
Camera order is preserved from CameraInfo to the render loop, so

    render file {idx:05d}.png   <->   metadata.json["test"]["file_path"][idx]

and convert_blender_data.py names those files "{:03d}_d{j}.png", where j is the
downsampling level: d0 = full, d1 = 1/2, d2 = 1/4, d3 = 1/8.

Grouping per_view.json by j therefore gives exactly the paper's four columns,
with no re-rendering and no resampling.

USAGE
-----
    python split_by_scale.py \
        --model-dir  benchmark_nerf_synthetic_ours_stmt/lego \
        --data-dir   multi-scale/lego \
        --method     ours_30000

    # all eight scenes, emit rows ready for results/runs.csv
    python split_by_scale.py --model-root benchmark_nerf_synthetic_ours_stmt \
        --data-root multi-scale --method ours_30000 --csv rows.csv \
        --tag method=mip-splatting,arm=A,train_scale=1x
"""
import argparse, json, os, sys, csv
from collections import defaultdict

SCALE_NAME = {0: "1x", 1: "1/2", 2: "1/4", 3: "1/8"}


def scale_of(path):
    """'.../003_d2.png' -> 2. Returns None if the name carries no _d<j> suffix."""
    stem = os.path.basename(path)
    stem = stem[:-4] if stem.lower().endswith(".png") else stem
    if "_d" not in stem:
        return None
    tail = stem.rsplit("_d", 1)[1]
    return int(tail) if tail.isdigit() else None


def load_order(data_dir, split="test"):
    """The camera order readMultiScale produced, as a list of scale indices."""
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        sys.exit(f"no metadata.json in {data_dir} — is this a multi-scale directory?")
    with open(meta_path) as f:
        meta = json.load(f)[split]
    paths = meta["file_path"]
    scales = [scale_of(p) for p in paths]
    if any(s is None for s in scales):
        sys.exit("some file_path entries carry no _d<j> suffix; cannot infer scale")
    return scales


def split_scene(model_dir, data_dir, method):
    pv_path = os.path.join(model_dir, "per_view.json")
    if not os.path.exists(pv_path):
        # metrics.py swallows every exception in a bare `except:` and merely prints
        # "Unable to compute metrics for model". A missing file is a FAILED run.
        sys.exit(f"{pv_path} missing — metrics.py failed silently for {model_dir}")
    with open(pv_path) as f:
        per_view = json.load(f)

    key = next(iter(per_view))                      # metrics.py nests under scene_dir
    if method not in per_view[key]:
        sys.exit(f"method {method!r} not in per_view.json; have {list(per_view[key])}")
    block = per_view[key][method]

    scales = load_order(data_dir)
    acc = defaultdict(lambda: defaultdict(list))
    missing = 0
    for idx, j in enumerate(scales):
        name = f"{idx:05d}.png"
        if name not in block["PSNR"]:
            missing += 1
            continue
        for m in ("PSNR", "SSIM", "LPIPS"):
            acc[j][m].append(block[m][name])
    if missing:
        print(f"  warning: {missing} of {len(scales)} test views absent from per_view.json",
              file=sys.stderr)

    out = {}
    for j in sorted(acc):
        n = len(acc[j]["PSNR"])
        out[SCALE_NAME.get(j, f"d{j}")] = {
            "n": n,
            **{m: sum(acc[j][m]) / n for m in ("PSNR", "SSIM", "LPIPS")},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir");  ap.add_argument("--data-dir")
    ap.add_argument("--model-root"); ap.add_argument("--data-root")
    ap.add_argument("--method", default="ours_30000")
    ap.add_argument("--csv", help="append rows here, ready for results/runs.csv")
    ap.add_argument("--tag", default="", help="k=v,k=v pairs copied into every row")
    a = ap.parse_args()

    tags = dict(kv.split("=", 1) for kv in a.tag.split(",") if kv) if a.tag else {}

    if a.model_dir:
        scenes = [(os.path.basename(a.model_dir.rstrip("/")), a.model_dir, a.data_dir)]
    elif a.model_root:
        scenes = []
        for s in sorted(os.listdir(a.model_root)):
            md = os.path.join(a.model_root, s)
            if os.path.isdir(md):
                scenes.append((s, md, os.path.join(a.data_root, s)))
    else:
        sys.exit("give --model-dir/--data-dir or --model-root/--data-root")

    rows, agg = [], defaultdict(lambda: defaultdict(list))
    for name, md, dd in scenes:
        print(f"\n{name}")
        res = split_scene(md, dd, a.method)
        for sc, v in res.items():
            print(f"  {sc:>4}  n={v['n']:<4} PSNR {v['PSNR']:7.3f}   "
                  f"SSIM {v['SSIM']:.4f}   LPIPS {v['LPIPS']:.4f}")
            for m in ("PSNR", "SSIM", "LPIPS"):
                agg[sc][m].append(v[m])
            rows.append({**tags, "scene": name, "dataset": "blender",
                         "test_scale": sc, "psnr": f"{v['PSNR']:.4f}",
                         "ssim": f"{v['SSIM']:.5f}", "lpips": f"{v['LPIPS']:.5f}",
                         "lpips_backbone": "vgg", "iterations":
                         a.method.split("_")[-1]})

    print("\nMEAN OVER SCENES")
    for sc in ("1x", "1/2", "1/4", "1/8"):
        if sc in agg:
            n = len(agg[sc]["PSNR"])
            print(f"  {sc:>4}  ({n} scenes)  PSNR {sum(agg[sc]['PSNR'])/n:7.3f}   "
                  f"SSIM {sum(agg[sc]['SSIM'])/n:.4f}   "
                  f"LPIPS {sum(agg[sc]['LPIPS'])/n:.4f}")

    if a.csv and rows:
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
