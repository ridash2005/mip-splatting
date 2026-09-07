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
    python tools/split_by_scale.py \
        --model-dir  benchmark_nerf_synthetic_ours_stmt/lego \
        --data-dir   multi-scale/lego \
        --method     ours_30000

    # all eight scenes, appended as full results/runs.csv rows
    python tools/split_by_scale.py --model-root benchmark_nerf_synthetic_ours_stmt \
        --data-root multi-scale --method ours_30000 --csv results/runs.csv \
        --tag method=mip-splatting,arm=A,train_scale=1x

CSV OUTPUT
----------
results/runs.csv has a fixed 28-column header (see §11 of the reproduction
prompt). This tool always writes exactly that header's columns, in that order,
filling anything it does not know (run_id, timestamp_utc, seed, n_gaussians,
...) with "" rather than guessing — a cell is either produced by a logged run
or it is empty, never invented. Pass those fields via --tag when you have them
(e.g. seed=0), or patch the row afterwards.
"""
import argparse, json, os, sys, csv
from collections import defaultdict
from datetime import datetime, timezone

SCALE_NAME = {0: "1x", 1: "1/2", 2: "1/4", 3: "1/8"}

# results/runs.csv header, verbatim from the reproduction prompt's §11 schema.
# split_by_scale.py never invents a value for a column it doesn't compute —
# unfilled columns are written as "".
RUNS_CSV_SCHEMA = [
    "run_id", "timestamp_utc", "method", "arm", "impl_commit", "rasteriser_commit",
    "dataset", "scene", "resolution_flag", "load_allres", "kernel_size", "disable_3D_filter",
    "train_scale", "test_scale", "iterations", "seed",
    "psnr", "ssim", "lpips", "lpips_backbone", "n_gaussians", "model_mb", "peak_vram_mb",
    "train_seconds", "render_fps", "gpu_model", "platform", "level_claimed", "notes",
]


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
    ap.add_argument("--csv", help="append rows here (results/runs.csv schema)")
    ap.add_argument("--tag", default="", help="k=v,k=v pairs copied into every row, "
                    "e.g. method=mip-splatting,arm=A,train_scale=1x,seed=0")
    a = ap.parse_args()

    tags = dict(kv.split("=", 1) for kv in a.tag.split(",") if kv) if a.tag else {}
    unknown_tags = set(tags) - set(RUNS_CSV_SCHEMA)
    if unknown_tags:
        sys.exit(f"--tag has keys not in results/runs.csv schema: {sorted(unknown_tags)}")

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

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, agg = [], defaultdict(lambda: defaultdict(list))
    for name, md, dd in scenes:
        print(f"\n{name}")
        res = split_scene(md, dd, a.method)
        for sc, v in res.items():
            print(f"  {sc:>4}  n={v['n']:<4} PSNR {v['PSNR']:7.3f}   "
                  f"SSIM {v['SSIM']:.4f}   LPIPS {v['LPIPS']:.4f}")
            for m in ("PSNR", "SSIM", "LPIPS"):
                agg[sc][m].append(v[m])
            row = {col: "" for col in RUNS_CSV_SCHEMA}
            row.update(tags)
            row.update({
                "timestamp_utc": now,
                "dataset": "blender",
                "scene": name,
                "test_scale": sc,
                "psnr": f"{v['PSNR']:.4f}",
                "ssim": f"{v['SSIM']:.5f}",
                "lpips": f"{v['LPIPS']:.5f}",
                "lpips_backbone": "vgg",
                "iterations": a.method.split("_")[-1],
            })
            rows.append(row)

    print("\nMEAN OVER SCENES")
    for sc in ("1x", "1/2", "1/4", "1/8"):
        if sc in agg:
            n = len(agg[sc]["PSNR"])
            print(f"  {sc:>4}  ({n} scenes)  PSNR {sum(agg[sc]['PSNR'])/n:7.3f}   "
                  f"SSIM {sum(agg[sc]['SSIM'])/n:.4f}   "
                  f"LPIPS {sum(agg[sc]['LPIPS'])/n:.4f}")

    if a.csv and rows:
        new = not os.path.exists(a.csv)
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RUNS_CSV_SCHEMA)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
