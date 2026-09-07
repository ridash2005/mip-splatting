#!/usr/bin/env python3
"""
End-to-end self-test of the reporting chain, on a CPU-only machine.

The expensive half of this project (train -> render -> metrics) needs a
CC 7.0+ GPU. The cheap half — per_view.json -> split_by_scale.py ->
results/runs.csv -> make_table.py / make_figures.py — does not, and it is
where a silent error is most dangerous: a wrong index mapping or a
method-slug mismatch produces a plausible-looking table rather than a crash.

This script builds a synthetic scene whose per-view metrics are constructed
so that every downstream number is known in advance, then asserts each stage
reproduces it. It touches nothing under results/ — everything happens in a
temporary directory.

    python tools/selftest_pipeline.py

Exit code 0 = the reporting chain is trustworthy. Non-zero = do not believe
any table this repo prints until it is fixed.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import split_by_scale as sbs  # noqa: E402

SCENES = ["lego", "chair"]
N_PER_SCALE = 5          # test views per scale, per scene
# Per-scale PSNR planted in the fixture, one value per arm. Arm B is given the
# published 3DGS collapse and arm A the Mip-Splatting plateau, so a chain that
# silently swaps or drops an arm shows up as an obviously wrong table.
PLANTED = {
    "B": {0: 33.33, 1: 26.95, 2: 21.38, 3: 17.69},
    "A": {0: 33.36, 1: 34.00, 2: 31.85, 3: 28.67},
}
ITERS = 30000
METHOD_DIR = f"ours_{ITERS}"

failures = []


def ok(desc, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {desc}" + (f" — {detail}" if detail else ""))
    if not passed:
        failures.append(desc)


def build_fixture(root):
    """Write a multi-scale data dir and a metrics.py-shaped per_view.json per arm.

    The interleaving below is the point of the test. convert_blender_data.py
    emits '{:03d}_d{j}.png' grouped by scale, and readMultiScale preserves
    metadata.json order, so index -> scale is NOT index % 4. The fixture uses
    the real grouped-by-scale layout; a splitter that assumed round-robin would
    pass on an interleaved fixture and fail on real data.
    """
    for scene in SCENES:
        data_dir = os.path.join(root, "multi-scale", scene)
        os.makedirs(data_dir, exist_ok=True)
        file_paths = [f"test/{i:03d}_d{j}.png"
                      for j in range(4) for i in range(N_PER_SCALE)]
        with open(os.path.join(data_dir, "metadata.json"), "w") as f:
            json.dump({"test": {"file_path": file_paths},
                       "train": {"file_path": [f"train/{i:03d}_d{j}.png"
                                               for j in range(4) for i in range(20)]}}, f)

        for arm, planted in PLANTED.items():
            model_dir = os.path.join(root, f"out_arm{arm}", scene)
            os.makedirs(model_dir, exist_ok=True)
            psnr, ssim, lp = {}, {}, {}
            for idx, p in enumerate(file_paths):
                j = int(p.rsplit("_d", 1)[1].split(".")[0])
                # A deterministic per-view spread around the planted mean whose
                # sum over the group is exactly zero, so the group mean is exact.
                off = (idx % N_PER_SCALE) - (N_PER_SCALE - 1) / 2
                psnr[f"{idx:05d}.png"] = planted[j] + off * 0.1
                ssim[f"{idx:05d}.png"] = 0.9 + j * 0.01 + off * 0.001
                lp[f"{idx:05d}.png"] = 0.05 + j * 0.01 + off * 0.001
            # The shape metrics.py actually writes. It does
            # `json.dump(full_dict[scene_dir], ...)` — the value, not the
            # enclosing dict — so the top-level key is the METHOD directory,
            # with no scene_dir above it. The fixture is built from that source
            # line rather than from what the splitter expected to find, because
            # the two disagreed and the mismatch does not fail loudly.
            # arm A is written scene-nested instead, to hold the compatibility
            # path in method_block() honest against forks that keep it.
            per_view = {METHOD_DIR: {"PSNR": psnr, "SSIM": ssim, "LPIPS": lp}}
            n = len(file_paths)
            results = {METHOD_DIR: {"PSNR": sum(psnr.values()) / n,
                                    "SSIM": sum(ssim.values()) / n,
                                    "LPIPS": sum(lp.values()) / n}}
            if arm == "A":
                per_view, results = {model_dir: per_view}, {model_dir: results}
            with open(os.path.join(model_dir, "per_view.json"), "w") as f:
                json.dump(per_view, f)
            with open(os.path.join(model_dir, "results.json"), "w") as f:
                json.dump(results, f)
    return file_paths


def main():
    with tempfile.TemporaryDirectory(prefix="btp-selftest-") as root:
        build_fixture(root)

        # -- 1: the index mapping recovers exactly the planted per-scale numbers
        for arm, planted in PLANTED.items():
            for scene in SCENES:
                res = sbs.split_scene(os.path.join(root, f"out_arm{arm}", scene),
                                      os.path.join(root, "multi-scale", scene),
                                      METHOD_DIR)
                got = {sc: round(v["PSNR"], 6) for sc, v in res.items()}
                want = {sbs.SCALE_NAME[j]: round(v, 6) for j, v in planted.items()}
                ok(f"split_by_scale recovers planted per-scale PSNR (arm {arm}, {scene})",
                   got == want, f"got {got}")
                ok(f"every scale group has n={N_PER_SCALE} (arm {arm}, {scene})",
                   all(v["n"] == N_PER_SCALE for v in res.values()),
                   str({k: v["n"] for k, v in res.items()}))

        # -- 2: §6 check 9 — the count-weighted mean reproduces metrics.py's pooled number
        for arm in PLANTED:
            scene = SCENES[0]
            model_dir = os.path.join(root, f"out_arm{arm}", scene)
            res = sbs.split_scene(model_dir, os.path.join(root, "multi-scale", scene),
                                  METHOD_DIR)
            with open(os.path.join(model_dir, "results.json")) as f:
                pooled = sbs.method_block(json.load(f), METHOD_DIR)["PSNR"]
            n_tot = sum(v["n"] for v in res.values())
            weighted = sum(v["n"] * v["PSNR"] for v in res.values()) / n_tot
            ok(f"count-weighted mean == pooled results.json PSNR (arm {arm})",
               abs(weighted - pooled) < 1e-9, f"{weighted:.9f} vs {pooled:.9f}")

        # -- 2b: a method name that is not there must fail loudly, not silently
        #        index into the wrong dictionary
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "split_by_scale.py"),
             "--model-dir", os.path.join(root, "out_armB", SCENES[0]),
             "--data-dir", os.path.join(root, "multi-scale", SCENES[0]),
             "--method", "ours_99999"], capture_output=True, text=True)
        ok("a wrong --method exits non-zero and names the keys it did find",
           r.returncode != 0 and METHOD_DIR in (r.stdout + r.stderr),
           (r.stderr or r.stdout).strip()[-160:])

        # -- 3: CSV round trip through the real CLI
        csv_path = os.path.join(root, "runs.csv")
        for arm, method in (("A", "mip-splatting"), ("B", "3dgs")):
            r = subprocess.run(
                [sys.executable, os.path.join(HERE, "split_by_scale.py"),
                 "--model-root", os.path.join(root, f"out_arm{arm}"),
                 "--data-root", os.path.join(root, "multi-scale"),
                 "--method", METHOD_DIR, "--csv", csv_path,
                 "--tag", f"method={method},arm={arm},train_scale=1x,"
                          f"load_allres=False,seed=0,level_claimed=L3"],
                capture_output=True, text=True)
            ok(f"split_by_scale.py CLI exits 0 (arm {arm})", r.returncode == 0,
               (r.stderr or r.stdout)[-300:])

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        ok("CSV header is exactly the §11 schema",
           list(rows[0]) == sbs.RUNS_CSV_SCHEMA if rows else False)
        ok(f"CSV has {len(SCENES) * 4 * 2} rows (scenes x scales x arms)",
           len(rows) == len(SCENES) * 4 * 2, str(len(rows)))
        ok("iterations column parsed out of the method dir name",
           all(r["iterations"] == str(ITERS) for r in rows),
           sorted({r["iterations"] for r in rows}))
        ok("no row invents a value for a column the splitter cannot know",
           all(r["n_gaussians"] == "" and r["train_seconds"] == "" for r in rows))

        # -- 4: make_table reproduces the planted numbers, and refuses the wrong protocol
        r = subprocess.run([sys.executable, os.path.join(HERE, "make_table.py"),
                            "--table", "1", "--runs", csv_path, "--iterations", str(ITERS)],
                           capture_output=True, text=True)
        ok("make_table.py --table 1 exits 0", r.returncode == 0, (r.stderr or "")[-300:])
        ok("table 1 prints the planted 1/8 collapse (17.69 vs 28.67)",
           "17.69" in r.stdout and "28.67" in r.stdout,
           [l for l in r.stdout.splitlines() if l.startswith("| 1/8")])

        r2 = subprocess.run([sys.executable, os.path.join(HERE, "make_table.py"),
                             "--table", "2", "--runs", csv_path, "--iterations", str(ITERS)],
                            capture_output=True, text=True)
        ok("make_table.py --table 2 refuses (no load_allres=True rows exist)",
           r2.returncode != 0, (r2.stderr or "").strip()[-200:])

        r3 = subprocess.run([sys.executable, os.path.join(HERE, "make_table.py"),
                             "--table", "1", "--runs", csv_path, "--iterations", "7000"],
                            capture_output=True, text=True)
        ok("make_table.py refuses an iteration count with no rows, and says which exist",
           r3.returncode != 0 and str(ITERS) in (r3.stderr or ""),
           (r3.stderr or "").strip()[-200:])

        # -- 5: make_figures 'measured' actually consumes the rows
        figdir = os.path.join(root, "figures")
        r4 = subprocess.run([sys.executable, os.path.join(HERE, "make_figures.py"),
                             "--mode", "measured", "--runs", csv_path,
                             "--out", figdir, "--iterations", str(ITERS)],
                            capture_output=True, text=True)
        ok("make_figures.py --mode measured exits 0", r4.returncode == 0,
           (r4.stderr or "")[-400:])
        series = [l for l in r4.stdout.splitlines() if l.startswith("measured series")]
        ok("measured mode bound BOTH arms (the method-slug mapping works)",
           any("/ 3DGS ->" in l for l in series)
           and any("/ Mip-Splatting ->" in l for l in series), series)
        expected_figs = ["fig1-scale-degradation.svg", "fig2-mipnerf360-zoom.svg",
                         "fig3-compute-budget.svg", "fig4-cost-quality.svg",
                         "fig5-fisher-geometry.svg", "fig6-anisotropy.svg"]
        present = sorted(os.listdir(figdir)) if os.path.isdir(figdir) else []
        ok("all six figures written (5 and 6 were previously unreachable)",
           set(expected_figs) <= set(present), str(present))

        # The strongest available statement that 'measured' is not just the
        # published figure under a different filename: draw both and compare.
        pubdir = os.path.join(root, "figures-published")
        subprocess.run([sys.executable, os.path.join(HERE, "make_figures.py"),
                        "--mode", "published", "--out", pubdir],
                       capture_output=True, text=True)
        fig1 = "fig1-scale-degradation.svg"
        m = open(os.path.join(figdir, fig1), "rb").read()
        p = open(os.path.join(pubdir, fig1), "rb").read()
        ok("measured fig1 differs from published fig1 (the CSV really drives the plot)",
           m != p, f"measured {len(m)} B vs published {len(p)} B")

        r5 = subprocess.run([sys.executable, os.path.join(HERE, "make_figures.py"),
                             "--mode", "measured", "--runs", os.path.join(root, "nope.csv"),
                             "--out", figdir],
                            capture_output=True, text=True)
        ok("make_figures.py refuses 'measured' with no rows (never draws published as measured)",
           r5.returncode != 0, (r5.stderr or "").strip()[-160:])

    check_kernel_helpers()

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + "; ".join(failures))
        return 1
    print("reporting chain verified end to end.")
    return 0


def load_kernel_helpers():
    """Exec kaggle/r0_smoke.py's preamble to get at its pure helper functions.

    The kernel is a top-to-bottom script: importing it would run nvidia-smi and
    `import torch`. Everything above the `check 1: GPU` banner is definitions
    only, so that prefix execs safely anywhere.
    """
    src = open(os.path.join(REPO, "kaggle", "r0_smoke.py"), encoding="utf-8").read()
    marker = "# =============================================================== check 1: GPU"
    if marker not in src:
        return None
    ns = {"__name__": "r0_smoke_preamble"}
    exec(compile(src[:src.index(marker)], "r0_smoke.py<preamble>", "exec"), ns)
    return ns


def check_kernel_helpers():
    """Exercise the kernel's build-time fixes against a copy of this repo.

    These run inside a compute-capability 7.0+ Kaggle session, which has proved
    to be the scarce resource in this project. A helper that raises there costs
    a whole session; here it costs nothing.
    """
    ns = load_kernel_helpers()
    if ns is None:
        ok("kaggle/r0_smoke.py preamble is loadable", False, "banner not found")
        return
    ok("kaggle/r0_smoke.py preamble execs with no GPU present", True)

    with tempfile.TemporaryDirectory(prefix="btp-kernel-") as d:
        repo = os.path.join(d, "repo")
        for rel in ("submodules/simple-knn", "."):
            os.makedirs(os.path.join(repo, rel), exist_ok=True)
        shutil.copy(os.path.join(REPO, "submodules", "simple-knn", "simple_knn.cu"),
                    os.path.join(repo, "submodules", "simple-knn", "simple_knn.cu"))
        shutil.copy(os.path.join(REPO, "train.py"), os.path.join(repo, "train.py"))

        knn = os.path.join(repo, "submodules", "simple-knn", "simple_knn.cu")
        ns["patch_simple_knn_flt_max"](repo)
        src = open(knn).read()
        ok("simple-knn <cfloat> fix applied, and FLT_MAX is really used there",
           "#include <cfloat>" in src and "FLT_MAX" in src)
        before = src
        ns["patch_simple_knn_flt_max"](repo)      # must be idempotent
        ok("simple-knn fix is idempotent (the kernel applies it to both arms)",
           open(knn).read() == before)

        train = os.path.join(repo, "train.py")
        ns["neutralise_unused_open3d_import"](repo)
        after = open(train).read()
        ok("open3d fallback comments out the import and nothing else",
           "# import open3d as o3d" in after
           and "\nimport open3d as o3d\n" not in after
           and len(after.splitlines()) == len(open(os.path.join(REPO, "train.py"),
                                                   encoding="utf-8").read().splitlines()))
        ns["neutralise_unused_open3d_import"](repo)      # must be idempotent
        ok("open3d fallback is idempotent (the kernel applies it to both arms)",
           open(train).read() == after)

        # The guard that matters: if o3d were actually used, removing the import
        # would change behaviour, and the fallback must refuse rather than
        # silently break training inside a scarce GPU session.
        used = os.path.join(d, "used")
        os.makedirs(used, exist_ok=True)
        with open(os.path.join(used, "train.py"), "w", encoding="utf-8") as f:
            f.write("import open3d as o3d\n\npcd = o3d.geometry.PointCloud()\n")
        try:
            ns["neutralise_unused_open3d_import"](used)
            refused = False
        except RuntimeError as e:
            refused = "NOT unused" in str(e)
        ok("open3d fallback REFUSES when o3d is actually used in the file", refused)

    # The scale parser both the kernel and the splitter depend on.
    ok("scale_of maps convert_blender_data's naming to 0..3",
       [sbs.scale_of(f"test/{i:03d}_d{j}.png") for i, j in
        ((0, 0), (7, 1), (42, 2), (199, 3))] == [0, 1, 2, 3])
    ok("scale_of returns None for a name with no _d<j> suffix",
       sbs.scale_of("test/00000.png") is None)


if __name__ == "__main__":
    sys.exit(main())
