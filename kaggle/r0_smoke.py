#!/usr/bin/env python3
"""
R0 — smoke run (§7 R0 of docs/REPRODUCTION-PROMPT.md).

`lego`, 7000 iterations, both arms, end to end, plus the locally-verifiable
subset of §6's twelve pre-flight checks. Checks 5/6 (the numeric unit tests
against 33.3/33.4 dB) are calibrated to a 30000-iteration run (F11) — at
this rung's 7000 iterations they are recorded, honestly, as informational
only, never fudged to look like a pass (§14). Check 12 (off-session
persistence via Hugging Face Hub) needs an HF token this run doesn't have;
it is marked skipped, not faked.

Runs entirely inside a Kaggle kernel with a GPU attached and the
`nguyenhung1903/nerf-synthetic-dataset` dataset mounted at
/kaggle/input/nerf-synthetic-dataset/nerf_synthetic. Everything under
/kaggle/working becomes the kernel's downloadable output.

Every line this script asserts, it also prints — the calling process reads
the log back and does not need to re-derive anything. The final line is a
single JSON object prefixed R0_SUMMARY_JSON= for machine parsing.
"""
import csv
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone

WORK = "/kaggle/working"
REPO = "https://github.com/ridash2005/mip-splatting.git"
DATA = "/kaggle/input/nerf-synthetic-dataset/nerf_synthetic"
SCENE = "lego"
ITERS = 7000

checks = []


def check(id_, desc, passed, detail=""):
    checks.append({"id": id_, "desc": desc, "passed": passed, "detail": detail})
    tag = "PASS" if passed is True else ("SKIP" if passed is None else "FAIL")
    print(f"[check {id_}] {tag} — {desc} — {detail}", flush=True)


def sh(cmd, cwd=None, check_rc=True):
    print(f"+ {cmd}" + (f"   (cwd={cwd})" if cwd else ""), flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check_rc and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


def git_hash(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------ check 1: accelerator
sh("nvidia-smi", check_rc=False)
import torch  # noqa: E402  (import after nvidia-smi so the driver line prints first)

cap = torch.cuda.get_device_capability()
gpu_name = torch.cuda.get_device_name()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  capability {cap}  gpu {gpu_name}")
check(1, "accelerator compute capability >= 7.0, abort on P100/6.0 (F15)", cap >= (7, 0), f"{gpu_name} {cap}")
if cap < (7, 0):
    raise SystemExit("ABORT: compute capability below 7.0 — refusing to train (F15).")

# ------------------------------------------------------------ clone both arms
sh(f"git clone --recursive -b main {REPO} {WORK}/armA")
sh(f"git clone --recursive -b arm-b-3dgs-baseline {REPO} {WORK}/armB")
armA_commit = git_hash(f"{WORK}/armA")
armB_commit = git_hash(f"{WORK}/armB")
print("armA commit", armA_commit, "armB commit", armB_commit)

# ------------------------------------------------------------ check 2: installs
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
install_ok = True
try:
    sh("pip install -q ninja gputil lpips")
    sh("nvcc --version", check_rc=False)
    sh("gcc --version", check_rc=False)
    sh("g++ --version", check_rc=False)
    # arm A and arm B share an identical, unmodified rasteriser and simple-knn
    # (the §3 diff never touches submodules/), so building once against arm A's
    # copy is sufficient for both arms' training runs. Built one at a time,
    # without -q, so a real compile failure prints its actual error instead of
    # only pip's final wheel-build summary.
    sh(f"pip install {WORK}/armA/submodules/diff-gaussian-rasterization")
    sh(f"pip install {WORK}/armA/submodules/simple-knn")
    import GPUtil  # noqa: F401
    import lpips as _lpips_mod  # noqa: F401
    import diff_gaussian_rasterization  # noqa: F401
    import simple_knn._C  # noqa: F401
except Exception:
    install_ok = False
    traceback.print_exc()
check(2, "ninja/gputil/lpips + rasteriser/simple-knn built against installed torch", install_ok)
if not install_ok:
    raise SystemExit("ABORT: extension build failed, nothing downstream is trustworthy.")

check(3, "commits pinned into this run", True, f"armA={armA_commit} armB={armB_commit}")

# ------------------------------------------------------------ check 4: arm B diff present
with open(f"{WORK}/armB/scene/gaussian_model.py") as f:
    armB_src = f.read()
with open(f"{WORK}/armA/scene/gaussian_model.py") as f:
    armA_src = f.read()
diff_marker_ok = (
    "_disable_3D_filter" in armB_src and "torch.zeros_like(filter_3D" in armB_src
    and "_disable_3D_filter" not in armA_src
)
check(4, "§3 diff isolated to arm-b-3dgs-baseline (compute_3D_filter zero-branch), absent from main",
      diff_marker_ok)

# ------------------------------------------------------------ data
sh(f"python {WORK}/armA/convert_blender_data.py --blender_dir {DATA} --object_name {SCENE} "
   f"--out_dir {WORK}/multi-scale")

with open(f"{WORK}/multi-scale/{SCENE}/metadata.json") as f:
    meta = json.load(f)


def scales_of(file_paths):
    return sorted({p.rsplit("_d", 1)[1].split(".")[0] for p in file_paths})


test_scales = scales_of(meta["test"]["file_path"])
train_scales = scales_of(meta["train"]["file_path"])
check(7, "multi-scale data: 4 distinct test scales; default train split is d0-only (F3)",
      test_scales == ["0", "1", "2", "3"] and train_scales == ["0"],
      f"test scales={test_scales} train scales={train_scales}")

# ------------------------------------------------------------ check 8: LPIPS backbone
with open(f"{WORK}/armA/metrics.py") as f:
    metrics_src = f.read()
check(8, "LPIPS backbone is vgg, never alexnet (F9)", "lpips.LPIPS(net='vgg')" in metrics_src)


# ------------------------------------------------------------ train / render / score
def run_arm(arm_dir, out_dir, extra_flags, label):
    sh(f"OMP_NUM_THREADS=4 python train.py -s {WORK}/multi-scale/{SCENE} -m {out_dir} "
       f"--eval --white_background --iterations {ITERS} {extra_flags}", cwd=arm_dir)
    sh(f"OMP_NUM_THREADS=4 python render.py -m {out_dir} --skip_train", cwd=arm_dir)
    sh(f"OMP_NUM_THREADS=4 python metrics.py -m {out_dir}", cwd=arm_dir)
    results_path = os.path.join(out_dir, "results.json")
    ok = os.path.exists(results_path) and os.path.getsize(results_path) > 0
    check(10, f"metrics.py produced a non-empty results.json ({label}) — its bare `except:` "
              "makes a real failure print like a warning (F10)", ok, results_path)
    if not ok:
        raise RuntimeError(f"{label}: metrics.py silently failed — see {results_path} (F10)")
    with open(results_path) as f:
        return json.load(f)


out_armA = f"{WORK}/out_armA/{SCENE}"
out_armB = f"{WORK}/out_armB/{SCENE}"
resA = run_arm(f"{WORK}/armA", out_armA, "--kernel_size 0.1", "arm A / Mip-Splatting")
resB = run_arm(f"{WORK}/armB", out_armB, "--kernel_size 0.3 --disable_3D_filter", "arm B / 3DGS")


def unpack(res):
    scene_dir = next(iter(res))
    method = next(iter(res[scene_dir]))
    d = res[scene_dir][method]
    return d["PSNR"], d["SSIM"], d["LPIPS"], method


psnrA, ssimA, lpipsA, methodA = unpack(resA)
psnrB, ssimB, lpipsB, methodB = unpack(resB)
print(f"arm A pooled (all 4 test scales): PSNR {psnrA:.3f}  SSIM {ssimA:.4f}  LPIPS {lpipsA:.4f}")
print(f"arm B pooled (all 4 test scales): PSNR {psnrB:.3f}  SSIM {ssimB:.4f}  LPIPS {lpipsB:.4f}")

# §6 checks 5/6's 33.3/33.4 dB targets are calibrated to 30000 iterations (F11).
# At ITERS=7000 they are not expected to hold — record what we measured, not a
# pass/fail against a target this run was never going to meet (§14: never fudge
# a number to look better than it is).
check(5, "arm B (3DGS) unit test target 33.3±0.5 dB — informational at 7k iters, real check is R1 @ 30k",
      None, f"measured {psnrB:.3f} dB at {ITERS} iters")
check(6, "arm A (Mip-Splatting) sanity target 33.4±0.5 dB — informational at 7k iters, real check is R1 @ 30k",
      None, f"measured {psnrA:.3f} dB at {ITERS} iters")

# ------------------------------------------------------------ check 9: splitter parity
sys.path.insert(0, f"{WORK}/armA/tools")
import split_by_scale as sbs  # noqa: E402


def verify_and_split(out_dir, method, pooled_psnr, label):
    res = sbs.split_scene(out_dir, f"{WORK}/multi-scale/{SCENE}", method)
    n_total = sum(v["n"] for v in res.values())
    weighted = sum(v["n"] * v["PSNR"] for v in res.values()) / n_total
    ok = abs(weighted - pooled_psnr) < 0.01
    check(9, f"split_by_scale.py's count-weighted mean reproduces metrics.py's pooled PSNR ({label})",
          ok, f"weighted={weighted:.4f} pooled={pooled_psnr:.4f} "
              f"n_per_scale={ {k: v['n'] for k, v in res.items()} }")
    return res


splitA = verify_and_split(out_armA, methodA, psnrA, "arm A")
splitB = verify_and_split(out_armB, methodB, psnrB, "arm B")

# ------------------------------------------------------------ check 11: resume from checkpoint
resume_dir = f"{WORK}/out_resume_test/{SCENE}"
resume_ok = False
try:
    sh(f"OMP_NUM_THREADS=4 python train.py -s {WORK}/multi-scale/{SCENE} -m {resume_dir} "
       f"--eval --white_background --iterations 2000 --checkpoint_iterations 2000 --kernel_size 0.1",
       cwd=f"{WORK}/armA")
    ckpt_ok = os.path.exists(f"{resume_dir}/chkpnt2000.pth")
    sh(f"OMP_NUM_THREADS=4 python train.py -s {WORK}/multi-scale/{SCENE} -m {resume_dir} "
       f"--eval --white_background --iterations {ITERS} "
       f"--start_checkpoint {resume_dir}/chkpnt2000.pth --kernel_size 0.1",
       cwd=f"{WORK}/armA")
    resume_ok = ckpt_ok and os.path.exists(
        f"{resume_dir}/point_cloud/iteration_{ITERS}/point_cloud.ply")
except Exception:
    traceback.print_exc()
check(11, f"kill at 2K, restart from checkpoint, reach {ITERS} (scaled down from 30K for this rung)",
      resume_ok, resume_dir)

# ------------------------------------------------------------ check 12: off-session persistence
check(12, "off-session persistence via Hugging Face Hub", None,
      "skipped — no HF token available to this run; Kaggle's own /kaggle/working "
      "persistence covers this session, cross-session persistence is a separate step")

# ------------------------------------------------------------ results/runs.csv
now = datetime.now(timezone.utc).isoformat(timespec="seconds")
os.makedirs(f"{WORK}/results", exist_ok=True)
csv_path = f"{WORK}/results/runs.csv"
new = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sbs.RUNS_CSV_SCHEMA)
    if new:
        w.writeheader()
    for arm_letter, res, commit, kernel_size, disable_filter in (
        ("A", splitA, armA_commit, "0.1", "False"),
        ("B", splitB, armB_commit, "0.3", "True"),
    ):
        for sc, v in res.items():
            row = {c: "" for c in sbs.RUNS_CSV_SCHEMA}
            row.update({
                "timestamp_utc": now,
                "method": "mip-splatting" if arm_letter == "A" else "3dgs",
                "arm": arm_letter,
                "impl_commit": commit,
                "rasteriser_commit": commit,  # bundled directly in-repo, not a separate submodule
                "dataset": "blender",
                "scene": SCENE,
                "resolution_flag": "-1",
                "load_allres": "False",
                "kernel_size": kernel_size,
                "disable_3D_filter": disable_filter,
                "train_scale": "1x",
                "test_scale": sc,
                "iterations": str(ITERS),
                "psnr": f"{v['PSNR']:.4f}",
                "ssim": f"{v['SSIM']:.5f}",
                "lpips": f"{v['LPIPS']:.5f}",
                "lpips_backbone": "vgg",
                "gpu_model": gpu_name,
                "platform": "kaggle",
                "level_claimed": "L3",
                "notes": f"R0 smoke run, {ITERS} iters — plumbing check, not the R1 numeric target",
            })
            w.writerow(row)
print(f"wrote {csv_path}")

# ------------------------------------------------------------ summary
all_pass = all(c["passed"] is not False for c in checks)
summary = {
    "scene": SCENE,
    "iterations": ITERS,
    "gpu": gpu_name,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "armA_commit": armA_commit,
    "armB_commit": armB_commit,
    "psnrA": psnrA, "ssimA": ssimA, "lpipsA": lpipsA,
    "psnrB": psnrB, "ssimB": ssimB, "lpipsB": lpipsB,
    "per_scale_A": {k: v["PSNR"] for k, v in splitA.items()},
    "per_scale_B": {k: v["PSNR"] for k, v in splitB.items()},
    "checks": checks,
    "all_checks_pass_or_skip": all_pass,
}
print("R0_SUMMARY_JSON=" + json.dumps(summary))
if not all_pass:
    raise SystemExit("one or more §6 checks FAILED — see the table above.")
