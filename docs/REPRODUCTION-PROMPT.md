# Claude Code — baseline reproduction operating prompt (v3, all flags resolved)

**How to use.** Open Claude Code in an empty directory that will become the reproduction repository, and
paste everything below the rule as the first message. It is self-contained.

**Before pasting**, fill the two bracketed values in §0. Nothing else is left open.

**v3 status.** Every ambiguity in v1/v2 has been resolved by reading repository source. There are **no
undecided branches left**: arm B is settled, the STMT/MTMT switch is a named flag, the per-scale table has a
shipped tool, and the only remaining unknowns are two facts that can be read off a screen in the first
session (§8). Two v2 decisions were reversed and are marked **↺** below — do not follow v2.

---

## ROLE

You are running the baseline reproduction for a B.Tech Project on sampling-geometry band-limits for
radiance fields. Make **3D Gaussian Splatting** (Kerbl et al., SIGGRAPH 2023) and **Mip-Splatting** (Yu et
al., CVPR 2024) train, render and score on free-tier GPU compute, and attach three read-only instruments.

You are reproducing, not improving. The only code change you may make is the one specified in §3, verbatim.

## 0 · SESSION CONSTANTS

```
KAGGLE_USERNAME = [ ... ]
HF_USERNAME     = [ ... ]
```

## 1 · THE ONE ARCHITECTURAL DECISION, ALREADY MADE

**Everything runs inside the `autonomousvision/mip-splatting` codebase. Both arms.**

- **Arm A — Mip-Splatting**: the repository as shipped.
- **Arm B — 3DGS**: the same repository with the 3D smoothing filter zeroed and `--kernel_size 0.3`.

**↺ This reverses v2**, which preferred running the original `graphdeco-inria/gaussian-splatting` repo for
arm B. That is **impossible** for R1/R2, and here is exactly why — do not re-litigate it:

`convert_blender_data.py` writes `images_train/ images_val/ images_test/ metadata.json` — the mip-NeRF
multiscale format. It does **not** write `transforms_train.json`. And `scene/__init__.py` dispatches on
filename:

```python
if   os.path.exists(os.path.join(args.source_path, "sparse")):                # Colmap
elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")): # Blender
elif os.path.exists(os.path.join(args.source_path, "metadata.json")):         # Multi-scale
```

`sceneLoadTypeCallbacks` in mip-splatting has three entries — `Colmap`, `Blender`, **`Multi-scale`**. The
original 3DGS repo has only the first two, so on a `multi-scale/{scene}` directory it falls through every
branch and dies with `NameError: scene_info`. It also could not work in principle: `readMultiScale` sets
`fovx = focal2fov(meta["focal"][idx], ...)` **per frame**, whereas the Blender reader takes one global
`camera_angle_x`.

**Consequences you get for free, and should state in the report:**
- Both arms share one `metrics.py`, one LPIPS implementation, one dataloader → differences are the method,
  not the plumbing. This is a stronger comparison than cross-repo, not a weaker one.
- Every risk about post-publication 3DGS flags (`--antialiasing`, sparse-Adam, exposure, depth) disappears,
  because you never run that repository on the critical path.
- The reduction is a **unit test**: with the filter zeroed, arm B must land at ≈33.3 dB full-res.

Optionally, and only after R3, run the real 3DGS repo on the COLMAP scenes (which it *can* read) as an
independent cross-check. Never on the critical path.

## 2 · VERIFIED FACTS — do not re-derive; if reality disagrees, stop and report

All read from source on 7 Sep 2026.

| # | Fact | File |
|---|---|---|
| F1 | `mip-splatting/scripts/` = `fused_ply.py`, `run_mipnerf360.py`, `run_mipnerf360_stmt.py`, `run_nerf_synthetic_mtmt.py`, `run_nerf_synthetic_stmt.py`. All are Mip-Splatting arms. **No 3DGS baseline ships.** | `scripts/` |
| F2 | Scene dispatch is by filename; only mip-splatting has a `Multi-scale` callback. | `scene/__init__.py`, `scene/dataset_readers.py` |
| F3 | **STMT vs MTMT is one flag: `--load_allres`.** `ModelParams.load_allres` defaults `False`; `readMultiScale(..., only_highres)` skips every file not ending `d0.png` for the training split. Default = single-scale train. | `arguments/__init__.py`, `scene/dataset_readers.py` |
| F4 | `ModelParams._kernel_size` default = **0.1** (Mip-Splatting's 2D filter). 3DGS's fixed dilation is **0.3**. | `arguments/__init__.py` |
| F5 | The 3D filter enters through exactly two properties — `get_scaling_with_3D_filter` = `sqrt(s²+f²)` and `get_opacity_with_3D_filter` with `coef = sqrt(det1/det2)`. **Setting `filter_3D ≡ 0` makes both identities.** | `scene/gaussian_model.py` |
| F6 | `filter_3D = distance / focal_length * (0.2 ** 0.5)`, shape `[N,1]`; `distance` = min clamped z over cameras seeing the point (z>0.2, within a 1.15× screen margin); `focal_length` = max `camera.focal_x` over cameras, taken **independently**. Unseen Gaussians get `distance[valid].max()`. | `scene/gaussian_model.py::compute_3D_filter` |
| F7 | `render_set` writes `test/ours_{iter}/test_preds_{r}/` and `gt_{r}/` where `{r} = dataset.resolution` (the `-r` flag), **not** the per-image scale. With default `-r -1` all four scales land in one directory. | `render.py` |
| F8 | `metrics.py` averages every image in that directory into **one** PSNR/SSIM/LPIPS → `results.json`; it also writes `per_view.json` keyed by render filename. **The shipped pipeline cannot produce the four-column table.** See §4. | `metrics.py` |
| F9 | `metrics.py` uses `lpips.LPIPS(net='vgg')` (the official `lpips` package). 3DGS uses a vendored `lpipsPyTorch`, also VGG. **Never AlexNet.** | `metrics.py` both repos |
| F10 | `metrics.py` wraps each scene in a **bare `except:`** and prints "Unable to compute metrics for model". A hard failure looks like a warning. | `metrics.py` |
| F11 | Blender flags in the shipped script: `--eval --white_background`. No iteration flag → `OptimizationParams.iterations` default **30000**. `densification_interval` = 100. | `run_nerf_synthetic_stmt.py`, `arguments/__init__.py` |
| F12 | **↺ The Blender test split is NOT every-8th.** It is the dataset's own test split (`metadata.json["test"]`, or `transforms_test.json`). `llffhold=8` lives only in `readColmapSceneInfo`, so every-8th applies to Mip-NeRF 360 / T&T / Deep Blending only. v2 stated this without qualification. | `scene/dataset_readers.py` |
| F13 | The benchmark scripts ship a `GPUtil` dispatcher (`ThreadPoolExecutor(max_workers=8)`, one scene per free GPU, polling `getAvailable(maxMemory=0.1)`). Kaggle's 2×T4 is used automatically. | `scripts/*.py` |
| F14 | Blender scenes: `ship drums ficus hotdog lego materials mic chair`. `convert_blender_data.py` names images `{:03d}_d{j}.png`, focal `f / 2**j`, j = 0..3. | `scripts/`, `convert_blender_data.py` |
| F15 | 3DGS requires Compute Capability **7.0+**. T4 = 7.5 ✅. **P100 = 6.0 ❌.** | 3DGS README |
| F16 | Real-scene settings: outdoor `-i images_4`, indoor `-i images_2`, T&T/DB default; `--disable_viewer --quiet --eval --test_iterations -1`; renders at 7000 and 30000. | `full_eval.py` |

## 3 · ARM B — the only authorised code change, verbatim

Two edits, on a branch named `arm-b-3dgs-baseline`. Commit them separately from everything else and put the
diff in the report.

**Edit 1** — `arguments/__init__.py`, in `ModelParams.__init__`, beside `self._kernel_size = 0.1`:
```python
self.disable_3D_filter = False
```

**Edit 2** — `scene/gaussian_model.py`, the last line of `compute_3D_filter`:
```python
# was:
self.filter_3D = filter_3D[..., None]
# becomes:
if getattr(self, "_disable_3D_filter", False):
    self.filter_3D = torch.zeros_like(filter_3D[..., None])
else:
    self.filter_3D = filter_3D[..., None]
```
Set `gaussians._disable_3D_filter = dataset.disable_3D_filter` in `train.py` and `render.py` immediately
after the `GaussianModel` is constructed, before the first `compute_3D_filter` call.

**Why this is exactly 3DGS, not an approximation (F5).** With `filter_3D = 0`:
`get_scaling_with_3D_filter = sqrt(s² + 0) = s`, and in `get_opacity_with_3D_filter`,
`det2 = det1` so `coef = 1` and the opacity is returned unchanged. Both properties become the identity.
Combined with `--kernel_size 0.3`, matching 3DGS's fixed `0.3f` 2D dilation, the model is 3DGS.

**Unit test, run before anything else depends on it:** train `lego` with the flag on, and assert the
full-resolution STMT PSNR lands within 0.5 dB of **33.3** — 3DGS's published Blender value, measured
independently as 33.32 and 33.33 by two groups. If it does not, stop; the arm is wrong.

## 4 · THE PER-SCALE TABLE — you must run the splitter

F7 and F8 mean `metrics.py` gives you **one** averaged number where the paper reports four. Without this
step there is no Table 2 reproduction. Use `tools/split_by_scale.py` (supplied alongside this prompt).

The mapping it relies on, so you can check it: `readMultiScale` iterates
`metadata.json["test"]["file_path"]` in order, appending one camera per entry; `render_set` then does
`for idx, view in enumerate(views)` and saves `'{0:05d}'.format(idx)`. Order is preserved end to end, so
render `{idx:05d}.png` ↔ `file_path[idx]`, whose `_d{j}` suffix is the scale. Grouping `per_view.json` by
`j` gives the four columns with no re-rendering and no resampling.

```bash
python tools/split_by_scale.py \
  --model-root benchmark_nerf_synthetic_ours_stmt --data-root multi-scale \
  --method ours_30000 --csv results/runs.csv \
  --tag method=mip-splatting,arm=A,train_scale=1x
```

Do **not** instead run `render.py`/`metrics.py` four times with `-r 1 2 4 8`. That resamples the images and
changes the experiment.

## 5 · WHAT COUNTS AS REPRODUCED

| Level | Claim | Criterion |
|---|---|---|
| **L1** | Numerical | mean PSNR ±**0.20 dB**, SSIM ±**0.005**, LPIPS(VGG) ±**0.010** vs published, at the paper's resolution / iterations / split |
| **L2** | Phenomenon | the effect reappears with correct sign, ordering and order of magnitude under a stated deviation; absolute values not compared |
| **L3** | Harness parity | your own re-run is the control for all later work; published numbers cited, never used as the control |

Expect L1 only on Blender. Everything real is L2 + L3. Every reported row states its level.

## 6 · PRE-FLIGHT — twelve checks, all green before R1

Verifications, not runs. **Report the table, then wait.**

1. **Accelerator.** `torch.cuda.get_device_capability()` → `(7, 5)`. **Abort on P100** (F15).
2. **Installs.** `ninja`, `gputil` (imports as `GPUtil`), `lpips`, then `pip install
   submodules/diff-gaussian-rasterization submodules/simple-knn` with `TORCH_CUDA_ARCH_LIST="7.5"`.
3. **Commits pinned.** mip-splatting HEAD + both submodule commits, into `results/runs.csv` from row 1.
4. **Arm B diff applied** exactly as §3, on its own branch, and the diff printed.
5. **Arm B unit test.** `lego`, filter disabled, `--kernel_size 0.3` → full-res PSNR within 0.5 dB of 33.3.
6. **Arm A sanity.** `lego`, as shipped → full-res PSNR within 0.5 dB of 33.4.
7. **Multi-scale data.** Per scene, `metadata.json["test"]["file_path"]` has four distinct `_d{j}` groups;
   the training split under default `load_allres=False` contains **only** `d0` files (F3).
8. **LPIPS backbone.** Assert `net='vgg'` at import; record it as a CSV column (F9).
9. **Splitter.** `tools/split_by_scale.py` reproduces `results.json`'s single number as the
   count-weighted mean of its four groups. This proves the index mapping.
10. **`metrics.py` failure detection.** After every metrics call, assert `results.json` exists and is
    non-empty — the bare `except:` makes failure look like a warning (F10).
11. **Resume.** Kill a run at 2 K, restart from checkpoint, reach 30 K.
12. **Off-session persistence.** Push a dummy checkpoint and CSV; retrieve from a fresh session.

## 7 · THE LADDER

Do not begin a rung until its predecessor's exit criterion is met. **Report and wait at G1, G2, G3.**

### R0 — Smoke · ≤ 1 GPU-h
`lego`, 7 000 iterations, both arms, end to end, plus all of §6.
```bash
git clone --recursive https://github.com/autonomousvision/mip-splatting && cd mip-splatting
export TORCH_CUDA_ARCH_LIST="7.5"
pip install ninja gputil lpips
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
```
Mip-Splatting's README pins `python=3.8 / torch==1.12.1+cu113`. **Do not recreate that on a 2026 image** —
build the extensions against the installed torch and record what you built against.

**Exit (G1):** both arms train, render and score without hand-editing; §6 fully green; measured
iterations/second recorded, replacing every estimate below. **Report and wait.**

### R1 — Blender STMT · PRIMARY TARGET · est. 10–16 GPU-h
```bash
python convert_blender_data.py --blender_dir nerf_synthetic/ --out_dir multi-scale

# Arm A — as shipped (dispatcher fills both T4s)
python scripts/run_nerf_synthetic_stmt.py

# Arm B — same script with two flags added to the train.py line:
#   --kernel_size 0.3 --disable_3D_filter
# copy the script to scripts/run_nerf_synthetic_stmt_3dgs.py, change output_dir, add the flags.

python tools/split_by_scale.py --model-root <out> --data-root multi-scale \
  --method ours_30000 --csv results/runs.csv --tag method=...,arm=...,train_scale=1x
```
Published target (mean PSNR, 8 scenes):

| test scale | 3DGS | Mip-Splatting |
|---|---|---|
| full | 33.33 | 33.36 |
| 1/2 | 26.95 | 34.00 |
| 1/4 | 21.38 | 31.85 |
| 1/8 | **17.69** | **28.67** |

**Exit (G2):** 3DGS falls ≳13 dB full→⅛; Mip-Splatting falls ≲6 dB; gap at ⅛ exceeds **9 dB**. If not,
**stop and do not proceed.** Diagnose in this order: (a) §6 check 9, the splitter's index mapping;
(b) whether arm B's `filter_3D` is actually zero at step 30 000 — print its max; (c) `--white_background`.
**Report and wait.**

### R2 — Blender MTMT · est. 14–22 GPU-h
Identical to R1 with **`--load_allres`** on both arms (F3); `scripts/run_nerf_synthetic_mtmt.py` for arm A.
Published: 3DGS 28.79 / 30.66 / 31.64 / 27.98 vs Mip-Splatting 32.81 / 34.49 / 35.45 / 35.50.
**Exit:** Mip-Splatting ahead at every scale, gap widening at ⅛. L1 if within 0.20 dB.

### R3 — Real scenes that fit · est. 20–30 GPU-h
`scripts/run_mipnerf360.py` for arm A; the same with `--kernel_size 0.3 --disable_3D_filter` for arm B.
Restrict to what fits 16 GB: Mip-NeRF 360 indoor (`room counter kitchen bonsai`) at `-i images_2`, plus
Tanks & Temples (`truck train`) and Deep Blending (`drjohnson playroom`) at default (F16). Every-8th test
split applies **here** (F12).

Data: `tandt_db.zip` from `repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/`; Mip-NeRF 360
from the Barron et al. project page. Reference: DB 29.41 / 0.903 / 0.243; T&T 23.14 / 0.841 / 0.183;
Mip-NeRF 360 all-9 average 27.21 / 0.815 / 0.214 — **not** comparable to a 4-indoor subset; report the
subset average and say so.
**Exit:** per-scene table for scenes that fit. **On OOM: record it and move on.** Do not reduce resolution,
iterations or Gaussian count to make a scene fit. Expect OOM during densification if at all.

### R5 — Seed spread · NOT OPTIONAL · est. 4–8 GPU-h
3 seeds × 2 scenes (one Blender, one real) × 2 arms → per-scene mean ± σ. This is the error bar that makes
the project's parity prediction falsifiable. Run it even if R3 is incomplete.

### R4 — Outdoor, reduced · OPTIONAL · est. 12–20 GPU-h
`bicycle flowers garden stump treehill` at `-i images_8` instead of `images_4`. **The only authorised
deviation from paper settings.** L2 only, separate table, resolution in the caption. Cut first if quota binds.

## 8 · THE ONLY TWO THINGS STILL UNKNOWN — resolve in session 1, then record

Everything else in this prompt is settled. These two cannot be read from source:

1. **Kaggle quota and session cap.** Secondary sources give **30 h/week**, **12 h max GPU session**,
   accelerators **T4 ×2** or P100, ~**20 GB** auto-saved `/kaggle/working`. Read the true values off the
   notebook sidebar in session 1 and write them into the README. The ladder does not depend on them; only
   the calendar does. With the dispatcher on 2×T4 (F13), R1 should be ~5–8 h wall-clock — **one session**.
2. **Whether `counter` and `drjohnson` fit 16 GB.** Unknowable without running. R3's exit rule already
   covers it: record the OOM, do not work around it.

Disk hygiene, since `/kaggle/working` is ~20 GB: 8 scenes × 4 scales × ~200 test views × 2 (render + gt)
is roughly 2.5 GB of PNGs per arm. After each scene's metrics succeed, delete `test_preds_*` and `gt_*`,
keeping `results.json`, `per_view.json`, the checkpoint, and a fixed set of qualitative views.

## 9 · INSTRUMENTS — attach to the runs above, change nothing

Read-only with respect to the optimiser. If any would alter a training decision, it is implemented wrongly.

**I-1 · Floor occupancy.** Log `self.filter_3D` (shape `[N,1]`, F6) at end of training, per scene, with each
Gaussian's distance to its nearest camera and final scale vector. **Exclude or flag the Gaussians assigned
`distance[valid].max()` because no camera saw them** (F6) — they are an artefact and will otherwise form a
spurious right tail. Deliverable: one histogram per scene plus a summary CSV.
*Decides:* B1's estimation term is visible only where `s²/λ_i > f_k²`. If the floor sits at or above the cap
almost everywhere, B1 collapses to the baseline and the project must pivot — the cheapest falsification
available, and a by-product of a run already scheduled.

**I-2 · Λ conditioning profile.** On a **trained, unmodified** arm-B checkpoint, accumulate
```
Λ_k = Σ_n  w_nk · (J_n W_n)^T Σ_pix^{-1} (J_n W_n)
```
`W_n` the viewing transform, `J_n` the Jacobian of the affine approximation to the projective transform —
the pair already formed for `Σ' = J W Σ W^T J^T` — `Σ_pix` isotropic pixel noise, `w_nk` the blending
contribution in view *n*. Each view contributes rank 2. Record `λ1 ≥ λ2 ≥ λ3`, `λ3/λ1`, and the angle
between the smallest eigenvector and the mean viewing direction. Pure PyTorch, no CUDA.

**I-3 · Run-to-run spread.** From R5: per-scene mean and standard deviation, both arms. One CSV, σ stated.

## 10 · CAMERA-SUBSET PROTOCOLS — build once, free

Four view selections from images already downloaded; no new capture, no new training data. **Low-parallax**
(narrow angular cone) · **one-sided arc** (half the orbit) · **mixed focal** (a random half downsampled
2–4×) · **grazing** (high-incidence views only). Seeded, reproducible JSON index lists in the repo.

## 11 · RESULTS SCHEMA — append-only, never edited by hand

`results/runs.csv`, one row per (arm, scene, test-scale, seed). Append only.

```
run_id, timestamp_utc, method, arm, impl_commit, rasteriser_commit,
dataset, scene, resolution_flag, load_allres, kernel_size, disable_3D_filter,
train_scale, test_scale, iterations, seed,
psnr, ssim, lpips, lpips_backbone, n_gaussians, model_mb, peak_vram_mb,
train_seconds, render_fps, gpu_model, platform, level_claimed, notes
```

`arm` is `A` or `B`. `lpips_backbone` is `vgg`. `level_claimed` is `L1`/`L2`/`L3`. Provide `make table1`,
`make table2`, … and `make figures`; `make figures` calls `make_figures.py --mode measured`, which **refuses
to draw** if the CSV has no usable rows — so a "measured" figure can never contain a published number.

## 12 · INFRASTRUCTURE

- **GitHub** — the repository. One branch per component; `arm-b-3dgs-baseline` holds the §3 diff and nothing
  else. Every run logs git hash, full config, seed, GPU model, wall-clock.
- **Kaggle Datasets** — download once, build `multi-scale/`, publish as a **private** dataset mounting
  read-only at `/kaggle/input/`. Never re-download inside a quota-metered GPU session.
- **Hugging Face Hub (private)** — checkpoints and `results/runs.csv`, pushed **before** the session ends.
- **W&B free tier** — optional; skip if it costs more than ten minutes. The CSV is the source of truth.

Ask before creating any account. Never write credentials into the repository — read them from Kaggle
Secrets, Colab userdata, or environment variables. Nothing paid, ever; if a step seems to need payment,
stop and report.

**Licence.** 3DGS is under the Gaussian-Splatting License (Inria / MPII): research and evaluation only,
redistribution must carry the licence and retain notices, no commercial use. Mip-Splatting follows it. Do
not publish derived code without the licence text; do not push dataset contents to any public repo.

## 13 · HOW TO REPORT

After every rung, and immediately on any failure: (1) what ran — arm, scene, iterations, resolution, seed,
GPU, commit; (2) the numbers as a table, with the claimed level per row; (3) comparison to published, delta,
and whether it meets that level's tolerance; (4) what failed, verbatim, including the full OOM or build
error; (5) measured GPU-hours consumed and remaining quota; (6) the one decision you need, or "none".
Short. No narration of steps visible in the log.

## 14 · PROHIBITIONS

- No code change beyond §3, which is applied verbatim on its own branch.
- Do not change resolution, iteration count or scene subset to make a number look better. R4 is the only
  authorised deviation and must be labelled in its caption.
- Do not produce the per-scale table by re-rendering at `-r 1 2 4 8`. That resamples. Use §4.
- Do not let LPIPS default to AlexNet (F9). Do not run on a P100 (F15).
- Do not trust `metrics.py`'s console output; assert `results.json` (F10).
- Do not say "every 8th image" about Blender — that is COLMAP-only (F12).
- Do not compare Mip-Splatting's Mip-NeRF 360 zoom numbers (1× PSNR 29.19) against 3DGS's Table 1 average
  (27.21). Different protocols. This exact error already occurred once in this project's earlier write-up.
- Do not treat `gsplat`'s `rasterize_mode="antialiased"` as Mip-Splatting: it applies the 2D opacity
  compensation ρ = √(det Σ / det(Σ+εI)), not the 3D smoothing filter.
- Do not fill any results cell you did not measure. A cell is either produced by a logged run or it is
  empty. There is no third state.

## 15 · FIRST ACTIONS

1. `nvidia-smi`; `python -c "import torch; print(torch.__version__, torch.version.cuda,
   torch.cuda.get_device_capability())"` — record the environment; abort on capability `(6, 0)`.
2. Initialise the repository, `results/runs.csv` header, `make` targets, and `tools/split_by_scale.py`.
3. Apply the §3 diff on its own branch and print it.
4. Work through §6. **Report the twelve-row table and wait.**
5. Then R0, and report at G1.
