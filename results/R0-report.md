# R0 — lego · 7000 iterations · both arms

## 1 · What ran

| | |
|---|---|
| scene | `lego` |
| iterations | 7000 |
| protocol | Blender STMT, `load_allres=False`, `-r -1`, `--eval --white_background` |
| test split | the dataset's own `metadata.json["test"]` — **not** every 8th (F12) |
| arm A | Mip-Splatting as shipped, `--kernel_size 0.1` |
| arm B | `--kernel_size 0.3 --disable_3D_filter` |
| GPU | Tesla T4 ×2 |
| torch / CUDA | 2.10.0+cu128 / 12.8 |
| arm A commit | `da225e8ee4b1` |
| arm B commit | `36a337e41af9` |
| LPIPS backbone | vgg (F9) |
| seed | 0 (`safe_state` seeds unconditionally; no `--seed` flag exists) |

## 2 · Measured

| test scale | 3DGS PSNR | Mip-Splatting PSNR | 3DGS SSIM | Mip-S SSIM | 3DGS LPIPS | Mip-S LPIPS | n views | level |
|---|---|---|---|---|---|---|---|---|
| 1x | 33.76 | 33.51 | 0.9770 | 0.9765 | 0.0261 | 0.0264 | 200 | L3 |
| 1/2 | 31.63 | 34.05 | 0.9777 | 0.9851 | 0.0194 | 0.0126 | 200 | L3 |
| 1/4 | 27.55 | 31.73 | 0.9595 | 0.9833 | 0.0341 | 0.0137 | 200 | L3 |
| 1/8 | 24.37 | 28.39 | 0.9364 | 0.9733 | 0.0643 | 0.0244 | 200 | L3 |
| *pooled (all 4)* | 29.33 | 31.92 | 0.9627 | 0.9796 | 0.0360 | 0.0193 | | L3 |

Cost and model size, measured:

| | arm A (Mip-Splatting) | arm B (3DGS) |
|---|---|---|
| train wall-clock (s) | 336 | 312 |
| iterations/second | 20.82 | 22.40 |
| render wall-clock (s) | 88 | 87 |
| render fps | 9.13 | 9.24 |
| test views rendered | 800 | 800 |
| Gaussians | 255772 | 224096 |
| model size (MB) | 64.5 | 56.5 |
| peak VRAM (MB) | 4275 | 4275 |
| max abs(filter_3D) in saved PLY | 0.00163636 | 0 |

## 3 · Against published

**Not comparable.** The published Blender numbers are measured at 30000 iterations (F11); this run is 7000. Every row above is claimed at **L3** (harness parity) — it is the control for later work, not a numerical reproduction. The targets are printed below for orientation only, and no delta against them means anything yet.

| test scale | arm | measured | published target | delta | within L1 (±0.2 dB)? |
|---|---|---|---|---|---|
| 1x | 3DGS | 33.76 | 33.33 | +0.43 | n/a — L3 run |
| 1x | Mip-Splatting | 33.51 | 33.36 | +0.15 | n/a — L3 run |
| 1/2 | 3DGS | 31.63 | 26.95 | +4.68 | n/a — L3 run |
| 1/2 | Mip-Splatting | 34.05 | 34.00 | +0.05 | n/a — L3 run |
| 1/4 | 3DGS | 27.55 | 21.38 | +6.17 | n/a — L3 run |
| 1/4 | Mip-Splatting | 31.73 | 31.85 | -0.12 | n/a — L3 run |
| 1/8 | 3DGS | 24.37 | 17.69 | +6.68 | n/a — L3 run |
| 1/8 | Mip-Splatting | 28.39 | 28.67 | -0.28 | n/a — L3 run |

The G2 exit criterion, evaluated on this run (it is *defined* on R1 at 30k, so at any other length this is the phenomenon check, not the gate):

- 3DGS falls **9.39 dB** full → 1/8 (G2 wants ≳13 dB) — below the threshold
- Mip-Splatting falls **5.13 dB** (G2 wants ≲6 dB) — meets the threshold
- gap at 1/8 is **4.02 dB** (G2 wants > 9 dB) — below the threshold

## 4 · §6 pre-flight

| # | result | check | detail |
|---|---|---|---|
| 1 | PASS | accelerator compute capability >= 7.0, abort on P100/6.0 (F15) | Tesla T4 x2 (7, 5) |
| 2 | PASS | ninja/gputil/lpips/open3d + rasteriser & simple-knn built against installed torch | open3d=installed |
| 3 | PASS | commits pinned into this run | armA=da225e8ee4b105dac444fe255c757513538b7f85 armB=36a337e41af9354a64e604285710c70499ae5fc7 torch=2.10.0+cu128 cuda=12.8 |
| 4 | PASS | §3 diff present on arm B, absent from main, and confined to its four files | files arm B introduces over the merge base: ['arguments/__init__.py', 'render.py', 'scene/gaussian_model.py', 'train.py']; commits behind main: 0 |
| 7 | PASS | multi-scale data written: 4 distinct test scales; metadata lists all 4 train scales, of which only d0 may survive readMultiScale (F3) | test=['0', '1', '2', '3'] n_test=800 train=['0', '1', '2', '3'] n_train=400 d0=100 |
| 8 | PASS | LPIPS backbone is vgg, never alexnet (F9) |  |
| 10 | PASS | metrics.py produced a non-empty results.json (arm A / Mip-Splatting) — its bare `except:` makes a real failure print like a warning (F10) | /kaggle/working/out_armA/lego/results.json |
| 9 | PASS | split_by_scale.py's count-weighted mean reproduces metrics.py's pooled PSNR (arm A / Mip-Splatting) — proves the render-index → scale mapping | weighted=31.9193 pooled=31.9193 n_total=800 n_per_scale={'1x': 200, '1/2': 200, '1/4': 200, '1/8': 200} |
| 10 | PASS | metrics.py produced a non-empty results.json (arm B / 3DGS) — its bare `except:` makes a real failure print like a warning (F10) | /kaggle/working/out_armB/lego/results.json |
| 9 | PASS | split_by_scale.py's count-weighted mean reproduces metrics.py's pooled PSNR (arm B / 3DGS) — proves the render-index → scale mapping | weighted=29.3262 pooled=29.3262 n_total=800 n_per_scale={'1x': 200, '1/2': 200, '1/4': 200, '1/8': 200} |
| 7.1 | PASS | readMultiScale loaded the d0-only training split (F3, load_allres=False) | armA loaded 100, armB loaded 100, d0 files in metadata = 100 (of 400 total) |
| 4.1 | PASS | arm B's saved model has filter_3D ≡ 0 and arm A's does not (F5) — this, not the diff text, is what makes arm B exactly 3DGS | max\|filter_3D\|: armA=0.00163636 armB=0 |
| 5 | skip | arm B (3DGS) unit test target 33.3±0.5 dB — informational at 7000 iters, the real check is R1 @ 30k full-res | pooled 29.326 dB; per-scale reported below |
| 6 | skip | arm A (Mip-Splatting) sanity target 33.4±0.5 dB — informational at 7000 iters, the real check is R1 @ 30k full-res | pooled 31.919 dB; per-scale reported below |
| 9.1 | PASS | tools/selftest_pipeline.py: reporting chain verified against a fixture with planted, known-in-advance answers | exit 0 |
| 11 | **FAIL** | kill at 2000, restart from checkpoint, reach 7000 (scaled down from 30K for this rung) | RuntimeError: command failed (1): OMP_NUM_THREADS=4 python train.py -s /kaggle/working/multi-scale/lego -m /kaggle/working/out_resume_test/lego --eval --white_background --iterations 7000 --test_iterations -1 --start_checkpoint /kaggle/working/out_resume_test/lego/chkpnt2000.pth --kernel_size 0.1
--- tail ---
Optimizing /kaggle/working/out_resume_test/lego
Output folder: /kaggle/working/out_resume |
| 12 | skip | off-session persistence | Kaggle leg satisfied — results/, logs/ and both models persist as kernel output and are pulled down by tools/kaggle_push.py. Hugging Face Hub leg skipped: no HF token is available to this run; not faked. |

**Failures, verbatim:**

- check 11: kill at 2000, restart from checkpoint, reach 7000 (scaled down from 30K for this rung) — `RuntimeError: command failed (1): OMP_NUM_THREADS=4 python train.py -s /kaggle/working/multi-scale/lego -m /kaggle/working/out_resume_test/lego --eval --white_background --iterations 7000 --test_iterations -1 --start_checkpoint /kaggle/working/out_resume_test/lego/chkpnt2000.pth --kernel_size 0.1
--- tail ---
Optimizing /kaggle/working/out_resume_test/lego
Output folder: /kaggle/working/out_resume`

Skipped, and why — none of these is recorded as a pass:

- check 5: pooled 29.326 dB; per-scale reported below
- check 6: pooled 31.919 dB; per-scale reported below
- check 12: Kaggle leg satisfied — results/, logs/ and both models persist as kernel output and are pulled down by tools/kaggle_push.py. Hugging Face Hub leg skipped: no HF token is available to this run; not faked.

## 5 · Compute

- `train_armA` — 5.6 min
- `train_armB` — 5.2 min
- `convert_blender_data` — 1.8 min
- `metrics_armA` — 1.3 min
- `metrics_armB` — 1.2 min

**Measured: 0.25 GPU-hours** for this rung (the ladder budgeted ≤ 1 for R0).

Measured throughput on Tesla T4: **20.82 iterations/second** for arm A, 22.40 for arm B. A 30 000-iteration scene is therefore ≈ 24 min of training per arm, before rendering and metrics — this replaces the estimates in §7 (G1).

Remaining quota: not machine-readable from the kernels API; read the notebook sidebar (§8.1).

