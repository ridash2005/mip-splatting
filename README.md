# BTP baseline reproduction — 3D Gaussian Splatting vs Mip-Splatting

B.Tech Project reproduction harness for sampling-geometry band-limits in radiance
fields. Reproduces **3D Gaussian Splatting** (Kerbl et al., SIGGRAPH 2023) and
**Mip-Splatting** (Yu et al., CVPR 2024) as two arms of one codebase, so the only
difference between them is the method, not the plumbing. Governing spec:
[`docs/REPRODUCTION-PROMPT.md`](docs/REPRODUCTION-PROMPT.md) — do not re-derive
anything it states as settled; if reality disagrees with it, stop and report.

## The one architectural decision

Both arms run inside this repository (a fork of `autonomousvision/mip-splatting`),
not the original `graphdeco-inria/gaussian-splatting`:

- **Arm A — Mip-Splatting**: the repository as shipped.
- **Arm B — 3DGS**: the same repository with the 3D smoothing filter zeroed
  (`--disable_3D_filter`), `--kernel_size 0.3`, **and** the 2D Mip filter's
  opacity compensation disabled in the rasteriser
  (`--disable_2D_mip_compensation`, branch `arm-b-3dgs-vanilla`). See §3 of the
  prompt for why the first two are exactly 3DGS's *3D* band-limit, and why the
  original repo cannot read the multi-scale Blender format at all.

  The third flag was added after gate G2 failed. Zeroing the 3D filter makes the
  arm 3DGS in one domain and says nothing about the other: both arms had
  inherited Mip-Splatting's screen-space opacity compensation
  `rho = sqrt(det(Sigma')/det(Sigma'+kI))`, which vanilla 3DGS does not apply, so
  the arm labelled 3DGS was anti-aliasing better than 3DGS. On `lego` at 30K,
  removing it leaves full resolution unchanged (36.08 -> 36.05) and drops 1/8
  scale from 24.10 to 17.46 against a published 17.69. Every arm-B row measured
  before that is marked `SUPERSEDED_BY_C2` in `results/runs.csv` and kept.

## Repo layout

```
arguments/ scene/ gaussian_renderer/ ...   mip-splatting, as forked (arm A)
scripts/                                    benchmark drivers shipped upstream
tools/split_by_scale.py                     recovers the per-scale table (§4) —
                                             never re-render at -r 1 2 4 8, it resamples
tools/make_table.py                         renders Table 1 / Table 2 from results/runs.csv
tools/make_figures.py                       renders every report figure; --mode measured
                                             refuses to draw without usable CSV rows
results/runs.csv                            append-only, one row per (arm, scene, test-scale, seed)
docs/REPRODUCTION-PROMPT.md                 the operating prompt this repo implements
```

Branches:
- `main` — arm A (repo as shipped) plus the tooling above.
- `arm-b-3dgs-baseline` — main plus **only** the §3 diff (2 files, 2 edit sites
  each: `arguments/__init__.py`, `scene/gaussian_model.py`, and the one-line
  wiring in `train.py` / `render.py`). Nothing else. Diff printed in git history.

Remotes: `origin` = https://github.com/ridash2005/mip-splatting (this fork),
`upstream` = https://github.com/autonomousvision/mip-splatting (read-only reference).

## Session constants

```
KAGGLE_ACCOUNTS = rickaryadas, ceoricky   (see .env; rotation order)
HF_USERNAME     = rickaryadas
```

Kaggle credentials live in `.env` at the repo root, which is gitignored and is
the one place a token is written. Each account is a `KAGGLE_ACCOUNT_<n>_USERNAME`
/ `_TOKEN` pair, numbered from 1; the pair is always resolved together, because
a kernel is pushed to `<username>/<slug>` and authenticated with `<token>`.
Copy `.env.example` to `.env` to set this up on a fresh checkout.

## Environment

**Local dev machine**: NVIDIA GeForce MX350, 2 GB VRAM, compute capability below
the repo's minimum. **Fails F15 (7.0+ required) — do not attempt training here.**
Used only for git, tooling, and orchestrating remote runs. The reporting half of
the pipeline does run here: `make selftest`.

**Training target**: Kaggle. `torch.cuda.get_device_capability()` is asserted
in-session (§6 check 1) before any run, and the kernel aborts on anything below
(7, 0).

**Choosing the accelerator.** Kaggle allocates a Tesla P100 (compute capability
6.0) to every plain `enable_gpu: true` push — eleven consecutive pushes got one,
and 3DGS cannot run on it (F15). The accelerator is selected by a `machineShape`
field on `/api/v1/kernels/push`, whose name and permitted values are read out of
kagglesdk 0.1.37's wire schema. Neither installed client can send it (kagglesdk
0.1.28 has no such field; newer kagglesdk and `kaggle` 2.x need Python ≥ 3.11,
this machine has 3.10), so `tools/kaggle_push.py --accelerator NvidiaTeslaT4`
posts the body directly with the field added. A probe kernel confirmed it
returns `GPU 0: Tesla T4 / GPU 1: Tesla T4`. The API validates none of it — a
bad field name or value returns HTTP 200 and silently yields a P100 — which is
why the kernel asserts the capability itself and prints the GPU it got. See
[`kaggle/README.md`](kaggle/README.md).

Kaggle quota (§8.1) still to be read off the notebook sidebar; the secondary
figures are 30 h/week, 12 h/session, ~20 GB `/kaggle/working`.

**When the week runs out.** The 30 h allowance is per account on a rolling
seven-day window, and exhausting it on 8 September 2026 stalled the ladder for
days. `tools/kaggle_push.py` now rotates: when Kaggle refuses a push for want of
quota, that account goes into cooldown (persisted in `.kaggle_quota_state.json`,
so a restarted driver does not rediscover it) and the push is retried as the
next account in `.env`. `tools/finish.py` only falls back to waiting once every
account is spent. Adding an account to `.env` is what buys more compute; see
`.env.example`. `make selftest` covers the rotation offline.

### Environment measured in-session

| | |
|---|---|
| image | Python 3.12, torch 2.10.0+cu128, CUDA 12.8, driver 580.159.04 |
| accelerator | `machineShape=NvidiaTeslaT4` → **Tesla T4 ×2** (CC 7.5), passes F15 |
| default (no `machineShape`) | Tesla P100 (CC 6.0) — aborts at check 1 |
| concurrency | 2 batch GPU sessions per account |
| weekly budget | 30 GPU-h per account, rolling; 2 accounts configured → 60 h |

### The two deviations from a clean `pip install`, and why

Neither touches the method; both are recorded in the run's notes and in the CSV.

1. **`#include <cfloat>` added to `submodules/simple-knn/simple_knn.cu`.** That
   file uses `FLT_MAX` without including it. Older toolchains pulled `<float.h>`
   in transitively; CUDA 12.8 with GCC 13 does not, and the build fails outright.
   A compile fix, applied identically to both arms.
2. **`open3d`.** `train.py` imports it and never uses it. The kernel installs it;
   only if the image cannot provide it does the run neutralise that single import
   — and only after proving at run time that `o3d`/`open3d` appear exactly twice
   in the file (the import line and nothing else).

An earlier revision also rewrote `auxiliary.h` to drop an unused projection
block. That silenced a warning and nothing more, so it was a rasteriser source
edit outside §3; it has been reverted.

## Status

- [x] Forked upstream, cloned, remotes set up.
- [x] Arm B diff applied verbatim on `arm-b-3dgs-baseline`. `git diff
      main...arm-b-3dgs-baseline` is exactly **4 files, 7 insertions, 1
      deletion**, enforced on every run by the kernel's check 4.
- [x] **Reporting chain verified end to end** — `make selftest`, 37 assertions,
      no GPU. See "What the self-test caught" below.
- [x] **R0 complete. All twelve §6 checks pass or are honestly skipped.**
      `lego`, 7000 iterations, both arms, Tesla T4 ×2, 0.47 GPU-hours.
      Skips are checks 5/6 (their 33.3/33.4 dB targets are 30 000-iteration
      values, so they are recorded, not scored) and check 12's Hugging Face leg
      (no token; the Kaggle persistence leg is satisfied).
- [ ] R1 (primary target), R2, R3, R5, R4 (optional).

### R0 result — `results/R0-report.md`

| test scale | 3DGS (arm B) | Mip-Splatting (arm A) | gap |
|---|---|---|---|
| 1x | 33.78 | 33.52 | −0.26 |
| 1/2 | 31.65 | 34.04 | +2.39 |
| 1/4 | 27.56 | 31.72 | +4.16 |
| 1/8 | 24.37 | 28.37 | +4.00 |

Mean of two runs at each scale; L3, and only L3. Sign and ordering reproduce —
3DGS falls 9.4 dB full→⅛ against Mip-Splatting's 5.1, and Mip-Splatting leads at
every reduced scale. The magnitudes do not, and are not claimed to: 7 000
iterations is not the 30 000 the published numbers are measured at (F11).

Mip-Splatting already sits within 0.3 dB of published at every scale, while
3DGS is **+4.7 to +6.7 dB above** its published targets. The aliasing collapse
deepens with training, so the gap should widen toward the published 10.98 dB at
30 K — which is exactly what G2 tests.

**§3 verified on the artefact, not the diff text.** `filter_3D` read back out of
the saved point clouds is **0.0 for arm B and 0.00163636 for arm A** (check
4.1). `render.py` never recomputes the filter — `load_ply` reads it from the PLY
— so the PLY is what decides whether arm B is really 3DGS.

### Measured, replacing §7's estimates (G1)

| | |
|---|---|
| training throughput | 20.8 it/s (arm A), 22.4 it/s (arm B) on a Tesla T4 |
| render | 9.1 fps over 800 test views |
| peak VRAM | **4 275 MB** — 16 GB is nowhere near binding on Blender |
| model | 256 k Gaussians / 64 MB (A) vs 224 k / 56 MB (B) |
| R0 cost | **0.47 GPU-hours** against the ≤ 1 budgeted |
| implied R1 | ~24 min training per scene per arm → 8 scenes × 2 arms ≈ **6–7 GPU-h**, against §7's 10–16 |

### Run-to-run spread, free

The rung was run twice at identical seed and config. The largest per-scale PSNR
difference is **0.039 dB** — five times below L1's ±0.20 dB tolerance, so the
harness's own nondeterminism (CUDA atomics in the rasteriser) cannot threaten an
L1 claim. Gaussian counts moved by 0.1–0.7 %. Both runs are in
`results/runs.csv`; it is append-only, and neither was removed to tidy the
table.

### What the self-test caught

Four defects, all silent — none of them would have raised an error where it
happened:

1. **`make figures --mode measured` drew the published curve.** `runs.csv` stores
   `3dgs` / `mip-splatting`; the figures key off `3DGS` / `Mip-Splatting`. Every
   measured lookup missed, the non-empty guard still passed, and fig 1 was
   written to a "measured" filename containing published numbers — the one thing
   §11 says must be impossible. Unknown slugs are now fatal.
2. **Figures 5 and 6 were never drawn.** They are defined below the old
   `if __name__ == "__main__"` block, so `main()` could not see them.
3. **The splitter's assumed JSON nesting does not exist.** `metrics.py` does
   `json.dump(full_dict[scene_dir], ...)` — the value, not the enclosing dict —
   so `results.json` is keyed by method, with no scene_dir above it.
   `split_by_scale.py` would have exited *after* both arms had trained.
4. **The open3d guard proved itself.** Its "this import is provably unused"
   check counted references across the whole file including comments, so once it
   had inserted its own comment the guard was a tautology.

Numbers 1 and 3 are the dangerous class: they produce a plausible table or a
late crash rather than an early error. Number 3 was found only after the fixture
was rebuilt from `metrics.py`'s actual dump line rather than from what the
splitter expected — a fixture that encodes the bug cannot detect it.

## Running the ladder on Kaggle

Training cannot happen on the local machine (see Environment). The plan is to
push a Kaggle kernel that clones this repo's `main` (arm A) and
`arm-b-3dgs-baseline` (arm B), runs §6 and the rung currently in scope, and
writes checkpoints + `results/runs.csv` rows back out — pulled down here via
the Kaggle API, and pushed to Hugging Face Hub before the session ends (§12).
No credentials are stored in this repository; they are read from Kaggle
Secrets / environment variables at run time.

## Tables and figures

Everything regenerates from `results/runs.csv` — no number is ever typed into
a document by hand.

```
make selftest   # the reporting chain, against a fixture with known answers
make table1     # R1 Blender STMT: 3DGS vs Mip-Splatting, 4 test scales
make table2     # R2 Blender MTMT
make figures    # figures/, redrawn from measured rows with the published curve
                # kept as a faint reference line
make thesis     # thesis/main.pdf, tables and figures regenerated first
make slides     # slides/BTP-Panel.pptx, from the same three sources
make all        # both
```

`make thesis` and `make slides` read the same inputs — `results/runs.csv`, the
rung summaries under `results/kaggle_runs/`, and `results/geometry/claims.json`
— so a slide cannot disagree with the document about a number. A table or figure
whose run has not happened is emitted as an explicit "not yet measured"
placeholder rather than drawn from the published column; `make_figures.py
--mode measured` additionally refuses to run at all if the CSV has no usable
rows.

Superseded rows are dropped from every average and kept in the file:

```
python tools/supersede.py --by C2 --where arm=B dataset=blender --dry-run
python tools/test_geometry_claims.py --json results/geometry/claims.json
```

`ITERS` selects which runs a table or figure averages, and defaults to 30000 —
the R1/R2 target. `make table1 ITERS=7000` reads the R0 smoke rows instead.
Without it, a 7 000-iteration smoke row and a 30 000-iteration R1 row would be
averaged into a number that describes neither.

## Licence

3DGS and Mip-Splatting are under the Gaussian-Splatting License (Inria/MPII):
research and evaluation only, redistribution must carry the licence and retain
notices, no commercial use. See `LICENSE.md`. Nothing in this repository is
published for redistribution beyond that licence.
