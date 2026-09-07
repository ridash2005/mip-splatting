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
  (`--disable_3D_filter`) and `--kernel_size 0.3`. See §3 of the prompt for why
  this is exactly 3DGS, not an approximation, and why the original repo cannot
  read the multi-scale Blender format at all.

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
KAGGLE_USERNAME = rickaryadas
HF_USERNAME     = rickaryadas
```

## Environment

**Local dev machine**: NVIDIA GeForce MX350, 2 GB VRAM, driver reports compute
capability below the repo's minimum. **Fails F15 (Compute Capability 7.0+
required) — do not attempt training here.** This machine is used only for git
operations, tooling, and orchestrating remote runs.

**Training target**: Kaggle, 2×T4 (CC 7.5), accelerator confirmed against
`torch.cuda.get_device_capability()` in-session (§6 check 1) before any run.
Kaggle quota (30 h/week, 12 h/session — secondary-source figures, to be read
off the notebook sidebar and confirmed in session 1 per §8.1) is not yet
verified from a live session.

## Status

- [x] Forked upstream, cloned, remotes set up.
- [x] F1/F3/F4/F5 spot-checked against source at the forked commit — match the
      prompt exactly (see commit history on `main`).
- [x] Arm B diff applied verbatim on `arm-b-3dgs-baseline`, diff printed in its
      commit.
- [x] Tooling scaffolded: `results/runs.csv` header, `tools/split_by_scale.py`,
      `tools/make_table.py`, `tools/make_figures.py`, `Makefile`.
- [ ] §6 pre-flight (12 checks) — blocked on a Kaggle GPU session; not runnable
      on the local machine (fails check 1).
- [ ] R0 smoke run.
- [ ] R1 (primary target), R2, R3, R5, R4 (optional).

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
make table1     # R1 Blender STMT: 3DGS vs Mip-Splatting, 4 test scales
make table2     # R2 Blender MTMT
make figures    # figures/, redrawn from measured rows with the published curve
                # kept as a faint reference line
```

## Licence

3DGS and Mip-Splatting are under the Gaussian-Splatting License (Inria/MPII):
research and evaluation only, redistribution must carry the licence and retain
notices, no commercial use. See `LICENSE.md`. Nothing in this repository is
published for redistribution beyond that licence.
