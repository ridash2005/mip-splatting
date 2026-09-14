# Claude Code — completion prompt (C-series, after the 8 Sep audit)

**How to use.** Open Claude Code in `D:\BTP\repro` and paste everything below the rule as the first message.
This is the *second* prompt of the project. The first (`docs/REPRODUCTION-PROMPT.md`) got the baselines
running; this one fixes what the audit found and finishes the science.

**Do not re-run anything the audit marked valid.** Arm A (Mip-Splatting) reproduces to within 0.26 dB and
its 64 R1/R2 rows are keepers. R5's seed spread is done. R0 is done.

---

## CONTEXT — where the project actually stands

184 measured rows in `results/runs.csv`. Both baselines run end to end on 2×T4. B1 is implemented in
`scene/fisher_filter.py` and routes correctly through `cov3D_precomp`. Two defects block submission:

**BLOCKER 1 — the 3DGS arm is not 3DGS.** `gaussian_renderer/__init__.py` passes `kernel_size` into
Mip-Splatting's `GaussianRasterizationSettings`, so the arm keeps the 2D Mip filter's *opacity
compensation* ρ = √(det Σ′/det(Σ′+kI)). Vanilla 3DGS adds a fixed `0.3f` dilation with **no** compensation.
Setting `kernel_size=0.3` reproduces the dilation magnitude but not the absence of compensation.

Measured consequence (Blender STMT, 8 scenes, mean PSNR):

| scale | 3DGS ours | 3DGS published | Δ | gap ours | gap published |
|---|---|---|---|---|---|
| 1× | 33.31 | 33.33 | −0.02 ✅ | −0.17 | 0.03 |
| 1/2 | 31.63 | 26.95 | **+4.68** ❌ | 2.21 | 7.05 |
| 1/4 | 27.80 | 21.38 | **+6.42** ❌ | 3.97 | 10.47 |
| 1/8 | 24.54 | 17.69 | **+6.85** ❌ | **4.05** | **10.98** |

Full-res is right to 0.02 dB, so data, split, per-scale demux, metric and optimiser are all correct. Only
the scale-degradation is wrong, in the direction of too much anti-aliasing. **Gate G2 required >9 dB and got
4.05 dB. It failed and the ladder continued anyway.** Every "3DGS" comparison is provisional until fixed.

**BLOCKER 2 — B1 has only been run where Proposition 2 forbids it from differing.** All three B1 runs
(`floor_diag`, `t2_b1/lego`, `t2_b1/chair`) used `protocol: full` (span ≈179°, 100 cameras) and all report
`frac_above_floor = 0.0`, `mean_anisotropy = 1.0`. That means Σ_filt = f_k²I identically — **"B1" in those
runs is Mip-Splatting under another name.** This is a *successful validation of Proposition 2*, and it is
not a test of the claim. The stress protocols were never instantiated.

## HARD RULES

1. **No fabricated numbers.** A results cell is measured by a logged run or it is empty.
2. **Keep valid work.** Do not re-run arm A, R5, or R0.
3. **Free tier only.** 2×T4, 16 GB, ~30 h/week, 12 h session cap. Checkpoint and resume on every run.
4. **Every row lands in `results/runs.csv`** with commit, flags, VRAM, seed and `level_claimed`.
5. **Report and wait** at the exit criterion of C2 and C5.

---

## C1 — Fix the 3DGS arm  ·  0 GPU-h

Pick **one**:

- **(a) Preferred — build the stock rasteriser alongside.** Clone
  `graphdeco-inria/diff-gaussian-rasterization` at the paper-era commit into
  `submodules/diff-gaussian-rasterization-vanilla`, install it under a distinct module name, and have
  `gaussian_renderer.render` dispatch to it when `disable_3D_filter` is set. Cleanest: the 3DGS arm then
  runs the actual 3DGS rasteriser, and the claim "we reproduce 3DGS" is literally true.
- **(b) Add a switch.** Thread `disable_2D_mip_compensation` through `GaussianRasterizationSettings` into
  the CUDA forward pass, and when set, skip the ρ opacity-compensation term while keeping the dilation.
  Requires a CUDA edit and a rebuild; record the diff on its own branch.

**Verification before spending any quota on the full sweep** — one scene, `lego`, 30 K:

```
arm B full-res PSNR ≈ 33.3   (must stay)
arm B 1/8   PSNR ≈ 17.7      (must fall — currently 24.5)
```

If 1/8 does not fall to ≈18, the compensation is still active. Do not proceed. Print the actual opacity
tensor statistics with and without the switch to confirm the term is gone.

## C2 — Re-run R1 and R2, arm B only  ·  6–8 GPU-h

8 Blender scenes × {STMT, MTMT} × arm B. Arm A is untouched. Use the existing
`kaggle/blender_rung.py` path and the existing splitter.

**Exit (gate G2): the ⅛-scale gap exceeds 9 dB.** Published is 10.98. Report the full table against
published and **wait for my go-ahead.** Mark the superseded arm-B rows in `runs.csv` as
`notes=SUPERSEDED_BY_C2` rather than deleting them — the audit trail matters.

## C3 — Build the stress protocols  ·  0 GPU-h

`tools/camera_protocols.py` already implements `full`, `arc`, `cone`, `grazing`. Instantiate them for at
least `lego` and `chair`, and **write `mixed_focal`** (a random half of the training views downsampled
2–4×, seeded). Version each as a JSON index list with its measured `span_deg` and camera count.

Target conditioning, from Equation 4.8 (λ₃/λ₁ = sin²(θ/2)):

| protocol | target span θ | predicted λ₃/λ₁ | predicted σ_depth/σ_lat |
|---|---|---|---|
| `full` (control) | ~179° | ≈ 1.0 | 1.0 |
| `arc` | ~20° | ≈ 0.03 | 5.8× |
| `cone` | ~6° | ≈ 0.003 | 19× |
| `grazing` | — | — | anisotropy along the surface normal |
| `mixed_focal` | ~179° | ≈ 1.0 | tests the mixed-camera defect, not parallax |

Sanity-check each built protocol by printing its actual angular span; if `cone` comes out at 40° it is not
a cone and the prediction will not apply.

## C4 — Run I-2 across protocols  ·  1–2 GPU-h

Compute λ₃/λ₁ per primitive on **trained, unmodified Mip-Splatting checkpoints** — no retraining, no method
code, pure PyTorch under `no_grad`. One histogram per (scene, protocol).

**Exit (gate G4):** λ₃/λ₁ ≈ 1 on `full` and ≪ 1 on `cone`/`arc`, in the predicted ordering. If Λ comes out
isotropic on the narrow cone, that contradicts a closed form verified to six figures — suspect the
visibility weighting `w_nk`, not the geometry, and diagnose before proceeding.

**Exclude primitives that no camera sees.** `compute_3D_filter` assigns them `distance[valid].max()`; left
in, they form a spurious right tail that reads as the floor binding.

## C5 — THE DECISIVE EXPERIMENT  ·  8–12 GPU-h

B1 **and** Mip-Splatting, on `cone`, `arc`, `grazing`, `mixed_focal`, ≥2 scenes, 30 K iterations. Report per
protocol:

- PSNR / SSIM / LPIPS(VGG) at 1×, ½, ¼, ⅛
- `frac_above_floor` and `mean_anisotropy` — **these are as important as the PSNR**
- Δ(B1 − Mip-Splatting) in dB **and in units of σ = 0.039 dB**

**Exit (gate G3):**

- `frac_above_floor > 0` on at least the `cone` protocol — otherwise the estimation term never binds
  anywhere and the claim must narrow to the mixed-focal case or pivot to B2.
- Δ ordered as `cone` > `arc` > `full` ≈ 0, each gain exceeding **3σ = 0.12 dB**.
- `full` must show **no loss** — Proposition 2 forbids it. A loss there is a bug, not a result.

**Report and wait.** This single table is the thesis's central empirical claim.

## C6 — Unify the results table  ·  0 GPU-h

B1 rows currently live only in `results/kaggle_runs/t2_b1/summary.json` and never reach
`results/runs.csv`. Merge them, add `arm=B1` and the protocol as columns, and make every thesis table and
figure regenerate from the CSV alone. `make_figures.py --mode measured` must refuse to draw if the CSV
lacks usable rows — verify that guard still fires.

## C7 — B1 seed spread  ·  3–4 GPU-h

3 seeds on one stress protocol (`cone`), one scene. Gives σ for the method so the gain in C5 is reported as
a multiple of its own noise, not the baseline's.

## C8 — B2 evaluation  ·  4–6 GPU-h

Run `kaggle/b2_eval.py`. Produce Table 4: full degree-3 baseline, magnitude pruning at matched model size,
and identifiability masking. Report PSNR, model MB, fraction of non-DC SH retained, and mean ℓ_max.

**Normalise the Gram matrix before thresholding**: use `Ā_k = A_k / Σ_n w_nk` so τ is scale-free across
primitives with different view counts. Without this, τ means different things for different primitives.

## C9 — Matched fps  ·  <1 GPU-h

Current `render_fps` of 8–10 is not comparable to the published 180–300 and almost certainly includes I/O
and per-scale rendering. Measure 3DGS, Mip-Splatting and B1 **in one run, same scene, same GPU, render loop
only**, and report the *ratio*. The thesis claims rendering is unchanged relative to baseline; that is the
quantity to defend.

## C10 — Complete R3  ·  8–10 GPU-h

Tanks & Temples (`truck`, `train`) and Deep Blending (`drjohnson`, `playroom`), both arms, settings per
`full_eval.py`. Reference: DB 29.41 / 0.903 / 0.243; T&T 23.14 / 0.841 / 0.183. On OOM, record it and move
on — do not reduce resolution to make a scene fit.

## C11 — Apply the thesis corrections  ·  0 GPU-h

From the audit, §05. All zero-GPU; do them while runs are queued.

- **M-1** Define `w_nk = T_nk·α_k ∈ [0,1]` and state that the weighting is a heuristic down-weighting of
  occluded views, so Λ is an *observability matrix* coinciding with the Fisher information at unit weights.
- **M-2** Fix `σ_pix = 0.447 px` by inheritance from Mip-Splatting; `s` is the dimensionless confidence
  multiplier, default 1. Removes the over-determination between "s default 1" and `s·σ_pix = √0.2`.
- **M-3** Present Λ as the Gauss–Newton approximation to the Hessian of the photometric cost under a
  point-feature approximation; keep Cramér–Rao as motivation, not as the claim.
- **M-4** **New subsection.** The rank-1 shortcut `(f/z)²(I − ddᵀ)` is exact *only on-axis*. Off-axis the two
  non-zero eigenvalues differ — measured λ₂/λ₁ = 0.512, 0.682, 0.956, 0.997 over four random poses. So
  perspective contributes anisotropy independently of parallax, and a scalar filter discards both. Reproduce
  those four numbers with a short script and cite it.
- **M-5** Replace Appendix B.2's descriptive proof with the exact spectrum. Two views aimed at the primitive
  give Λ ∝ 2I − d₁d₁ᵀ − d₂d₂ᵀ, diagonal with eigenvalues **2, 2cos²(θ/2), 2sin²(θ/2)** — so
  λ₃/λ₁ = sin²(θ/2) in two lines. **State the middle eigenvalue**: the in-plane lateral direction is also
  degraded, so the filter is triaxial, not merely elongated in depth. Verified to 6×10⁻¹¹ at
  θ = 2°, 6°, 15°, 45°, 90°, 120°.
- **M-6** Add a one-line lemma: α ← α√(|Σ|/|Σ+Σ_filt|) preserves total mass for *any* positive-definite
  Σ_filt, so Mip-Splatting's compensation carries over to the anisotropic case unchanged. Verified to machine
  precision.
- **M-7** Normalise A_k before thresholding (see C8).
- **M-8** State that an anisotropic Σ_filt does not commute with Σ_k, so the per-axis √(s²+f²) update is
  unavailable and the full 3×3 must go to the rasteriser — as `gaussian_renderer` already does via
  `cov3D_precomp`. Also state that Λ is computed under `no_grad` and held constant between recomputations,
  which is what removes any question of differentiating an eigendecomposition.

## C12 — Literature due-diligence  ·  0 GPU-h  ·  OVERDUE

Still no evidence of this in the repository, and it is the only task that can invalidate the project.
Determine precisely whether anyone has already made the 3D filter anisotropic or per-direction. Priority:

- **FisherRF** (ECCV 2024) and **PUP 3D-GS** (CVPR 2025) — both use Fisher information in 3DGS. Establish
  exactly what they use it *for* (view selection? pruning?) and how that differs from a per-primitive
  band-limit. This is the highest-value comparison in the whole related-work chapter.
- **3DGS-MCMC** (NeurIPS 2024) — nearest principled density-control work.
- **Analytic-Splatting**, **Multi-Scale 3DGS**, **AAA-Gaussians**, and any 2025–26 successor to
  Mip-Splatting.

Write three lines per paper: the claim, the mechanism, the experiment that convinced you. If any of them
already computes a per-primitive anisotropic band-limit from capture geometry, **stop and report
immediately** — the response is to narrow the contribution, and it is far cheaper to learn now.

---

## ORDER OF EXECUTION

```
today, no GPU:      C12 (start), C11, C3, C1
then, critical path: C2  -> report, wait -> C4 -> C5 -> report, wait
then:                C6, C7, C9
then, if quota:      C8, C10
```

C1–C5 are the critical path, ~15–22 GPU-h. Total ≈30–43 GPU-h, one to one-and-a-half weeks of quota.

## REPORTING

After C2 and C5, and immediately on any failure: what ran (arm, scene, protocol, seed, commit, GPU); the
table with `level_claimed` per row; comparison to published with deltas and whether the tolerance is met;
`frac_above_floor` and `mean_anisotropy` wherever B1 is involved; measured GPU-hours and remaining quota;
and the one decision you need from me, or "none". Keep it short.

## PROHIBITIONS

- Do not report `t2_b1`'s current numbers as a method result — they are the baseline under another name.
- Do not delete superseded rows; mark them.
- Do not reduce resolution, iterations or scene count to make something fit or look better.
- Do not let LPIPS default to AlexNet; VGG is pinned and is a CSV column.
- Do not run on a P100 — the compute-capability guard is correct, leave it in.
- Do not claim G2 or G3 passed without the numbers in this document's exit criteria.
