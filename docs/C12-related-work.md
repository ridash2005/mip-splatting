# C12 — Literature due-diligence

**Question this task exists to answer.** Has anyone already made 3DGS's *3D* smoothing filter
anisotropic or per-direction? If so, the contribution of §4 has to narrow, and it is much cheaper
to learn that before the stress suite runs than after.

**Answer: partly, and the contribution must narrow — but it does not collapse.**
One 2025 paper (AAA-Gaussians) already makes the 3D filter a full 3×3. It derives the anisotropy
from the *current view ray*, recomputed per rendered frame. Nobody derives it from the
*conditioning of the training capture*, which is what §4 does. Details and the exact line of
separation are below.

Three lines per paper: the claim, the mechanism, the experiment that convinced me.

---

## The one that forces a change — AAA-Gaussians

**AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering**, arXiv:2504.12811 (2025).
<https://arxiv.org/abs/2504.12811>

- **Claim.** 3DGS's aliasing and its popping artifacts are both consequences of doing the filtering
  in screen space; replacing the 2D Mip filter with a fully 3D formulation fixes both, and beats
  Mip-Splatting out of distribution.
- **Mechanism.** An *adaptive* 3D smoothing filter that is a full 3×3 covariance, not a scalar σ.
  The effective sampling rate is `v̂′ = min(v̂_train, v̂)` (their Eq. 13) — the training-set rate
  clamped by the rate of the view being rendered. The anisotropy comes from decomposing the
  covariance about the ray: with `d` the normalised vector from the primitive mean `μ` to the
  camera origin `o`, and `Σ_⊥` the 2×2 covariance projected onto the subspace orthogonal to `d`,
  the opacity renormalisation is `√(|Σ_⊥| dᵀΣ⁻¹d / |Σ̂_⊥| dᵀΣ̂⁻¹d)` (their Eq. 10), which
  accounts only for the scale change *perpendicular to the ray*.
- **What convinced me.** Their §3.2 states the replacement of the 2D Mip filter outright, and Eq. 13
  makes the per-view dependence explicit — `v̂` is the rate of the *current* camera, so the filter
  cannot be a quantity stored once per primitive.

### The line of separation, stated precisely

| | AAA-Gaussians | This thesis (B1) |
|---|---|---|
| filter shape | full 3×3 | full 3×3 |
| **what sets the anisotropy** | **the current view ray `d`** | **the spectrum of Λ_k over the training cameras** |
| **when it is computed** | **every rendered view** | **once per recomputation, stored per primitive** |
| quantity it band-limits by | sampling rate of the view being rendered | conditioning of the capture that was actually available |
| degenerate case | isotropic when the projection is symmetric about the ray | isotropic ⟺ Λ_k isotropic (Proposition 2) |

Both make the 3D filter anisotropic. They answer *different questions*: AAA-Gaussians asks "how
finely can I resolve this primitive **from where I am standing now**", B1 asks "how finely was this
primitive **ever constrained by the cameras that saw it**". A primitive seen from a narrow baseline
is under-determined along depth *no matter which view renders it*; AAA-Gaussians' filter does not
see that, because at render time `d` carries no memory of the capture's spread. That is exactly the
regime the `cone` and `arc` protocols isolate.

**Consequence for the write-up (actioned):** §1 and §4 may no longer claim "an anisotropic 3D
filter" as the contribution. The contribution is *the source of the anisotropy* — a per-primitive
band-limit read off the capture geometry's conditioning — and AAA-Gaussians must be cited as
concurrent prior art for the anisotropic filter itself. §7 should also note that the two are
composable rather than competing: B1 sets the floor from the capture, AAA-Gaussians clamps it by
the view.

---

## Fisher information in 3DGS — related use, different object

**FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information**,
ECCV 2024 (oral). <https://jiangwenpl.github.io/FisherRF/>

- **Claim.** Fisher information quantifies how much a candidate view would tell you about the
  radiance field's parameters, so the next-best-view problem is solved by maximising Expected
  Information Gain rather than by a proxy uncertainty head.
- **Mechanism.** Fisher information over the *model parameters*, used as an acquisition function to
  **choose which camera to add next**. With a 3DGS backend it selects views at ~70 fps.
- **What convinced me.** The task is active view selection and active mapping; the information
  matrix scores *candidate cameras*, never a primitive's covariance. Nothing in the rendering path
  changes — the trained model is ordinary 3DGS.

**Difference from B1.** Same statistical object, opposite direction of use. FisherRF asks which
camera to *acquire*; B1 takes the cameras as given and asks what the ones you already have leave
unresolved. FisherRF changes the dataset, B1 changes the primitive.

---

**PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting**, CVPR 2025.
<https://arxiv.org/abs/2406.10219>

- **Claim.** A second-order sensitivity score prunes 90 % of primitives with less fidelity loss than
  magnitude or opacity heuristics, giving 3.56× faster rendering.
- **Mechanism.** A second-order (Hessian/Fisher) approximation of the training-view reconstruction
  error with respect to each Gaussian's **spatial parameters — mean position and scale**, reduced to
  a scalar sensitivity score, applied in a multi-round prune-refine loop on any pretrained model.
- **What convinced me.** The score is a scalar per primitive and its only consumer is the pruning
  decision; the covariance is never modified by it. The pipeline is explicitly post-hoc and leaves
  training unchanged.

**Difference from B1 — and this is the closest call in the chapter.** PUP's per-primitive matrix is
built from the same ingredients as Λ_k: second-order photometric sensitivity with respect to
spatial parameters, accumulated over training views. The divergence is what happens to it. PUP
**collapses the matrix to a scalar and thresholds it** — the eigen*structure* is discarded, which is
precisely the information B1 keeps. B1 never prunes; it uses the eigenvectors to shape a filter.
So PUP is the right paper to cite for "this matrix is a meaningful per-primitive quantity", and the
honest statement of novelty is *retaining its spectrum instead of its trace/determinant*. §7 should
report the overlap explicitly rather than let a reader find it.

---

## Density control and anti-aliasing — background, no collision

**3D Gaussian Splatting as Markov Chain Monte Carlo**, NeurIPS 2024 (spotlight).
<https://arxiv.org/abs/2404.09591>

- **Claim.** Densify/prune/reset heuristics are unnecessary; treating the primitive set as MCMC
  samples removes them and also removes the dependence on point-cloud initialisation.
- **Mechanism.** SGLD updates (gradient + noise), with cloning rewritten as a relocation that
  approximately preserves sample probability.
- **What convinced me.** The contribution is entirely in *how many primitives there are and where
  they go*. No filter, no band-limit, no covariance floor.

**Difference from B1.** Orthogonal, and composable. Nearest principled density-control work, cited
as such.

---

**Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration**, ECCV 2024.
<https://arxiv.org/abs/2403.11056>

- **Claim.** Approximating the pixel window as a 2D Gaussian low-pass (Mip-Splatting) over-suppresses
  high frequencies; integrating the Gaussian response over the pixel window analytically keeps the
  detail.
- **Mechanism.** A closed-form approximation to the integral of the 2D Gaussian over the square pixel
  footprint, replacing the screen-space convolution.
- **What convinced me.** The whole method is a 2D/screen-space pixel-response argument; the 3D
  representation is untouched.

**Difference from B1.** Improves the *2D* half of Mip-Splatting. B1 changes the *3D* half. Directly
composable, and worth stating as such.

---

**Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering** (CVPR 2024) — the multi-scale
baseline whose STMT/MTMT protocol this reproduction already uses. Trains a per-scale set of
primitives and picks by scale at render time; no per-primitive band-limit, and its cost grows with
the number of scales. Cited in §2 as the alternative to filtering rather than a competitor to B1.

---

## Verdict

**No stop.** No paper computes a per-primitive anisotropic band-limit *from the capture geometry's
conditioning*. The claim survives.

**Two narrowings, both mandatory:**

1. **Novelty of "anisotropic 3D filter" is gone** (AAA-Gaussians, April 2025). What remains is the
   *source* of the anisotropy — capture conditioning, fixed per primitive — versus their per-view
   ray decomposition. §1/§4 reworded; the two are complementary and that should be said.
2. **The matrix itself is not new** (PUP 3D-GS). Second-order photometric sensitivity in the spatial
   parameters, accumulated over training views, is exactly their pruning score's ingredient. B1's
   novelty is *keeping the spectrum* where PUP keeps a scalar. Say so before a reviewer does.

Neither narrowing touches the experimental design: the stress suite still tests a claim nobody has
tested, because nobody else's filter is a function of the training-camera spread.
