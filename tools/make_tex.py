#!/usr/bin/env python3
"""
Generate every measured table and every inline number in the thesis, from
results/runs.csv and the rung summaries.

The thesis claims that no number in it is typed by hand. This is the file that
makes that true: it writes thesis/generated/*.tex, and main.tex \\input{}s them.
Nothing under thesis/ that a human edits contains a measurement.

    python tools/make_tex.py --runs results/runs.csv --out thesis/generated

Discipline, matching tools/make_figures.py:
  * A table with no measured rows is emitted as an explicit "not yet measured"
    placeholder carrying the published targets and NOTHING in the measured
    columns. It is never quietly filled from the published column.
  * Every macro this writes is defined whether or not the run exists, so the
    thesis always compiles; an unmeasured macro expands to a visible marker
    rather than to a plausible-looking number.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCALES = ["1x", "1/2", "1/4", "1/8"]
SCALE_TEX = {"1x": r"full", "1/2": r"$\tfrac{1}{2}$", "1/4": r"$\tfrac{1}{4}$",
             "1/8": r"$\tfrac{1}{8}$"}
ARM_NAME = {"A": "Mip-Splatting", "B": "3DGS"}

# Mip-Splatting (Yu et al., CVPR 2024), Blender, mean PSNR over 8 scenes.
# Cited as targets. Never used as a measurement, never copied into a result cell.
PUBLISHED = {
    "STMT": {"B": [33.33, 26.95, 21.38, 17.69], "A": [33.36, 34.00, 31.85, 28.67]},
    "MTMT": {"B": [28.79, 30.66, 31.64, 27.98], "A": [32.81, 34.49, 35.45, 35.50]},
}
NOT_MEASURED = r"\notmeasured"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_geometry_claims import cap_ratio  # noqa: E402  the verified closed form


def cap_sigma_ratio(span_deg):
    """sigma_D/sigma_L predicted for cameras spread over a cap of this span.

    The two-view form 1/sin(theta/2) is exact only for two views; a capture
    protocol is a distribution over a cap, and for that the two-view form
    understates the depth anisotropy by sqrt(2) in sigma. Every prediction
    quoted against a measured span therefore uses this, not Equation (4.8).
    """
    r = cap_ratio(math.radians(span_deg) / 2.0)
    return 1.0 / math.sqrt(max(r, 1e-12))


def load(runs_path):
    """Every measured row, minus the ones a later stage has replaced.

    results/runs.csv is append-only, so a re-measured configuration leaves its
    predecessor in the file with SUPERSEDED in `notes` (tools/supersede.py).
    Dropping those here is what keeps a superseded run on the record without
    letting it into an average.
    """
    if not os.path.exists(runs_path):
        return []
    with open(runs_path, newline="") as f:
        return [r for r in csv.DictReader(f)
                if r.get("psnr") and "SUPERSEDED" not in (r.get("notes") or "")]


def select(rows, *, iterations, load_allres, dataset="blender", seed=None,
           train_scale=("1x", "multi")):
    """Rows of the STANDARD benchmark, unless train_scale says otherwise.

    train_scale is the filter that keeps the reproduction tables free of the
    stress suite. method_eval writes arm "A" for Mip-Splatting exactly as
    blender_rung does, and tags the protocol in train_scale ("1x/cone"), so
    without this a mip row trained on ten cameras is averaged into Table 1 --
    which pulled arm A's full-resolution figure down by 2.4 dB the first time
    the stress rows were merged. Pass None to take every protocol.
    """
    out = []
    for r in rows:
        if r.get("dataset") != dataset:
            continue
        if r.get("iterations") != str(iterations):
            continue
        if r.get("load_allres") != load_allres:
            continue
        if seed is not None and r.get("seed") != str(seed):
            continue
        if train_scale is not None and r.get("train_scale") not in train_scale:
            continue
        out.append(r)
    return out


def by_arm_scale(rows):
    """{arm: {scale: [row, ...]}} — one row per scene."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("arm") in ARM_NAME:
            acc[r["arm"]][r["test_scale"]].append(r)
    return acc


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def fmt(v, spec="{:.2f}"):
    return spec.format(v) if isinstance(v, (int, float)) else NOT_MEASURED


# --------------------------------------------------------------- main tables
def placeholder(label, caption, note=None):
    """A labelled, visibly-empty table.

    Emitted instead of a bare \emph{} so that \ref{} to it still resolves: a
    chapter may legitimately reference a table whose run has not happened yet,
    and an undefined reference would be a build error rather than an honest
    "not measured".
    """
    return "\n".join([
        r"\begin{table}[htbp]", r"  \centering",
        r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
        r"  \small",
        r"  \begin{tabular}{c}", r"    \toprule",
        r"    \textit{not yet measured} \\", r"    \bottomrule",
        r"  \end{tabular}",
        r"  \par\vspace{2pt}\footnotesize "
        + (note or r"This run has not been performed. The cell is empty by design; "
                   r"see Chapter~\ref{ch:results}."),
        r"\end{table}", ""])


def matched(rows, arms=("A", "B")):
    """Restrict to the scenes every listed arm has, and say which they are.

    A gap column between a six-scene arm and an eight-scene one is not a gap; it
    is the difference between two sets of objects. Runs do fail -- C2 lost two
    scenes to a CUDA fault -- so this has to be enforced rather than assumed.
    """
    per = {}
    for r in rows:
        if r.get("arm") in arms:
            per.setdefault(r["arm"], set()).add(r["scene"])
    present = [a for a in arms if per.get(a)]
    if not present:
        return [], set()
    common = set.intersection(*(per[a] for a in present))
    return [r for r in rows if r.get("scene") in common], common


def scale_table(rows, protocol, label, caption):
    """The headline table: published target beside our measurement, per scale."""
    rows, common = matched(rows)
    acc = by_arm_scale(rows)
    scenes = sorted(common)
    pub = PUBLISHED[protocol]
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small",
         r"  \begin{tabular}{lcccccc}", r"    \toprule",
         r"    & \multicolumn{2}{c}{3DGS (arm B)} & \multicolumn{2}{c}{Mip-Splatting (arm A)}"
         r" & \multicolumn{2}{c}{gap, ours} \\",
         r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"    test scale & published & \textbf{ours} & published & \textbf{ours}"
         r" & published & \textbf{ours} \\",
         r"    \midrule"]
    for i, sc in enumerate(SCALES):
        cells = [SCALE_TEX[sc]]
        got = {}
        for arm in ("B", "A"):
            p = pub[arm][i]
            vals = [float(r["psnr"]) for r in acc.get(arm, {}).get(sc, [])]
            m = mean(vals)
            got[arm] = m
            cells += [f"{p:.2f}", (r"\textbf{" + f"{m:.2f}" + "}") if m else NOT_MEASURED]
        gap_pub = pub["A"][i] - pub["B"][i]
        gap_ours = (got["A"] - got["B"]) if (got["A"] and got["B"]) else None
        cells += [f"{gap_pub:+.2f}", (r"\textbf{" + f"{gap_ours:+.2f}" + "}")
                  if gap_ours is not None else NOT_MEASURED]
        L.append("    " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}"]
    if scenes:
        L.append(r"  \par\vspace{2pt}\footnotesize Mean over " + str(len(scenes))
                 + r" scene(s): \texttt{" + ", ".join(scenes) + r"}. PSNR in dB.")
    else:
        L.append(r"  \par\vspace{2pt}\footnotesize \emph{Not yet measured.} "
                 r"Published targets shown so the table states what it is aiming at; "
                 r"the measured columns are deliberately empty.")
    L += [r"\end{table}", ""]
    return "\n".join(L)


def per_scene_table(rows, protocol, label, caption):
    """Every scene, every scale, both arms — the appendix table."""
    acc = defaultdict(dict)
    for r in rows:
        acc[(r["scene"], r["arm"])][r["test_scale"]] = r
    scenes = sorted({s for s, _ in acc})
    if not scenes:
        return placeholder(label, caption)
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \footnotesize",
         r"  \begin{tabular}{ll" + "r" * 4 + "rr}", r"    \toprule",
         r"    scene & arm & " + " & ".join(SCALE_TEX[s] for s in SCALES)
         + r" & \#Gauss & MB \\", r"    \midrule"]
    for scene in scenes:
        for arm in ("B", "A"):
            d = acc.get((scene, arm), {})
            if not d:
                continue
            any_row = next(iter(d.values()))
            cells = [scene.replace("_", r"\_") if arm == "B" else "",
                     ARM_NAME[arm]]
            cells += [fmt(float(d[s]["psnr"]) if s in d else None) for s in SCALES]
            cells += [f"{int(any_row['n_gaussians']):,}".replace(",", r"\,")
                      if any_row.get("n_gaussians") else NOT_MEASURED,
                      fmt(float(any_row["model_mb"]) if any_row.get("model_mb") else None,
                          "{:.1f}")]
            L.append("    " + " & ".join(cells) + r" \\")
        L.append(r"    \addlinespace[2pt]")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize PSNR in dB per test scale.",
          r"\end{table}", ""]
    return "\n".join(L)


def cost_table(rows, label, caption):
    """Cost and model size — measured on the same GPU in the same run."""
    acc = defaultdict(list)
    for r in rows:
        if r.get("test_scale") != "1x" or r.get("arm") not in ARM_NAME:
            continue
        acc[r["arm"]].append(r)
    if not acc:
        return placeholder(label, caption)
    fields = [("n_gaussians", "primitives", "{:,.0f}"),
              ("model_mb", "model (MB)", "{:.1f}"),
              ("train_seconds", "train (s)", "{:.0f}"),
              ("render_fps", "render (fps)", "{:.2f}"),
              ("peak_vram_mb", "peak VRAM (MB)", "{:,.0f}")]
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{lrr}", r"    \toprule",
         r"    & 3DGS (arm B) & Mip-Splatting (arm A) \\", r"    \midrule"]
    for key, name, spec in fields:
        cells = [name]
        for arm in ("B", "A"):
            vals = [float(r[key]) for r in acc.get(arm, []) if r.get(key)]
            m = mean(vals)
            cells.append(spec.format(m).replace(",", r"\,") if m is not None
                         else NOT_MEASURED)
        L.append("    " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Mean over scenes; same GPU, same run. "
          r"Training is 30\,000 iterations unless stated.",
          r"\end{table}", ""]
    return "\n".join(L)


def seed_table(rows, label, caption):
    """I-3: per-scene mean and standard deviation across seeds."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["scene"], r["arm"], r["test_scale"])].append(float(r["psnr"]))
    multi = {k: v for k, v in acc.items() if len(v) > 1}
    if not multi:
        return placeholder(label, caption, "No configuration has been repeated "
                           "yet, so no spread can be quoted.")
    scenes = sorted({s for s, _, _ in multi})
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{lll" + "c" * 4 + "}", r"    \toprule",
         r"    scene & arm & $n$ & " + " & ".join(SCALE_TEX[s] for s in SCALES)
         + r" \\", r"    \midrule"]
    worst = 0.0
    for scene in scenes:
        for arm in ("B", "A"):
            cells = [scene.replace("_", r"\_") if arm == "B" else "", ARM_NAME[arm]]
            ns, body = [], []
            for s in SCALES:
                v = multi.get((scene, arm, s))
                if not v:
                    body.append(NOT_MEASURED)
                    continue
                m = mean(v)
                sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
                worst = max(worst, sd)
                ns.append(len(v))
                body.append(f"{m:.2f}\\,$\\pm$\\,{sd:.3f}")
            cells.append(str(max(ns) if ns else 0))
            L.append("    " + " & ".join(cells + body) + r" \\")
        L.append(r"    \addlinespace[2pt]")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Mean\,$\pm$\,sample standard deviation "
          r"in dB over seeds. Largest $\sigma$ observed: "
          + f"{worst:.3f}" + r"\,dB.", r"\end{table}", ""]
    return "\n".join(L)


def preflight_table(summary, label, caption):
    """The §6 twelve checks, from the R0 kernel's own summary."""
    if not summary:
        return placeholder(label, caption, "The rung has not been run.")
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \footnotesize", r"  \begin{tabular}{clp{0.62\linewidth}}", r"    \toprule",
         r"    \# & result & check \\", r"    \midrule"]
    for c in summary.get("checks", []):
        tag = {True: r"\pass", False: r"\fail", None: r"\skipped"}[c["passed"]]
        desc = tex_escape(c["desc"])
        L.append(f"    {c['id']} & {tag} & {desc} " + r"\\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Emitted by the run itself "
          r"(\texttt{results/kaggle\_runs/*/summary.json}); not transcribed.",
          r"\end{table}", ""]
    return "\n".join(L)


def tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return (s.replace("≥", r"$\geq$").replace("≡", r"$\equiv$").replace("±", r"$\pm$")
             .replace("→", r"$\rightarrow$").replace("§", r"\S"))



PROTO_LABEL = {"full": "full orbit", "arc": "one-sided arc",
               "cone": "low-parallax cone", "pencil": "narrow pencil",
               "mixed": "mixed focal", "grazing": "grazing"}
# Ordered by conditioning, best first. A protocol missing from this list is
# silently dropped from every table and figure, which is how `pencil` -- the one
# protocol that reaches the regime -- would have vanished from the results it was
# added to produce.
PROTO_ORDER = ["full", "grazing", "mixed", "arc", "cone", "pencil"]


def instruments_table(inst, label, caption):
    """I-1 and I-2 per capture protocol, from the instrument run's own JSON."""
    if not inst or not inst.get("by_protocol"):
        return placeholder(label, caption, "The instruments have not been run.")
    bp = inst["by_protocol"]
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small",
         r"  \begin{tabular}{lrrrrrr}", r"    \toprule",
         r"    & & \multicolumn{2}{c}{I-1: est.\ term $>$ floor}"
         r" & \multicolumn{2}{c}{I-2: $\sigma_D/\sigma_L$} & \\",
         r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"    protocol & span & worst dir & all dirs & measured & predicted"
         r" & est./floor \\", r"    \midrule"]
    for k in PROTO_ORDER:
        v = bp.get(k)
        if not v:
            continue
        L.append("    " + " & ".join([
            PROTO_LABEL[k],
            f"${v['median_span_deg']:.0f}^\\circ$",
            f"{v['median_frac_exceeding_floor_worst_dir'] * 100:.0f}\\%",
            f"{v['median_frac_exceeding_floor_all_dirs'] * 100:.0f}\\%",
            f"{v['median_sigma_ratio_measured']:.2f}$\\times$",
            f"{cap_sigma_ratio(v['median_span_deg']):.2f}$\\times$",
            f"{v['median_ratio_est_over_floor_p50']:.2f}",
        ]) + r" \\")
    n = max((v["n_scenes"] for v in bp.values()), default=0)
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Median over " + str(n) +
          r" Blender scenes. \emph{Span} is the maximum pairwise angle "
          r"subtended at the capture centre by the selected cameras. "
          r"\emph{est./floor} is the median of "
          r"$\sqrt{s^2/\lambda_{\min}}\,/\,f_k$: below one the floor binds and "
          r"Proposition~\ref{prop:reduction} applies; above one the estimation "
          r"term governs. \emph{predicted} is the cap form of "
          r"\S\ref{sec:capprediction} evaluated at each protocol's own measured "
          r"span --- not Equation~\eqref{eq:anisotropy}, which is exact for two "
          r"views and understates a distributed capture by $\sqrt{2}$.",
          r"\end{table}", ""]
    return "\n".join(L)



# Published references for the real scenes, from the 3DGS paper. Cited as
# context, never as the control: these are 30K single-seed numbers on a 24 GB
# card, and the subset here is not the set they average over.
REAL_REF = {"tandt": ("Tanks & Temples (2-scene avg)", 23.14, 0.841, 0.183),
            "db": ("Deep Blending (2-scene avg)", 29.41, 0.903, 0.243)}


def real_scene_table(rows, label, caption):
    """R3: one row per (scene, arm). No per-scale split -- one test resolution."""
    sel = [r for r in rows if r.get("dataset") in ("mipnerf360", "tandt", "db")]
    if not sel:
        return placeholder(label, caption, "The rung has not been run.")
    by = {}
    for r in sel:
        by[(r["dataset"], r["scene"], r["arm"])] = r
    scenes = sorted({(d, s) for d, s, _ in by})
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{llrrrrr}", r"    \toprule",
         r"    scene & arm & PSNR & SSIM & LPIPS & primitives & peak VRAM \\",
         r"    \midrule"]
    last_ds = None
    for ds, scene in scenes:
        if ds != last_ds:
            if last_ds is not None:
                L.append(r"    \addlinespace[2pt]")
            last_ds = ds
        for arm in ("B", "A"):
            r = by.get((ds, scene, arm))
            if not r:
                continue
            L.append("    " + " & ".join([
                scene.replace("_", r"\_") if arm == "B" else "",
                ARM_NAME[arm],
                f"{float(r['psnr']):.2f}",
                f"{float(r['ssim']):.4f}",
                f"{float(r['lpips']):.4f}",
                f"{int(r['n_gaussians']):,}".replace(",", r"\,")
                if r.get("n_gaussians") else NOT_MEASURED,
                f"{int(r['peak_vram_mb']):,}".replace(",", r"\,") + r"\,MB"
                if r.get("peak_vram_mb") else NOT_MEASURED,
            ]) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize 30\,000 iterations, every-8th test "
          r"split (F12), Mip-NeRF 360 indoor at \texttt{-i images\_2} and the rest "
          r"at their defaults (F16). L2: a subset average is not comparable to a "
          r"published all-scene average, and none is quoted against it.",
          r"\end{table}", ""]
    return "\n".join(L)



METHOD_LABEL_TEX = {"3dgs": "3DGS", "mip-splatting": "Mip-Splatting",
                    "b1-fisher": "B1 --- Fisher band-limit"}


def method_table(rows, inst, label, caption):
    """Table 2: the method against both baselines on the standard benchmark."""
    sel = [r for r in rows if r.get("dataset") == "blender"
           and r.get("iterations") == "30000" and r.get("load_allres") == "False"
           and r.get("seed") == "0"
           and r.get("train_scale", "").split("/")[0] == "1x"
           and (r.get("train_scale") == "1x" or r.get("train_scale").endswith("/full"))]
    order = ["3dgs", "mip-splatting", "b1-fisher"]
    # Matched scenes, or the table is not a comparison. B1 was measured on two
    # scenes before it was measured on eight; averaging its two against the
    # baselines' eight compares lego and chair to a different set of objects and
    # reads as a gain that is nothing but scene difficulty. Restrict every row to
    # the scenes all three methods have, and say how many that is.
    per_method_scenes = {}
    for r in sel:
        if r.get("method") in order:
            per_method_scenes.setdefault(r["method"], set()).add(r["scene"])
    present = [m for m in order if per_method_scenes.get(m)]
    if not present:
        return placeholder(label, caption, "The method has not been evaluated.")
    common = set.intersection(*(per_method_scenes[m] for m in present))
    if not common:
        return placeholder(label, caption,
                           "No scene has been measured under every method, so no "
                           "matched comparison exists yet.")
    acc = {}
    for r in sel:
        m = r.get("method")
        if m not in order or r["scene"] not in common:
            continue
        acc.setdefault(m, {}).setdefault(r["test_scale"], []).append(r)

    floor = (inst or {}).get("by_protocol", {}).get("full", {})
    floor_pct = floor.get("median_frac_exceeding_floor_worst_dir")

    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{lrrrrrr}", r"    \toprule",
         r"    method & full & $\tfrac12$ & $\tfrac14$ & $\tfrac18$ & MB"
         r" & floor-dominated \\", r"    \midrule"]
    for m in order:
        d = acc.get(m)
        if not d:
            continue
        cells = [METHOD_LABEL_TEX[m]]
        for sc in SCALES:
            v = [float(r["psnr"]) for r in d.get(sc, [])]
            cells.append(f"{mean(v):.2f}" if v else NOT_MEASURED)
        mb = [float(r["model_mb"]) for r in d.get("1x", []) if r.get("model_mb")]
        cells.append(f"{mean(mb):.1f}" if mb else NOT_MEASURED)
        if m == "b1-fisher" and floor_pct is not None:
            cells.append(f"{(1 - floor_pct) * 100:.0f}\\%")
        elif m == "mip-splatting":
            cells.append(r"100\% by construction")
        else:
            cells.append(r"n/a")
        L.append("    " + " & ".join(cells) + r" \\")
    # The paired quantity the chapter actually argues about.
    d_b1 = acc.get("b1-fisher", {})
    d_mip = acc.get("mip-splatting", {})
    if d_b1 and d_mip:
        L.append(r"    \midrule")
        cells = [r"$\Delta$ (B1 $-$ Mip-Splatting)"]
        for sc in SCALES:
            a = [float(r["psnr"]) for r in d_b1.get(sc, [])]
            b = [float(r["psnr"]) for r in d_mip.get(sc, [])]
            cells.append(f"{mean(a) - mean(b):+.3f}" if a and b else NOT_MEASURED)
        cells += ["", ""]
        L.append("    " + " & ".join(cells) + r" \\")

    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Mean PSNR over the "
          + str(len(common)) + r" scene(s) measured under \emph{every} method "
          r"(" + ", ".join(r"\texttt{" + s + "}" for s in sorted(common)) + r"), "
          r"30\,000 iterations, single-scale train. \emph{floor-dominated} is the "
          r"fraction of primitives where the Nyquist floor exceeds the estimation "
          r"term in every direction, measured by I-1 on the same capture; where it "
          r"is 100\,\% Proposition~\ref{prop:reduction} makes B1 and Mip-Splatting "
          r"identical.",
          r"\end{table}", ""]
    return "\n".join(L)


# Run-to-run noise of the harness, from R5 (§7.5). The stress-suite delta is
# reported as a multiple of it, because a gain smaller than the noise is not a
# gain. Overwritten from the measured repeat spread in main().
SIGMA_FALLBACK = 0.039

# Every column that makes two rows a different configuration rather than a
# repeat of the same one. Leaving any of these out silently merges unrelated
# runs and reports their difference as harness noise -- dropping load_allres
# alone pools STMT with MTMT and inflates a 0.039 dB spread to 2.4 dB.
CONFIG_KEY = ("dataset", "scene", "method", "arm", "train_scale", "test_scale",
              "iterations", "load_allres", "kernel_size", "seed")


def _run_spans(summaries_dir="results/kaggle_runs"):
    """Median realised span per protocol, from the runs that produced the PSNR.

    The instruments are a separate session and their protocol definitions can
    lag the runner's -- they did: `cone` was re-defined from a 20-degree wedge to
    a tenth of the cameras, which lands at 36-54 degrees, and the chapter was
    quoting conditioning measured on the first beside PSNR measured on the
    second. Taking the span from the session that trained the model makes the
    row self-consistent by construction.
    """
    out = {}
    for d, _, files in os.walk(summaries_dir):
        for fn in files:
            if fn not in ("summary.json", "method_summary.json"):
                continue
            try:
                s = json.load(open(os.path.join(d, fn)))
            except Exception:
                continue
            for key, job in (s.get("done") or {}).items():
                parts = key.split("/")
                if len(parts) == 3 and job.get("span_deg"):
                    out.setdefault(parts[2], []).append(float(job["span_deg"]))
    return {k: sorted(v)[len(v) // 2] for k, v in out.items() if v}


def _b1_diagnostics(summaries_dir="results/kaggle_runs"):
    """frac_above_floor and mean_anisotropy per protocol, from the runs' own JSON.

    The filter prints both during training and method_eval.py records them per
    job; they never reach runs.csv because they are properties of the filter
    rather than of a rendered image. §6.6 asks for them beside the PSNR, so they
    are read back from the summaries here rather than retyped.
    """
    out = {}
    for d, _, files in os.walk(summaries_dir):
        for fn in files:
            if fn not in ("summary.json", "method_summary.json"):
                continue
            try:
                s = json.load(open(os.path.join(d, fn)))
            except Exception:
                continue
            for key, job in (s.get("done") or {}).items():
                if "frac_above_floor" not in job:
                    continue
                parts = key.split("/")
                if len(parts) != 3:
                    continue
                method, _scene, proto = parts
                if method != "b1":
                    continue
                e = out.setdefault(proto, {"floor": [], "aniso": []})
                e["floor"].append(job["frac_above_floor"])
                e["aniso"].append(job["mean_anisotropy"])
    return out


def stress_table(rows, inst, label, caption, sigma=SIGMA_FALLBACK):
    """Table 3: the stress suite -- the thesis's central empirical claim.

    Per protocol: the paired PSNR delta, that delta in units of the harness's own
    run-to-run noise, and the two filter diagnostics (§6.6) without which a
    positive delta cannot be attributed to the estimation term binding.
    """
    sel = [r for r in rows if "/" in (r.get("train_scale") or "")]
    if not sel:
        return placeholder(label, caption,
                           "The stress-suite training runs have not been performed.")
    # Paired per scene: both methods train on the same protocol subset with the
    # same seed in the same session, so the delta is a within-scene difference
    # and must not be taken between two different scene means.
    acc = {}
    for r in sel:
        proto = r["train_scale"].split("/")[1]
        acc.setdefault(proto, {}).setdefault(r["method"], {}) \
           .setdefault((r["scene"], r["test_scale"]), []).append(float(r["psnr"]))
    bp = (inst or {}).get("by_protocol", {})
    spans = _run_spans()
    diag = _b1_diagnostics()
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \setlength{\tabcolsep}{4pt}",
         r"  \begin{tabular}{lrrrrrrrr}", r"    \toprule",
         r"    protocol & span & $\sigma_D/\sigma_L$ & Mip-Spl. & B1 & $\Delta$"
         r" & $\Delta/\sigma$ & above floor & aniso. \\", r"    \midrule"]
    for proto in PROTO_ORDER:
        d = acc.get(proto)
        if not d:
            continue
        mip_d = d.get("mip-splatting", {})
        b1_d = d.get("b1-fisher", {})
        keys = sorted(set(mip_d) & set(b1_d))
        paired = [mean(b1_d[k]) - mean(mip_d[k]) for k in keys]
        mip = [x for k in keys for x in mip_d[k]]
        b1 = [x for k in keys for x in b1_d[k]]
        delta = mean(paired) if paired else None
        v = bp.get(proto) or {}
        span = spans.get(proto) or v.get("median_span_deg")
        # I-2's anisotropy belongs to this row only if the instruments measured
        # the same capture. When the two spans disagree by more than a couple of
        # degrees they are different protocols under one name, and the cell is
        # left empty rather than filled from the wrong one.
        aniso = None
        if v.get("median_span_deg") and span:
            if abs(v["median_span_deg"] - span) <= 2.0:
                aniso = v.get("median_sigma_ratio_measured")
        dg = diag.get(proto) or {}
        L.append("    " + " & ".join([
            PROTO_LABEL[proto],
            f"${span:.0f}^\\circ$" if span else NOT_MEASURED,
            f"{aniso:.2f}$\\times$" if aniso else NOT_MEASURED,
            f"{mean(mip):.2f}" if mip else NOT_MEASURED,
            f"{mean(b1):.2f}" if b1 else NOT_MEASURED,
            (r"\textbf{" + f"{delta:+.3f}" + "}") if delta is not None else NOT_MEASURED,
            f"{delta / sigma:+.1f}" if delta is not None and sigma else NOT_MEASURED,
            f"{mean(dg.get('floor', [])) * 100:.0f}\\%" if dg.get("floor") else NOT_MEASURED,
            f"{mean(dg.get('aniso', [])):.2f}$\\times$" if dg.get("aniso") else NOT_MEASURED,
        ]) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Mean PSNR over scenes and test "
          r"scales; $\Delta$ is paired within scene and test scale, never between "
          r"two means. Training cameras are restricted to the protocol and the "
          r"test set is left whole, so only the capture geometry varies. "
          r"$\sigma = " + f"{sigma:.3f}" + r"$\,dB is the harness's own "
          r"run-to-run spread (\S\ref{sec:spread}), so $\Delta/\sigma$ says "
          r"whether a difference is a result. \emph{above floor} is the fraction "
          r"of primitives on which the estimation term exceeds the Nyquist floor "
          r"in at least one direction, and \emph{aniso.} the mean ratio of the "
          r"filter's largest to smallest axis --- both printed by the filter "
          r"during training. Where \emph{above floor} is $0\,\%$, "
          r"Proposition~\ref{prop:reduction} makes the two rows the same filter "
          r"and $\Delta$ measures nothing but noise.",
          r"\end{table}", ""]
    return "\n".join(L)


def load_all(runs_path):
    """Every measured row, superseded ones included. Only C1 wants this."""
    if not os.path.exists(runs_path):
        return []
    with open(runs_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("psnr")]


def c1_table(all_rows, c1, label, caption):
    """C1: the same arm and scene with and without the 2D opacity compensation.

    The left column is deliberately read from the SUPERSEDED rows -- it is the
    measurement the compensation produced, which is the thing being compared
    against, so it has to come from the file rather than from memory.
    """
    scene = (c1.get("scenes") or ["lego"])[0]
    new = {}
    for job in (c1.get("done") or {}).values():
        for k, v in (job.get("split") or {}).items():
            new[k] = v.get("PSNR")
    if not new:
        return placeholder(label, caption, "C1 has not been run.")
    cand = [r for r in all_rows
            if r.get("scene") == scene and r.get("arm") == "B"
            and r.get("dataset") == "blender" and r.get("seed") == "0"
            and r.get("iterations") == str(c1.get("iterations", 30000))
            and r.get("load_allres") == "False"
            and r.get("train_scale") == "1x"]
    # Once C2 has landed the compensated rows are the marked ones; before that
    # they are the only ones. Both cases select the same measurement.
    marked = [r for r in cand if "SUPERSEDED" in (r.get("notes") or "")]
    old = {r["test_scale"]: float(r["psnr"]) for r in (marked or cand)}

    pub = PUBLISHED["STMT"]["B"]
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{lrrr}", r"    \toprule",
         r"    test scale & arm B, with $\rho$ & arm B, without $\rho$"
         r" & published 3DGS \\", r"    \midrule"]
    for i, sc in enumerate(SCALES):
        a, b = old.get(sc), new.get(sc)
        cell_b = (r"$\mathbf{" + f"{b:.2f}" + "}$") if (b and sc == "1/8") \
            else (f"${b:.2f}$" if b else NOT_MEASURED)
        L.append("    " + " & ".join([
            SCALE_TEX[sc],
            f"${a:.2f}$" if a else NOT_MEASURED,
            cell_b,
            f"${pub[i]:.2f}$",
        ]) + r" \\")
    drop = ((old.get("1x", 0) - old.get("1/8", 0)) if old else None,
            (new.get("1x", 0) - new.get("1/8", 0)) if new else None)
    L += [r"    \midrule",
          "    full $\\rightarrow \\tfrac18$ & "
          + (f"${drop[0]:.2f}$" if drop[0] else NOT_MEASURED) + " & "
          + (f"${drop[1]:.2f}$" if drop[1] else NOT_MEASURED)
          + f" & ${pub[0] - pub[3]:.2f}$ \\\\",
          r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize \texttt{" + scene + r"}, seed 0, "
          + str(c1.get("iterations", 30000)) + r" iterations, identical in every "
          r"other respect --- same branch but for the rasteriser flag, same "
          r"densification, same data, same seed. The published column is an "
          r"eight-scene mean and \texttt{" + scene + r"} is not the mean, so it "
          r"is a shape to compare against rather than a target; this is why the "
          r"gate was written scene-relative. The left column is read from the "
          r"rows this run supersedes, which are retained in "
          r"\texttt{results/runs.csv}.",
          r"\end{table}", ""]
    return "\n".join(L)


def g2_table(rows, label, caption):
    """Gate G2, scored from the CSV against thresholds set before the run.

    Superseded rows are excluded by `load`, so this scores whichever arm B is
    current: before C2 that is the arm carrying Mip-Splatting's opacity
    compensation, after it the vanilla one. The thresholds never move.
    """
    sel = [r for r in rows if r.get("dataset") == "blender"
           and r.get("iterations") == "30000"
           and r.get("load_allres") == "False" and r.get("seed") == "0"
           and (r.get("train_scale") == "1x")]
    sel, _common = matched(sel)
    acc = {}
    for r in sel:
        if r.get("arm") in ("A", "B"):
            acc.setdefault(r["arm"], {}).setdefault(r["test_scale"], []) \
               .append(float(r["psnr"]))

    def g(arm, sc):
        v = acc.get(arm, {}).get(sc)
        return mean(v) if v else None

    a1, a8 = g("A", "1x"), g("A", "1/8")
    b1, b8 = g("B", "1x"), g("B", "1/8")
    if not all(v is not None for v in (a1, a8, b1, b8)):
        return placeholder(label, caption,
                           "Arm A or arm B has no current rows at both full and "
                           "$\\tfrac18$ scale.")
    bdrop, adrop, gap = b1 - b8, a1 - a8, a8 - b8
    crit = [
        (r"3DGS falls, full $\rightarrow \tfrac18$", r"$\gtrsim 13$\,dB",
         f"{bdrop:.2f}\\,dB", bdrop >= 13.0),
        (r"Mip-Splatting falls, full $\rightarrow \tfrac18$", r"$\lesssim 6$\,dB",
         f"{adrop:.2f}\\,dB", adrop <= 6.0),
        (r"gap at $\tfrac18$", r"$> 9$\,dB", f"{gap:.2f}\\,dB", gap > 9.0),
    ]
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{lrrl}", r"    \toprule",
         r"    criterion & threshold & measured & verdict \\", r"    \midrule"]
    for name, thr, got, ok in crit:
        L.append(f"    {name} & {thr} & {got} & "
                 + (r"\pass" if ok else r"\fail") + r" \\")
    n_scene = len({r["scene"] for r in sel if r.get("arm") == "B"})
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Mean over " + str(n_scene) +
          r" scene(s), 30\,000 iterations, seed 0. Published 3DGS falls "
          r"$15.64$\,dB and the published gap at $\tfrac18$ is $10.98$\,dB. "
          r"Thresholds were set before any run and have not been moved. Rows "
          r"marked superseded in \texttt{results/runs.csv} are excluded here and "
          r"everywhere else, and are retained in the file.",
          r"\end{table}", ""]
    return "\n".join(L)


PROTO_SELECTION = {
    "full": ("all training views",
             "the control. Parity is required by "
             "Proposition~\\ref{prop:reduction}, not hoped for."),
    "grazing": ("high-incidence views only",
                "obliquity at near-full parallax; isolates the off-axis term of "
                "\\S\\ref{sec:offaxis}."),
    "mixed": ("all views, a seeded random half downsampled $2$--$4\\times$",
              "the mixed-camera defect of \\S\\ref{sec:defects}, not parallax."),
    "arc": ("contiguous azimuthal wedge",
            "the anisotropic regime: the floor binds in some directions and not "
            "in others."),
    "cone": ("tightest angular cluster, a tenth of the cameras",
             "a degraded but trainable capture. As built it lands at 36--54$^\circ$, "
             "where the floor still binds."),
    "pencil": ("every camera within a $20^\circ$ wedge",
               "the regime itself: the narrowest capture the suite contains, and "
               "the only one where the estimation term exceeds the floor."),
}


def protocol_table(inst, label, caption):
    """The five capture protocols at the spans and camera counts they realise.

    Generated because they were not what they were asked for: `arc` and `cone`
    select by angular EXTENT, so the camera count is an outcome and varies by
    scene. A hand-written count is a claim about a protocol that was never built.
    """
    res = (inst or {}).get("results") or {}
    bp = (inst or {}).get("by_protocol") or {}
    if not bp:
        return placeholder(label, caption, "The protocols have not been built.")
    counts = {}
    for key, v in res.items():
        proto = key.rsplit("/", 1)[-1]
        if v.get("n_cameras"):
            counts.setdefault(proto, []).append(int(v["n_cameras"]))
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{llrrp{0.29\linewidth}}", r"    \toprule",
         r"    protocol & selection & cameras & span & what it tests \\",
         r"    \midrule"]
    for k in PROTO_ORDER:
        v = bp.get(k)
        if not v or k not in PROTO_SELECTION:
            continue
        sel, tests = PROTO_SELECTION[k]
        c = counts.get(k) or []
        cam = (f"{min(c)}" if c and min(c) == max(c)
               else (f"{min(c)}--{max(c)}" if c else NOT_MEASURED))
        L.append("    " + " & ".join([
            PROTO_LABEL[k], sel, cam,
            f"${v['median_span_deg']:.0f}^\\circ$", tests,
        ]) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Span is the median over the "
          r"8 Blender scenes of the maximum pairwise angle the selected cameras "
          r"subtend at the capture centre --- measured from the subset that was "
          r"built, not specified. \emph{arc} and \emph{cone} select by angular "
          r"extent rather than by a camera count, so the count is an outcome and "
          r"differs between scenes; the range is given. A subset asked for at "
          r"$6^\circ$ that comes out at $20^\circ$ is not the capture the "
          r"prediction was made for, which is why the prediction is re-evaluated "
          r"at the span actually realised.",
          r"\end{table}", ""]
    return "\n".join(L)


def budget_table(summaries_dir, label, caption):
    """Every GPU session this project ran, and what it cost.

    Read from the sessions' own summaries rather than maintained by hand, because
    a budget table that is edited is a budget table that stops being true the
    first time a run is repeated -- and several were.
    """
    seen, rows_ = {}, []
    for d, _, files in os.walk(summaries_dir):
        for fn in files:
            if fn not in ("summary.json", "b2.json", "fps.json",
                          "method_summary.json"):
                continue
            path = os.path.join(d, fn)
            try:
                s = json.load(open(path))
            except Exception:
                continue
            hours = s.get("gpu_hours_used")
            if hours is None:
                continue
            name = os.path.basename(d)
            if name in seen:
                continue
            seen[name] = True
            scenes = s.get("scenes") or s.get("only") or []
            protos = s.get("protocols") or []
            what = []
            if scenes:
                what.append(f"{len(scenes)} scene" + ("s" if len(scenes) > 1 else ""))
            if protos:
                what.append(f"{len(protos)} protocol"
                            + ("s" if len(protos) > 1 else ""))
            if s.get("methods"):
                what.append(" + ".join(s["methods"]))
            rows_.append((name, s.get("rung") or "", ", ".join(what),
                          float(hours), bool(s.get("complete", True))))
    if not rows_:
        return placeholder(label, caption, "No session summary carries a cost.")
    rows_.sort(key=lambda t: -t[3])
    total = sum(r[3] for r in rows_)
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{llrr}", r"    \toprule",
         r"    session & rung & what it covered & GPU-h \\", r"    \midrule"]
    for name, rung, what, hours, complete in rows_:
        L.append("    " + " & ".join([
            r"\texttt{" + name.replace("_", r"\_") + "}",
            rung, what or r"---",
            f"{hours:.2f}" + ("" if complete else r"$^{*}$"),
        ]) + r" \\")
    L += [r"    \midrule",
          r"    \multicolumn{3}{l}{\textbf{total}} & \textbf{"
          + f"{total:.1f}" + r"} \\",
          r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Session wall-clock, which is how the "
          r"free-tier weekly allowance of 30\,h per account is counted; a "
          r"two-GPU session does twice the work in one hour of it. Read from "
          r"each session's own JSON summary, so a repeated run appears at its "
          r"real cost. $^{*}$ session did not report itself complete --- either "
          r"it was a deliberately narrowed run or it ended early, and "
          r"\S\ref{sec:budget} says which.",
          r"\end{table}", ""]
    return "\n".join(L)


def fps_table(fps, label, caption):
    """C9: render-call fps for the three arms, and the ratio that is the claim."""
    if not fps or not fps.get("results"):
        return placeholder(label, caption,
                           "The matched render-speed benchmark has not been run. "
                           "The \\texttt{render\\_fps} column of "
                           "Table~\\ref{tab:cost} is wall time over a whole "
                           "\\texttt{render.py} invocation and is deliberately "
                           "not quoted here in its place.")
    res = fps["results"]
    base = (res.get("mip") or {}).get("fps")
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{lrrrr}", r"    \toprule",
         r"    method & primitives & fps & vs.\ Mip-Splatting & filter setup \\",
         r"    \midrule"]
    for key in ("3dgs", "mip", "b1"):
        v = res.get(key)
        if not v:
            continue
        ratio = (v["fps"] / base) if base else None
        setup = v.get("filter_setup_seconds") or 0.0
        L.append("    " + " & ".join([
            v["label"],
            f"{v['n_gaussians']:,}".replace(",", r"\,"),
            f"{v['fps']:.1f}",
            f"{ratio:.3f}$\\times$" if ratio else NOT_MEASURED,
            (f"{setup:.2f}\\,s" if setup else r"---"),
        ]) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize \texttt{" + str(fps.get("scene", "")) +
          r"}, " + str((res.get("mip") or res.get("b1") or {}).get("n_views", 0)) +
          r" test views, " + str(fps.get("gpu", "")) + r", median of "
          + str(fps.get("repeats", 0)) + r" passes after "
          + str(fps.get("warmup_views", 0)) + r" warm-up views, CUDA "
          r"synchronised around each pass. The render \emph{call} only: no scene "
          r"load, no PNG encode, no disk write --- which is why these numbers are "
          r"an order of magnitude above the \texttt{render\_fps} column of "
          r"Table~\ref{tab:cost}, and why the ratio rather than the absolute "
          r"figure is what this thesis defends. \emph{filter setup} is the "
          r"one-off cost of building $\Sigma_{\mathrm{filt}}$ from the training "
          r"cameras before the loop begins.",
          r"\end{table}", ""]
    return "\n".join(L)


def geometry_tables(geo, inst):
    """M-4, M-5 and M-5b, from tools/test_geometry_claims.py's own JSON.

    These are closed forms, not measurements, but they are still generated
    rather than typed: the script that verifies them is the only place their
    digits exist, so a change to the derivation cannot fail to reach the
    document.
    """
    if not geo:
        return placeholder("offaxis", r"Off-axis anisotropy of a single view.",
                           "tools/test\\_geometry\\_claims.py has not been run.")
    out = []

    m4 = geo.get("m4") or {}
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{M-4 --- a single view is already anisotropic off-axis. The "
         r"rank-1 shortcut $(f/z)^2(I-dd^{\!\top})$ is a projector, so its two "
         r"non-zero eigenvalues are equal by construction; the true "
         r"$J^{\!\top}\!J$ separates them by exactly $\cos^2\varphi$. Four "
         r"randomly drawn poses, and the law verified over "
         + str(m4.get("n_sweep", 0)) + r" more.}",
         r"  \label{tab:offaxis}", r"  \small",
         r"  \begin{tabular}{rrrr}", r"    \toprule",
         r"    $\varphi$ (off-axis) & $\lambda_2/\lambda_1$ measured "
         r"& $\cos^2\varphi$ & $|\text{diff}|$ \\", r"    \midrule"]
    for phi, ratio, pred in m4.get("poses", []):
        L.append(f"    ${phi:.1f}^\\circ$ & {ratio:.4f} & {pred:.4f} & "
                 f"$<10^{{-15}}$ \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Maximum error of the law over the "
          r"sweep: $" + f"{m4.get('law_max_abs_err', 0):.1e}".replace("e-", r"\times10^{-")
          + r"}$. At a representative off-axis point the shortcut differs from "
          r"the truth by " + f"{m4.get('offaxis_rel_diff', 0) * 100:.0f}" +
          r"\,\% of the largest entry --- and differs in \emph{shape}, not by a "
          r"scalar, so no choice of scalar filter absorbs it.",
          r"\end{table}", ""]
    out.append("\n".join(L))

    m5 = geo.get("m5") or {}
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{M-5 --- the exact two-view spectrum. $\Lambda \propto 2I - "
         r"d_1d_1^{\!\top} - d_2d_2^{\!\top}$ is diagonal in the frame of the two "
         r"rays with eigenvalues $\{2,\,2\cos^2(\theta/2),\,2\sin^2(\theta/2)\}$. "
         r"Three distinct values: the filter is \emph{triaxial}, and the in-plane "
         r"lateral direction is degraded as well as depth.}",
         r"  \label{tab:twoviewspectrum}", r"  \small",
         r"  \begin{tabular}{rrrrrr}", r"    \toprule",
         r"    $\theta$ & $\lambda_1$ & $\lambda_2$ & $\lambda_3$ "
         r"& $\lambda_{\min}/\lambda_{\max}$ & $\sin^2(\theta/2)$ \\",
         r"    \midrule"]
    for e in m5.get("spectrum", []):
        m = e["measured"]
        flag = "" if abs(e["true_ratio"] - e["naive"]) < 1e-6 else r"$^{\dagger}$"
        L.append(f"    ${e['theta_deg']}^\\circ$ & {m[0]:.4f} & {m[1]:.4f} & "
                 f"{m[2]:.6f} & {e['true_ratio']:.6f}{flag} & {e['naive']:.6f} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Measured against numerical "
          r"eigendecomposition; maximum absolute error $"
          + f"{m5.get('max_abs_err', 0):.0e}".replace("e-", r"\times10^{-") + r"}$. "
          r"$^{\dagger}$ past $\theta = 90^\circ$ the depth direction is better "
          r"constrained than the in-plane lateral one and the two exchange roles, "
          r"so the ratio of smallest to largest is "
          r"$\min(\sin^2(\theta/2), \cos^2(\theta/2))$ and not $\sin^2(\theta/2)$. "
          r"Equation~\eqref{eq:anisotropy} is stated for $\theta \le 90^\circ$, "
          r"which is the whole of the regime the stress protocols occupy.",
          r"\end{table}", ""]
    out.append("\n".join(L))

    m5b = geo.get("m5b") or {}
    bp = (inst or {}).get("by_protocol", {})
    by_span = {}
    for k in PROTO_ORDER:
        v = bp.get(k)
        if v:
            by_span[k] = (v["median_span_deg"], v["median_sigma_ratio_measured"])
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{M-5b --- a protocol is a distribution, not two views. For "
         r"cameras spread over a cap of half-angle $\alpha$, "
         r"$\lambda_3/\lambda_1 = 2(1-E)/(1+E)$ with "
         r"$E = (1+c+c^2)/3$, $c = \cos\alpha$ --- exact, and asymptotically "
         r"\emph{half} the two-view value, because a cap's cameras are spread "
         r"across the interval rather than sitting at its two ends. The two-view "
         r"form therefore \emph{understates} a real capture's depth anisotropy by "
         r"$\sqrt{2}$ in $\sigma$.}",
         r"  \label{tab:cap}", r"  \small",
         r"  \begin{tabular}{rrrr}", r"    \toprule",
         r"    span & cap $\sigma_D/\sigma_L$ & two-view $\sigma_D/\sigma_L$ "
         r"& measured \\", r"    \midrule"]
    for e in m5b.get("cap", []):
        near = [k for k, (sp, _) in by_span.items()
                if abs(sp - e["span_deg"]) <= 3.0]
        meas = (f"{by_span[near[0]][1]:.2f}$\\times$ (" + PROTO_LABEL[near[0]] + ")"
                if near else "")
        L.append(f"    ${e['span_deg']}^\\circ$ & {e['cap_sigma_ratio']:.2f}"
                 r"$\times$ & " + f"{e['two_view_sigma_ratio']:.2f}" +
                 r"$\times$ & " + meas + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Closed form checked against "
          + f"{m5b.get('n_samples', 0):,}".replace(",", r"\,") +
          r" sampled directions per span, maximum relative error $"
          + f"{m5b.get('mc_max_rel_err', 0):.0e}".replace("e-", r"\times10^{-")
          + r"}$. The measured column is I-2 on the protocol whose median span "
          r"matches the row, from Table~\ref{tab:instruments}.",
          r"\end{table}", ""]
    out.append("\n".join(L))
    return "\n".join(out)


def b2_table(b2, label, caption):
    """Table 4: B2 against the unmasked control and magnitude pruning."""
    if not b2 or not b2.get("results"):
        return placeholder(label, caption, "B2 has not been evaluated.")
    res = b2["results"]
    rows_by_cfg = {"full": [], "b2": [], "magnitude": []}
    for k, v in res.items():
        if "_sweep" in k:
            continue
        cfg = k.split("/")[1]
        if cfg in rows_by_cfg:
            rows_by_cfg[cfg].append(v)
    if not rows_by_cfg["b2"]:
        return placeholder(label, caption, "B2 has not been evaluated.")

    def agg(vs, key):
        xs = [v[key] for v in vs if key in v and v[key] is not None]
        return mean(xs) if xs else None

    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{lrrrr}", r"    \toprule",
         r"    configuration & PSNR & model MB & non-DC SH retained"
         r" & mean $\ell_{\max}$ \\", r"    \midrule"]
    labels = [("full", "full degree 3 (control)"),
              ("magnitude", "magnitude pruning, matched size"),
              ("b2", "B2 --- identifiability masking")]
    for cfg, name in labels:
        vs = rows_by_cfg[cfg]
        if not vs:
            continue
        psnr, mb = agg(vs, "PSNR"), agg(vs, "model_mb")
        ret = agg(vs, "retained")
        lmax = agg(vs, "mean_l_max")
        L.append("    " + " & ".join([
            name,
            f"{psnr:.2f}" if psnr is not None else NOT_MEASURED,
            f"{mb:.1f}" if mb is not None else NOT_MEASURED,
            f"{ret * 100:.0f}\\%" if ret is not None else (r"100\%" if cfg == "full"
                                                          else NOT_MEASURED),
            f"{lmax:.2f}" if lmax is not None else (r"3.00" if cfg == "full"
                                                    else NOT_MEASURED),
        ]) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Mean over the scenes measured, "
          r"post-hoc on trained models. Magnitude pruning is matched to the exact "
          r"fraction B2 retained, so the two rows are the same size and differ only "
          r"in WHICH coefficients they keep.",
          r"\end{table}", ""]
    return "\n".join(L)



def b2_protocol_table(proto, sweep, label, caption):
    """B2's criterion across capture protocols, measured without a GPU.

    Equation 5.1 depends on primitive positions and training view directions
    only, so the retained fraction and mean l_max are measurable from a trained
    point cloud alone. PSNR is not -- that needs a render pass -- and its column
    stays empty rather than estimated.
    """
    if not proto:
        return placeholder(label, caption, "B2 has not been evaluated.")
    by = {}
    for k, v in proto.items():
        by.setdefault(v["protocol"], []).append(v)
    L = [r"\begin{table}[htbp]", r"  \centering",
         r"  \caption{" + caption + "}", r"  \label{tab:" + label + "}",
         r"  \small", r"  \begin{tabular}{lrrrrr}", r"    \toprule",
         r"    protocol & cameras & span & views/prim & mean $\ell_{\max}$"
         r" & non-DC retained \\", r"    \midrule"]
    for k in PROTO_ORDER:
        vs = by.get(k)
        if not vs:
            continue
        L.append("    " + " & ".join([
            PROTO_LABEL[k],
            f"{mean([v['n_cameras'] for v in vs]):.0f}",
            f"${mean([v['span_deg'] for v in vs]):.0f}^\\circ$",
            f"{mean([v['mean_views'] for v in vs]):.0f}",
            f"{mean([v['mean_l_max'] for v in vs]):.2f}",
            f"{mean([v['non_dc_retained'] for v in vs]) * 100:.1f}\\%",
        ]) + r" \\")
    n = len({v["scene"] for v in proto.values()})
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \par\vspace{2pt}\footnotesize Mean over " + str(n) +
          r" scene(s), $\tau = 0.01$, post-hoc on trained models. Training "
          r"cameras are restricted to the protocol; the criterion is a function "
          r"of geometry, so no retraining is involved and no GPU was used.",
          r"\end{table}", ""]
    body = "\n".join(L)

    if sweep:
        S = [r"\begin{table}[htbp]", r"  \centering",
             r"  \caption{Sensitivity of Equation~\eqref{eq:lmax} to $\tau$, on the "
             r"low-parallax cone. The criterion is a cliff rather than a graded "
             r"response: a tenfold change in $\tau$ moves the retained fraction "
             r"from a fifth of the coefficients to none of them.}",
             r"  \label{tab:btwosweep}", r"  \small",
             r"  \begin{tabular}{rrr}", r"    \toprule",
             r"    $\tau$ & mean $\ell_{\max}$ & non-DC retained \\",
             r"    \midrule"]
        for t in sorted(sweep, key=float):
            v = sweep[t]
            S.append(f"    {t} & {v['mean_l_max']:.2f} & "
                     f"{v['non_dc_retained'] * 100:.1f}\\% " + r"\\")
        S += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
        body += "\n" + "\n".join(S)
    return body


# ------------------------------------------------------------------- macros
def macros(rows, summaries, inst=None):
    """Inline numbers, as \\newcommand. Undefined data yields a visible marker."""
    M = {}

    def put(name, value, spec="{:.2f}"):
        M[name] = spec.format(value) if isinstance(value, (int, float)) else NOT_MEASURED

    for proto, allres in (("STMT", "False"), ("MTMT", "True")):
        sel = select(rows, iterations=30000, load_allres=allres, seed=0)
        acc = by_arm_scale(sel)
        # LaTeX control sequences cannot contain digits, so the arm is named
        # ArmB/ArmA rather than 3dgs/mips.
        for arm, tag in (("B", "ArmB"), ("A", "ArmA")):
            vals = {}
            for sc in SCALES:
                vals[sc] = mean([float(r["psnr"]) for r in acc.get(arm, {}).get(sc, [])])
            key = proto.lower().capitalize() + tag
            for sc, nm in zip(SCALES, ("Full", "Half", "Quarter", "Eighth")):
                put(key + nm, vals[sc])
            put(key + "Drop",
                (vals["1x"] - vals["1/8"]) if vals["1x"] and vals["1/8"] else None)
        aF = mean([float(r["psnr"]) for r in acc.get("A", {}).get("1/8", [])])
        bF = mean([float(r["psnr"]) for r in acc.get("B", {}).get("1/8", [])])
        put(proto.capitalize() + "GapEighth", (aF - bF) if aF and bF else None)
        M[proto.capitalize() + "Scenes"] = str(len({r["scene"] for r in sel})) or "0"

    r0 = summaries.get("R0") or {}
    sa, sb = r0.get("stats_A") or {}, r0.get("stats_B") or {}
    put("rZeroItersPerSec", sa.get("iters_per_second"), "{:.1f}")
    put("rZeroRenderFps", sa.get("render_fps"), "{:.1f}")
    put("rZeroPeakVram", sa.get("peak_vram_mb"), "{:,.0f}")
    put("rZeroGpuHours", r0.get("gpu_hours_used"), "{:.2f}")
    put("armAFilterMax", sa.get("filter_3D_max"), "{:.3g}")
    put("armBFilterMax", sb.get("filter_3D_max"), "{:.0f}")

    # Largest per-scale spread between any two runs of the same configuration.
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["scene"], r["arm"], r["test_scale"], r["iterations"],
                 r["load_allres"], r["seed"])].append(float(r["psnr"]))
    spreads = [max(v) - min(v) for v in grouped.values() if len(v) > 1]
    put("repeatSpread", max(spreads) if spreads else None, "{:.3f}")

    # Seed-to-seed sigma: a different initial point cloud and view order, not the
    # same configuration run twice. Larger than the atomics spread, and the bar
    # a single-seed L1 claim actually has to clear.
    seedacc = defaultdict(list)
    for r in rows:
        if (r.get("dataset") == "blender" and r.get("iterations") == "30000"
                and r.get("load_allres") == "False"
                and r.get("train_scale") == "1x"):
            seedacc[(r["scene"], r["arm"], r["test_scale"])].append(float(r["psnr"]))
    sds = {}
    for k, v in seedacc.items():
        if len(v) > 1:
            m = sum(v) / len(v)
            sds[k] = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    put("seedSigma", max(sds.values()) if sds else None, "{:.3f}")
    armB = [s for k, s in sds.items() if k[1] == "B"]
    armA = [s for k, s in sds.items() if k[1] == "A"]
    put("seedSigmaArmB", max(armB) if armB else None, "{:.3f}")
    put("seedSigmaArmA", max(armA) if armA else None, "{:.3f}")
    worst = max(sds, key=sds.get) if sds else None
    M["seedSigmaWhere"] = (f"{ARM_NAME[worst[1]]} on \\texttt{{{worst[0]}}} at "
                           f"{SCALE_TEX[worst[2]]}") if worst else NOT_MEASURED

    bad = [k for k in M if not k.isalpha()]
    if bad:
        raise SystemExit(f"macro name(s) LaTeX cannot accept (letters only): {bad}")
    real = [r for r in rows if r.get("dataset") in ("mipnerf360", "tandt", "db")
            and r.get("peak_vram_mb")]
    put("realScenes", len({r["scene"] for r in real}) if real else None, "{:.0f}")
    put("realPeakVram", max((float(r["peak_vram_mb"]) for r in real), default=None),
        "{:,.0f}")
    put("realMaxGauss", max((float(r["n_gaussians"]) for r in real
                             if r.get("n_gaussians")), default=None), "{:,.0f}")

    # C1: the compensation switch, on one scene. Quoted in §7.2's prose, so the
    # prose cannot drift from the run that produced it.
    c1 = summaries.get("C1") or {}
    c1new = {}
    for job in (c1.get("done") or {}).values():
        for k, v in (job.get("split") or {}).items():
            c1new[k] = v.get("PSNR")
    c1scene = (c1.get("scenes") or [None])[0]
    c1old = {}
    if c1scene:
        for r in rows:
            if (r.get("scene") == c1scene and r.get("arm") == "B"
                    and r.get("dataset") == "blender" and r.get("seed") == "0"
                    and r.get("iterations") == "30000"
                    and r.get("load_allres") == "False"
                    and r.get("train_scale") == "1x"):
                c1old[r["test_scale"]] = float(r["psnr"])
    for sc, nm in zip(SCALES, ("Full", "Half", "Quarter", "Eighth")):
        put("cOne" + nm, c1new.get(sc))
        put("cOneBase" + nm, c1old.get(sc))
    put("cOneCollapse",
        (c1old["1/8"] - c1new["1/8"])
        if c1old.get("1/8") and c1new.get("1/8") else None)
    put("cOneDrop",
        (c1new["1x"] - c1new["1/8"])
        if c1new.get("1x") and c1new.get("1/8") else None)
    M["cOneScene"] = (c1scene or NOT_MEASURED).replace("_", "")

    bp = (inst or {}).get("by_protocol", {})
    for k, name in (("full", "Full"), ("arc", "Arc"), ("cone", "Cone"),
                    ("mixed", "Mixed"), ("grazing", "Grazing")):
        v = bp.get(k) or {}
        put("inst" + name + "Aniso", v.get("median_sigma_ratio_measured"))
        put("inst" + name + "Span", v.get("median_span_deg"), "{:.0f}")
        put("inst" + name + "Cap",
            cap_sigma_ratio(v["median_span_deg"]) if v else None)
        put("inst" + name + "ExceedWorst",
            None if not v else v["median_frac_exceeding_floor_worst_dir"] * 100, "{:.0f}")
        put("inst" + name + "ExceedAll",
            None if not v else v["median_frac_exceeding_floor_all_dirs"] * 100, "{:.0f}")
        put("inst" + name + "EstOverFloor", v.get("median_ratio_est_over_floor_p50"))

    lines = [r"% Generated by tools/make_tex.py — do not edit.",
             r"% Every measured number in the thesis comes from here.",
             r"\providecommand{\notmeasured}{\textcolor{gray}{--}}"]
    for k, v in sorted(M.items()):
        lines.append(r"\newcommand{\%s}{%s}" % (k, v.replace(",", r"\,")))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--out", default="thesis/generated")
    ap.add_argument("--summaries", default="results/kaggle_runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.runs)

    summaries = {}
    for d, _, files in os.walk(a.summaries):
        for fn in files:
            if fn.endswith("summary.json"):
                try:
                    s = json.load(open(os.path.join(d, fn)))
                except Exception:
                    continue
                rung = s.get("rung")
                if rung and (rung not in summaries
                             or s.get("all_checks_pass_or_skip") or s.get("complete")):
                    summaries[rung] = s

    inst = {}
    for d, _, files in os.walk(a.summaries):
        for fn_ in files:
            if fn_ == "summary.json" and "instrument" in d.replace("\\", "/"):
                try:
                    cand = json.load(open(os.path.join(d, fn_)))
                except Exception:
                    continue
                if cand.get("by_protocol"):
                    inst = cand

    # seed=0 for the headline tables. R5 adds seeds 1 and 2 for two scenes only,
    # so including them would silently weight the 8-scene mean toward lego and
    # chair -- it moved arm A's full-res figure by nearly a decibel before this
    # filter was added. The seed table below deliberately takes all seeds.
    b2proto, b2sweep = {}, {}
    # The B2 run rebuilds the protocol sweep from the same camera_protocols.py
    # the instruments use, so it supersedes the local file, which was produced
    # under an earlier definition of arc and cone and disagrees with
    # Table~\ref{tab:instruments} about what a "cone" is.
    for d, _, files in os.walk(a.summaries):
        for fn_ in files:
            if fn_ == "b2_protocols.json":
                try:
                    b2proto = json.load(open(os.path.join(d, fn_)))
                except Exception:
                    pass
    for path, tgt in (("results/b2_protocols/protocols.json", "proto"),
                      ("results/b2_protocols/tau_sweep.json", "sweep")):
        if tgt == "proto" and b2proto:
            continue
        if os.path.exists(path):
            try:
                loaded = json.load(open(path))
            except Exception:
                continue
            if tgt == "proto":
                b2proto = loaded
            else:
                b2sweep = loaded

    b2 = {}
    for d, _, files in os.walk(a.summaries):
        for fn_ in files:
            if fn_ in ("b2.json", "summary.json"):
                try:
                    cand = json.load(open(os.path.join(d, fn_)))
                except Exception:
                    continue
                if cand.get("rung") == "R8" or (cand.get("results")
                                                and "b2" in str(list(cand["results"])[:3])):
                    b2 = cand

    fps = {}
    for d, _, files in os.walk(a.summaries):
        for fn_ in files:
            if fn_ not in ("fps.json", "summary.json"):
                continue
            try:
                cand = json.load(open(os.path.join(d, fn_)))
            except Exception:
                continue
            if cand.get("rung") == "C9" and cand.get("results"):
                fps = cand

    geo = {}
    if os.path.exists("results/geometry/claims.json"):
        try:
            geo = json.load(open("results/geometry/claims.json"))
        except Exception:
            geo = {}

    stmt = select(rows, iterations=30000, load_allres="False", seed=0)
    mtmt = select(rows, iterations=30000, load_allres="True", seed=0)
    seeds = select(rows, iterations=30000, load_allres="False")

    # The noise the stress-suite delta is measured against. Largest spread
    # between repeats of one identical configuration -- measured, not assumed,
    # and falling back to the R0 figure only if no repeat exists yet.
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r.get(k) for k in CONFIG_KEY)].append(float(r["psnr"]))
    reps = [max(v) - min(v) for v in grouped.values() if len(v) > 1]
    seed_sigma = max(reps) if reps else SIGMA_FALLBACK

    written = {
        "table1-stmt.tex": scale_table(
            stmt, "STMT", "stmt",
            r"Blender, single-scale train $\rightarrow$ multi-scale test (R1). "
            r"Mean PSNR over scenes, 30\,000 iterations. Published values are "
            r"targets cited from Mip-Splatting, never used as the control (L3)."),
        "table2-mtmt.tex": scale_table(
            mtmt, "MTMT", "mtmt",
            r"Blender, multi-scale train and test, \texttt{--load\_allres} (R2). "
            r"Mean PSNR over scenes, 30\,000 iterations."),
        "table-perscene-stmt.tex": per_scene_table(
            stmt, "STMT", "perscenestmt",
            r"R1 per scene. Every cell is one logged run."),
        "table-cost.tex": cost_table(
            stmt, "cost", r"Cost and model size, R1, measured on the same GPU."),
        "table-seedspread.tex": seed_table(
            seeds, "seedspread",
            r"Run-to-run spread (I-3). Repeats of an identical configuration."),
        "table-preflight.tex": preflight_table(
            summaries.get("R0"), "preflight",
            r"The twelve \S6 pre-flight checks, as the R0 kernel reported them."),
        "table-method.tex": method_table(rows, inst, "method",
            r"R7 Table 2 --- the method against both baselines on the standard "
            r"Blender benchmark."),
        "table-stress.tex": stress_table(rows, inst, "stress",
            r"R7 Table 3 --- the stress suite. Where the claim lives.",
            sigma=seed_sigma),
        "table-geometry.tex": geometry_tables(geo, inst),
        "table-protocols.tex": protocol_table(inst, "protocols",
            r"The five capture protocols, as built."),
        "table-budget.tex": budget_table(a.summaries, "budget",
            r"Every GPU session this project ran, and what it cost."),
        "table-g2.tex": g2_table(rows, "g2",
            r"Gate G2, scored against the thresholds set before any run."),
        "table-c1.tex": c1_table(
            load_all(a.runs), summaries.get("C1") or {}, "c1",
            r"C1 --- the same arm, scene and seed, with and without "
            r"Mip-Splatting's 2D opacity compensation. The densification is "
            r"identical in both columns, so whatever the difference is, it is "
            r"not densification."),
        "table-fps.tex": fps_table(fps, "fps",
            r"C9 --- matched render speed. The render call alone, all three arms "
            r"in one session on one GPU."),
        "table-b2-protocols.tex": b2_protocol_table(
            b2proto, b2sweep, "btwoproto",
            r"B2 --- the identifiability criterion across capture protocols."),
        "table-b2.tex": b2_table(b2, "b2",
            r"R8 Table 4 --- B2, angular identifiability of spherical harmonics."),
        "table-real.tex": real_scene_table(
            [r for r in rows if r.get("iterations") == "30000"], "real",
            r"R3 --- real scenes that fit 16\,GB, both arms."),
        "table-instruments.tex": instruments_table(
            inst, "instruments",
            r"I-1 (floor occupancy) and I-2 (conditioning) across the five "
            r"capture protocols of \S\ref{sec:stress}. Computed from trained "
            r"point clouds and camera geometry; no retraining."),
        "measured.tex": macros(rows, summaries, inst),
    }
    for name, body in written.items():
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            f.write(body)
        print(f"wrote {os.path.join(a.out, name)}")

    print(f"\nrows: {len(rows)} | R1(STMT) {len(stmt)} | R2(MTMT) {len(mtmt)} "
          f"| summaries: {sorted(summaries)}")
    if not stmt:
        print("NOTE: no 30k STMT rows yet — Table 1 emitted as an explicit "
              "'not yet measured' placeholder, never filled from the published column.")


if __name__ == "__main__":
    main()
