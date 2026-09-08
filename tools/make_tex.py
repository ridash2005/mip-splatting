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


def load(runs_path):
    if not os.path.exists(runs_path):
        return []
    with open(runs_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("psnr")]


def select(rows, *, iterations, load_allres, dataset="blender", seed=None):
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


def scale_table(rows, protocol, label, caption):
    """The headline table: published target beside our measurement, per scale."""
    acc = by_arm_scale(rows)
    scenes = sorted({r["scene"] for r in rows})
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
               "cone": "low-parallax cone", "mixed": "mixed focal",
               "grazing": "grazing"}
PROTO_ORDER = ["full", "arc", "cone", "mixed", "grazing"]


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
            f"{v['median_sigma_ratio_predicted']:.2f}$\\times$",
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
          r"term governs. \emph{predicted} is Equation~\eqref{eq:anisotropy} at "
          r"the measured span, exact for two symmetric views.",
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

    bp = (inst or {}).get("by_protocol", {})
    for k, name in (("full", "Full"), ("arc", "Arc"), ("cone", "Cone"),
                    ("mixed", "Mixed"), ("grazing", "Grazing")):
        v = bp.get(k) or {}
        put("inst" + name + "Aniso", v.get("median_sigma_ratio_measured"))
        put("inst" + name + "Span", v.get("median_span_deg"), "{:.0f}")
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
    stmt = select(rows, iterations=30000, load_allres="False", seed=0)
    mtmt = select(rows, iterations=30000, load_allres="True", seed=0)
    seeds = select(rows, iterations=30000, load_allres="False")

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
