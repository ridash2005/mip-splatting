#!/usr/bin/env python3
"""
Render a rung's report in the shape §13 of the reproduction prompt asks for,
straight from the summary.json a kernel wrote.

    python tools/report_rung.py results/kaggle_runs/r0_smoke_v3/summary.json \
        --md results/R0-report.md

§13 wants six things: what ran; the numbers as a table with the claimed level
per row; the comparison to published with delta and whether it meets that
level's tolerance; what failed, verbatim; measured GPU-hours; and the one
decision needed. Everything here comes from the summary — nothing is typed in
by hand, and a cell with no measurement behind it is left empty (§14).

The published columns are stated as targets, never as results, and the tolerance
verdict names the level it is testing against (§5): L1 is PSNR ±0.20 dB, and it
only applies at the paper's iterations and resolution. A smoke run at 7 000
iterations is L3 and is scored as such — the published numbers appear beside it
for orientation and are labelled as not comparable.
"""
import argparse
import json
import os
import sys

# The report is UTF-8 (arrows, section signs, en dashes). A Windows console
# defaults to cp1252 and would raise UnicodeEncodeError on the way out, so the
# stream is reconfigured rather than the text degraded.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Mip-Splatting (Yu et al., CVPR 2024), Blender, mean over 8 scenes, 30k iters.
PUBLISHED = {
    "STMT": {"B": {"1x": 33.33, "1/2": 26.95, "1/4": 21.38, "1/8": 17.69},
             "A": {"1x": 33.36, "1/2": 34.00, "1/4": 31.85, "1/8": 28.67}},
    "MTMT": {"B": {"1x": 28.79, "1/2": 30.66, "1/4": 31.64, "1/8": 27.98},
             "A": {"1x": 32.81, "1/2": 34.49, "1/4": 35.45, "1/8": 35.50}},
}
ARM_NAME = {"A": "Mip-Splatting", "B": "3DGS"}
L1_PSNR_TOL = 0.20
SCALES = ["1x", "1/2", "1/4", "1/8"]


def fmt(v, spec="{:.3f}"):
    return spec.format(v) if isinstance(v, (int, float)) else ""


def build(s, protocol="STMT"):
    L = []
    iters = s.get("iterations")
    ref_iters = 30000
    comparable = iters == ref_iters
    scenes = s.get("scene")
    level = "L1" if comparable else "L3"

    L.append(f"# {s.get('rung', 'rung')} — {scenes} · {iters} iterations · both arms")
    L.append("")

    # -- (1) what ran
    L.append("## 1 · What ran")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| scene | `{scenes}` |")
    L.append(f"| iterations | {iters} |")
    L.append(f"| protocol | Blender {protocol}, `load_allres=False`, `-r -1`, "
             "`--eval --white_background` |")
    L.append(f"| test split | the dataset's own `metadata.json[\"test\"]` — "
             "**not** every 8th (F12) |")
    L.append(f"| arm A | Mip-Splatting as shipped, `--kernel_size 0.1` |")
    L.append(f"| arm B | `--kernel_size 0.3 --disable_3D_filter` |")
    L.append(f"| GPU | {s.get('gpu')} ×{s.get('n_gpu')} |")
    L.append(f"| torch / CUDA | {s.get('torch')} / {s.get('cuda')} |")
    L.append(f"| arm A commit | `{s.get('armA_commit', '')[:12]}` |")
    L.append(f"| arm B commit | `{s.get('armB_commit', '')[:12]}` |")
    L.append(f"| LPIPS backbone | vgg (F9) |")
    L.append(f"| seed | 0 (`safe_state` seeds unconditionally; no `--seed` flag exists) |")
    L.append("")

    # -- (2) the numbers
    L.append("## 2 · Measured")
    L.append("")
    stA, stB = s.get("stats_A", {}), s.get("stats_B", {})
    fullA = s.get("per_scale_A_full", {})
    fullB = s.get("per_scale_B_full", {})
    L.append("| test scale | 3DGS PSNR | Mip-Splatting PSNR | 3DGS SSIM | Mip-S SSIM "
             "| 3DGS LPIPS | Mip-S LPIPS | n views | level |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for sc in SCALES:
        a, b = fullA.get(sc), fullB.get(sc)
        if not a and not b:
            continue
        L.append(f"| {sc} | {fmt(b and b['PSNR'], '{:.2f}')} "
                 f"| {fmt(a and a['PSNR'], '{:.2f}')} "
                 f"| {fmt(b and b['SSIM'], '{:.4f}')} | {fmt(a and a['SSIM'], '{:.4f}')} "
                 f"| {fmt(b and b['LPIPS'], '{:.4f}')} | {fmt(a and a['LPIPS'], '{:.4f}')} "
                 f"| {(a or b).get('n', '')} | {level} |")
    pooled = s.get("pooled", {})
    if pooled:
        L.append(f"| *pooled (all 4)* | {fmt(pooled['B']['psnr'], '{:.2f}')} "
                 f"| {fmt(pooled['A']['psnr'], '{:.2f}')} "
                 f"| {fmt(pooled['B']['ssim'], '{:.4f}')} | {fmt(pooled['A']['ssim'], '{:.4f}')} "
                 f"| {fmt(pooled['B']['lpips'], '{:.4f}')} | {fmt(pooled['A']['lpips'], '{:.4f}')} "
                 f"| | {level} |")
    L.append("")
    L.append("Cost and model size, measured:")
    L.append("")
    L.append("| | arm A (Mip-Splatting) | arm B (3DGS) |")
    L.append("|---|---|---|")
    for key, label, spec in (("train_seconds", "train wall-clock (s)", "{:.0f}"),
                             ("iters_per_second", "iterations/second", "{:.2f}"),
                             ("render_seconds", "render wall-clock (s)", "{:.0f}"),
                             ("render_fps", "render fps", "{:.2f}"),
                             ("n_rendered", "test views rendered", "{:.0f}"),
                             ("n_gaussians", "Gaussians", "{:.0f}"),
                             ("model_mb", "model size (MB)", "{:.1f}"),
                             ("peak_vram_mb", "peak VRAM (MB)", "{:.0f}"),
                             ("filter_3D_max", "max abs(filter_3D) in saved PLY", "{:.6g}")):
        L.append(f"| {label} | {fmt(stA.get(key), spec)} | {fmt(stB.get(key), spec)} |")
    L.append("")

    # -- (3) comparison to published
    L.append("## 3 · Against published")
    L.append("")
    if not comparable:
        L.append(f"**Not comparable.** The published Blender numbers are measured at "
                 f"{ref_iters} iterations (F11); this run is {iters}. Every row above is "
                 f"claimed at **L3** (harness parity) — it is the control for later work, "
                 f"not a numerical reproduction. The targets are printed below for "
                 f"orientation only, and no delta against them means anything yet.")
        L.append("")
    pub = PUBLISHED.get(protocol, {})
    L.append("| test scale | arm | measured | published target | delta | "
             f"within L1 (±{L1_PSNR_TOL} dB)? |")
    L.append("|---|---|---|---|---|---|")
    for sc in SCALES:
        for arm, full in (("B", fullB), ("A", fullA)):
            m = full.get(sc)
            p = pub.get(arm, {}).get(sc)
            if m is None or p is None:
                continue
            d = m["PSNR"] - p
            verdict = ("n/a — L3 run" if not comparable
                       else ("yes" if abs(d) <= L1_PSNR_TOL else "no"))
            L.append(f"| {sc} | {ARM_NAME[arm]} | {m['PSNR']:.2f} | {p:.2f} "
                     f"| {d:+.2f} | {verdict} |")
    L.append("")
    if fullB.get("1x") and fullB.get("1/8") and fullA.get("1x") and fullA.get("1/8"):
        dropB = fullB["1x"]["PSNR"] - fullB["1/8"]["PSNR"]
        dropA = fullA["1x"]["PSNR"] - fullA["1/8"]["PSNR"]
        gap = fullA["1/8"]["PSNR"] - fullB["1/8"]["PSNR"]
        L.append("The G2 exit criterion, evaluated on this run (it is *defined* on R1 at "
                 "30k, so at any other length this is the phenomenon check, not the gate):")
        L.append("")
        L.append(f"- 3DGS falls **{dropB:.2f} dB** full → 1/8 (G2 wants ≳13 dB) "
                 f"— {'meets' if dropB >= 13 else 'below'} the threshold")
        L.append(f"- Mip-Splatting falls **{dropA:.2f} dB** (G2 wants ≲6 dB) "
                 f"— {'meets' if dropA <= 6 else 'above'} the threshold")
        L.append(f"- gap at 1/8 is **{gap:.2f} dB** (G2 wants > 9 dB) "
                 f"— {'meets' if gap > 9 else 'below'} the threshold")
        L.append("")

    # -- (4) checks and failures
    L.append("## 4 · §6 pre-flight")
    L.append("")
    L.append("| # | result | check | detail |")
    L.append("|---|---|---|---|")
    for c in s.get("checks", []):
        tag = {True: "PASS", False: "**FAIL**", None: "skip"}[c["passed"]]
        detail = (c.get("detail") or "").replace("|", "\\|")
        L.append(f"| {c['id']} | {tag} | {c['desc'].replace('|', chr(92) + '|')} | {detail} |")
    L.append("")
    failed = [c for c in s.get("checks", []) if c["passed"] is False]
    skipped = [c for c in s.get("checks", []) if c["passed"] is None]
    if failed:
        L.append("**Failures, verbatim:**")
        L.append("")
        for c in failed:
            L.append(f"- check {c['id']}: {c['desc']} — `{c.get('detail')}`")
        L.append("")
    else:
        L.append("No check failed.")
        L.append("")
    if skipped:
        L.append("Skipped, and why — none of these is recorded as a pass:")
        L.append("")
        for c in skipped:
            L.append(f"- check {c['id']}: {c.get('detail')}")
        L.append("")
    if s.get("notes"):
        L.append("Deviations recorded by the run:")
        L.append("")
        for n in s["notes"]:
            L.append(f"- {n}")
        L.append("")

    # -- (5) cost
    L.append("## 5 · Compute")
    L.append("")
    t = s.get("timings_seconds", {})
    total_h = s.get("gpu_hours_used") or (sum(t.values()) / 3600 if t else None)
    for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
        L.append(f"- `{k}` — {v / 60:.1f} min")
    if total_h:
        L.append("")
        L.append(f"**Measured: {total_h:.2f} GPU-hours** for this rung "
                 f"(the ladder budgeted ≤ 1 for R0).")
        ips = (s.get("stats_A") or {}).get("iters_per_second")
        if ips:
            L.append("")
            L.append(f"Measured throughput on {s.get('gpu')}: "
                     f"**{ips:.2f} iterations/second** for arm A, "
                     f"{(s.get('stats_B') or {}).get('iters_per_second', 0):.2f} for arm B. "
                     f"A 30 000-iteration scene is therefore ≈ "
                     f"{30000 / ips / 60:.0f} min of training per arm, before rendering "
                     f"and metrics — this replaces the estimates in §7 (G1).")
    L.append("")
    L.append("Remaining quota: not machine-readable from the kernels API; read the "
             "notebook sidebar (§8.1).")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="path to a kernel's summary.json")
    ap.add_argument("--protocol", default="STMT", choices=["STMT", "MTMT"])
    ap.add_argument("--md", help="write markdown here in addition to stdout")
    a = ap.parse_args()
    if not os.path.exists(a.summary):
        sys.exit(f"{a.summary} not found — the run has not produced a summary yet.")
    with open(a.summary) as f:
        s = json.load(f)
    md = build(s, a.protocol)
    print(md)
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\nwrote {a.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
