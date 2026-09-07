#!/usr/bin/env python3
"""
BTP baseline reproduction — figure generation.

Two modes:
  published   figures drawn from the numbers printed in the 3DGS and Mip-Splatting
              papers. These are NOT our measurements. Used to state the target.
  measured    the same figures redrawn from results/runs.csv once our runs land,
              with the published curve kept as a faint reference line.

Every figure in the report regenerates from this file. No number is ever typed
into a document by hand.

    python make_figures.py --mode published --out figures/
    python make_figures.py --mode measured  --runs results/runs.csv --out figures/
"""
import argparse, os, csv, math, sys

# The report is UTF-8 (arrows, section signs, en dashes). A Windows console
# defaults to cp1252 and would raise UnicodeEncodeError on the way out, so the
# stream is reconfigured rather than the text degraded.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------- design tokens
SURFACE     = "#fcfcfb"
INK         = "#1a1d21"
INK_2       = "#52514e"
MUTED       = "#8b939d"
GRID        = "#e3e7ec"
S1          = "#2a78d6"   # slot 1 — Mip-Splatting
S2          = "#eb6834"   # slot 2 — 3DGS
REF         = "#b9c0c8"   # published reference, when overlaid on measured

plt.rcParams.update({
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family":       "DejaVu Sans",
    "font.size":         8.0,
    "axes.edgecolor":    GRID,
    "axes.linewidth":    0.8,
    "axes.labelcolor":   INK_2,
    "axes.titlecolor":   INK,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelcolor":  INK_2,
    "ytick.labelcolor":  INK_2,
    "xtick.major.size":  0,
    "ytick.major.size":  0,
    "grid.color":        GRID,
    "grid.linewidth":    0.7,
    "svg.fonttype":      "path",
})

def _clean(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True); ax.xaxis.grid(False)

def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name + ".svg")
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", p)

# ------------------------------------------------------- published reference data
# Source: Mip-Splatting (Yu et al., CVPR 2024), arXiv:2311.16493.
SCALES   = ["1x", "1/2", "1/4", "1/8"]
STMT     = {"3DGS": [33.33, 26.95, 21.38, 17.69],
            "Mip-Splatting": [33.36, 34.00, 31.85, 28.67]}
MTMT     = {"3DGS": [28.79, 30.66, 31.64, 27.98],
            "Mip-Splatting": [32.81, 34.49, 35.45, 35.50]}
# Mip-NeRF 360, single-scale train / zoom-in test.
ZOOM     = {"3DGS": {"1x": 29.19, "8x": 19.59},
            "Mip-Splatting": {"1x": 29.39, "8x": 26.22}}
# 3DGS Table 1, Mip-NeRF 360 column: (label, train minutes, PSNR)
COST     = [("Plenoxels",      25.8,  23.08),
            ("INGP-Base",       5.6,  25.30),
            ("INGP-Big",        7.5,  25.59),
            ("Mip-NeRF 360",  2880.0, 27.69),
            ("3DGS 7K",         6.4,  25.60),
            ("3DGS 30K",       41.6,  27.21)]
# §05 ladder: (rung, low GPU-h, high GPU-h)
LADDER   = [("R0 smoke",        0.5,  1),
            ("R1 Blender STMT",10,   16),
            ("R2 Blender MTMT",14,   22),
            ("R3 real scenes", 20,   30),
            ("R5 seed spread",  4,    8),
            ("R4 outdoor (opt)",12,  20)]
WEEKLY_QUOTA_H = 30.0


# ------------------------------------------------------------------- fig 1
def fig_scale_degradation(out, measured=None):
    """The headline figure: what happens when test sampling rate leaves the training one."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    x = range(4)
    for ax, data, title in (
        (axes[0], STMT, "Single-scale train → multi-scale test"),
        (axes[1], MTMT, "Multi-scale train → multi-scale test"),
    ):
        for name, colour in (("3DGS", S2), ("Mip-Splatting", S1)):
            y = data[name]
            src = (measured or {}).get((title, name))
            if src:
                ax.plot(x, y, color=REF, lw=1.4, ls=(0, (3, 2)), zorder=2)
                ax.plot(x, src, color=colour, lw=2.0, marker="o", ms=4.5,
                        mec=SURFACE, mew=1.2, zorder=4, label=name)
            else:
                ax.plot(x, y, color=colour, lw=2.0, marker="o", ms=4.5,
                        mec=SURFACE, mew=1.2, zorder=4, label=name)
        ax.set_xticks(list(x)); ax.set_xticklabels(SCALES)
        ax.set_xlim(-0.28, 3.28); ax.set_ylim(15, 39)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlabel("test resolution", fontsize=7.5)
        ax.set_title(title, fontsize=8.2, pad=8, loc="left")
        _clean(ax)
    axes[0].set_ylabel("PSNR (dB)", fontsize=7.5)

    # direct labels, placed inside the axes above/below their own curve
    axes[0].annotate("Mip-Splatting", (2, STMT["Mip-Splatting"][2]), xytext=(-2, 9),
                     textcoords="offset points", color=INK, fontsize=7.5,
                     weight="bold", ha="center")
    axes[0].annotate("3DGS", (2, STMT["3DGS"][2]), xytext=(-2, -13),
                     textcoords="offset points", color=INK, fontsize=7.5,
                     weight="bold", ha="center")
    # the gap that matters, drawn on the 1/8 column
    axes[0].annotate("", xy=(3, STMT["3DGS"][3]), xytext=(3, STMT["Mip-Splatting"][3]),
                     arrowprops=dict(arrowstyle="<->", color=INK_2, lw=0.9,
                                     shrinkA=3, shrinkB=3))
    axes[0].text(2.9, (STMT["3DGS"][3] + STMT["Mip-Splatting"][3]) / 2,
                 "10.98 dB", ha="right", va="center", fontsize=7.5,
                 color=INK, weight="bold")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower left",
                   labelcolor=INK_2, handlelength=1.6)
    _save(fig, out, "fig1-scale-degradation")


# ------------------------------------------------------------------- fig 2
def fig_zoom(out):
    """Mip-NeRF 360, single-scale train, zoom-in test: 1x vs 8x."""
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    for i, (name, colour) in enumerate((("3DGS", S2), ("Mip-Splatting", S1))):
        a, b = ZOOM[name]["1x"], ZOOM[name]["8x"]
        ax.plot([0, 1], [a, b], color=colour, lw=2.0, marker="o", ms=5,
                mec=SURFACE, mew=1.2, label=name, zorder=3)
        ax.annotate(f"{b:.2f}", (1, b), xytext=(7, -2), textcoords="offset points",
                    fontsize=7.5, color=INK, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["1×", "8×"])
    ax.set_xlim(-0.42, 1.42); ax.set_ylim(17, 32)
    ax.set_ylabel("PSNR (dB)", fontsize=7.5)
    ax.set_xlabel("zoom-in factor at test time", fontsize=7.5)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    _clean(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", labelcolor=INK_2,
              handlelength=1.6)
    _save(fig, out, "fig2-mipnerf360-zoom")


# ------------------------------------------------------------------- fig 3
def fig_budget(out):
    """GPU-hour ladder against a weekly free-tier quota."""
    fig, ax = plt.subplots(figsize=(7.1, 2.6))
    names = [r[0] for r in LADDER]
    # cumulative: each rung starts where its predecessor ended, so the quota
    # lines mean what they appear to mean.
    cum_lo, cum_hi, lo_acc, hi_acc = [], [], 0.0, 0.0
    for _, lo, hi in LADDER:
        cum_lo.append((lo_acc, lo_acc + lo)); cum_hi.append((hi_acc, hi_acc + hi))
        lo_acc += lo; hi_acc += hi
    y = list(range(len(LADDER)))[::-1]
    for yi, (l0, l1), (h0, h1), nm in zip(y, cum_lo, cum_hi, names):
        optional = "(opt)" in nm
        c = S2 if optional else S1
        # band between the optimistic and pessimistic cumulative positions
        ax.barh(yi, h1 - l0, left=l0, height=0.44, color=c,
                alpha=0.16, edgecolor="none", zorder=2)
        ax.plot([l1, h1], [yi, yi], color=c, lw=2.4, solid_capstyle="round",
                zorder=3, alpha=0.95)
        ax.plot([l1, h1], [yi, yi], "o", color=c, ms=5, mec=SURFACE, mew=1.2, zorder=4)
        ax.text(h1 + 1.6, yi, f"cum. {l1:g}–{h1:g} h", va="center",
                fontsize=7.2, color=INK_2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.8)
    ax.tick_params(axis="y", pad=2)
    for k in (1, 2, 3):
        ax.axvline(k * WEEKLY_QUOTA_H, color=MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1)
        ax.text(k * WEEKLY_QUOTA_H, len(LADDER) - 0.28,
                f"week {k}" if k > 1 else "1 week of quota",
                fontsize=6.9, color=MUTED, ha="center")
    ax.set_xlim(0, 118); ax.set_ylim(-0.7, len(LADDER) + 0.15)
    ax.set_xlabel("cumulative GPU-hours (estimated; replaced by R0's measurement)",
                  fontsize=7.5)
    _clean(ax, ygrid=False)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.spines["left"].set_visible(False)
    _save(fig, out, "fig3-compute-budget")


# ------------------------------------------------------------------- fig 4
def fig_cost_quality(out):
    """3DGS Table 1, Mip-NeRF 360: quality against training cost."""
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    # hand-placed labels: several points sit within 0.3 dB of each other
    OFF = {"Plenoxels":    ( 9,  -3, "left"),
           "INGP-Base":    ( 2, -14, "center"),
           "INGP-Big":     ( 9,  -3, "left"),
           "3DGS 7K":      ( 0,  10, "center"),
           "3DGS 30K":     ( 9,  -3, "left"),
           "Mip-NeRF 360": (-9,  -3, "right")}
    for name, mins, psnr in COST:
        gs = name.startswith("3DGS")
        c = S1 if gs else MUTED
        ax.plot(mins, psnr, "o", ms=7 if gs else 5.5, color=c,
                mec=SURFACE, mew=1.3, zorder=4 if gs else 3)
        dx, dy, ha = OFF[name]
        ax.annotate(name, (mins, psnr), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.0, ha=ha,
                    color=INK if gs else INK_2,
                    weight="bold" if gs else "normal")
    ax.set_xscale("log")
    ax.set_xlim(3.2, 6000); ax.set_ylim(22.4, 28.6)
    ax.set_xticks([10, 60, 600, 3000])
    ax.set_xticklabels(["10 min", "1 h", "10 h", "48 h"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("training time, single GPU (log)", fontsize=7.5)
    ax.set_ylabel("PSNR (dB)", fontsize=7.5)
    _clean(ax)
    _save(fig, out, "fig4-cost-quality")


# ------------------------------------------------------------------- measured
# results/runs.csv stores the method as a lowercase slug; the figures key off the
# display names used in STMT/MTMT above. Without this mapping every measured key
# misses, load_runs still returns a non-empty dict, the "refuses to draw" gate
# passes, and fig 1 silently draws the PUBLISHED curve under a "measured" filename
# — exactly what §11/§14 forbid. Any slug not listed here is a hard error.
METHOD_LABEL = {"3dgs": "3DGS", "mip-splatting": "Mip-Splatting"}


def load_runs(path, iterations=None):
    """Collapse results/runs.csv into {(protocol, method): [psnr @ 1x, 1/2, 1/4, 1/8]}.

    `iterations` restricts to one training length. Averaging a 7 000-iteration
    smoke row together with a 30 000-iteration R1 row would produce a number that
    describes neither run, so the caller must say which one it wants.
    """
    if not path or not os.path.exists(path):
        return None
    order = {"1x": 0, "1/2": 1, "1/4": 2, "1/8": 3}
    acc, seen_iters, bad_methods = {}, set(), set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("dataset") != "blender" or not r.get("psnr"):
                continue
            seen_iters.add(r.get("iterations", ""))
            if iterations is not None and r.get("iterations") != str(iterations):
                continue
            label = METHOD_LABEL.get((r.get("method") or "").strip().lower())
            if label is None:
                bad_methods.add(r.get("method"))
                continue
            proto = ("Single-scale train → multi-scale test"
                     if r.get("train_scale") == "1x"
                     else "Multi-scale train → multi-scale test")
            key = (proto, label)
            acc.setdefault(key, {}).setdefault(r["test_scale"], []).append(float(r["psnr"]))
    if bad_methods:
        raise SystemExit(
            f"unrecognised method slug(s) {sorted(bad_methods)} in {path}; "
            f"known: {sorted(METHOD_LABEL)}. Refusing to draw rather than drop rows.")
    out = {}
    for key, byscale in acc.items():
        if len(byscale) < 4:
            continue
        out[key] = [sum(byscale[s]) / len(byscale[s]) for s in sorted(byscale, key=order.get)]
    if not out:
        print(f"note: no complete 4-scale group in {path} at iterations={iterations}; "
              f"iteration counts present: {sorted(i for i in seen_iters if i)}")
    return out or None




# ------------------------------------------------------------------- fig 5/6
# Theory figures. The geometry below is exact, not illustrative: Lambda is
# computed from real pinhole Jacobians and the ellipses are its inverse.

def _cam_fisher(p, C, f=1000.0, sig=1.0, up=None):
    import numpy as np
    up = np.array([0., 1., 0.]) if up is None else up
    z_ax = p - C; z_ax = z_ax / np.linalg.norm(z_ax)
    x_ax = np.cross(up, z_ax); x_ax = x_ax / np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    R = np.stack([x_ax, y_ax, z_ax])
    x, y, z = R @ (p - C)
    J = np.array([[f/z, 0, -f*x/z**2], [0, f/z, -f*y/z**2]])
    A = J @ R
    return A.T @ A / sig**2


def fig_fisher_geometry(out):
    """Filter shape from accumulated Fisher information, two capture geometries.

    Both filters are normalised to the SAME lateral extent, because the floor
    sigma_i^2 = max(f_k^2, s^2/lambda_i) makes them agree there by construction.
    The whole difference is along depth, and equals 1/sin(theta/2) exactly.
    """
    import numpy as np
    t = np.linspace(0, 2*np.pi, 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))
    LIM = 6.6
    for ax, th_deg, title in ((axes[0], 20, "One-sided arc · θ = 20°"),
                              (axes[1], 90, "Full orbit · θ = 90°")):
        th = np.radians(th_deg)
        ratio = 1.0 / np.sin(th / 2)            # sigma_depth / sigma_lat, exact
        # camera rays, drawn to the primitive at the origin
        for a in (-th/2, th/2):
            d = np.array([np.sin(a), -np.cos(a)])
            ax.plot([d[0]*LIM*0.93, 0], [d[1]*LIM*0.93, 0], color=MUTED,
                    lw=0.9, ls=(0, (3, 3)), zorder=1)
            ax.plot(d[0]*LIM*0.93, d[1]*LIM*0.93, "s", color=MUTED, ms=5, zorder=2)
        ax.plot(np.cos(t), np.sin(t), color=S2, lw=2.0, zorder=4,
                label="Mip-Splatting — isotropic $f_k$")
        ax.plot(np.cos(t), ratio*np.sin(t), color=S1, lw=2.0, zorder=3,
                label="this work — $\\Sigma_{filt}$ from $\\Lambda^{-1}$")
        ax.plot(0, 0, "o", color=INK, ms=3.5, zorder=5)
        ax.annotate("", xy=(0, ratio), xytext=(0, 1),
                    arrowprops=dict(arrowstyle="<->", color=INK_2, lw=0.9))
        ax.text(0.45, (1+ratio)/2, f"{ratio:.1f}×", fontsize=7.6,
                color=INK, weight="bold", va="center", ha="left")
        ax.set_xlim(-LIM*1.15, LIM*1.15); ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8.2, pad=8, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ("top", "right", "left", "bottom"):
            ax.spines[s_].set_color(GRID)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.4, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.045), labelcolor=INK_2, handlelength=1.8,
               columnspacing=2.4)
    axes[0].text(-LIM*1.08, LIM*0.90, "depth axis —\nno single view\nconstrains it",
                 fontsize=7.0, color=MUTED, va="top", linespacing=1.4)
    _save(fig, out, "fig5-fisher-geometry")


def fig_anisotropy_curve(out):
    """sigma_depth/sigma_lat = 1/sin(theta/2), with the capture protocols marked."""
    import numpy as np
    th = np.radians(np.linspace(1.5, 120, 400))
    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    ax.plot(np.degrees(th), 1/np.sin(th/2), color=S1, lw=2.0, zorder=3)
    ax.axhline(1.0, color=S2, lw=1.6, zorder=2)
    ax.text(112, 1.14, "Mip-Splatting assumes 1", fontsize=7.0, color=S2, ha="right")
    for deg, name in ((6, "low-parallax"), (20, "one-sided arc"), (75, "full orbit")):
        v = 1/np.sin(np.radians(deg)/2)
        ax.plot(deg, v, "o", color=INK, ms=5, mec=SURFACE, mew=1.2, zorder=4)
        ax.annotate(f"{name}\n{v:.1f}×", (deg, v), xytext=(7, 3),
                    textcoords="offset points", fontsize=7.0, color=INK_2)
    ax.set_yscale("log"); ax.set_ylim(0.8, 60); ax.set_xlim(0, 120)
    ax.set_yticks([1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["1×","2×","5×","10×","20×","50×"])
    ax.set_xlabel("parallax angle θ subtended at the primitive", fontsize=7.5)
    ax.set_ylabel("depth / lateral uncertainty", fontsize=7.5)
    _clean(ax)
    _save(fig, out, "fig6-anisotropy")


# ------------------------------------------------------------------- entry point
# Defined last on purpose: figs 5/6 are declared below fig 4, and a main() placed
# above them would run before those names exist — which is why `make figures`
# previously emitted four figures and silently skipped the two theory ones.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["published", "measured"], default="published")
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--out",  default="figures")
    ap.add_argument("--iterations", type=int, default=30000,
                    help="only average rows from runs of this length (default 30000, "
                         "the R1/R2 target). Pass 7000 to draw the R0 smoke rows.")
    a = ap.parse_args()
    measured = load_runs(a.runs, a.iterations) if a.mode == "measured" else None
    if a.mode == "measured" and not measured:
        raise SystemExit(
            "no usable rows in %s at iterations=%s — refusing to draw a 'measured' "
            "figure with published numbers in it. Run the ladder first."
            % (a.runs, a.iterations))
    if measured:
        # ASCII only: this runs on a Windows cp1252 console too, and a figure
        # tool must not die reporting what it is about to draw.
        for (proto, name), vals in sorted(measured.items()):
            proto_code = "STMT" if proto.startswith("Single") else "MTMT"
            print(f"measured series: {proto_code} / {name} -> "
                  + ", ".join(f"{v:.2f}" for v in vals))
    fig_scale_degradation(a.out, measured)
    fig_zoom(a.out)
    fig_budget(a.out)
    fig_cost_quality(a.out)
    fig_fisher_geometry(a.out)
    fig_anisotropy_curve(a.out)


if __name__ == "__main__":
    main()
