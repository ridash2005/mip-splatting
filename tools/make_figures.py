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
import argparse, os, csv, json, math, sys

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

# LaTeX embeds PDF; the web/report path uses SVG. Both are written so the two
# consumers never drift apart.
FORMATS = ["svg", "pdf"]


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in FORMATS:
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        print("wrote", p)
    plt.close(fig)

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
    # The gap that matters, drawn on the 1/8 column -- and drawn on whichever
    # pair of curves is in the foreground. The arrow used to be pinned to the
    # published points with the published 10.98 typed beside it, which on a
    # panel whose solid curves are this project's own measurement reads as a
    # measured gap. Both numbers are quoted; neither is typed.
    t0 = "Single-scale train → multi-scale test"
    m_b = (measured or {}).get((t0, "3DGS"))
    m_a = (measured or {}).get((t0, "Mip-Splatting"))
    if m_b and m_a:
        lo, hi, tag = m_b[3], m_a[3], "ours"
    else:
        lo, hi, tag = STMT["3DGS"][3], STMT["Mip-Splatting"][3], "published"
    axes[0].annotate("", xy=(3, lo), xytext=(3, hi),
                     arrowprops=dict(arrowstyle="<->", color=INK_2, lw=0.9,
                                     shrinkA=3, shrinkB=3))
    axes[0].text(2.9, (lo + hi) / 2,
                 f"{hi - lo:.2f} dB" + chr(10) + f"({tag})", ha="right", va="center",
                 fontsize=7.5, color=INK, weight="bold", linespacing=1.25)
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
# Figure 1 is the two-baseline scale-degradation plot and draws only these two.
# b1-fisher is listed so the unknown-slug guard below stays a guard: it exists to
# catch a method nobody has taught the figures about, and it started firing on a
# method this project produces on purpose, which left every figure stale against
# runs.csv without failing the build.
METHOD_LABEL = {"3dgs": "3DGS", "mip-splatting": "Mip-Splatting"}
METHOD_KNOWN = set(METHOD_LABEL) | {"b1-fisher"}


def load_runs(path, iterations=None):
    """Collapse results/runs.csv into {(protocol, method): [psnr @ 1x, 1/2, 1/4, 1/8]}.

    `iterations` restricts to one training length. Averaging a 7 000-iteration
    smoke row together with a 30 000-iteration R1 row would produce a number that
    describes neither run, so the caller must say which one it wants.

    Reads the record through make_tex's loaders, deliberately. This used to walk
    the CSV itself, and so it kept superseded rows, pooled all three seeds, and
    counted a scene measured twice twice -- which is why fig1 showed 34.61 dB
    where Table 1 showed 33.65 for the same quantity. A figure and a table that
    disagree about the same number are worse than either alone, and the only
    durable fix is one loader.
    """
    if not path or not os.path.exists(path):
        return None
    import make_tex as mt
    rows_in = mt.collapse_repeats(mt.load(path))
    rows_in = [r for r in rows_in if r.get("seed") == "0"]
    # Matched on scenes, per training protocol and independently, exactly as the
    # tables are. ship completes on the multi-scale protocol and not on the
    # single-scale one, so the two groups legitimately have different scene
    # sets; what neither may have is one curve over eight scenes against another
    # over seven.
    keep = []
    for ts in ("1x", "multi"):
        grp = [r for r in rows_in if r.get("train_scale") == ts]
        keep += mt.matched(grp, ("A", "B"))[0]
    rows_in = keep
    order = {"1x": 0, "1/2": 1, "1/4": 2, "1/8": 3}
    acc, seen_iters, bad_methods = {}, set(), set()
    if True:
        for r in rows_in:
            if r.get("dataset") != "blender" or not r.get("psnr"):
                continue
            seen_iters.add(r.get("iterations", ""))
            if iterations is not None and r.get("iterations") != str(iterations):
                continue
            slug = (r.get("method") or "").strip().lower()
            if slug not in METHOD_KNOWN:
                bad_methods.add(r.get("method"))
                continue
            label = METHOD_LABEL.get(slug)
            if label is None:
                continue            # known, but not one of this figure's two
            # Only the two standard training protocols. A stress row carries
            # train_scale like "1x/cone", which is neither, and the old
            # either/or put every one of them in the multi-scale group -- a
            # three-camera pencil averaged into a curve captioned
            # "multi-scale train".
            ts = r.get("train_scale")
            if ts == "1x":
                proto = "Single-scale train → multi-scale test"
            elif ts == "multi":
                proto = "Multi-scale train → multi-scale test"
            else:
                continue
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
    # The spans the built protocols actually realise, and the cap prediction at
    # those spans -- not a two-view formula read off a nominal angle.
    for ax, th_deg, title in ((axes[0], 20, "Low-parallax cone · span 20°"),
                              (axes[1], 80, "One-sided arc · span 80°")):
        th = np.radians(th_deg)
        ratio = _cap_sigma_ratio(th_deg)        # sigma_depth / sigma_lat
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


def _cap_sigma_ratio(span_deg):
    """sigma_D/sigma_L for views uniform on a cap of this angular span.

    The closed form of §4.9, verified in tools/test_geometry_claims.py. Imported
    from there rather than reimplemented, so a figure can never disagree with the
    derivation it illustrates.
    """
    import math
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_geometry_claims import cap_ratio
    return 1.0 / math.sqrt(max(cap_ratio(math.radians(span_deg) / 2.0), 1e-12))


def _instrument_protocols(path="results/kaggle_runs/instruments/summary.json"):
    """(span, measured sigma_D/sigma_L, label) per protocol, or [] if unmeasured."""
    if not os.path.exists(path):
        return []
    try:
        bp = (json.load(open(path)) or {}).get("by_protocol") or {}
    except Exception:
        return []
    names = {"full": "full orbit", "grazing": "grazing", "arc": "one-sided arc",
             "cone": "low-parallax cone", "mixed": "mixed focal"}
    out = []
    for k, v in bp.items():
        if k in names and v.get("median_span_deg"):
            out.append((v["median_span_deg"], v["median_sigma_ratio_measured"],
                        names[k]))
    return sorted(out)


def fig_anisotropy_curve(out):
    """The prediction, both forms, against what the protocols actually measured.

    Two curves, because the difference between them is a correction this project
    had to make: the two-view closed form is exact for two cameras and understates
    a distributed capture by sqrt(2). Plotting only the two-view curve would put
    every measured point above the prediction and invite the reading that the
    model is wrong, when the model being applied was the wrong one.
    """
    import numpy as np
    deg = np.linspace(2.0, 179, 500)
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ax.plot(deg, [_cap_sigma_ratio(d) for d in deg], color=S1, lw=2.0, zorder=4,
            label="prediction — cameras over a cap (Eq. 4.10)")
    ax.plot(deg, 1/np.sin(np.radians(deg)/2), color=S1, lw=1.4, ls=(0, (4, 3)),
            zorder=3, label="two-view closed form (Eq. 4.8) — understates by $\\sqrt{2}$")
    ax.axhline(1.0, color=S2, lw=1.6, zorder=2,
               label="Mip-Splatting assumes 1 everywhere")

    meas = _instrument_protocols()
    for i, (span, val, name) in enumerate(meas):
        ax.plot(span, val, "o", color=INK, ms=5.5, mec=SURFACE, mew=1.2, zorder=6,
                label="measured (I-2, 8 scenes)" if i == 0 else None)
    # full orbit and mixed focal sit at the same span and within 0.02x of each
    # other -- which is itself the mixed-focal result -- so they are labelled as
    # one point rather than as two overlapping ones.
    by_name = {n: (s, v) for s, v, n in meas}
    groups = []
    if "full orbit" in by_name and "mixed focal" in by_name:
        s, v = by_name["full orbit"]
        groups.append(((s, v), f"full orbit {v:.2f}×\n"
                               f"mixed focal {by_name['mixed focal'][1]:.2f}×",
                       (-72, 30)))
    for n, off in (("grazing", (-52, 4)), ("one-sided arc", (8, 12)),
                   ("low-parallax cone", (10, -20))):
        if n in by_name:
            s, v = by_name[n]
            groups.append(((s, v), f"{n}\n{v:.2f}×", off))
    for (s, v), text, off in groups:
        lead = abs(off[0]) > 30 or abs(off[1]) > 20
        ax.annotate(text, (s, v), xytext=off, textcoords="offset points",
                    fontsize=7.0, color=INK_2, linespacing=1.3, zorder=7,
                    arrowprops=(dict(arrowstyle="-", color=MUTED, lw=0.6,
                                     shrinkA=1, shrinkB=3) if lead else None))
    ax.set_yscale("log"); ax.set_ylim(0.82, 40); ax.set_xlim(0, 190)
    ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1×", "2×", "5×", "10×", "20×"])
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("angular span of the training cameras, at the primitive", fontsize=7.5)
    ax.set_ylabel("depth / lateral uncertainty", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.9, loc="upper right", labelcolor=INK_2,
              handlelength=2.0)
    _clean(ax)
    _save(fig, out, "fig6-anisotropy")


# ----------------------------------------------------------- figs 7-10, measured
# Everything below draws only from results/runs.csv and the runs' own summaries.
# Each returns without writing anything if its measurement does not exist yet: a
# figure is a claim, and an empty panel is preferable to an invented one.

PROTO_ORDER = ["full", "grazing", "mixed", "arc", "cone", "pencil"]
PROTO_LABEL = {"full": "full orbit", "grazing": "grazing", "mixed": "mixed focal",
               "arc": "one-sided arc", "cone": "low-parallax cone",
               "pencil": "narrow pencil"}


def _placeholder_panel(out, name, msg):
    """A visibly empty figure, so the document builds and says what is missing.

    The alternative --- emitting nothing --- breaks the LaTeX build on a missing
    \\includegraphics, and the alternative to that is drawing the figure from
    something other than a measurement, which is the one thing this file exists
    to prevent.
    """
    fig, ax = plt.subplots(figsize=(5.6, 1.5))
    ax.axis("off")
    ax.text(0.5, 0.62, "not yet measured", ha="center", va="center",
            fontsize=11, color=MUTED, style="italic")
    ax.text(0.5, 0.26, msg, ha="center", va="center", fontsize=7.2, color=MUTED)
    _save(fig, out, name)


def _rows(path="results/runs.csv"):
    """The record, through the one loader. See load_runs for why not a copy."""
    if not os.path.exists(path):
        return []
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_tex as mt
    return mt.load(path)


def _summaries(root="results/kaggle_runs"):
    out = []
    for d, _, files in os.walk(root):
        for fn in files:
            if fn.endswith("summary.json") or fn == "fps.json":
                try:
                    out.append(json.load(open(os.path.join(d, fn))))
                except Exception:
                    pass
    return out


# Every column that makes two rows a different configuration rather than a
# repeat of the same one. Leaving any of these out silently merges unrelated
# runs and reports their difference as harness noise: dropping load_allres alone
# turned the 0.039 dB atomics spread into 2.4 dB, because it pooled STMT with
# MTMT.
CONFIG_KEY = ("dataset", "scene", "method", "arm", "train_scale", "test_scale",
              "iterations", "load_allres", "kernel_size", "seed")


def _seed_sigma(rows):
    """Largest spread between repeats of one identical configuration, in dB."""
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[tuple(r.get(k) for k in CONFIG_KEY)].append(float(r["psnr"]))
    sp = [max(v) - min(v) for v in g.values() if len(v) > 1]
    return max(sp) if sp else None


def fig_stress(out):
    """The decisive experiment: paired Delta per protocol against the noise floor.

    Paired within (scene, test scale) because both methods train on the same
    protocol subset in the same session; a difference of two scene means would
    not be the same quantity.
    """
    import numpy as np
    rows = [r for r in _rows() if "/" in (r.get("train_scale") or "")]
    if not rows:
        print("fig7: no stress-suite rows yet — placeholder")
        _placeholder_panel(out, "fig7-stress-delta",
                           "generated from results/runs.csv once the stress "
                           "suite has run")
        return
    sigma = _seed_sigma(_rows()) or 0.039

    acc = {}
    for r in rows:
        proto = r["train_scale"].split("/")[1]
        acc.setdefault(proto, {}).setdefault(r["method"], {}) \
           .setdefault((r["scene"], r["test_scale"]), []).append(float(r["psnr"]))

    floor = {}
    for s in _summaries():
        for key, job in (s.get("done") or {}).items():
            parts = key.split("/")
            if len(parts) == 3 and parts[0] == "b1" and "frac_above_floor" in job:
                floor.setdefault(parts[2], []).append(job["frac_above_floor"])

    protos, deltas, above = [], [], []
    for p in PROTO_ORDER:
        d = acc.get(p) or {}
        mip, b1 = d.get("mip-splatting", {}), d.get("b1-fisher", {})
        keys = sorted(set(mip) & set(b1))
        if not keys:
            continue
        paired = [sum(b1[k]) / len(b1[k]) - sum(mip[k]) / len(mip[k]) for k in keys]
        protos.append(PROTO_LABEL[p])
        deltas.append(sum(paired) / len(paired))
        f = floor.get(p)
        above.append(sum(f) / len(f) if f else None)
    if not protos:
        print("fig7: no protocol has both methods — placeholder")
        _placeholder_panel(out, "fig7-stress-delta",
                           "no capture protocol has been run under both methods "
                           "yet")
        return

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    y = np.arange(len(protos))
    ax.axvspan(-3 * sigma, 3 * sigma, color=GRID, zorder=1)
    ax.axvline(0, color=MUTED, lw=1.0, zorder=2)
    ax.barh(y, deltas, height=0.55, zorder=3,
            color=[S1 if v > 0 else S2 for v in deltas])
    for i, (v, a) in enumerate(zip(deltas, above)):
        off = 4 if v >= 0 else -4
        ax.annotate(f"{v:+.3f} dB   ({v/sigma:+.1f}σ)", (v, i), xytext=(off, 0),
                    textcoords="offset points", fontsize=7.0, color=INK,
                    va="center", ha="left" if v >= 0 else "right")
        if a is not None:
            ax.annotate(f"{a*100:.0f}% above floor", (0, i), xytext=(4, -11),
                        textcoords="offset points", fontsize=6.6, color=MUTED,
                        va="center", ha="left")
    ax.set_yticks(y); ax.set_yticklabels(protos, fontsize=7.6)
    ax.invert_yaxis()
    lim = max(0.16, max(abs(v) for v in deltas) * 1.9, 4 * sigma)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("PSNR difference, B1 − Mip-Splatting (dB), paired per scene and scale",
                  fontsize=7.4)
    # Inside the axes, always. Placed at 3*sigma it lands outside the limits the
    # moment sigma is large, and bbox_inches="tight" then stretches the figure to
    # contain it -- which produced a 15000-pixel-wide panel.
    ax.text(min(3 * sigma, lim * 0.92), len(protos) - 0.35,
            f"  ±3σ = ±{3*sigma:.3f} dB",
            fontsize=6.8, color=MUTED, va="center", ha="right")
    _clean(ax, ygrid=False); ax.xaxis.grid(True)
    _save(fig, out, "fig7-stress-delta")


def fig_floor(out):
    """Where the Nyquist floor stops binding — the mechanism, in one panel.

    The method can only differ from its baseline where the estimation term
    exceeds the floor. This is that quantity, measured, across the capture
    protocols, with unity marked: below it Proposition 2 applies and the two are
    the same filter.
    """
    p = "results/kaggle_runs/instruments/summary.json"
    if not os.path.exists(p):
        print("fig8: instruments not run — placeholder")
        _placeholder_panel(out, "fig8-floor-binding", "the instruments have not run")
        return
    bp = (json.load(open(p)) or {}).get("by_protocol") or {}
    items = [(k, bp[k]) for k in PROTO_ORDER if k in bp]
    if not items:
        print("fig8: no protocols measured — placeholder")
        _placeholder_panel(out, "fig8-floor-binding", "no protocol measured")
        return

    import numpy as np
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(items))
    vals = [v["median_ratio_est_over_floor_p50"] for _, v in items]
    ax.axhline(1.0, color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax.bar(x, vals, width=0.55, zorder=4,
           color=[S1 if v > 1 else REF for v in vals])
    for i, val in enumerate(vals):
        ax.annotate(f"{val:.2f}", (i, val), xytext=(0, 3), textcoords="offset points",
                    fontsize=7.2, color=INK, ha="center", weight="bold")
    ax.set_yscale("log"); ax.set_ylim(0.08, 400)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    # The span and the floor occupancy belong with the protocol's name, not
    # written over its bar where the contrast fights the fill.
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{PROTO_LABEL[k]}\nspan {v['median_span_deg']:.0f}°\n"
         f"{v['median_frac_exceeding_floor_worst_dir']*100:.0f}% / "
         f"{v['median_frac_exceeding_floor_all_dirs']*100:.0f}%"
         for k, v in items], fontsize=6.9, linespacing=1.5)
    ax.set_ylabel("estimation term ÷ Nyquist floor  (median)", fontsize=7.5)
    # One caption block, in the only region no bar occupies. Splitting it above
    # and below the line reads better but puts grey text across three bars.
    ax.text(-0.42, 78,
            "dashed line: the Nyquist floor.\n"
            "above it the estimation term governs, and B1 can differ;\n"
            "below it Proposition 2 makes B1 $\\equiv$ Mip-Splatting exactly.",
            fontsize=7.0, color=INK_2, ha="left", va="top", linespacing=1.5)
    _clean(ax)
    ax.tick_params(axis="x", pad=4)
    _save(fig, out, "fig8-floor-binding")


def fig_parity(out):
    """Proposition 2 as a picture: B1 against its baseline on matched scenes."""
    rows = [r for r in _rows()
            if r.get("dataset") == "blender" and r.get("iterations") == "30000"
            and r.get("load_allres") == "False" and r.get("seed") == "0"
            and (r.get("train_scale") == "1x"
                 or (r.get("train_scale") or "").endswith("/full"))]
    per = {}
    for r in rows:
        per.setdefault(r["method"], set()).add(r["scene"])
    order = [m for m in ("3dgs", "mip-splatting", "b1-fisher") if per.get(m)]
    if len(order) < 2:
        print("fig9: fewer than two methods measured — placeholder")
        _placeholder_panel(out, "fig9-parity", "fewer than two methods measured")
        return
    common = set.intersection(*(per[m] for m in order))
    if not common:
        print("fig9: no scene measured under every method — placeholder")
        _placeholder_panel(out, "fig9-parity",
                           "no scene has been measured under every method")
        return

    import numpy as np
    scales = ["1x", "1/2", "1/4", "1/8"]
    acc = {}
    for r in rows:
        if r["scene"] in common and r["method"] in order:
            acc.setdefault(r["method"], {}).setdefault(r["test_scale"], []) \
               .append(float(r["psnr"]))
    sigma = _seed_sigma(_rows()) or 0.039

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(scales))
    style = {"3dgs": (S2, "3DGS"), "mip-splatting": (S1, "Mip-Splatting"),
             "b1-fisher": (INK, "B1 — Fisher band-limit")}
    for m in order:
        c, lab = style[m]
        vals = [sum(acc[m][s]) / len(acc[m][s]) if acc[m].get(s) else None
                for s in scales]
        ls = (0, (5, 2)) if m == "b1-fisher" else "-"
        axes[0].plot(x, vals, color=c, lw=1.9, ls=ls, marker="o", ms=4.5,
                     mec=SURFACE, mew=1.0, label=lab, zorder=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["full", "½", "¼", "⅛"])
    axes[0].set_xlabel("test scale", fontsize=7.5)
    axes[0].set_ylabel("PSNR (dB)", fontsize=7.5)
    axes[0].legend(frameon=False, fontsize=7.0, labelcolor=INK_2, loc="lower left")
    axes[0].set_title(f"{len(common)} scene(s) measured under every method",
                      fontsize=8.0, loc="left", pad=8)
    _clean(axes[0])

    # The delta panel is the parity claim, and it has to be PAIRED: same capture,
    # same harness invocation, same build on both sides. Selecting on method
    # alone let R1's standard-orbit Mip-Splatting rows stand in for
    # Mip-Splatting at the full protocol, three commits older, which is the same
    # defect the methodParityMax macro carried.
    pair = {}
    for r in rows:
        if r["method"] not in ("mip-splatting", "b1-fisher"):
            continue
        if not (r.get("train_scale") or "").endswith("/full"):
            continue
        pair.setdefault((r["scene"], r["test_scale"], r.get("train_scale"),
                         r.get("impl_commit")), {})[r["method"]] = float(r["psnr"])
    both = {k: v for k, v in pair.items() if len(v) == 2}
    if both:
        per_scale = {}
        for (scene, scale, _, _), v in both.items():
            per_scale.setdefault(scale, []).append(
                v["b1-fisher"] - v["mip-splatting"])
        d = [sum(per_scale[s]) / len(per_scale[s]) if per_scale.get(s) else 0.0
             for s in scales]
        n_pair = len({k[0] for k in both})
        axes[1].axhspan(-sigma, sigma, color=GRID, zorder=1)
        axes[1].axhline(0, color=MUTED, lw=1.0, zorder=2)
        axes[1].bar(x, d, width=0.5, color=S1, zorder=3)
        for i, v in enumerate(d):
            axes[1].annotate(f"{v:+.3f}", (i, v), xytext=(0, 3 if v >= 0 else -10),
                             textcoords="offset points", fontsize=6.8, color=INK,
                             ha="center")
        axes[1].set_xticks(x); axes[1].set_xticklabels(["full", "½", "¼", "⅛"])
        axes[1].set_ylim(-max(0.25, max(abs(v) for v in d) * 2.2),
                         max(0.25, max(abs(v) for v in d) * 2.2))
        axes[1].set_ylabel("B1 − Mip-Splatting (dB)", fontsize=7.5)
        axes[1].set_title(
            f"Proposition 2 requires 0; paired on {n_pair} scene(s); "
            f"shaded band is ±σ = {sigma:.3f} dB",
            fontsize=7.6, loc="left", pad=8)
        _clean(axes[1], ygrid=False); axes[1].yaxis.grid(True)
    else:
        axes[1].text(0.5, 0.5, "no paired measurement yet" + chr(10) +
                               "(both methods, same run, same build)",
                     ha="center", va="center", fontsize=7.5, color=MUTED,
                     transform=axes[1].transAxes)
        axes[1].set_xticks([]); axes[1].set_yticks([])
        _clean(axes[1], ygrid=False)
    _save(fig, out, "fig9-parity")


def fig_b2(out):
    """B2: the fraction of non-DC SH the criterion keeps, by capture protocol."""
    # Prefer the B2 run's own sweep: it is built from the same camera_protocols
    # definition the instruments use, where the checked-in file predates the
    # switch from fixed camera counts to angular extent.
    p = None
    for d, _, files in os.walk("results/kaggle_runs"):
        for fn in files:
            if fn == "b2_protocols.json":
                p = os.path.join(d, fn)
    if p is None:
        p = "results/b2_protocols/protocols.json"
    if not os.path.exists(p):
        print("fig10: B2 protocol sweep not present — placeholder")
        _placeholder_panel(out, "fig10-b2-retention", "the B2 sweep has not run")
        return
    try:
        data = json.load(open(p))
    except Exception:
        return
    # Keyed "<scene>/<protocol>"; average over whatever scenes were evaluated.
    acc = {}
    for key, v in data.items():
        if not isinstance(v, dict) or "non_dc_retained" not in v:
            continue
        p = v.get("protocol") or (key.split("/")[-1])
        e = acc.setdefault(p, {"frac": [], "lmax": [], "span": []})
        e["frac"].append(v["non_dc_retained"])
        e["lmax"].append(v.get("mean_l_max"))
        if v.get("span_deg"):
            e["span"].append(v["span_deg"])
    rows = []
    for k in PROTO_ORDER:
        e = acc.get(k)
        if not e or not e["frac"]:
            continue
        lm = [x for x in e["lmax"] if x is not None]
        rows.append((PROTO_LABEL[k], sum(e["frac"]) / len(e["frac"]),
                     (sum(lm) / len(lm)) if lm else None,
                     (sum(e["span"]) / len(e["span"])) if e["span"] else None))
    if not rows:
        print("fig10: no usable B2 rows — placeholder")
        _placeholder_panel(out, "fig10-b2-retention", "no usable B2 rows")
        return

    import numpy as np
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] * 100 for r in rows], width=0.55, color=S1, zorder=3)
    for i, r in enumerate(rows):
        ax.annotate(f"{r[1]*100:.1f}%", (i, r[1] * 100), xytext=(0, 3),
                    textcoords="offset points", fontsize=7.0, color=INK, ha="center")
        if r[2] is not None:
            ax.annotate(f"$\\ell_{{max}}$ = {r[2]:.2f}", (i, 2.5), fontsize=6.7,
                        color=MUTED, ha="center", va="bottom", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=7.2)
    ax.set_ylim(0, 112)
    ax.set_ylabel("non-DC SH coefficients retained (%)", fontsize=7.5)
    ax.set_title("B2 keeps degree 3 where the views support it, and not otherwise",
                 fontsize=8.0, loc="left", pad=8)
    _clean(ax)
    _save(fig, out, "fig10-b2-retention")


# ------------------------------------------------------------------- entry point
# Defined last on purpose: figs 5/6 are declared below fig 4, and a main() placed
# above them would run before those names exist — which is why `make figures`
# previously emitted four figures and silently skipped the two theory ones.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["published", "measured"], default="published")
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--out",  default="figures")
    ap.add_argument("--formats", default="svg,pdf",
                    help="comma-separated output formats (pdf is what LaTeX embeds)")
    ap.add_argument("--iterations", type=int, default=30000,
                    help="only average rows from runs of this length (default 30000, "
                         "the R1/R2 target). Pass 7000 to draw the R0 smoke rows.")
    a = ap.parse_args()
    FORMATS[:] = [x.strip() for x in a.formats.split(",") if x.strip()]
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
    # Measured-result figures. Each refuses to draw if its run has not happened,
    # so the figure set is always a truthful picture of what exists.
    fig_stress(a.out)
    fig_floor(a.out)
    fig_parity(a.out)
    fig_b2(a.out)


if __name__ == "__main__":
    main()
