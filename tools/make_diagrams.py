#!/usr/bin/env python3
r"""
Explanatory diagrams --- the pictures that carry the ideas.

These are different in kind from the figures in tools/make_figures.py. Those plot
measurements. These plot *concepts*: what goes wrong when a primitive is smaller
than a pixel, what a camera does and does not tell you about where a point is,
what six capture protocols actually look like. Nothing here is data; everything
here is an argument someone would otherwise have to draw on a whiteboard.

They exist because this thesis is meant to be given as a talk as well as read,
and because the central objects --- an observability matrix, an anisotropic
covariance floor, a capture geometry --- are all spatial, and none of them is
obvious from an equation on first meeting.

    python tools/make_diagrams.py --out thesis/figures
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Circle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import SURFACE, INK, INK_2, MUTED, GRID, S1, S2, REF  # noqa: E402

FORMATS = ["pdf", "png"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in FORMATS:
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03, dpi=220)
        print("wrote", p)
    plt.close(fig)


def _bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


# ===================================================================== D1
def d_aliasing(out):
    """Why a sub-pixel primitive is fine at one resolution and wrong at another.

    The single most important intuition in the thesis, and the one an equation
    conveys worst. The primitives are the same size in both panels -- they are a
    property of the model, not of the camera. What changes is the pixel grid.
    At the training rate a pixel holds one or two of them and the fit tunes
    their opacities so the total is right. Sample more coarsely and the same
    pixel holds nine, each still drawn at the strength that was correct when it
    had a pixel to itself. The image gains energy it should not have, which is
    why the reconstruction gets brighter and fatter rather than merely blurrier.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))
    rng = np.random.default_rng(11)
    pts = rng.normal(0, 0.75, size=(60, 2))
    pts = pts[(np.abs(pts) < 1.95).all(axis=1)]
    r = 0.075

    for ax, (step, title) in zip(axes, [(0.5, "sampled at the training rate"),
                                        (1.0, "sampled half as finely")]):
        for g in np.arange(-2, 2.001, step):
            ax.axvline(g, color=GRID, lw=0.9, zorder=1)
            ax.axhline(g, color=GRID, lw=0.9, zorder=1)
        # the pixel we count in: the one just below-left of the origin
        x0, y0 = -step, -step
        inside = [(x, y) for (x, y) in pts if x0 <= x < x0 + step and y0 <= y < y0 + step]
        ax.add_patch(plt.Rectangle((x0, y0), step, step, facecolor=S1,
                                   alpha=0.16, ec=S1, lw=1.5, zorder=2))
        for (x, y) in pts:
            hot = (x, y) in inside
            ax.add_patch(Circle((x, y), r, color=S2, zorder=4, ec="none",
                                alpha=0.95 if hot else 0.42))
        ax.set_xlim(-2.05, 2.05); ax.set_ylim(-2.05, 2.05)
        _bare(ax)
        ax.set_title(title, fontsize=8.6, color=INK, pad=8)
        ax.text(0, -2.28, f"{len(inside)} primitives in the marked pixel",
                ha="center", va="top", fontsize=8.0, color=S1, weight="bold")

    axes[0].text(0, -2.62, "each contributes about what it should",
                 ha="center", va="top", fontsize=7.5, color=INK_2)
    axes[1].text(0, -2.62, "each still contributes the same amount,\n"
                           "so the pixel receives far too much",
                 ha="center", va="top", fontsize=7.5, color=INK_2,
                 linespacing=1.35)
    fig.suptitle("The primitives do not change size. The pixels do.",
                 fontsize=9.2, color=INK, y=1.015)
    _save(fig, out, "d1-aliasing")


# ===================================================================== D2
def d_bandlimit_locations(out):
    """Where each method puts its floor, and what each floor is made of."""
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    rows = [
        ("3DGS", "no world-space floor at all",
         "a fixed dilation of the projected ellipse,\napplied at render time", S2),
        ("Mip-Splatting", "one number per primitive,\nfrom the nearest camera",
         "the same fixed dilation,\nplus an opacity correction", S1),
        ("this work", "a $3{\\times}3$ matrix per primitive,\n"
         "from every camera that saw it",
         "unchanged — the screen-space\nterm is not what this touches", INK),
    ]
    y = 2.35
    ax.text(0.44, y + 0.62, "world-space limit\n(how small may a primitive be?)",
            ha="center", fontsize=8.0, color=INK, linespacing=1.3)
    ax.text(0.82, y + 0.62, "screen-space limit\n(how small may its image be?)",
            ha="center", fontsize=8.0, color=INK, linespacing=1.3)
    for name, world, screen, col in rows:
        ax.text(0.015, y, name, fontsize=8.0, color=col, weight="bold",
                va="center", ha="left")
        ax.text(0.44, y, world, fontsize=7.5, color=INK_2, va="center",
                ha="center", linespacing=1.35)
        ax.text(0.82, y, screen, fontsize=7.5, color=INK_2, va="center",
                ha="center", linespacing=1.35)
        y -= 1.0
    ax.axhline(2.92, xmin=0.02, xmax=0.98, color=GRID, lw=1.0)
    for yy in (1.85, 0.85):
        ax.axhline(yy, xmin=0.02, xmax=0.98, color=GRID, lw=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0.0, 3.35)
    _bare(ax); ax.set_aspect("auto")
    ax.text(0.5, -0.08, "The column on the left is the subject of this thesis. "
            "Nothing here changes the column on the right.",
            ha="center", fontsize=7.4, color=MUTED, transform=ax.transAxes)
    _save(fig, out, "d2-where-the-limit-lives")


# ===================================================================== D3
def d_one_camera(out):
    """What one camera tells you, and what it does not.

    Left: a single camera pins a point well across its line of sight and badly
    along it -- the uncertainty is a long thin cigar pointing at the camera.
    Right: a second camera from elsewhere is strong exactly where the first is
    weak, and the intersection is compact. This is the whole idea of the method
    in one picture.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    P = np.array([0.0, 0.0])

    def cam(ax, pos, col, label, lab_off=(0, -0.28), half_deg=17.0, reach=1.45):
        """Draw a camera at `pos` looking at the origin, with its view cone."""
        d = P - pos
        L = np.linalg.norm(d)
        d = d / L
        for s in (-1, 1):
            a = np.radians(s * half_deg)
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            e = pos + (R @ d) * L * reach
            ax.plot([pos[0], e[0]], [pos[1], e[1]], color=col, lw=0.9,
                    alpha=0.40, zorder=2)
        ax.add_patch(Circle(pos, 0.085, color=col, zorder=5, ec="none"))
        ax.text(pos[0] + lab_off[0], pos[1] + lab_off[1], label, fontsize=7.4,
                color=col, ha="center", va="top")
        return d, np.array([-d[1], d[0]])

    # --- left: one camera
    ax = axes[0]
    d, n = cam(ax, np.array([0.0, -2.35]), S1, "camera")
    ang = np.degrees(np.arctan2(d[1], d[0]))
    ax.add_patch(Ellipse(P, 2.30, 0.34, angle=ang, facecolor=S1, alpha=0.20,
                         ec=S1, lw=1.2, zorder=3))
    ax.annotate("", xy=(0, 1.05), xytext=(0, -1.05),
                arrowprops=dict(arrowstyle="<->", color=INK_2, lw=0.9))
    ax.text(0.10, 0.72, "poorly known\n(depth)", fontsize=7.2, color=INK,
            ha="left", linespacing=1.3)
    ax.annotate("", xy=(0.52, 0), xytext=(-0.52, 0),
                arrowprops=dict(arrowstyle="<->", color=INK_2, lw=0.9))
    ax.text(0.60, -0.16, "well known\n(across the view)", fontsize=7.2,
            color=INK, ha="left", linespacing=1.3)
    ax.set_title("one camera", fontsize=8.6, color=INK, pad=8)

    # --- right: two cameras
    ax = axes[1]
    cam(ax, np.array([-1.85, -1.55]), S1, "camera 1", (-0.10, -0.22))
    cam(ax, np.array([1.85, -1.55]), S2, "camera 2", (0.10, -0.22))
    ax.add_patch(Ellipse(P, 2.30, 0.34, angle=40, facecolor=S1, alpha=0.15,
                         ec=S1, lw=1.0, zorder=3))
    ax.add_patch(Ellipse(P, 2.30, 0.34, angle=-40, facecolor=S2, alpha=0.15,
                         ec=S2, lw=1.0, zorder=3))
    ax.add_patch(Ellipse(P, 0.52, 0.40, angle=0, facecolor=INK, alpha=0.30,
                         ec=INK, lw=1.3, zorder=4))
    ax.text(0.34, 0.34, "what both\nagree on", fontsize=7.2, color=INK,
            ha="left", linespacing=1.3)
    ax.set_title("two cameras, well separated", fontsize=8.6, color=INK, pad=8)

    for ax in axes:
        ax.add_patch(Circle(P, 0.055, color=INK, zorder=6, ec="none"))
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.75, 1.65)
        _bare(ax)
    fig.suptitle("The uncertainty in a primitive's position is a shape, "
                 "not a number", fontsize=8.8, color=INK, y=1.02)
    _save(fig, out, "d3-one-camera-two-cameras")


# ===================================================================== D4
def d_protocols(out):
    """The six capture protocols, drawn as what they are: camera layouts."""
    specs = [
        ("full orbit", 100, 179.0, "the control"),
        ("grazing", 50, 168.0, "high-incidence views only"),
        ("mixed focal", 100, 179.0, "half the views downsampled (hollow)"),
        ("one-sided arc", 25, 94.0, "a contiguous wedge"),
        ("low-parallax cone", 10, 51.0, "a tight cluster"),
        ("narrow pencil", 3, 20.0, "three cameras, 20 degrees"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.5))
    for ax, (name, n, span, note) in zip(axes.ravel(), specs):
        half = np.radians(span) / 2.0
        ang = np.linspace(-half, half, max(n, 2)) + np.pi / 2
        R = 1.32
        ax.add_patch(Circle((0, 0), 0.115, color=INK, zorder=5, ec="none"))
        # the object's silhouette, for scale
        ax.add_patch(Circle((0, 0), 0.40, facecolor=GRID, ec="none", zorder=1))
        col = S2 if n <= 10 else S1
        for i, a in enumerate(ang):
            x, y = R * np.cos(a), R * np.sin(a)
            ax.plot([x, 0], [y, 0], color=col, lw=0.5, alpha=0.30, zorder=2)
            # mixed focal differs from the full orbit only in that half the
            # images are downsampled, which no camera layout can show -- so the
            # downsampled half is drawn hollow.
            small = (name == "mixed focal" and i % 2 == 0)
            if small:
                ax.add_patch(Circle((x, y), 0.052, facecolor=SURFACE, ec=col,
                                    lw=0.8, zorder=4))
            else:
                ax.add_patch(Circle((x, y), 0.058, color=col, zorder=4, ec="none"))
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.95, 1.75)
        _bare(ax)
        ax.set_title(name, fontsize=8.2, color=INK, pad=5)
        ax.text(0, -1.42, f"{n} cameras · {span:.0f}°", ha="center",
                fontsize=7.2, color=INK_2)
        ax.text(0, -1.74, note, ha="center", fontsize=6.9, color=MUTED)
    fig.suptitle("Six capture protocols. Only the training cameras change; "
                 "the test set is always the full orbit.",
                 fontsize=8.8, color=INK, y=1.005)
    fig.tight_layout(rect=(0, 0.01, 1, 0.975), h_pad=3.2)
    _save(fig, out, "d4-protocols")


# ===================================================================== D5
def d_floor_vs_estimate(out):
    """The exact reduction, as a picture: which of two floors is larger.

    The method takes the larger of two limits. On a good capture the resolution
    floor wins everywhere, and the method IS its baseline. As the capture
    narrows, the estimation limit rises past it -- and only then does anything
    change.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    span = np.linspace(179, 12, 400)
    # Calibrated so the crossing falls between the cone and the pencil,
    # which is where the measurement puts it: on the cone the floor still
    # dominates every primitive, on the pencil it does not on 93% of them.
    est = 0.012 * (1.0 / np.sin(np.radians(span) / 2.0)) ** 2
    floor = np.full_like(span, 0.16)
    x = np.arange(len(span))

    ax.plot(x, floor, color=S1, lw=2.0, zorder=4, label="resolution floor (Nyquist)")
    ax.plot(x, est, color=S2, lw=2.0, zorder=4, label="estimation limit (this thesis)")
    ax.fill_between(x, np.maximum(est, floor), 0, color=INK, alpha=0.055, zorder=1)
    cross = int(np.argmax(est > floor))
    ax.axvline(cross, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.text(cross - 8, 0.40, "below this width the capture,\nnot the pixel grid,\n"
            "is the binding limit", ha="right", fontsize=7.3, color=INK,
            linespacing=1.35)
    ax.text(cross * 0.42, 0.205, "the two methods are identical here",
            ha="center", fontsize=7.4, color=S1)

    marks = [(179, "full orbit"), (94, "arc"), (51, "cone"), (20, "pencil")]
    for deg, lab in marks:
        i = int(np.argmin(np.abs(span - deg)))
        ax.plot([i], [0.03], marker="^", ms=5, color=INK_2, clip_on=False, zorder=6)
        ax.text(i, -0.02, lab, ha="center", va="top", fontsize=6.9, color=INK_2)
    ax.set_ylim(0, 0.78); ax.set_xlim(0, len(span) - 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("angular width of the capture   (wide  →  narrow)",
                  fontsize=7.8, labelpad=14)
    ax.set_ylabel("smallest structure the limit permits", fontsize=7.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left", labelcolor=INK_2)
    ax.set_title("The method takes whichever limit is larger — which is why it "
                 "does nothing on a normal benchmark",
                 fontsize=8.6, color=INK, loc="left", pad=9)
    _save(fig, out, "d5-floor-vs-estimate")


# ===================================================================== D6
def d_splatting(out):
    """What splatting actually does, for a reader meeting it for the first time.

    Three stages, left to right: an ellipsoid in the world, its projection to an
    ellipse on the image plane, and the accumulation of many such ellipses into
    pixels. The band-limit this thesis is about acts at stage one; the one both
    baselines already have acts at stage two.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # --- 1: the primitive in space
    ax = axes[0]
    for (cx, cy, w, h, a, al) in [(-0.35, 0.25, 1.5, 0.62, 28, 0.30),
                                  (0.45, -0.30, 1.1, 0.48, -18, 0.30),
                                  (0.05, 0.55, 0.8, 0.34, 62, 0.26)]:
        ax.add_patch(Ellipse((cx, cy), w, h, angle=a, facecolor=S2, alpha=al,
                             ec=S2, lw=1.1))
    ax.text(0, -1.32, "a primitive is an ellipsoid\nwith a colour and an opacity",
            ha="center", va="top", fontsize=7.3, color=INK_2, linespacing=1.35)
    ax.set_title("1. in the world", fontsize=8.4, color=INK, pad=6)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)

    # --- 2: projection
    ax = axes[1]
    ax.plot([-1.25, 1.25], [-0.95, -0.95], color=INK_2, lw=1.1)
    ax.text(0, -1.14, "image plane", ha="center", va="top", fontsize=7.0,
            color=INK_2)
    ax.add_patch(Ellipse((0, 0.45), 1.5, 0.62, angle=28, facecolor=S2,
                         alpha=0.28, ec=S2, lw=1.1))
    for s in (-1, 1):
        ax.plot([s * 0.72, s * 0.42], [0.62, -0.95], color=MUTED, lw=0.7,
                ls=(0, (3, 2)))
    ax.add_patch(Ellipse((0, -0.95), 0.84, 0.17, angle=0, facecolor=S1,
                         alpha=0.45, ec=S1, lw=1.1))
    ax.text(0, -1.42, "it projects to an ellipse\n(the EWA transform)",
            ha="center", va="top", fontsize=7.3, color=INK_2, linespacing=1.35)
    ax.set_title("2. onto the image", fontsize=8.4, color=INK, pad=6)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)

    # --- 3: accumulation into pixels
    ax = axes[2]
    for g in np.arange(-1.2, 1.21, 0.4):
        ax.axvline(g, color=GRID, lw=0.8); ax.axhline(g, color=GRID, lw=0.8)
    rng = np.random.default_rng(3)
    for _ in range(14):
        c = rng.normal(0, 0.45, 2)
        ax.add_patch(Ellipse(c, 0.42, 0.20, angle=rng.uniform(0, 180),
                             facecolor=S1, alpha=0.30, ec="none"))
    ax.text(0, -1.32, "many ellipses are blended,\nnearest first, into each pixel",
            ha="center", va="top", fontsize=7.3, color=INK_2, linespacing=1.35)
    ax.set_title("3. into pixels", fontsize=8.4, color=INK, pad=6)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)

    for ax in axes:
        _bare(ax)
    fig.suptitle("Splatting in three stages. This thesis is about a limit at "
                 "stage 1; both baselines already have one at stage 2.",
                 fontsize=8.6, color=INK, y=1.04)
    _save(fig, out, "d6-splatting")


# ===================================================================== D7
def d_sh_coverage(out):
    """Why a narrow capture cannot determine high-degree view dependence.

    Spherical harmonics of degree l oscillate l times around the sphere. If every
    camera that saw a primitive lies inside a small angular window, then over
    that window the basis functions are nearly the same shape, and no fit can
    tell which of them the data is asking for. The right panel is the same three
    curves as the left, drawn over the window alone.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    curves = ((1, S1, "degree 1"), (2, S2, "degree 2"), (3, INK, "degree 3"))
    half = np.radians(11.0)

    ax = axes[0]
    th = np.linspace(-np.pi, np.pi, 700)
    for l, col, lab in curves:
        ax.plot(th, np.cos(l * th), color=col, lw=1.6, label=lab)
    ax.axvspan(-half, half, color=INK, alpha=0.12, zorder=0)
    ax.annotate("this window", xy=(0, -1.16), xytext=(1.5, -1.33),
                fontsize=7.1, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8))
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-1.55, 1.9)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["$-\pi$", "0", "$\pi$"], fontsize=7.2)
    ax.set_title("seen from all around", fontsize=8.4, color=INK, pad=6)
    ax.text(0, -2.30, "the three are easy to tell apart, so a fit" + chr(10) +
                      "can say how much of each the data wants",
            ha="center", va="top", fontsize=7.3, color=INK_2, linespacing=1.35)
    ax.legend(frameon=False, fontsize=7.0, ncol=3, loc="upper center",
              labelcolor=INK_2, columnspacing=1.1, handlelength=1.3)

    ax = axes[1]
    thz = np.linspace(-half, half, 400)
    # Drawn with different dash patterns and widths: the three curves lie on top
    # of one another, and the reader needs to see that there are three of them
    # rather than assume only one was plotted.
    styles = [("-", 3.4), ((0, (5, 3)), 2.2), ((0, (1.2, 2.2)), 1.6)]
    for (l, col, lab), (ls, lw) in zip(curves, styles):
        y = np.cos(l * thz)
        ax.plot(np.degrees(thz), (y - y.mean()) / max(np.ptp(y), 1e-9),
                color=col, lw=lw, ls=ls)
    ax.set_xlim(-np.degrees(half), np.degrees(half)); ax.set_ylim(-1.55, 1.9)
    ax.set_xticks([-10, 0, 10])
    ax.set_xticklabels(["$-10^\circ$", "0", "$+10^\circ$"], fontsize=7.2)
    ax.set_title("seen from an 11-degree window", fontsize=8.4, color=INK, pad=6)
    ax.text(0, -2.30, "rescaled to the window, all three are the" + chr(10) +
                      "same downward arc: the data cannot separate them",
            ha="center", va="top", fontsize=7.3, color=INK_2, linespacing=1.35)

    for ax in axes:
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("viewing direction", fontsize=7.6)
    fig.suptitle("View-dependent colour is a series in the viewing direction. "
                 "A narrow capture samples too little of it.",
                 fontsize=8.6, color=INK, y=1.04)
    _save(fig, out, "d7-sh-coverage")


# ===================================================================== D8
def d_two_arms(out):
    """The experimental construction: what is held fixed, and what is varied."""
    fig, ax = plt.subplots(figsize=(6.9, 3.1))
    shared = ["the same dataloader", "the same metrics code",
              "the same perceptual backbone (VGG)",
              "the same densification", "the same seed and iteration count"]
    ax.add_patch(plt.Rectangle((0.03, 0.05), 0.94, 0.34, facecolor=GRID,
                               alpha=0.45, ec="none"))
    ax.text(0.5, 0.335, "held fixed — one checkout, one set of flags apart",
            ha="center", fontsize=8.0, color=INK, weight="bold")
    for i, t in enumerate(shared):
        ax.text(0.5, 0.255 - i * 0.045, t, ha="center", fontsize=7.2,
                color=INK_2)

    boxes = [(0.05, "Mip-Splatting arm", "the published\nband-limit", S1),
             (0.29, "3DGS arm", "3DGS's band-limit\nsemantics exactly", S2),
             (0.53, "anisotropic filter", "the proposed\nworld-space limit", INK),
             (0.77, "angular mask", "the proposed\nSH criterion", MUTED)]
    for x, name, what, col in boxes:
        ax.add_patch(plt.Rectangle((x, 0.52), 0.18, 0.30, facecolor=SURFACE,
                                   ec=col, lw=1.4))
        ax.text(x + 0.09, 0.765, name, ha="center", fontsize=7.2, color=col,
                weight="bold")
        ax.text(x + 0.09, 0.635, what, ha="center", fontsize=7.1, color=INK_2,
                linespacing=1.3)
        ax.annotate("", xy=(x + 0.09, 0.40), xytext=(x + 0.09, 0.52),
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=1.0))
    ax.text(0.5, 0.92, "varied: the band-limit, and nothing else",
            ha="center", fontsize=8.2, color=INK, weight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _bare(ax); ax.set_aspect("auto")
    ax.text(0.5, -0.06, "Any difference between two of these columns is the "
            "band-limit, because it is the only thing that differs.",
            ha="center", fontsize=7.4, color=MUTED, transform=ax.transAxes)
    _save(fig, out, "d8-two-arms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="thesis/figures")
    a = ap.parse_args()
    d_aliasing(a.out)
    d_bandlimit_locations(a.out)
    d_one_camera(a.out)
    d_protocols(a.out)
    d_floor_vs_estimate(a.out)
    d_splatting(a.out)
    d_sh_coverage(a.out)
    d_two_arms(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
