#!/usr/bin/env python3
r"""
The qualitative figures: what the band-limit does to an actual picture.

Every number in this thesis is a mean over scenes and views, and a mean is the
wrong instrument for the question "what does this actually look like". A 10 dB
PSNR gap is a fact about a table; the panel should be able to see it. These
figures are built from the renders the rungs themselves produced --
tools/fetch_renders.py pulls the handful each kernel kept after metrics consumed
the rest -- so they are the same pixels the PSNR column was computed from, not a
re-render made for the picture.

Three figures, and the third is there to stop the first two overselling:

  fig11  one scene down the scale ladder, both arms. The divergence appears as
         the test sampling rate leaves the training rate, which is Table 1 as a
         picture.
  fig12  the 1/8 column across scenes, so the effect is visibly not one scene.
  fig13  the narrow pencil, B1 against Mip-Splatting. Both reconstructions are
         bad, and the 0.197 dB B1 gains there is not visible. That is the
         honest state of the method's own result and it belongs in the document
         next to the number.

Every caption number is recomputed here from the two images being displayed, so
a label cannot drift from the pixels above it.

    python tools/make_qualitative.py --out thesis/figures
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (  # noqa: E402  one palette for the whole document
    SURFACE, INK, INK_2, MUTED, S1, S2, FORMATS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDERS = os.path.join(ROOT, "results", "renders")

# index -> test scale. The multi-scale test set interleaves the four scales, so
# the kept renders 0..3 are one view at full, 1/2, 1/4 and 1/8 -- verified by
# their pixel dimensions (800, 400, 200, 100), not assumed.
# matplotlib mathtext has \frac but not \tfrac, which is LaTeX-only.
SCALE_OF = {0: "full", 1: r"$\frac{1}{2}$", 2: r"$\frac{1}{4}$", 3: r"$\frac{1}{8}$"}

# Where each arm's renders live. Arm B must come from the VANILLA run: the R1
# kernel's arm B still carried Mip-Splatting's opacity compensation and its rows
# are superseded, so using its pictures would show a 3 dB gap beside a table
# reporting 10.8.
ARM_A = os.path.join(RENDERS, "r1", "armA_{scene}")
ARM_B = os.path.join(RENDERS, "c2v", "armB_{scene}")
PENCIL = os.path.join(RENDERS, "pencil", "{method}_{scene}_pencil")


def _img(path):
    return Image.open(path).convert("RGB")


def _arr(path):
    return np.asarray(_img(path), np.float64) / 255.0


def psnr(pred_path, gt_path):
    a, b = _arr(pred_path), _arr(gt_path)
    if a.shape != b.shape:
        return None
    mse = ((a - b) ** 2).mean()
    return 10.0 * np.log10(1.0 / mse) if mse > 0 else 99.0


def show(ax, path, size=320, interp=Image.NEAREST):
    """Draw one render, magnified so a rendered pixel is a visible block.

    NEAREST on purpose. A 1/8-scale render is 100x100, and any smooth resampler
    would average away the very aliasing the figure exists to show -- it would
    make 3DGS look better than it is. The magnification factor is stated in the
    caption.
    """
    ax.imshow(np.asarray(_img(path).resize((size, size), interp)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#d8dde3"); s.set_linewidth(0.6)


def label(ax, text, colour=INK, size=7.0, weight="normal"):
    ax.set_xlabel(text, fontsize=size, color=colour, labelpad=2.5,
                  fontweight=weight)


def have(*paths):
    return all(os.path.exists(p) for p in paths)


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in FORMATS:
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02, dpi=220)
        print("wrote", p)
    # A PNG as well, always: python-pptx embeds neither PDF nor SVG, and the
    # deck must show the same picture as the document.
    png = os.path.join(out, f"{name}.png")
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02, dpi=220)
    print("wrote", png)
    plt.close(fig)


# ------------------------------------------------------------------ fig 11
def fig_ladder(out, scene="lego", indices=(0, 1, 2, 3),
               name="fig11-qualitative-ladder"):
    """One scene, four test scales, both arms. Table 1 as a picture.

    `indices` exists for the deck: four rows is the right shape for a page and
    too tall for a 16:9 slide, where it overran the footer. The compact variant
    keeps the two rows that carry the argument -- the training scale, where the
    arms agree, and 1/8, where they do not.
    """
    a_dir, b_dir = ARM_A.format(scene=scene), ARM_B.format(scene=scene)
    rows = []
    for i in indices:
        gt = f"{a_dir}/gt_-1/{i:05d}.png"
        pa = f"{a_dir}/test_preds_-1/{i:05d}.png"
        pb = f"{b_dir}/test_preds_-1/{i:05d}.png"
        if have(gt, pa, pb):
            rows.append((i, gt, pb, pa))
    if not rows:
        print("fig11: no renders for", scene)
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.0, 2.05 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (i, gt, pb, pa) in enumerate(rows):
        show(axes[r][0], gt)
        show(axes[r][1], pb)
        show(axes[r][2], pa)
        axes[r][0].set_ylabel(f"test scale {SCALE_OF[i]}", fontsize=7.6,
                              color=INK, labelpad=6)
        label(axes[r][0], "ground truth", MUTED)
        label(axes[r][1], f"3DGS   {psnr(pb, gt):.2f} dB", S2, weight="bold")
        label(axes[r][2], f"Mip-Splatting   {psnr(pa, gt):.2f} dB", S1,
              weight="bold")
    which = "four test scales" if len(rows) == 4 else (
        "the training scale and " + SCALE_OF[rows[-1][0]].replace("$", "")
        .replace(chr(92) + "frac{1}{", "1/").replace("}", ""))
    fig.suptitle(f"Single-scale training, {which} - {scene}",
                 fontsize=8.6, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save(fig, out, name)


# ------------------------------------------------------------------ fig 12
def fig_scenes(out, scenes=("lego", "drums", "materials", "mic"), index=3,
               name="fig12-qualitative-scenes"):
    """The 1/8 column across scenes: the effect is not one object."""
    rows = []
    for sc in scenes:
        a_dir, b_dir = ARM_A.format(scene=sc), ARM_B.format(scene=sc)
        gt = f"{a_dir}/gt_-1/{index:05d}.png"
        pa = f"{a_dir}/test_preds_-1/{index:05d}.png"
        pb = f"{b_dir}/test_preds_-1/{index:05d}.png"
        if have(gt, pa, pb):
            rows.append((sc, gt, pb, pa))
    if not rows:
        print("fig12: no renders")
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.0, 2.05 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (sc, gt, pb, pa) in enumerate(rows):
        show(axes[r][0], gt); show(axes[r][1], pb); show(axes[r][2], pa)
        axes[r][0].set_ylabel(sc, fontsize=7.8, color=INK, labelpad=6)
        label(axes[r][0], "ground truth", MUTED)
        label(axes[r][1], f"3DGS   {psnr(pb, gt):.2f} dB", S2, weight="bold")
        label(axes[r][2], f"Mip-Splatting   {psnr(pa, gt):.2f} dB", S1,
              weight="bold")
    fig.suptitle(f"Trained at full resolution, tested at 1/8 - the same view, "
                 f"{len(rows)} scenes", fontsize=8.6, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save(fig, out, name)


# ------------------------------------------------------------------ fig 13
def fig_pencil(out, scenes=("chair", "lego"), index=0):
    """The method's own result, and what it does not fix.

    B1 leads Mip-Splatting by 0.197 dB on this protocol, on every paired cell.
    It is also invisible, on reconstructions that are both very bad. Both facts
    are true and the figure shows the second so the first is not over-read.
    """
    rows = []
    for sc in scenes:
        m = PENCIL.format(method="mip", scene=sc)
        b = PENCIL.format(method="b1", scene=sc)
        gt = f"{m}/gt_-1/{index:05d}.png"
        pm = f"{m}/test_preds_-1/{index:05d}.png"
        pb = f"{b}/test_preds_-1/{index:05d}.png"
        if have(gt, pm, pb):
            rows.append((sc, gt, pm, pb))
    if not rows:
        print("fig13: no pencil renders")
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.0, 2.05 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (sc, gt, pm, pb) in enumerate(rows):
        show(axes[r][0], gt, interp=Image.LANCZOS)
        show(axes[r][1], pm, interp=Image.LANCZOS)
        show(axes[r][2], pb, interp=Image.LANCZOS)
        vm, vb = psnr(pm, gt), psnr(pb, gt)
        axes[r][0].set_ylabel(sc, fontsize=7.8, color=INK, labelpad=6)
        label(axes[r][0], "ground truth", MUTED)
        label(axes[r][1], f"Mip-Splatting   {vm:.2f} dB", S1, weight="bold")
        label(axes[r][2], f"B1   {vb:.2f} dB   ({vb - vm:+.2f})", INK,
              weight="bold")
    fig.suptitle("Three cameras spanning 29° — where the estimation term binds",
                 fontsize=8.6, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save(fig, out, "fig13-qualitative-pencil")


def archive(out_dir):
    """Every render we hold, as one browsable sheet per scene.

    The three figures in the document are four scenes and two protocols, chosen
    because they show the effect clearly -- which is exactly the kind of choice
    a reader should be able to audit. This writes the rest: every scene, every
    kept view, both arms, with the per-view PSNR under each. If the figures had
    been flattering, these sheets would say so.
    """
    from PIL import ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    scenes = sorted({d.split("_", 1)[1] for d in os.listdir(os.path.join(RENDERS, "r1"))
                     if "_" in d} if os.path.isdir(os.path.join(RENDERS, "r1")) else [])
    cell, pad, strip = 190, 8, 16
    written = 0
    for sc in scenes:
        a_dir, b_dir = ARM_A.format(scene=sc), ARM_B.format(scene=sc)
        cols = []
        for i in range(6):
            gt = f"{a_dir}/gt_-1/{i:05d}.png"
            pa = f"{a_dir}/test_preds_-1/{i:05d}.png"
            pb = f"{b_dir}/test_preds_-1/{i:05d}.png"
            if not have(gt, pa):
                continue
            cols.append((i, gt, pb if have(pb) else None, pa))
        if not cols:
            continue
        w = len(cols) * (cell + pad) + pad
        h = 3 * (cell + strip + pad) + pad
        sheet = Image.new("RGB", (w, h), "white")
        dr = ImageDraw.Draw(sheet)
        for c, (i, gt, pb, pa) in enumerate(cols):
            x = pad + c * (cell + pad)
            for r, (path, name) in enumerate(((gt, "ground truth"),
                                              (pb, "3DGS"),
                                              (pa, "Mip-Splatting"))):
                y = pad + r * (cell + strip + pad)
                if path is None:
                    dr.text((x + 4, y + cell // 2), "not measured", fill="#8b939d")
                    continue
                sheet.paste(_img(path).resize((cell, cell), Image.NEAREST), (x, y))
                cap = name if r == 0 else f"{name}  {psnr(path, gt):.2f} dB"
                dr.text((x + 2, y + cell + 3), cap, fill="#1a1d21")
            dr.text((x + 2, 1), f"view {i}  ({_img(gt).size[0]}px)", fill="#52514e")
        sheet.save(os.path.join(out_dir, f"{sc}.png"))
        written += 1
    print(f"wrote {written} contact sheet(s) under {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="thesis/figures")
    ap.add_argument("--scene", default="lego")
    a = ap.parse_args()
    if not os.path.isdir(RENDERS):
        print(f"no renders under {RENDERS}; run tools/fetch_renders.py first")
        return 1
    fig_ladder(a.out, a.scene)
    fig_ladder(a.out, a.scene, indices=(0, 3),
               name="fig11c-qualitative-ladder-compact")
    fig_scenes(a.out)
    # Three scenes for the deck: four rows is taller than a 16:9 slide can hold
    # at a legible width, and the fourth row was being cut off.
    fig_scenes(a.out, scenes=("lego", "materials", "mic"),
               name="fig12c-qualitative-scenes-compact")
    fig_pencil(a.out)
    archive(os.path.join(ROOT, "results", "qualitative"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
