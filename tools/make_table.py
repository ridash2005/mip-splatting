#!/usr/bin/env python3
"""
Render Table 1 (R1, Blender STMT) / Table 2 (R2, Blender MTMT) from
results/runs.csv. Never prints a number it did not read from a logged row —
mirrors make_figures.py's "measured" refusal: an empty result set is an
error, not a table with published numbers quietly substituted in.

    python tools/make_table.py --table 1 --runs results/runs.csv
    python tools/make_table.py --table 2 --runs results/runs.csv --md results/table2.md
"""
import argparse, csv, os, sys

# The report is UTF-8 (arrows, section signs, en dashes). A Windows console
# defaults to cp1252 and would raise UnicodeEncodeError on the way out, so the
# stream is reconfigured rather than the text degraded.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from collections import defaultdict

SCALE_ORDER = {"1x": 0, "1/2": 1, "1/4": 2, "1/8": 3}
ARM_NAME = {"A": "Mip-Splatting", "B": "3DGS"}

TABLE_LOAD_ALLRES = {1: "False", 2: "True"}  # R1 = STMT (single-scale train), R2 = MTMT


def load(runs_path, load_allres, iterations):
    """Average the per-scene rows into one row per (arm, test_scale).

    `iterations` is required: results/runs.csv legitimately holds 7 000-iteration
    R0 smoke rows next to 30 000-iteration R1/R2 rows, and averaging across them
    yields a number that describes neither run.
    """
    if not os.path.exists(runs_path):
        return None, set()
    acc = defaultdict(lambda: defaultdict(list))
    seen_iters = set()
    # Through make_tex's loader: superseded rows out, probe rows out, repeats of
    # one configuration collapsed to a single row so a twice-measured scene does
    # not count twice. This file used to walk the CSV raw, which meant the
    # markdown tables it writes could disagree with the thesis tables built from
    # the same file -- and did, by nearly a decibel.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_tex as mt
    picked = []
    for r in mt.collapse_repeats(mt.load(runs_path)):
        if r.get("dataset") != "blender" or not r.get("psnr"):
            continue
        if r.get("seed") != "0" or r.get("train_scale") not in ("1x", "multi"):
            continue
        if r.get("load_allres") != load_allres:
            continue
        seen_iters.add(r.get("iterations", ""))
        if r.get("iterations") != str(iterations):
            continue
        if r.get("arm") not in ARM_NAME:
            continue
        picked.append(r)
    # NOT matched. This table sets each arm beside its published figure, and the
    # published figure is a mean over all eight Blender scenes, so each column
    # has to be a mean over the scenes that arm measured. Matching them here
    # would drop `ship` from arm A -- its weakest scene at full resolution --
    # and inflate the column by 0.42 dB against a target it is supposed to be
    # compared with. The between-arm quantity is the gap, and that one is
    # matched; it lives in the thesis table, not here.
    for r in picked:
        acc[r["arm"]][r["test_scale"]].append(
            (float(r["psnr"]), float(r["ssim"] or 0), float(r["lpips"] or 0)))
    out = {}
    for arm, byscale in acc.items():
        if len(byscale) < 4:
            continue
        out[arm] = {}
        for sc, vals in byscale.items():
            n = len(vals)
            out[arm][sc] = tuple(sum(v[i] for v in vals) / n for i in range(3)) + (n,)
    return (out or None), {i for i in seen_iters if i}


def render(rows, table_num):
    lines = [f"| test scale | {ARM_NAME['B']} PSNR | {ARM_NAME['A']} PSNR |"
             f" {ARM_NAME['B']} SSIM | {ARM_NAME['A']} SSIM |"
             f" {ARM_NAME['B']} LPIPS | {ARM_NAME['A']} LPIPS |",
             "|---|---|---|---|---|---|---|"]
    for sc in sorted({sc for arm in rows.values() for sc in arm}, key=SCALE_ORDER.get):
        cells = [sc]
        for arm in ("B", "A"):
            v = rows.get(arm, {}).get(sc)
            cells.append(f"{v[0]:.2f}" if v else "—")
        for arm in ("B", "A"):
            v = rows.get(arm, {}).get(sc)
            cells.append(f"{v[1]:.4f}" if v else "—")
        for arm in ("B", "A"):
            v = rows.get(arm, {}).get(sc)
            cells.append(f"{v[2]:.4f}" if v else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=int, choices=[1, 2], required=True,
                     help="1 = R1 Blender STMT (load_allres=False), "
                          "2 = R2 Blender MTMT (load_allres=True)")
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--md", help="write markdown here in addition to stdout")
    ap.add_argument("--iterations", type=int, default=30000,
                     help="only average rows from runs of this length (default 30000, "
                          "the R1/R2 target). Pass 7000 for the R0 smoke rows.")
    a = ap.parse_args()

    rows, seen = load(a.runs, TABLE_LOAD_ALLRES[a.table], a.iterations)
    if not rows:
        raise SystemExit(
            f"no usable rows in {a.runs} for table {a.table} "
            f"(need both arms, all 4 test scales, load_allres={TABLE_LOAD_ALLRES[a.table]}, "
            f"iterations={a.iterations}) — run the ladder first. "
            f"Iteration counts present for this protocol: {sorted(seen) or 'none'}.")

    md = render(rows, a.table)
    n = {arm: max(v[3] for v in sc.values()) for arm, sc in rows.items()}
    md += ("\n\n"
           f"R{a.table} · Blender {'STMT' if a.table == 1 else 'MTMT'} · "
           f"{a.iterations} iterations · rows averaged per scale: "
           + ", ".join(f"arm {k} n={v}" for k, v in sorted(n.items())))
    print(md)
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        # Explicit UTF-8: the table body carries "·" and "→", and the default
        # locale encoding on Windows is cp1252, which wrote a file no UTF-8
        # reader could open.
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\nwrote {a.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
