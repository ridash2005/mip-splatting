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
from collections import defaultdict

SCALE_ORDER = {"1x": 0, "1/2": 1, "1/4": 2, "1/8": 3}
ARM_NAME = {"A": "Mip-Splatting", "B": "3DGS"}

TABLE_LOAD_ALLRES = {1: "False", 2: "True"}  # R1 = STMT (single-scale train), R2 = MTMT


def load(runs_path, load_allres):
    if not os.path.exists(runs_path):
        return None
    acc = defaultdict(lambda: defaultdict(list))
    with open(runs_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("dataset") != "blender" or not r.get("psnr"):
                continue
            if r.get("load_allres") != load_allres:
                continue
            arm = r.get("arm")
            if arm not in ARM_NAME:
                continue
            acc[arm][r["test_scale"]].append(
                (float(r["psnr"]), float(r["ssim"] or 0), float(r["lpips"] or 0)))
    out = {}
    for arm, byscale in acc.items():
        if len(byscale) < 4:
            continue
        out[arm] = {}
        for sc, vals in byscale.items():
            n = len(vals)
            out[arm][sc] = tuple(sum(v[i] for v in vals) / n for i in range(3)) + (n,)
    return out or None


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
    a = ap.parse_args()

    rows = load(a.runs, TABLE_LOAD_ALLRES[a.table])
    if not rows:
        raise SystemExit(
            f"no usable rows in {a.runs} for table {a.table} "
            f"(need both arms, all 4 test scales, load_allres={TABLE_LOAD_ALLRES[a.table]}) "
            "— run the ladder first.")

    md = render(rows, a.table)
    print(md)
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        with open(a.md, "w") as f:
            f.write(md + "\n")
        print(f"\nwrote {a.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
