#!/usr/bin/env python3
r"""
Mark rows in results/runs.csv as superseded, without deleting them.

results/runs.csv is append-only (§11), and the 8 September audit's instruction on
the C-series was explicit: mark the superseded arm-B rows rather than removing
them, because the audit trail is the point. A deleted row is a run that cannot be
re-examined; a marked one is a run whose successor is named in the file.

Nothing else about the row changes, and the tables already skip any row whose
notes contain SUPERSEDED (see tools/finish.py's `rows`, tools/make_tex.py and
tools/make_figures.py), so marking is what takes a row out of every average
while leaving it on the record.

    python tools/supersede.py --by C2 --where arm=B dataset=blender \
        --unless-run-id-prefix c2- --dry-run

Every filter is an exact match on a column. A row already marked is left alone.
"""
import argparse
import csv
import os
import sys

MARK = "SUPERSEDED_BY_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--by", required=True,
                    help="the stage that replaces these rows, e.g. C2")
    ap.add_argument("--where", nargs="+", required=True, metavar="COL=VALUE",
                    help="exact-match filters; a row must satisfy all of them")
    ap.add_argument("--unless-run-id-prefix", default=None,
                    help="never mark a row whose run_id starts with this -- the "
                         "replacement rows themselves")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    where = dict(kv.split("=", 1) for kv in a.where)
    if not os.path.exists(a.runs):
        sys.exit(f"{a.runs}: not found")
    with open(a.runs, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    n = 0
    for r in rows:
        if MARK in (r.get("notes") or ""):
            continue
        if a.unless_run_id_prefix and r["run_id"].startswith(a.unless_run_id_prefix):
            continue
        if not all(r.get(k) == v for k, v in where.items()):
            continue
        r["notes"] = ((r.get("notes") or "") + f" {MARK}{a.by}").strip()
        n += 1
        if a.dry_run and n <= 5:
            print(f"  would mark {r['run_id']}  ({r['method']} {r['scene']} "
                  f"{r['test_scale']})")

    if a.dry_run:
        print(f"{n} row(s) would be marked {MARK}{a.by}; nothing written")
        return
    with open(a.runs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"marked {n} row(s) {MARK}{a.by} in {a.runs}")


if __name__ == "__main__":
    main()
