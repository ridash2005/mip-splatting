#!/usr/bin/env python3
"""
Append a finished run's rows to results/runs.csv.

results/runs.csv is append-only (§11). This checks the schema matches, skips
run_ids already present so re-running is safe, and never edits an existing row.

    python tools/merge_run.py results/kaggle_runs/t3_stress/runs.csv
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import split_by_scale as sbs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="+")
    ap.add_argument("--runs", default="results/runs.csv")
    a = ap.parse_args()

    existing = []
    if os.path.exists(a.runs):
        with open(a.runs, newline="") as f:
            existing = list(csv.DictReader(f))
    have = {r["run_id"] for r in existing}
    total = 0

    for src in a.source:
        if not os.path.exists(src):
            print(f"skip {src}: not found")
            continue
        with open(src, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"skip {src}: empty")
            continue
        if list(rows[0]) != sbs.RUNS_CSV_SCHEMA:
            sys.exit(f"{src}: header is not the §11 schema; refusing to append")
        add = [r for r in rows if r["run_id"] not in have]
        have |= {r["run_id"] for r in add}
        with open(a.runs, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sbs.RUNS_CSV_SCHEMA)
            if not existing and not total:
                w.writeheader()
            w.writerows(add)
        total += len(add)
        print(f"{src}: {len(rows)} rows, appended {len(add)}, "
              f"{len(rows) - len(add)} already present")
    print(f"total appended {total}; {a.runs} now has {len(existing) + total} rows")


if __name__ == "__main__":
    main()
