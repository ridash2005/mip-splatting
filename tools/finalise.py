#!/usr/bin/env python3
r"""
Fold every finished run into the record, then rebuild the document and the deck.

The steps are ordered, and the order is the whole point:

  1. merge   append each run's rows to results/runs.csv (append-only, idempotent)
  2. mark    flag the rows a re-run replaces, but ONLY once the replacement is
             actually in the file -- `--require` refuses otherwise, because
             marking early empties a table instead of updating it
  3. verify  re-run the closed-form checks, so the generated geometry tables
             cannot drift from the derivation they illustrate
  4. build   regenerate every table, macro and figure, then the PDF and the deck

Run it as often as you like; every step is idempotent.

    python tools/finalise.py            # --dry-run to see the plan
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KRUNS = os.path.join(ROOT, "results", "kaggle_runs")

# Which rows each re-run replaces. `require` is the number of replacement rows
# that must already be present -- 8 scenes x 4 test scales for a full sweep,
# 2 scenes x 3 seeds x 4 scales minus the seed-0 rows for the spread.
SUPERSEDES = [
    dict(by="C2", where=["arm=B", "dataset=blender", "load_allres=False",
                         "iterations=30000", "seed=0"],
         unless="C2", require=28,
         why="arm B, single-scale train, seed 0: re-measured without the 2D "
             "opacity compensation (C1). Scoped to seed 0 so the R5 seed rows "
             "are not orphaned -- C2S replaces those, separately, and marking "
             "them before it lands would leave the spread table with one seed."),
    dict(by="C2S", where=["arm=B", "dataset=blender", "load_allres=False",
                          "iterations=30000", "seed=1"],
         unless="C2S", require=8,
         why="arm B, seed 1: the seed spread was measured on the compensated "
             "arm, so it is a spread of a configuration no table reports."),
    dict(by="C2S", where=["arm=B", "dataset=blender", "load_allres=False",
                          "iterations=30000", "seed=2"],
         unless="C2S", require=8,
         why="arm B, seed 2: as above."),
    dict(by="C2M", where=["arm=B", "dataset=blender", "load_allres=True",
                          "iterations=30000"],
         unless="C2M", require=32,
         why="arm B, multi-scale train and test: same defect, same arm."),
    # These two are keyed on the COMMIT, not the rung name. A re-run that keeps
    # its rung has the rung in its notes just like the run it replaces, so
    # --unless-notes-contains exempts both and marks neither -- which is how 32
    # arm-B cells in Table 2 came to be averages over two different rasterisers,
    # one of them the defective build C1 was written to replace.
    dict(by="C2M-rebuild",
         where=["arm=B", "dataset=blender", "load_allres=True",
                "iterations=30000"],
         unless_commit="7896e78d", require=32,
         why="arm B, multi-scale: the first C2M sweep ran on 061a9084, before "
             "the forward and backward degeneracy guards were matched. The "
             "re-run on 7896e78d is the same rung and the same eight scenes, "
             "so only the commit separates them."),
    dict(by="R7-floor",
         where=["arm=C", "dataset=blender", "iterations=30000",
                "train_scale=1x/full"],
         unless_commit="22a5fbba", require=8, only_cells=True,
         why="B1 at the full protocol: 6693604d re-derived Mip-Splatting's "
             "floor instead of taking it, which differed by 0.5%% at the median "
             "on the first call and seeded a different densification "
             "trajectory. 22a5fbba takes theirs, which is what makes "
             "Proposition 2 exact in the floor as well."),
    dict(by="T3M", where=["train_scale=1x/mixed", "dataset=blender"],
         unless="T3M", require=16,
         why="the mixed-focal protocol: the first stress run passed only the "
             "index list, so it trained the full orbit under another name. The "
             "re-run applies the focal override the protocol actually "
             "specifies."),
]


def sh(cmd, dry):
    print("  " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args()
    py = sys.executable

    print("\n1. merge finished runs into results/runs.csv")
    found = []
    for d in sorted(os.listdir(KRUNS)) if os.path.isdir(KRUNS) else []:
        p = os.path.join(KRUNS, d, "runs.csv")
        if os.path.exists(p):
            found.append(p)
    if not found:
        print("  (no run has written a runs.csv yet)")
    else:
        sh([py, "tools/merge_run.py"] + found, a.dry_run)

    print("\n2. mark superseded rows (refused until the replacement is present)")
    for s in SUPERSEDES:
        print(f"  -- {s['by']}: {s['why']}")
        guard = (["--unless-impl-commit-prefix", s["unless_commit"]]
                 if s.get("unless_commit")
                 else ["--unless-notes-contains", s["unless"]])
        if s.get("only_cells"):
            # The replacement covers two scenes of the eight; without this the
            # other six are marked too and the table loses rows nothing replaced.
            guard += ["--only-replacement-cells"]
        rc = sh([py, "tools/supersede.py", "--by", s["by"], "--where"]
                + s["where"] + guard
                + ["--require", str(s["require"])], a.dry_run)
        if rc:
            print(f"     not applied yet; {s['by']} has not landed.")

    print("\n3. re-verify the closed forms the generated tables quote")
    sh([py, "tools/test_geometry_claims.py", "--json",
        "results/geometry/claims.json"], a.dry_run)

    if a.skip_build:
        print("\n(skipping the build)")
        return 0
    print("\n4. rebuild the thesis and the deck")
    sh(["make", "all"], a.dry_run)
    print("\n--> thesis/main.pdf and slides/BTP-Panel.pptx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
