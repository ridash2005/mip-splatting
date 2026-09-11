#!/usr/bin/env python3
"""
Run the remaining experimental programme end to end, unattended.

Everything here is already built and tested; what it needs is GPU quota. The
weekly Kaggle allowance (30 h) was exhausted on 8 September 2026 and returns on
a rolling seven-day window, so this driver waits for it rather than asking
someone to notice that it came back.

What it is for, beyond convenience: the 8 September audit found that gate G2
failed and *the ladder continued anyway*. Every stage below carries its exit
criterion as code. A failed gate halts the run and says so. Nothing downstream
of a red gate is allowed to quietly produce numbers that would then be read as
if the gate had passed.

    python tools/finish.py                  # the whole programme, in order
    python tools/finish.py --dry-run        # print the plan, touch nothing
    python tools/finish.py --only t3_stress # one stage
    python tools/finish.py --from c2_armb   # resume part-way

Cost is about 19 GPU-h against a 30 h week, so the whole programme fits in one
reset if nothing has to be repeated.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "results", "runs.csv")
KRUNS = os.path.join(ROOT, "results", "kaggle_runs")

QUOTA_MSG = "Maximum weekly GPU quota"
DATASET = "nguyenhung1903/nerf-synthetic-dataset"


# ----------------------------------------------------------------- CSV helpers
def rows(**where):
    """Rows of results/runs.csv matching every key=value, superseded ones out."""
    if not os.path.exists(RUNS):
        return []
    out = []
    for r in csv.DictReader(open(RUNS)):
        if "SUPERSEDED" in (r.get("notes") or ""):
            continue
        if all(str(r.get(k)) == str(v) for k, v in where.items()):
            out.append(r)
    return out


def psnr_at(scale, **where):
    """Mean PSNR over matching rows at one test scale, or None."""
    vals = [float(r["psnr"]) for r in rows(test_scale=scale, **where)
            if r.get("psnr") not in (None, "", "nan")]
    return sum(vals) / len(vals) if vals else None


def summary_of(outdir):
    p = os.path.join(outdir, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ---------------------------------------------------------------------- gates
def gate_c1(outdir):
    """C1: the vanilla rasteriser must collapse under downsampling, and only there.

    Scene-relative on purpose. The published 3DGS targets (33.3 full, 17.7 at
    1/8) are eight-scene means and lego is not the mean -- its arm-B full-res
    sits at 36.05. So compare against this project's own measured arm-B lego
    rows instead of against a number lego was never supposed to hit.
    """
    where = dict(arm="B", scene="lego", dataset="blender",
                 iterations="30000", load_allres="False")
    base_1x, base_18 = psnr_at("1x", **where), psnr_at("1/8", **where)
    if base_1x is None or base_18 is None:
        return False, "no baseline arm-B lego rows to compare against"

    s = summary_of(outdir)
    new = {}
    for job in (s.get("done") or {}).values():
        for k, v in (job.get("split") or {}).items():
            new[k] = v.get("PSNR")
    got_1x, got_18 = new.get("1x"), new.get("1/8")
    if got_1x is None or got_18 is None:
        return False, f"run produced no 1x / 1-8 split (got {sorted(new)})"

    keeps_full = abs(got_1x - base_1x) <= 0.75
    collapses = got_18 <= base_18 - 3.0
    msg = (f"1x {got_1x:.2f} vs {base_1x:.2f} (must stay within 0.75) -> "
           f"{'ok' if keeps_full else 'FAIL'}; "
           f"1/8 {got_18:.2f} vs {base_18:.2f} (must fall >=3.0) -> "
           f"{'ok' if collapses else 'FAIL'}")
    if keeps_full and not collapses:
        msg += "\n  the compensation is still active -- do not run the sweep"
    return keeps_full and collapses, msg


def gate_g2(outdir):
    """G2: arm B's 1/8-scale gap against Mip-Splatting must exceed 9 dB.

    Published is 10.98. The pre-C1 arm measured 4.05, which is what failed.
    """
    s = summary_of(outdir)
    if not s.get("done"):
        return False, "no completed jobs"
    a = psnr_at("1/8", arm="A", dataset="blender", iterations="30000",
                load_allres="False")
    b = []
    for job in (s.get("done") or {}).values():
        v = (job.get("split") or {}).get("1/8", {}).get("PSNR")
        if v is not None:
            b.append(v)
    if a is None or not b:
        return False, "missing arm A or arm B 1/8 rows"
    gap = a - (sum(b) / len(b))
    ok = gap > 9.0
    return ok, (f"1/8 gap {gap:.2f} dB (arm A {a:.2f} - arm B "
                f"{sum(b)/len(b):.2f}); need >9.0, published 10.98 -> "
                f"{'PASS' if ok else 'FAIL'}")


def gate_g3(outdir):
    """G3: the decisive experiment. Three conditions, all of them required.

    - frac_above_floor > 0 on `cone`, or the estimation term never binds and
      the claim has to narrow or pivot to B2.
    - Delta(B1 - Mip) ordered cone > arc > full, each gain above 3 sigma.
    - `full` shows no loss. Proposition 2 forbids it; a loss there is a bug.

    sigma = 0.139 dB is this project's measured worst-case seed spread (R5),
    not an assumption, so 3 sigma = 0.417 dB.
    """
    SIGMA3 = 3 * 0.139
    s = summary_of(outdir)
    done = s.get("done") or {}
    if not done:
        return False, "no completed jobs"

    # done keys look like "<method>/<scene>/<protocol>"
    by = {}
    for k, v in done.items():
        parts = k.split("/")
        if len(parts) != 3:
            continue
        method, scene, proto = parts
        by.setdefault(proto, {}).setdefault(method, []).append(v)

    lines, ok = [], True

    cone = by.get("cone", {})
    fracs = [j.get("frac_above_floor") for m in cone.values() for j in m
             if j.get("frac_above_floor") is not None]
    if not fracs:
        return False, "no cone rows -- the stress protocols did not run"
    if max(fracs) <= 0:
        ok = False
        lines.append(f"frac_above_floor on cone = {max(fracs)} -- the floor "
                     "never binds; narrow the claim or pivot to B2 (FAIL)")
    else:
        lines.append(f"frac_above_floor on cone = {max(fracs):.3f} > 0 (ok)")

    deltas = {}
    for proto, methods in by.items():
        def mean_psnr(m):
            vs = [j.get("pooled_psnr") for j in methods.get(m, [])
                  if j.get("pooled_psnr") is not None]
            return sum(vs) / len(vs) if vs else None
        b1, mip = mean_psnr("b1"), mean_psnr("mip")
        if b1 is not None and mip is not None:
            deltas[proto] = b1 - mip

    for proto in ("cone", "arc", "full"):
        if proto in deltas:
            lines.append(f"delta {proto:5s} = {deltas[proto]:+.3f} dB "
                         f"({deltas[proto]/0.139:+.1f} sigma)")

    if "full" in deltas and deltas["full"] < -SIGMA3:
        ok = False
        lines.append(f"full shows a loss of {deltas['full']:.3f} dB -- "
                     "Proposition 2 forbids it; this is a bug (FAIL)")

    for narrow in ("cone", "arc"):
        if narrow in deltas and deltas[narrow] <= SIGMA3:
            ok = False
            lines.append(f"{narrow} gain {deltas[narrow]:+.3f} dB does not "
                         f"exceed 3 sigma = {SIGMA3:.3f} (FAIL)")

    if "cone" in deltas and "arc" in deltas and deltas["cone"] <= deltas["arc"]:
        ok = False
        lines.append("ordering cone > arc does not hold (FAIL)")

    return ok, "\n  ".join(lines)


def gate_none(outdir):
    s = summary_of(outdir)
    n = len(s.get("done") or {})
    return n > 0, f"{n} jobs completed, no numeric exit criterion"


# --------------------------------------------------------------------- stages
STAGES = [
    dict(
        name="c1_verify",
        why="Does removing the 2D Mip opacity compensation actually change the "
            "arm? Nothing downstream is worth quota until this is answered.",
        hours=0.8,
        script="kaggle/blender_rung.py",
        slug="btp-c1-verify", title="BTP C1 verify",
        sets=["RUNG=C1", "SCENES=lego", "ITERS=30000", "ARMS_ENABLED=B",
              "ARM_B_BRANCH=arm-b-3dgs-vanilla",
              "ARM_B_EXTRA=--disable_2D_mip_compensation"],
        out="c1_verify", gate=gate_c1, merge=False,
    ),
    dict(
        name="t3_stress",
        why="The thesis's only untested claim. Independent of C1 -- it compares "
            "B1 against Mip-Splatting, and neither arm involves the 3DGS arm.",
        hours=8.0,
        script="kaggle/method_eval.py",
        slug="btp-t3-stress", title="BTP T3 stress",
        sets=["TABLE=3", "PROTOCOLS=full,arc,cone,mixed,grazing",
              "METHODS=mip,b1", "SCENES=lego,chair"],
        out="t3_stress", gate=gate_g3, merge=True,
    ),
    dict(
        name="c2_armb",
        why="Re-measure arm B on the fixed rasteriser so G2 can be scored. Arm A "
            "is untouched and is not re-run.",
        hours=6.0,
        script="kaggle/blender_rung.py",
        slug="btp-c2-armb", title="BTP C2 arm B",
        sets=["RUNG=C2", "ITERS=30000", "ARMS_ENABLED=B",
              "ARM_B_BRANCH=arm-b-3dgs-vanilla",
              "ARM_B_EXTRA=--disable_2D_mip_compensation"],
        out="c2_armb", gate=gate_g2, merge=True,
        requires="c1_verify",
    ),
    dict(
        name="t2_full",
        why="B1 on the six remaining standard scenes. Prediction on record: "
            "parity, because the floor binds on 100% of primitives.",
        hours=3.0,
        script="kaggle/method_eval.py",
        slug="btp-t2-b1-full", title="BTP T2 b1 full",
        sets=["TABLE=2", "PROTOCOLS=full", "METHODS=b1",
              "SCENES=ship,drums,ficus,hotdog,materials,mic"],
        out="t2_full", gate=gate_none, merge=True,
    ),
    dict(
        name="b2_eval",
        why="Table 4's PSNR and model-size columns.",
        hours=1.0,
        script="kaggle/b2_eval.py",
        slug="btp-b2-eval", title="BTP b2 eval",
        sets=[], kernel_sources=["rickaryadas/btp-r1-blender-stmt"],
        out="b2", gate=gate_none, merge=True,
    ),
]


# ---------------------------------------------------------------------- driver
def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def push(stage, wait_minutes, dry_run):
    outdir = os.path.join(KRUNS, stage["out"])
    cmd = [sys.executable, "tools/kaggle_push.py", stage["script"],
           "--slug", stage["slug"], "--title", stage["title"],
           "--dataset", DATASET, "--accelerator", "NvidiaTeslaT4",
           "--session-timeout", "43200", "--poll-timeout", "43200",
           "--retry-wrong-gpu", "4", "--out", outdir]
    for s in stage.get("sets", []):
        cmd += ["--set", s]
    for k in stage.get("kernel_sources", []):
        cmd += ["--kernel-source", k]

    if dry_run:
        print("   " + " ".join(cmd))
        return True, outdir

    while True:
        log(f"pushing {stage['name']} (~{stage['hours']} GPU-h)")
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        blob = (p.stdout or "") + (p.stderr or "")
        if QUOTA_MSG not in blob:
            sys.stdout.write(blob[-4000:])
            return p.returncode == 0, outdir
        log(f"weekly GPU quota exhausted; retrying in {wait_minutes} min. "
            "The window is rolling, so this clears on its own.")
        time.sleep(wait_minutes * 60)


def merge(outdir):
    csv_path = os.path.join(outdir, "runs.csv")
    if not os.path.exists(csv_path):
        log(f"no runs.csv in {outdir} -- nothing to merge")
        return
    subprocess.run([sys.executable, "tools/merge_run.py", csv_path],
                   cwd=ROOT, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--wait-minutes", type=int, default=30,
                    help="how long to sleep when the weekly quota is exhausted")
    a = ap.parse_args()

    plan = STAGES
    if a.only:
        plan = [s for s in plan if s["name"] == a.only]
    elif a.start:
        names = [s["name"] for s in plan]
        if a.start not in names:
            sys.exit(f"unknown stage {a.start}; have {names}")
        plan = plan[names.index(a.start):]
    if not plan:
        sys.exit("nothing to run")

    print(f"\nPlan -- {sum(s['hours'] for s in plan):.1f} GPU-h of a 30 h week\n")
    for s in plan:
        print(f"  {s['name']:11s} ~{s['hours']:>4.1f} h   {s['why']}")
    print()
    if a.dry_run:
        for s in plan:
            print(f"\n-- {s['name']}")
            push(s, a.wait_minutes, True)
        return

    passed = set()
    for s in plan:
        req = s.get("requires")
        if req and req not in passed and not a.only:
            log(f"SKIP {s['name']}: needs {req}, which did not pass")
            continue

        ok, outdir = push(s, a.wait_minutes, False)
        if not ok:
            log(f"HALT: {s['name']} did not complete")
            return 1
        if s.get("merge"):
            merge(outdir)

        gate_ok, msg = s["gate"](outdir)
        log(f"gate {s['name']}: {'PASS' if gate_ok else 'FAIL'}\n  {msg}")
        if not gate_ok:
            log(f"HALT at {s['name']}. The gate is red; nothing downstream "
                "runs. Report the table as measured and decide -- do not "
                "continue the ladder past a failed gate.")
            return 1
        passed.add(s["name"])

    log("all stages passed; rebuilding the thesis")
    subprocess.run(["make", "thesis"], cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
