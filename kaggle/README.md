# Kaggle kernels

`r0_smoke.py` is the R0 rung (§7): `lego`, 7 000 iterations, both arms, end to
end, plus the runnable subset of §6's twelve pre-flight checks. It is a Kaggle
*script* kernel — it runs top to bottom, prints every assertion it makes, and
ends with a single `R0_SUMMARY_JSON=` line that `tools/kaggle_push.py` parses
into `summary.json`, which `tools/report_rung.py` turns into the §13 report.

It needs a GPU with **compute capability 7.0+** (F15) and the
`nguyenhung1903/nerf-synthetic-dataset` dataset attached. It clones both arms
from GitHub, so anything not pushed is not in the run.

## Launching

```bash
python tools/kaggle_push.py kaggle/r0_smoke.py \
    --slug btp-r0-smoke-lego-7k-v2 --title "BTP R0 smoke lego 7k v2" \
    --dataset nguyenhung1903/nerf-synthetic-dataset \
    --session-timeout 21600 --poll-timeout 21600 --retry-wrong-gpu 12 \
    --out results/kaggle_runs/r0_smoke_v3
```

## The accelerator problem

**The API cannot choose the GPU.** `ApiSaveKernelRequest` carries a boolean
`enable_gpu` and nothing else — kagglesdk 0.1.28 has no accelerator field
anywhere in it — and every API-pushed launch so far has been allocated a Tesla
P100 (compute capability 6.0), which 3DGS cannot run on:

| launch | GPU | outcome |
|---|---|---|
| 1 | Tesla P100 | aborted at check 1 in 7 s (F15) |
| 2 | **Tesla T4 ×2** | got as far as `train.py`, died on the missing `open3d` |
| 3–8 | Tesla P100 ×6 | aborted at check 1 |

Launch 2 is the only one that got a T4, and it is the only one not pushed from
this tool in this session. So the accelerator looks like a per-notebook setting
that an API push resets to the default, not a per-launch lottery.

The kernel prints `R0_ABORT=WRONG_ACCELERATOR` on that path so the retry is
never confused with a real failure, and `--retry-wrong-gpu N` relaunches on that
marker alone. Each wrong-GPU launch costs about 30 seconds, so retrying is
nearly free — but if the setting really is sticky, retrying cannot fix it.

### Fallback: launch from the web UI

The accelerator dropdown exists only there.

1. Open <https://www.kaggle.com/code/rickaryadas/btp-r0-smoke-lego-7k-v2>. It
   already holds the current script — every push updates it.
2. Settings → **Accelerator** → **GPU T4 x2**. Confirm the
   `nguyenhung1903/nerf-synthetic-dataset` input and **Internet: on** (the
   kernel clones from GitHub and pip-installs).
3. **Save & Run All (Commit)**.
4. Pull the result down without pushing anything over it:

```bash
python tools/kaggle_push.py kaggle/r0_smoke.py --fetch-only \
    --slug btp-r0-smoke-lego-7k-v2 --poll-timeout 21600 \
    --out results/kaggle_runs/r0_smoke_v3
make report SUMMARY=results/kaggle_runs/r0_smoke_v3/summary.json OUT=results/R0-report.md
```

## What the run costs

Roughly 1 GPU-hour, within R0's ≤ 1 GPU-h budget: two arms × (7 000 iterations
+ 800 test-view renders + LPIPS-VGG over 800 pairs), plus the check-11 resume
test, which trains a further 2 000 + 7 000 iterations.

`/kaggle/working` is the kernel's downloadable output, so the run prunes itself
before exiting (§8): both `.git` directories, the regenerated `multi-scale`
dataset, the optimiser checkpoints and all but `KEEP_QUALITATIVE` renders per
directory. What survives is `results/`, `logs/`, both point clouds, and the
per-scene `results.json` / `per_view.json`.

Each arm's rows are appended to `results/runs.csv` and the summary is written
as soon as both arms finish — before the resume test — so a session that hits
its cap partway through still yields usable results.
