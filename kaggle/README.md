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

## Choosing the accelerator

Kaggle allocates a **Tesla P100 (compute capability 6.0)** to every plain
`enable_gpu: true` push, and 3DGS cannot run on it (F15). Eleven consecutive
pushes got one. It is not a lottery and retrying does not help.

The accelerator is selected by a **`machineShape`** field on
`/api/v1/kernels/push`. That name is not guesswork — it is read out of
kagglesdk 0.1.37's wire schema, whose docstring says:

> The machine shape to use for this session. Currently supported options:
> `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`.

Neither installed client can send it. kagglesdk 0.1.28's `ApiSaveKernelRequest`
has no `machine_shape` field and rejects unknown attributes client-side, and
both newer kagglesdk and `kaggle` 2.x require Python ≥ 3.11 while this machine
has 3.10. So `tools/kaggle_push.py --accelerator` serialises the body kagglesdk
would have sent, adds the field, and posts it directly with the same Bearer
token.

Confirmed by running it — a probe kernel pushed with
`machineShape=NvidiaTeslaT4` reported:

```
GPU 0: Tesla T4 (UUID: GPU-1e1b2616-...)
GPU 1: Tesla T4 (UUID: GPU-1afb1f2e-...)
```

So `NvidiaTeslaT4` yields the two-T4 configuration, which is what the shipped
`GPUtil` dispatcher wants for R1 (F13).

**The API validates none of this.** An unrecognised field name and an invalid
`machineShape` value both return HTTP 200 with an empty `error`, and you get a
P100 anyway. `accelerator` and `acceleratorId` were both tried first and are
silently ignored. That is why the kernel asserts the capability itself in
check 1 and prints the GPU it actually got, instead of trusting the request —
and why `--accelerator`'s choices are limited to the three the schema documents.

`--retry-wrong-gpu N` remains as a backstop for a genuinely transient
allocation, keyed off the kernel's `R0_ABORT=WRONG_ACCELERATOR` marker and
nothing else.

### Fallback: launch from the web UI

Still available if the field ever stops working.

1. Open <https://www.kaggle.com/code/rickaryadas/btp-r0-smoke-lego-7k-v2>. It
   already holds the current script — every push updates it.
2. Settings → **Accelerator** → **GPU T4 x2**. Confirm the
   `nguyenhung1903/nerf-synthetic-dataset` input and **Internet: on** (the
   kernel clones from GitHub and pip-installs).
3. **Save & Run All (Commit)**.
4. Pull the result down without pushing anything over it:

```bash
python tools/kaggle_push.py kaggle/r0_smoke.py --fetch-only     --slug btp-r0-smoke-lego-7k-v2 --poll-timeout 21600     --out results/kaggle_runs/r0_smoke_v3
make report SUMMARY=results/kaggle_runs/r0_smoke_v3/summary.json OUT=results/R0-report.md
```

Note: only **two** concurrent batch GPU sessions are allowed per account —
a third push returns `Maximum batch GPU session count of 2 reached.`

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
