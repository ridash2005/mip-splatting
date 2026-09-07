# Kaggle run logs

Evidence for the reproduction report: the decoded stdout/stderr of every kernel
launch, kept whether it succeeded or not. `tools/kaggle_push.py` writes
`log.txt` (Kaggle's JSON event stream) into the run's directory; the files here
are those streams decoded to plain text and named for what happened.

| file | what it shows |
|---|---|
| `r0_smoke_v2/01_p100_abort.log` | Kaggle allocated a Tesla P100 (CC 6.0). The kernel refused to train (F15) and exited in 7 s. |
| `r0_smoke_v2/02_t4x2_open3d_failure.log` | Tesla T4 ×2. Checks 1–4 and 8 pass, **both CUDA extensions build against torch 2.10 / CUDA 12.8**, `convert_blender_data.py` writes the four scales — then `train.py` dies on `ModuleNotFoundError: No module named 'open3d'`. Check 7 also failed, against a correct pipeline: it asserted on `metadata.json`, but F3's d0-only filtering happens in `readMultiScale(..., only_highres)` at load time. Both fixed in the following commit. |
| `r0_smoke_v3/01_p100_abort.log` | P100 again, with the `R0_ABORT=WRONG_ACCELERATOR` marker that `--retry-wrong-gpu` keys off. |
| `r0_smoke_v3/log_decoded.txt` | First run on T4 ×2 via `machineShape`. Checks 1–4, 7, 8 green; **arm A trained 7000 iterations in 284 s**; then died in the run's own instrumentation — `VramProbe` shadowed `threading.Thread._stop` with an Event, so `join()` raised `TypeError: 'Event' object is not callable`. |
| `r0_smoke_v4/01_t4x2_both_arms_resume_failed.log` | **Both arms complete end to end.** Every check green except 11: torch 2.6+ defaults `torch.load(weights_only=True)` and `train.py` calls it bare, so resuming from a checkpoint containing a numpy scalar fails. Fixed with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, an environment setting rather than a source edit. |
| `r0_smoke_v4/runs.csv`, `r0_smoke_v4/summary.json` | The kernel's own outputs, downloaded from the session. `runs.csv` merged into `results/runs.csv`; `summary.json` drives `results/R0-report.md`. |

The accelerator is not selectable: `ApiSaveKernelRequest` carries a boolean
`enable_gpu` and kagglesdk 0.1.28 has no accelerator field at all. See the
README's Environment section.
