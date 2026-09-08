# Finishing the remaining runs

The weekly Kaggle GPU quota (30 h) was exhausted on 8 September 2026. Everything
below is built, tested and committed; each is one command once the quota resets.

The quota is per rolling week, so these become available again automatically.
Check it in the notebook sidebar, or by pushing anything and reading the error.

## What is outstanding

| run | command | cost |
|---|---|---|
| Table 2, remaining 6 scenes | see below | ~3 GPU-h |
| Table 3, the stress suite | see below | ~8 GPU-h |
| Table 4, B2 render + metrics | see below | ~1 GPU-h |
| R3 Tanks & Temples / Deep Blending | was running at the cutoff; fetch only | 0 |

```bash
# Table 2 -- B1 on the standard benchmark. lego and chair are already measured,
# so this covers the other six. Prediction on record: parity, because I-1
# measured the floor binding on 100% of primitives and Proposition 2 then makes
# B1 and Mip-Splatting the same filter.
python tools/kaggle_push.py kaggle/method_eval.py \
    --slug btp-t2-b1-full --title "BTP T2 b1 full" \
    --dataset nguyenhung1903/nerf-synthetic-dataset --accelerator NvidiaTeslaT4 \
    --set TABLE=2 --set PROTOCOLS=full --set METHODS=b1 \
    --set SCENES=ship,drums,ficus,hotdog,materials,mic \
    --session-timeout 43200 --poll-timeout 43200 --retry-wrong-gpu 4 \
    --out results/kaggle_runs/t2_full

# Table 3 -- the stress suite. Both methods, five protocols, two scenes.
# This is where §4.6 predicts the method must help; the conditioning column is
# already measured (see the instruments), the PSNR columns are what these runs
# produce.
python tools/kaggle_push.py kaggle/method_eval.py \
    --slug btp-t3-stress --title "BTP T3 stress" \
    --dataset nguyenhung1903/nerf-synthetic-dataset --accelerator NvidiaTeslaT4 \
    --set TABLE=3 --set PROTOCOLS=full,arc,cone,mixed,grazing \
    --set METHODS=mip,b1 --set SCENES=lego,chair \
    --session-timeout 43200 --poll-timeout 43200 --retry-wrong-gpu 4 \
    --out results/kaggle_runs/t3_stress

# Table 4 -- B2. The criterion itself is already measured on all eight scenes
# without a GPU (results/b2_local/); this adds the PSNR and model-size columns
# by rendering the masked models and scoring them.
python tools/kaggle_push.py kaggle/b2_eval.py \
    --slug btp-b2-eval --title "BTP b2 eval" \
    --dataset nguyenhung1903/nerf-synthetic-dataset \
    --kernel-source rickaryadas/btp-r1-blender-stmt --accelerator NvidiaTeslaT4 \
    --session-timeout 43200 --poll-timeout 43200 --retry-wrong-gpu 4 \
    --out results/kaggle_runs/b2

# R3's remaining scenes -- the session was already running when the quota was
# reached and Kaggle lets those finish, so this only collects the output.
python tools/kaggle_push.py kaggle/r3_real_scenes.py --fetch-only \
    --slug btp-r3-tandt-db --poll-timeout 43200 \
    --out results/kaggle_runs/r3_tandtdb
```

## After any of them

```bash
python tools/merge_run.py results/kaggle_runs/<dir>/runs.csv   # append, never edit
make thesis                                                     # regenerates everything
```

Each table fills itself from `results/runs.csv`; nothing needs editing by hand,
and a table with no rows stays an explicit "not yet measured" placeholder rather
than quietly borrowing the published column.
