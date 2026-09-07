#!/usr/bin/env python3
"""
Run train.py under a chosen seed, without editing train.py.

§7's R5 rung ("Seed spread · NOT OPTIONAL") needs three seeds, but the code
exposes none: utils/general_utils.safe_state() calls random.seed(0),
np.random.seed(0) and torch.manual_seed(0) unconditionally, and train.py has no
--seed flag. Adding one would be a source change outside §3.

So the seed is applied from outside instead. train.py does
`from utils.general_utils import safe_state`, binding the name at import time —
so patching the attribute on the module *before* train.py is executed is enough,
and train.py itself is untouched on disk.

    BTP_SEED=1 python tools/seeded_train.py -s <scene> -m <out> --eval ...

Takes exactly the arguments train.py takes and forwards them unchanged.

One trap this handles: readMultiScaleNerfSyntheticInfo caches the random initial
point cloud as `points3d.ply` in the SOURCE directory and reuses it whenever the
file exists. Two seeds would then start from an identical initialisation and the
spread would be understated. The file is removed for any non-zero seed, so each
seed draws its own, and the seed used is printed so the log carries it.
"""
import os
import random
import runpy
import sys

SEED = int(os.environ.get("BTP_SEED", "0"))


def main():
    import numpy as np
    import torch
    import utils.general_utils as gu

    original = gu.safe_state

    def seeded_safe_state(silent):
        original(silent)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        print(f"[seeded_train] reseeded to {SEED} after safe_state", flush=True)

    gu.safe_state = seeded_safe_state

    # Drop the cached initial point cloud so this seed draws its own. Only for a
    # non-zero seed: seed 0 must stay bit-identical to a plain train.py run, so
    # the R5 sweep is anchored to the R1/R2 numbers rather than to a fresh draw.
    #
    # BTP_KEEP_INIT=1 suppresses this. A sweep runs both arms of one (scene,
    # seed) concurrently on two GPUs, and both would race to delete and rewrite
    # the same file -- one could read a half-written PLY, and the two arms would
    # no longer share an initialisation, which is the whole point of pairing
    # them. The rung kernel therefore pre-generates each seed's point cloud
    # serially into its own source directory and sets this.
    if SEED and os.environ.get("BTP_KEEP_INIT") != "1":
        try:
            i = sys.argv.index("-s")
        except ValueError:
            i = sys.argv.index("--source_path")
        ply = os.path.join(sys.argv[i + 1], "points3d.ply")
        if os.path.exists(ply):
            os.remove(ply)
            print(f"[seeded_train] removed cached {ply} so seed {SEED} draws its own",
                  flush=True)

    runpy.run_path("train.py", run_name="__main__")


if __name__ == "__main__":
    main()
