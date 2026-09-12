#!/usr/bin/env python3
r"""
Download the qualitative renders a finished Kaggle kernel kept, by path pattern.

Every rung deletes all but the first `KEEP_QUALITATIVE` rendered PNGs per model
directory once metrics have consumed them (tools/kernel_common.prune_renders) --
8 scenes x 4 scales x 200 views x 2 x 2 arms is several gigabytes and
/kaggle/working ships as the kernel's output. What survives is a handful of
matched (prediction, ground truth) pairs per model, which is exactly what a
qualitative figure needs and what kaggle_push.py never downloaded: its
WANTED_OUTPUT list is the small result files, by design, because pulling every
PNG on every fetch would dominate the run time.

    python tools/fetch_renders.py rickaryadas btp-r1-blender-stmt \
        --out results/renders/r1 --match "out_arm"

The layout mirrors the kernel's, so a later tool can find a prediction and its
ground truth by construction rather than by guessing:

    <out>/<model dir>/<gt_-1|test_preds_-1>/<index>.png
"""
import argparse
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_client import accounts, get_client  # noqa: E402
from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E402
    ApiListKernelSessionOutputRequest,
)


def listing(user, slug, max_pages=60):
    """Every output file of a kernel session, walking the pagination.

    The listing is capped per page and a rung's output runs to hundreds of
    entries with the cloned repository in front of the renders, so a single
    page contains the checkout and none of the pictures.
    """
    acct = [a for a in accounts() if a.username == user]
    if not acct:
        sys.exit(f"{user}: not configured in .env")
    with get_client(acct[0]) as c:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug = user, slug
        resp = c.kernels.kernels_api_client.list_kernel_session_output(req)
        files = list(resp.files or [])
        token, pages = getattr(resp, "next_page_token", ""), 1
        while token and pages < max_pages:
            more = ApiListKernelSessionOutputRequest()
            more.user_name, more.kernel_slug, more.page_token = user, slug, token
            r2 = c.kernels.kernels_api_client.list_kernel_session_output(more)
            files += list(r2.files or [])
            token, pages = getattr(r2, "next_page_token", ""), pages + 1
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("slug")
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", default="", help="substring the path must contain")
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()

    files = listing(a.user, a.slug)
    want = []
    for f_ in files:
        path = f_.file_name.replace("\\", "/")
        if not path.lower().endswith(".png") or "/test/" not in path:
            continue
        if a.match and a.match not in path:
            continue
        if not getattr(f_, "url", ""):
            continue
        want.append((path, f_.url))
    print(f"{a.user}/{a.slug}: {len(files)} output files, {len(want)} renders match")

    got = 0
    for path, url in sorted(want)[: a.limit]:
        # Keep the model directory and the gt/pred directory, drop the rest of
        # the kernel's working-directory prefix: the pairing is what matters.
        parts = path.split("/")
        model = parts[parts.index("test") - 1]
        arm = "armA" if "out_armA" in path else ("armB" if "out_armB" in path else "")
        leaf = "/".join(parts[-2:])
        dst = os.path.join(a.out, (arm + "_" if arm else "") + model, *leaf.split("/"))
        if os.path.exists(dst):
            got += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            with open(dst, "wb") as fh:
                fh.write(r.content)
            got += 1
        except Exception as e:
            print(f"  could not fetch {path}: {type(e).__name__}: {e}")
    print(f"wrote {got} file(s) under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
