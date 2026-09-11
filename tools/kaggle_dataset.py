#!/usr/bin/env python3
r"""
Upload a local directory to Kaggle as a dataset, under any configured account.

Why this exists. A Kaggle kernel's *output* is visible only to the account that
produced it, so a stage that mounts another rung's checkpoints can only run as
that rung's owner (tools/kaggle_push.py says so, and it is verified against the
live API). When that account is inside its rolling GPU-quota window and another
account is not, the programme stalls on a permissions detail rather than on
compute. A *dataset*, unlike a kernel output, can be created by one account and
mounted by any kernel that account owns -- so pulling a rung's checkpoints down
and re-uploading them as a dataset moves the work onto whichever account has
quota, at the cost of bandwidth and no GPU at all.

Nothing about the science changes: the PLYs uploaded are byte-identical to the
ones the original kernel wrote, and the directory layout is preserved, so a
kernel that discovers models by walking /kaggle/input finds them unchanged.

    python tools/kaggle_dataset.py <dir> --slug btp-r1-arma-ckpt \
        --title "BTP R1 arm A checkpoints" --username ceoricky

Prints the dataset ref to pass to `kaggle_push.py --dataset`.
"""
import argparse
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kaggle_client as kc  # noqa: E402

from kagglesdk.blobs.types.blob_api_service import (  # noqa: E402
    ApiBlobType, ApiStartBlobUploadRequest)
from kagglesdk.datasets.types.dataset_api_service import (  # noqa: E402
    ApiCreateDatasetRequest, ApiCreateDatasetVersionRequest, ApiDatasetNewFile)


def walk(root):
    """Every file under root, as (absolute path, posix path relative to root)."""
    out = []
    for d, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(d, fn)
            out.append((p, os.path.relpath(p, root).replace(os.sep, "/")))
    return sorted(out, key=lambda t: t[1])


def upload_one(client, path, rel):
    """Push one file to Kaggle's blob store; return the token that names it."""
    req = ApiStartBlobUploadRequest()
    req.type = ApiBlobType.DATASET
    req.name = rel
    req.content_length = os.path.getsize(path)
    req.last_modified_epoch_seconds = int(os.path.getmtime(path))
    resp = client.blobs.blob_api_client.start_blob_upload(req)
    with open(path, "rb") as fh:
        put = requests.put(resp.create_url, data=fh, timeout=3600)
    put.raise_for_status()
    return resp.token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--username", required=True,
                    help="the account to own the dataset; must be in .env")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--new-version", action="store_true",
                    help="add a version to a dataset that already exists")
    a = ap.parse_args()

    acct = kc.account_for(a.username)
    files = walk(a.directory)
    total = sum(os.path.getsize(p) for p, _ in files)
    print(f"{len(files)} files, {total / 1e6:.0f} MB -> {acct.username}/{a.slug}",
          flush=True)

    with kc.get_client(acct) as client:
        tokens = []
        t0 = time.time()
        for i, (path, rel) in enumerate(files, 1):
            tokens.append((upload_one(client, path, rel), rel))
            done = sum(os.path.getsize(p) for p, _ in files[:i])
            print(f"  [{i}/{len(files)}] {rel}  "
                  f"{done / 1e6:.0f}/{total / 1e6:.0f} MB  "
                  f"{time.time() - t0:.0f}s", flush=True)

        new_files = []
        for token, rel in tokens:
            f = ApiDatasetNewFile()
            f.token = token
            new_files.append(f)

        if a.new_version:
            req = ApiCreateDatasetVersionRequest()
            req.owner_slug = acct.username
            req.slug = a.slug
            req.version_notes = "refresh"
            req.files = new_files
            resp = client.datasets.dataset_api_client.create_dataset_version(req)
        else:
            req = ApiCreateDatasetRequest()
            req.owner_slug = acct.username
            req.slug = a.slug
            req.title = a.title
            req.license_name = "CC0-1.0"
            req.is_private = True
            req.files = new_files
            resp = client.datasets.dataset_api_client.create_dataset(req)

    err = getattr(resp, "error", None)
    if err:
        sys.exit(f"Kaggle refused the dataset: {err}")
    print(f"\ndataset ref: {acct.username}/{a.slug}")
    print(f"url: {getattr(resp, 'url', '')}")


if __name__ == "__main__":
    main()
