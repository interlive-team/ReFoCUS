"""Download the ReFoCUS-393K video shards into the HF cache and unpack them into video_root.

Shards are fetched concurrently and each one is unpacked as soon as its download finishes,
so that <video_root>/<sample["video"]> resolves for every training sample. The tars stay in
the HF cache (583 GiB on top of the unpacked videos); a re-run reuses them. Request access
to the gated dataset on the Hub first and log in with `hf auth login`.

    python scripts/download_videos.py --video_root VIDEO_FOLDER
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

DATASET = "interlive/ReFoCUS-393K"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    shards = sorted(f for f in list_repo_files(DATASET, repo_type="dataset") if f.endswith(".tar"))
    assert shards, f"no tar shards in {DATASET}"
    args.video_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.max_workers) as pool:
        futures = {
            pool.submit(hf_hub_download, DATASET, shard, repo_type="dataset"): shard
            for shard in shards
        }
        for done, future in enumerate(as_completed(futures), 1):
            # Members are stored as transcoded/<source>/...; drop that prefix.
            subprocess.run(
                ["tar", "-xf", future.result(), "-C", args.video_root, "--strip-components=1"],
                check=True,
            )
            print(f"[{done}/{len(shards)}] unpacked {futures[future]}")
    print(f"{sum(1 for _ in args.video_root.rglob('*.mp4'))} videos under {args.video_root}")


if __name__ == "__main__":
    main()
