#!/usr/bin/env python
"""
Extract and save frame indices from a shelve-based video DB with two strategies.

Input
-----
Each DB item must contain a key "frame_idx" whose 0th and -1st elements are the video's first and last frame indices (closed interval).

Strategies
----------
- endpoints : Uniform sampling including both ends over [first, last].
- centers   : Split [first, last] into N equal segments and pick the center of each segment.

Usage
-----
python database_to_frameidx_uniform.py \
    /path/to/in_db \
    /path/to/out_db \
    --numframes 32 \
    --mode end   # or 'mid'
"""

import argparse
import shelve
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export frame indices from DB with uniform or segment-center sampling"
    )
    p.add_argument("in_db", type=str, help="Path to input shelve DB")
    p.add_argument("out_db", type=str, help="Destination shelve DB for sampled indices")
    p.add_argument(
        "--numframes", type=int, default=32, help="Number of frames to sample per item"
    )
    p.add_argument(
        "--mode",
        type=str,
        default="end",
        choices=["end", "mid"],
        help="Sampling strategy",
    )
    return p.parse_args()


def uniform_endpoints(first: int, last: int, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([first], dtype=int)
    return np.linspace(first, last, num=n, dtype=int)


def uniform_centers(first: int, last: int, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([int(round((first + last) / 2))], dtype=int)
    span = float(last - first)
    step = span / n
    centers = first + (np.arange(n, dtype=float) + 0.5) * step
    centers = np.rint(centers)
    centers = np.clip(centers, first, last).astype(int)
    return centers


def main() -> None:
    args = parse_args()
    Path(args.out_db).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    with shelve.open(args.in_db, "r") as db_in, shelve.open(args.out_db, "n") as db_out:

        count = 0
        for key, val in db_in.items():
            frame_idx = np.asarray(val["frame_idx"], dtype=int)
            first = int(frame_idx[0])
            last = int(frame_idx[-1])

            if args.mode == "end":
                indices = uniform_endpoints(first, last, args.numframes)
            elif args.mode == "mid":
                indices = uniform_centers(first, last, args.numframes)
            else:
                raise ValueError(f"unknown {args.mode=}")

            db_out[key] = indices
            count += 1

        print(f"[OK] {count} items written → {args.out_db}")


if __name__ == "__main__":
    main()
