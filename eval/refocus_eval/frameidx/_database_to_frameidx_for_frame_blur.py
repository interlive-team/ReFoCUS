#!/usr/bin/env python
"""
Segment blur/deblur writer (simplified).

- Split [first, last] into N equal segments.
- Use segment centers (uniform mid) as representative frame indices.
- Aggregate all 'candidates' (assumed to index into frame_idx) to segment frequencies.
- Decide blur flags by mode:
    * top_blur    : blur top-k frequent segments
    * top_deblur  : deblur top-k frequent segments, others blur
    * random_blur : blur k random segments (no frequency collection)

Output per item:
    db_out[key] = {"indices": list[int], "blur": list[bool]}
"""

import argparse
import random
import shelve
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write segment-wise blur/deblur with uniform-mid indices (simplified)"
    )
    p.add_argument("in_db", type=str, help="Path to input shelve DB")
    p.add_argument("out_db", type=str, help="Path to output shelve DB")
    p.add_argument(
        "--numframes", type=int, default=32, help="Number of segments/indices"
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["top_blur", "top_deblur", "random_blur"],
        default="top_blur",
    )
    p.add_argument("--k", type=int, default=8, help="Number of segments to blur/deblur")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--numquery",
        type=int,
        default=32,
        help="Generated query length (last dim of candidates)",
    )
    p.add_argument(
        "--usefirst",
        type=int,
        default=None,
        help="Use only the first K positions per candidate row (default: use all numquery)",
    )
    return p.parse_args()


def flatten(xs: Iterable) -> Iterable:
    for x in xs:
        if isinstance(x, (list, tuple, np.ndarray)):
            yield from flatten(x)
        else:
            yield x


def uniform_centers(first: int, last: int, n: int) -> np.ndarray:
    span = float(last - first)
    step = span / n
    centers = first + (np.arange(n, dtype=float) + 0.5) * step
    centers = np.rint(centers)
    centers = np.clip(centers, first, last).astype(int)
    return centers


def frame_to_segment(frame: int, first: int, last: int, nseg: int) -> int:
    assert first <= frame <= last
    length = last - first + 1
    seg = (frame - first) * nseg // length
    return int(seg)


def collect_counts_from_candidates(
    candidates,
    frame_idx: np.ndarray,
    first: int,
    last: int,
    nseg: int,
    numquery: int,
    usefirst: Optional[int],
) -> np.ndarray:
    counts = np.zeros(nseg, dtype=int)

    if candidates is not None:
        cand = np.asarray(candidates)
        if cand.ndim == 1:
            cand = cand.reshape(-1, numquery)
        else:
            assert cand.ndim == 2
            assert cand.shape[-1] == numquery

        if usefirst is not None:
            cand = cand[:, :usefirst]

        for cid in cand.ravel():
            s = frame_to_segment(frame_idx[cid], first, last, nseg)
            counts[s] += 1

    return counts


def choose_topk_flags(counts: np.ndarray, k: int, invert: bool = False) -> List[bool]:
    assert k > 0
    n = len(counts)
    tie = list(range(n))
    random.shuffle(tie)
    order = sorted(range(n), key=lambda i: (-int(counts[i]), tie[i]))
    topk = set(order[:k])
    return [((i in topk) ^ invert) for i in range(n)]


def choose_random_flags(n: int, k: int) -> List[bool]:
    k = max(0, min(k, n))
    chosen = set(random.sample(range(n), k)) if k else set()
    return [(i in chosen) for i in range(n)]


def main() -> None:
    args = parse_args()
    assert 1 < args.numframes
    assert 1 <= args.k <= args.numframes
    assert 0 < args.numquery
    assert args.usefirst is None or 0 < args.usefirst <= args.numquery

    random.seed(args.seed)
    Path(args.out_db).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    with shelve.open(args.in_db, "r") as db_in, shelve.open(args.out_db, "n") as db_out:
        for key, val in db_in.items():
            frame_idx = np.asarray(val["frame_idx"], dtype=int)
            first, last = int(frame_idx[0]), int(frame_idx[-1])
            centers = uniform_centers(first, last, args.numframes)

            if args.mode == "random_blur":
                blur_flags = choose_random_flags(args.numframes, args.k)
            else:
                counts = collect_counts_from_candidates(
                    val["candidates"],
                    frame_idx,
                    first,
                    last,
                    args.numframes,
                    args.numquery,
                    args.usefirst,
                )
                if args.mode == "top_blur":
                    blur_flags = choose_topk_flags(counts, args.k, invert=False)
                else:  # top_deblur
                    blur_flags = choose_topk_flags(counts, args.k, invert=True)

            db_out[key] = {
                "indices": centers.tolist(),
                "blur": [bool(x) for x in blur_flags],
            }

        print(f"[OK] {len(db_out)} items written → {args.out_db}")
        print(
            f"[INFO] mode={args.mode} numframes={args.numframes} k={args.k} "
            f"seed={args.seed} numquery={args.numquery} usefirst={args.usefirst}"
        )


if __name__ == "__main__":
    main()
