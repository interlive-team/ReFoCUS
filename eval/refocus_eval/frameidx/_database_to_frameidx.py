#!/usr/bin/env python
"""
Extract and save a set of frame indices from a shelve-based video DB.

Usage
-----
python database_to_frameidx.py \
    /path/to/database_file \
    /path/to/save/indices.npy \
    --candidate 0          # which RL candidate to adopt (0-63)

Arguments
---------
db_path      : positional – shelve DB file created by the evaluation scripts
out_path     : positional – file to write the selected indices to (.npy or .txt)
--candidate  : optional  – candidate row (0-63) to adopt; default = 0
--sample-id  : optional  – item number in the DB to export; default = 0
"""

import argparse
import shelve
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export chosen frame indices from DB")
    p.add_argument("in_db", type=str, help="Path to shelve DB")
    p.add_argument("out_db", type=str, help="Destination file for indices")
    p.add_argument("--candidate", type=int, default=0, help="Candidate idx (0–63)")
    p.add_argument(
        "--minframes",
        type=int,
        default=0,
        help="If frame count <= this, use uniform sampling",
    )
    p.add_argument(
        "--sort", action="store_true", help="Sort output indices before saving"
    )
    p.add_argument(
        "--numquery", type=int, default=32, help="Generated query length for in_db"
    )
    p.add_argument(
        "--numframes", type=int, default=32, help="Number of frames to export"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert args.numframes <= args.numquery

    Path(args.out_db).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    with shelve.open(args.in_db, "r") as db_in, shelve.open(args.out_db, "n") as db_out:

        cand_count = 0
        total_count = 0

        for key, val in db_in.items():
            total_count += 1
            frame_idx = np.asarray(val["frame_idx"])
            if val["candidates"] is None or len(frame_idx) <= args.minframes:
                indices = np.linspace(
                    frame_idx[0], frame_idx[-1], args.numframes, dtype=int
                )
            else:
                cand_count += 1
                candidates = np.asarray(val["candidates"])

                if candidates.ndim == 1:
                    candidates = candidates.reshape(-1, args.numquery)
                else:
                    assert candidates.ndim == 2
                    assert candidates.shape[-1] == args.numquery

                indices = frame_idx[candidates[args.candidate][: args.numframes]]

            if args.sort:
                indices = np.sort(indices)

            db_out[key] = indices

        print(f"[OK] {len(db_out)} items written → {args.out_db}")

        if total_count > 0:
            print(
                f"[INFO] selected with candidates : {cand_count} / {total_count} ({cand_count/total_count:.2%})"
            )
        else:
            print("[INFO] No items processed.")


if __name__ == "__main__":
    main()
