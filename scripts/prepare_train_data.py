"""Filter the released per-stage meta jsons down to samples whose video exists locally.

The training loader and the reward servers both open <video_root>/<sample["video"]>
directly, so a sample whose video failed to download must be dropped here rather
than at runtime (it would otherwise shift the step count silently).

    python scripts/prepare_train_data.py --meta_dir META --video_root VIDEO_FOLDER --output_dir DATA_ROOT

META is the downloaded interlive/ReFoCUS-393K-metadata dataset; every train_stage*.json in it
is written to DATA_ROOT under the same name, which train_refocus.sh reads by STAGE.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def filter_meta(meta: Path, video_root: Path, output: Path):
    samples = json.loads(meta.read_text())
    kept, dropped = [], Counter()
    for sample in samples:
        if (video_root / sample["video"]).is_file():
            kept.append(sample)
        else:
            dropped[sample["video"].split("/", 1)[0]] += 1
    output.write_text(json.dumps(kept))
    print(f"{meta.name}: kept {len(kept)} / {len(samples)} -> {output}")
    for source, n in sorted(dropped.items()):
        print(f"  dropped {n} from {source} (video missing)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_dir", type=Path, required=True)
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    metas = sorted(args.meta_dir.glob("train_stage*.json"))
    assert metas, f"no train_stage*.json under {args.meta_dir}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for meta in metas:
        filter_meta(meta, args.video_root, args.output_dir / meta.name)


if __name__ == "__main__":
    main()
