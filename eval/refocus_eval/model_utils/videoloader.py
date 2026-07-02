import concurrent.futures
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from decord import VideoReader, cpu

TargetSize = Optional[
    Union[Tuple[int, int], Callable[[int, int], Optional[Tuple[int, int]]]]
]


class VideoLoader:
    def __init__(
        self,
        target_size: TargetSize = (448, 448),
        num_workers: int = 4,
        access_cost: float = 0.002,
        decode_cost: float = 0.01,
        multiprocessing: bool = True,
    ):
        """
        :param target_size: (height, width) for resize
        :param num_workers: number of parallel workers
        :param access_cost: per-frame-range access time weight
        :param decode_cost: per-frame decode+resize time weight
        """
        if target_size is not None and not callable(target_size):
            assert target_size[0] > 0 and target_size[1] > 0
        assert num_workers > 0

        self.target_size = target_size
        self.num_workers = num_workers
        self.access_cost = access_cost
        self.decode_cost = decode_cost
        self.executor = (
            concurrent.futures.ProcessPoolExecutor
            if multiprocessing
            else concurrent.futures.ThreadPoolExecutor
        )(max_workers=num_workers)

    def close(self) -> None:
        """Shut down the worker pool. Safe to call multiple times."""
        self.executor.shutdown(wait=True)

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self):
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass

    def _split_indices_dp(self, idx_list: List[int], k: int) -> List[List[int]]:
        """
        O(n·k) two-pointer DP.
        Minimises max segment-cost where
        cost = access_cost*(range) + decode_cost*(count).
        """
        if not idx_list:
            return []

        n = len(idx_list)
        k = min(k, n)

        f = [[float("inf")] * (n + 1) for _ in range(k + 1)]
        P = [[0] * (n + 1) for _ in range(k + 1)]
        f[0][0] = 0.0

        for s in range(1, k + 1):
            pointer = 0
            for j in range(1, n + 1):
                pointer = min(pointer, j - 1)
                while pointer + 1 < j:
                    cost_curr = max(
                        f[s - 1][pointer],
                        self.access_cost * (idx_list[j - 1] - idx_list[pointer])
                        + self.decode_cost * (j - pointer),
                    )
                    cost_next = max(
                        f[s - 1][pointer + 1],
                        self.access_cost * (idx_list[j - 1] - idx_list[pointer + 1])
                        + self.decode_cost * (j - pointer - 1),
                    )
                    if cost_next <= cost_curr:
                        pointer += 1
                    else:
                        break
                seg_cost = self.access_cost * (
                    idx_list[j - 1] - idx_list[pointer]
                ) + self.decode_cost * (j - pointer)
                f[s][j] = max(f[s - 1][pointer], seg_cost)
                P[s][j] = pointer

        # backtrace to reconstruct segments
        segments, s, j = [], k, n
        while s:
            p = P[s][j]
            segments.append(idx_list[p:j])
            j, s = p, s - 1
        return segments[::-1]

    @staticmethod
    def _load_and_resize(
        video_path: str,
        indices: List[int],
        target_size: TargetSize,
    ) -> np.ndarray:
        """
        Load frames with decord (num_threads=1) and resize:
        - Resize strategy:
            * both downscale -> INTER_AREA
            * both upscale   -> INTER_CUBIC
            * mixed          -> AREA then CUBIC
        Returns array shape (N, H, W, C).
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        frames = vr.get_batch(indices).asnumpy()
        b, h, w, c = frames.shape
        ts = target_size(h, w) if callable(target_size) else target_size
        if ts is None:
            return frames

        th, tw = ts
        assert (
            int(th) == th and th > 0 and int(tw) == tw and tw > 0
        ), "resolved target_size (height, width) must be positive integers"
        th, tw = int(th), int(tw)

        if (tw, th) == (w, h):
            return frames

        output = np.empty_like(frames, shape=(b, th, tw, c))
        for f, o in zip(frames, output):
            match (tw <= w, th <= h):
                case (True, True):
                    cv2.resize(f, (tw, th), dst=o, interpolation=cv2.INTER_AREA)
                case (False, False):
                    cv2.resize(f, (tw, th), dst=o, interpolation=cv2.INTER_CUBIC)
                case (True, False):
                    tmp = cv2.resize(f, (tw, h), interpolation=cv2.INTER_AREA)
                    cv2.resize(tmp, (tw, th), dst=o, interpolation=cv2.INTER_CUBIC)
                case (False, True):
                    tmp = cv2.resize(f, (w, th), interpolation=cv2.INTER_AREA)
                    cv2.resize(tmp, (tw, th), dst=o, interpolation=cv2.INTER_CUBIC)
        return output

    def run(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """
        Process a single video's frames:
        1. Split frame_indices into ≤num_workers segments via DP.
        2. Decode+resize each segment in parallel.
        3. Reassemble and return as a (N, H, W, C) array.
        """
        assert len(frame_indices) > 0
        assert all(int(i) == i and i >= 0 for i in frame_indices)

        pos_map: Dict[int, List[int]] = {}
        for idx, f in enumerate(frame_indices):
            pos_map.setdefault(int(f), []).append(idx)
        uniq_frames = sorted(pos_map.keys())

        n = len(frame_indices)

        # split into balanced segments
        segments = self._split_indices_dp(uniq_frames, self.num_workers)

        # submit each segment to the pool
        futures = {
            self.executor.submit(
                VideoLoader._load_and_resize, video_path, seg, self.target_size
            ): seg
            for seg in segments
        }

        # collect per-frame results
        output: Optional[np.ndarray] = None
        for fut in concurrent.futures.as_completed(futures):
            seg = futures[fut]
            arr = fut.result()  # shape (len(seg), H, W, C)
            if output is None:
                output = np.empty_like(arr, shape=(n, *arr.shape[1:]))
            for a, i in zip(arr, seg):
                output[pos_map[i]] = a

        return output


if __name__ == "__main__":
    # example usage
    tasks = [
        ("/path/to/video1.mp4", [0, 5, 10, 20]),
        ("/path/to/video2.mp4", [3, 4, 7, 15, 30]),
    ]

    with VideoLoader(target_size=(224, 224), num_workers=3) as loader:
        for path, indices in tasks:
            frames = loader.run(path, indices)
            print(f"{path}: {frames.shape}")
