from math import ceil, sqrt
from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def apply_gaussian_blur(frames: np.ndarray, sigma: Tuple[float, float]):
    _, H, W, _ = frames.shape
    sy, sx = sigma
    ky, kx = max(1, ceil(sqrt(sy))), max(1, ceil(sqrt(sx)))
    Ht, Wt = max(1, int(round(H / ky))), max(1, int(round(W / kx)))
    sigma = (sy * Ht / H, sx * Wt / W)
    Y = np.empty_like(frames)
    for f, y in zip(frames, Y):
        vmin, vmax = f.min(), f.max()
        x = f.astype(np.float32, copy=False)
        x = cv2.resize(x, (Wt, Ht), interpolation=cv2.INTER_AREA)
        x = gaussian_filter(x, sigma=sigma, axes=(0, 1))
        x = cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC)
        np.clip(x, vmin, vmax, out=x)
        y[:] = x.astype(frames.dtype, copy=False)
    return Y
