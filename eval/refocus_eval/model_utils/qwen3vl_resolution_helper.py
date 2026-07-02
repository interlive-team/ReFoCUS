import math
from typing import Optional, Tuple


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    max_ratio: int = 200,
) -> Tuple[int, int]:
    assert max_pixels >= min_pixels, "max_pixels must be >= min_pixels."
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def video_resolution(
    nframes: int,
    height: int,
    width: int,
    image_patch_size: int = 14,
    *,
    spatial_merge_size: int = 2,
    video_min_token_num: int = 128,
    video_max_token_num: int = 768,
    frame_factor: int = 2,
    model_seq_len: int = 128000,
    max_ratio: int = 200,
) -> Tuple[int, int]:
    """
    Compute final (H, W) from frame count and source size using the same
    budgeting logic as the original pipeline (frames + context budget).
    """
    image_factor = image_patch_size * spatial_merge_size

    min_pixels = video_min_token_num * (image_factor**2)
    max_frame_pixels_hard = video_max_token_num * (image_factor**2)
    total_pixels = int(model_seq_len * (image_factor**2) * 0.9)
    per_frame_total_cap = (total_pixels / nframes) * frame_factor
    max_pixels = max(
        min(max_frame_pixels_hard, per_frame_total_cap),
        int(min_pixels * 1.05),
    )

    resized_h, resized_w = smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
        max_ratio=max_ratio,
    )
    return int(resized_h), int(resized_w)
