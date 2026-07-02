import math


def image_resolution(
    height: int,
    width: int,
    factor: int = 28,
    min_tokens: int = 4 * 4,
    max_tokens: int = 16384,
):
    """
    Single-image version assuming `max_tokens` applies to this image only.
    Keeps the original rounding/scaling logic.
    """
    min_pixels = min_tokens * factor * factor
    max_pixels = max_tokens * factor * factor

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # Downscale if exceeding the per-image max
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor

    # Enforce per-image minimum
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return (h_bar, w_bar)
