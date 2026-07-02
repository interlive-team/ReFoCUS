"""ReFoCUS blur ablation with HuggingFace-native LLaVA-OneVision.

frame_set per sample is {"indices": list[int], "blur": list[bool]}; flagged frames are
gaussian-blurred (sigma=blur_sigma) before being fed to the model.
"""

import numpy as np
from lmms_eval.api.registry import register_model
from refocus_eval.model_utils.gaussian_blur import apply_gaussian_blur
from refocus_eval.model_utils.llava_ov_hf_base import LlavaOVHFBase, load_frames_native


@register_model("LLAVAONEVISION_FRAME_SELECTION_BLUR")
class LLAVAONEVISION_FRAME_SELECTION_BLUR(LlavaOVHFBase):
    def __init__(self, *args, blur_sigma: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.blur_sigma = float(blur_sigma)

    def _frames_for(self, video_path, frame_set):
        indices = frame_set["indices"]
        blur_flags = np.asarray(frame_set["blur"], dtype=bool)
        assert len(blur_flags) == len(indices)
        frames_np = np.ascontiguousarray(load_frames_native(video_path, indices))
        if self.blur_sigma > 0.0 and blur_flags.any():
            frames_np[blur_flags] = apply_gaussian_blur(
                frames=frames_np[blur_flags], sigma=(self.blur_sigma, self.blur_sigma)
            )
        return frames_np
