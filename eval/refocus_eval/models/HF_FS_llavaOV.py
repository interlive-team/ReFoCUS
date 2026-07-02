"""ReFoCUS frame-selection evaluation with HuggingFace-native LLaVA-OneVision.

Use an HF-format checkpoint, e.g. llava-hf/llava-onevision-qwen2-7b-ov-hf.
frame_set per sample is a flat list of selected frame indices.
"""

from lmms_eval.api.registry import register_model
from refocus_eval.model_utils.llava_ov_hf_base import LlavaOVHFBase


@register_model("LLAVAONEVISION_FRAME_SELECTION")
class LLAVAONEVISION_FRAME_SELECTION(LlavaOVHFBase):
    pass
