import os
import warnings

from .siglip_encoder import SigLipVisionTower

warnings.filterwarnings("ignore")


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    if "siglip" in vision_tower:
        return SigLipVisionTower(
            vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs
        )
    raise ValueError(f"Unknown vision tower: {vision_tower}")
